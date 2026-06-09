"""
aligner.py — Face alignment using eye landmarks.

Why alignment matters:
  Even with perfect detection, a tilted face reduces ArcFace accuracy by
  5-10% on real-world data. Alignment rotates the crop so both eyes are
  at a canonical horizontal position, giving the CNN a consistent input
  regardless of head roll.

Pipeline per detected face:
  1. Receive bounding-box crop + 5 facial keypoints from YOLOv8.
  2. Compute the angle between the two eye centres.
  3. Compute the scale so the inter-eye distance matches the desired
     output width fraction.
  4. Apply an affine warp (getRotationMatrix2D) centred on the eyes.
  5. Return the aligned, resized crop.

If keypoints are unavailable (low-quality detection), falls back to
a plain centre-crop + resize without rotation.
"""

import cv2
import numpy as np
import config


class FaceAligner:
    """
    Align a face crop using eye landmark positions.

    Args:
        output_size:      (H, W) of the output face patch.
        desired_left_eye: (x_frac, y_frac) position of the left eye
                          in the output image. Default (0.35, 0.40).
    """

    def __init__(
        self,
        output_size: tuple   = config.IMAGE_SIZE,
        desired_left_eye: tuple  = config.DESIRED_LEFT_EYE,
        desired_right_eye: tuple = config.DESIRED_RIGHT_EYE,
    ):
        self.output_size      = output_size          # (H, W)
        self.desired_left_eye = desired_left_eye
        self.desired_right_eye = desired_right_eye

        # Desired inter-eye distance in output pixels.
        desired_dist = (desired_right_eye[0] - desired_left_eye[0])
        self.desired_dist_px = desired_dist * output_size[1]

    # ── Public API ─────────────────────────────────────────────────────────

    def align(
        self,
        image: np.ndarray,
        left_eye: tuple,
        right_eye: tuple,
    ) -> np.ndarray:
        """
        Align a face given the full image and both eye coordinates.
        """
        left_eye  = np.array(left_eye,  dtype=np.float32)
        right_eye = np.array(right_eye, dtype=np.float32)

        # ── 1. Rotation angle ──────────────────────────────────────────────
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # ── 2. Scale ───────────────────────────────────────────────────────
        dist  = np.linalg.norm(right_eye - left_eye)
        scale = self.desired_dist_px / max(dist, 1e-5)

        # ── 3. Centre of rotation = midpoint between eyes ──────────────────
        eyes_centre = (
            float((left_eye[0] + right_eye[0]) / 2.0),
            float((left_eye[1] + right_eye[1]) / 2.0),
        )

        # ── 4. Rotation + scale matrix ─────────────────────────────────────
        M = cv2.getRotationMatrix2D(eyes_centre, angle, scale)

        # ── 5. Adjust translation so the eyes land at the desired position ─
        # Output width is index [1], Height is index [0]
        tX = self.output_size[1] * 0.5
        tY = self.output_size[0] * self.desired_left_eye[1]
        
        M[0, 2] += (tX - eyes_centre[0])
        M[1, 2] += (tY - eyes_centre[1])

        # ── 6. Warp ────────────────────────────────────────────────────────
        (W, H) = (self.output_size[1], self.output_size[0])
        aligned = cv2.warpAffine(
            image, M, (W, H),
            flags       = cv2.INTER_CUBIC,
            borderMode  = cv2.BORDER_REPLICATE,
        )

        return aligned

    def align_from_bbox_and_kpts(
        self,
        image: np.ndarray,
        bbox: tuple,
        keypoints: np.ndarray,
    ) -> np.ndarray:
        """
        Convenience wrapper: crop from bbox first, then align using
        eye keypoints that are in the original image coordinate space.
        """
        h, w = image.shape[:2]
        
        if keypoints is not None and len(keypoints) >= 2:
            left_eye_pt  = keypoints[config.LEFT_EYE_IDX,  :2].astype(float)
            right_eye_pt = keypoints[config.RIGHT_EYE_IDX, :2].astype(float)

            # Auto-denormalize coordinates if they arrive as values between 0 and 1
            if left_eye_pt[0] <= 1.0 and left_eye_pt[1] <= 1.0:
                left_eye_pt[0] *= w
                left_eye_pt[1] *= h
            if right_eye_pt[0] <= 1.0 and right_eye_pt[1] <= 1.0:
                right_eye_pt[0] *= w
                right_eye_pt[1] *= h

            # Spatial boundary check
            valid = (
                left_eye_pt[0] > 0 and left_eye_pt[1] > 0 and
                right_eye_pt[0] > 0 and right_eye_pt[1] > 0 and
                left_eye_pt[0] < w and left_eye_pt[1] < h and
                right_eye_pt[0] < w and right_eye_pt[1] < h
            )
            if valid:
                return self.align(image, left_eye_pt, right_eye_pt)

        # Fallback: plain crop + resize
        return self.crop_and_resize(image, bbox)

    def crop_and_resize(
        self,
        image: np.ndarray,
        bbox: tuple,
    ) -> np.ndarray:
        """
        Fallback: crop bounding box and resize (no rotation).
        """
        x1, y1, x2, y2 = bbox
        h_img, w_img   = image.shape[:2]

        # Add 10% margin
        bw = x2 - x1
        bh = y2 - y1
        mx = int(bw * 0.10)
        my = int(bh * 0.10)

        x1 = max(0, int(x1) - mx)
        y1 = max(0, int(y1) - my)
        x2 = min(w_img, int(x2) + mx)
        y2 = min(h_img, int(y2) + my)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((*self.output_size, 3), dtype=np.uint8)

        return cv2.resize(crop, (self.output_size[1], self.output_size[0]),
                          interpolation=cv2.INTER_CUBIC)


# ── Module-level convenience function ─────────────────────────────────────

_default_aligner = None


def get_aligner() -> FaceAligner:
    """Return (and lazily create) the module-level default aligner."""
    global _default_aligner
    if _default_aligner is None:
        _default_aligner = FaceAligner()
    return _default_aligner


def align_face(
    image: np.ndarray,
    bbox: tuple,
    keypoints: np.ndarray | None = None,
) -> np.ndarray:
    """
    Module-level convenience wrapper.
    """
    aligner = get_aligner()
    
    if keypoints is not None:
        if hasattr(keypoints, 'cpu'):
            keypoints = keypoints.cpu().numpy()
            
        keypoints = np.squeeze(keypoints)
        
        if keypoints.ndim == 2 and len(keypoints) >= 2:
            return aligner.align_from_bbox_and_kpts(image, bbox, keypoints)
            
    return aligner.crop_and_resize(image, bbox)