"""
VisiGuard Prediction / Inference
=================================
Two operating modes:

  1. Single-image mode  – classify one face image file from disk.
  2. Webcam mode        – real-time face detection + recognition via OpenCV.

Usage
-----
  # Classify a single image:
  python predict.py --image path/to/face.jpg

  # Real-time webcam (press Q to quit):
  python predict.py --webcam

  # Programmatic (import as module):
  from predict import VisiGuardPredictor
  p = VisiGuardPredictor()
  result = p.predict_image("face.jpg")
  print(result)   # → {"identity": "Colin_Powell", "confidence": 0.97}
"""

import argparse
import os
import numpy as np
import cv2
import tensorflow as tf

import config
import utils

logger = utils.get_logger()


# ─────────────────────────────────────────────
# Predictor class
# ─────────────────────────────────────────────

class VisiGuardPredictor:
    """
    Encapsulates model loading, preprocessing, and prediction.
    Load once; call predict_image() or predict_frame() many times.
    """

    def __init__(self,
                 model_path: str = config.CHECKPOINT_PATH,
                 encoder_path: str = config.LABEL_ENCODER_PATH):
        """
        Load model and label encoder from disk.

        Parameters
        ----------
        model_path   : path to saved .keras model file
        encoder_path : path to pickled sklearn LabelEncoder
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train.py first."
            )
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"Label encoder not found at {encoder_path}. Run train.py first."
            )

        logger.info(f"Loading model: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

        logger.info(f"Loading label encoder: {encoder_path}")
        self.le = utils.load_pickle(encoder_path)
        self.class_names = list(self.le.classes_)
        self.num_classes = len(self.class_names)
        logger.info(f"Ready — {self.num_classes} identities loaded.")

    # ── Low-level: numpy array → prediction ───

    def _preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Convert a BGR OpenCV image (any size) to a model-ready batch tensor.

        Steps:
          BGR → RGB  (OpenCV reads BGR by default)
          Resize     to IMAGE_SIZE
          Cast       to float32  (EfficientNet takes [0,255])
          Expand     batch dimension
        """
        rgb   = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, config.IMAGE_SIZE[::-1])  # cv2: (W, H)
        tensor  = resized.astype(np.float32)
        return np.expand_dims(tensor, axis=0)               # (1, H, W, 3)

    def _decode_prediction(self, probs: np.ndarray) -> dict:
        """
        Convert softmax probability vector to a human-readable result.

        Returns
        -------
        dict with keys:
          identity    : str  – predicted name (or "Unknown")
          confidence  : float – probability of the top class
          top3        : list of (identity, confidence) for top-3
        """
        top3_idx  = np.argsort(probs)[::-1][:3]
        top3      = [(self.class_names[i], float(probs[i])) for i in top3_idx]
        top_conf  = top3[0][1]
        top_name  = top3[0][0] if top_conf >= config.CONFIDENCE_THRESHOLD \
                    else "Unknown"

        return {
            "identity":   top_name,
            "confidence": top_conf,
            "top3":       top3,
        }

    # ── Public: predict from a file path ──────

    def predict_image(self, image_path: str) -> dict:
        """
        Classify a face image stored on disk.

        Parameters
        ----------
        image_path : path to a .jpg / .png image

        Returns
        -------
        dict with keys: identity, confidence, top3
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Could not decode image: {image_path}")

        batch  = self._preprocess(bgr)
        probs  = self.model.predict(batch, verbose=0)[0]   # (num_classes,)
        result = self._decode_prediction(probs)

        logger.info(
            f"Prediction → {result['identity']} "
            f"({result['confidence']*100:.1f}%)"
        )
        logger.info(
            "  Top-3: " +
            ", ".join(f"{n} {c*100:.1f}%" for n, c in result["top3"])
        )
        return result

    # ── Public: predict from a raw OpenCV frame ─

    def predict_frame(self, bgr_frame: np.ndarray) -> dict:
        """
        Classify a face region from a video frame (already cropped to face).
        Same interface as predict_image() but accepts a numpy array.
        """
        batch  = self._preprocess(bgr_frame)
        probs  = self.model.predict(batch, verbose=0)[0]
        return self._decode_prediction(probs)


# ─────────────────────────────────────────────
# Webcam real-time recognition
# ─────────────────────────────────────────────

def run_webcam(predictor: VisiGuardPredictor) -> None:
    """
    Open the default webcam and run face detection + recognition in real time.

    Face detection uses OpenCV's Haar Cascade (CPU-friendly, no extra deps).
    Each detected face region is classified by VisiGuardPredictor.

    Press Q to quit.
    """
    # Load Haar Cascade from OpenCV's bundled data
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade  = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam. Check camera permissions.")
        return

    logger.info("Webcam started. Press Q to quit.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Run inference every 3rd frame to keep display smooth
        if frame_count % 3 == 0:
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces  = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            for (x, y, w, h) in faces:
                # Crop detected face region
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue

                result     = predictor.predict_frame(face_crop)
                identity   = result["identity"]
                confidence = result["confidence"]

                # Choose bounding-box colour: green=known, red=unknown
                colour = (0, 200, 0) if identity != "Unknown" else (0, 0, 220)
                label  = f"{identity}  {confidence*100:.0f}%"

                cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
                cv2.putText(
                    frame, label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2
                )

        cv2.imshow("VisiGuard – Real-time Recognition (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam closed.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="VisiGuard – Face Recognition Inference"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str,
                       help="Path to a face image file to classify.")
    group.add_argument("--webcam", action="store_true",
                       help="Run real-time recognition via webcam.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predictor = VisiGuardPredictor()

    if args.image:
        result = predictor.predict_image(args.image)
        print("\n──────────────────────────────────")
        print(f"  Identity   : {result['identity']}")
        print(f"  Confidence : {result['confidence']*100:.1f}%")
        print("  Top-3 predictions:")
        for name, conf in result["top3"]:
            print(f"    {name:<30} {conf*100:.2f}%")
        print("──────────────────────────────────\n")

    elif args.webcam:
        run_webcam(predictor)
