"""
inference.py — Three inference modes:

  compare   — Compare two face images; return similarity + verdict.
  identify  — Detect + identify all faces in a photo against a gallery.
  webcam    — Live webcam recognition (Colab-compatible with JS capture).

Usage:
    python inference.py --mode compare --img1 alice.jpg --img2 bob.jpg
    python inference.py --mode identify --img group.jpg --gallery data/gallery/
    python inference.py --mode webcam --gallery data/gallery/
"""

import os
import sys
import argparse
import json
import time

import numpy as np
import cv2
import tensorflow as tf

import config
from detector import FaceDetector, get_detector
from utils    import (
    cosine_similarity,
    load_image_for_inference,
    load_class_catalogue_safe,
    build_gallery_embeddings,
)

# ── פונקציית עזר לנרמול (פותרת את שגיאת ה-Deserialization) ──────────────

def _l2_norm(x):
    return tf.math.l2_normalize(x, axis=1)

# ── Embedding model singleton ──────────────────────────────────────────────

_embedding_model = None


def get_embedding_model(path: str = config.BEST_EMBEDDING_MODEL):
    global _embedding_model
    if _embedding_model is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Embedding model not found at {path}. "
                "Please run train.py first."
            )
        print(f"[inference] Loading model from {path}…")
        
        # עדכון מילון ה-custom_objects עם פונקציית הנרמול החסרה
        custom_objs = {
            "ArcFaceLayer": __import__("arcface").ArcFaceLayer,
            "_l2_norm": _l2_norm
        }
        
        _embedding_model = tf.keras.models.load_model(
            path,
            compile=False,
            custom_objects=custom_objs,
        )
        print("[inference] Model ready.")
    return _embedding_model


def embed_face_crop(face_crop_bgr: np.ndarray, model) -> np.ndarray:
    """
    Preprocess a 112×112 BGR face crop and extract its 512-d embedding.

    Args:
        face_crop_bgr: (112, 112, 3) uint8 BGR array.
        model:         Keras embedding model.

    Returns:
        embedding: (512,) float32 L2-normalised vector.
    """
    # Convert BGR → RGB, scale to [0, 255] float32
    img_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    batch   = np.expand_dims(img_rgb, 0)           # (1, 112, 112, 3)
    emb     = model.predict(batch, verbose=0)[0]   # (512,)
    # Re-normalise (model outputs L2-normed, but let's be safe)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)


# ── Mode: compare ──────────────────────────────────────────────────────────

def compare_two_images(img1_path: str, img2_path: str) -> dict:
    """
    Compare two face images and return a structured result.

    If multiple faces are detected in an image, the largest/most-confident
    face is used.
    """
    model    = get_embedding_model()
    detector = get_detector()

    # הגדרת סף ההחלטה האופטימלי מהאימון של VisiGuard
    OPTIMAL_THRESHOLD = 0.1974

    results = {}
    for label, path in [("img1", img1_path), ("img2", img2_path)]:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {path}")

        det = detector.detect_largest(img_bgr)
        if det is None:
            raise RuntimeError(
                f"No face detected in {path}. "
                "Try lowering FACE_CONF_THRESHOLD in config.py."
            )
        results[label] = {
            "embedding": embed_face_crop(det["face_crop"], model),
            "detection": det,
        }

    emb1 = results["img1"]["embedding"]
    emb2 = results["img2"]["embedding"]
    sim  = float(np.dot(emb1, emb2))   # cosine sim (already L2-normed)

    same_person = sim >= OPTIMAL_THRESHOLD
    
    # Confidence: scale similarity to [0, 1] for the output range
    if same_person:
        conf = min(1.0, (sim - OPTIMAL_THRESHOLD) / 0.30 + 0.5)
    else:
        conf = min(1.0, (OPTIMAL_THRESHOLD - sim) / 0.30 + 0.5)
    conf = max(0.0, conf)

    verdict = {
        "similarity":    round(sim,  4),
        "same_person":   same_person,
        "verdict":       "Same Person ✓" if same_person else "Different Person ✗",
        "confidence":    round(conf * 100, 1),
        "threshold":     OPTIMAL_THRESHOLD,
    }

    # Print nice report
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"   VisiGuard Face Comparison Result")
    print(sep)
    print(f"  Similarity (Cosine) : {verdict['similarity']:.4f}")
    print(f"  Decision Threshold  : {verdict['threshold']:.4f}")
    print(f"  Verdict             : {verdict['verdict']}")
    print(f"  Confidence Score    : {verdict['confidence']}%")
    print(sep)

    return verdict


# ── Mode: identify ─────────────────────────────────────────────────────────

class FaceIdentifier:
    """
    Identify faces in an image against a gallery of known identities.
    """

    def __init__(
        self,
        gallery_dir:   str   = config.GALLERY_DIR,
        threshold:     float = 0.1974,  # שימוש בסף האופטימלי כברירת מחדל
        model_path:    str   = config.BEST_EMBEDDING_MODEL,
    ):
        self.threshold = threshold
        self.model     = get_embedding_model(model_path)
        self.detector  = get_detector()

        # Build gallery
        print(f"[identifier] Building gallery from {gallery_dir}…")
        self.gallery_names, self.gallery_embeddings = build_gallery_embeddings(
            gallery_dir = gallery_dir,
            model       = self.model,
            detector    = self.detector,
        )
        print(f"[identifier] Gallery ready: {len(self.gallery_names)} identities, "
              f"{len(self.gallery_embeddings)} embeddings.")

    def identify(self, image_bgr: np.ndarray) -> list[dict]:
        detections = self.detector.detect(image_bgr)
        if not detections:
            return []

        identified = []
        for det in detections:
            emb = embed_face_crop(det["face_crop"], self.model)

            if len(self.gallery_embeddings) == 0:
                identity   = "Unknown"
                best_sim   = 0.0
            else:
                gallery_mat = np.array(self.gallery_embeddings)  # (M, 512)
                sims        = gallery_mat @ emb                   # (M,)
                best_idx    = int(np.argmax(sims))
                best_sim    = float(sims[best_idx])
                identity    = (
                    self.gallery_names[best_idx]
                    if best_sim >= self.threshold else "Unknown"
                )

            identified.append({
                "bbox":       det["bbox"],
                "confidence": det["confidence"],
                "identity":   identity,
                "similarity": round(best_sim, 4),
                "detection":  det,
            })

        return identified

    def identify_and_draw(
        self,
        image_bgr: np.ndarray,
        output_path: str = None,
    ) -> np.ndarray:
        results = self.identify(image_bgr)
        if not results:
            print("[identifier] No faces detected.")
            return image_bgr

        labels = [
            f"{r['identity']} ({r['similarity']:.2f})"
            for r in results
        ]
        vis = self.detector.draw_detections(
            image_bgr,
            [r["detection"] for r in results],
            labels=labels,
        )

        if output_path:
            cv2.imwrite(output_path, vis)
            print(f"[identifier] Result saved to {output_path}")

        # Print table
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"   Identification Results  ({len(results)} face(s))")
        print(sep)
        for i, r in enumerate(results):
            print(f"  Face {i+1}: {r['identity']:<25} sim={r['similarity']:.4f}  "
                  f"det={r['confidence']:.2f}")
        print(sep)

        return vis


# ── Mode: webcam (Colab-compatible) ────────────────────────────────────────

COLAB_CAPTURE_JS = """
(async () => {
  const div = document.createElement('div');
  const video = document.createElement('video');
  const capture = document.createElement('button');
  const canvas = document.createElement('canvas');

  capture.textContent = '📸 Capture Frame';
  capture.style.cssText = 'font-size:1.2rem;padding:0.4rem 1rem;margin:0.5rem;cursor:pointer';
  div.appendChild(video);
  div.appendChild(document.createElement('br'));
  div.appendChild(capture);
  div.appendChild(canvas);
  document.body.appendChild(div);

  const stream = await navigator.mediaDevices.getUserMedia({video: true});
  video.srcObject = stream;
  await video.play();
  google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

  return new Promise(resolve => {
    capture.onclick = () => {
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      const data = canvas.toDataURL('image/jpeg', 0.9);
      stream.getTracks().forEach(t => t.stop());
      div.remove();
      resolve(data.split(',')[1]);
    };
  });
})()
"""


def run_webcam_colab(gallery_dir: str = config.GALLERY_DIR):
    try:
        from IPython.display import display, Javascript
        from google.colab.output import eval_js
        import base64
    except ImportError:
        print("[webcam] Not running in Colab. Use OpenCV webcam mode instead.")
        run_webcam_opencv(gallery_dir)
        return

    identifier = FaceIdentifier(gallery_dir=gallery_dir)

    print("[webcam] Starting Colab webcam capture. "
          "Click 'Capture Frame' to identify faces.")

    while True:
        try:
            b64_data = eval_js(COLAB_CAPTURE_JS)
        except KeyboardInterrupt:
            break

        img_bytes = base64.b64decode(b64_data)
        img_arr   = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("[webcam] Could not decode captured image.")
            continue

        vis = identifier.identify_and_draw(frame)

        _, encoded = cv2.imencode(".jpg", vis)
        b64_img = __import__("base64").b64encode(encoded).decode("utf-8")
        display(__import__("IPython.display", fromlist=["HTML"]).HTML(
            f'<img src="data:image/jpeg;base64,{b64_img}" style="max-width:100%"/>'
        ))

        again = input("\nCapture another? [y/N]: ").strip().lower()
        if again != "y":
            break


def run_webcam_opencv(gallery_dir: str = config.GALLERY_DIR, camera_idx: int = 0):
    identifier = FaceIdentifier(gallery_dir=gallery_dir)
    cap        = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print(f"[webcam] Could not open camera {camera_idx}.")
        return

    print("[webcam] Webcam running. Press Q to quit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis = identifier.identify_and_draw(frame)

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Face recognition inference.")
    p.add_argument("--mode",    required=True,
                   choices=["compare", "identify", "webcam"])
    p.add_argument("--img1",    default=None, help="(compare mode) First image path.")
    p.add_argument("--img2",    default=None, help="(compare mode) Second image path.")
    p.add_argument("--img",     default=None, help="(identify mode) Input image path.")
    p.add_argument("--gallery", default=config.GALLERY_DIR)
    p.add_argument("--output",  default=None, help="(identify mode) Output image path.")
    p.add_argument("--model",   default=config.BEST_EMBEDDING_MODEL)
    p.add_argument("--colab",   action="store_true",
                   help="Use Colab JS webcam capture instead of OpenCV.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "compare":
        if not args.img1 or not args.img2:
            print("Error: --img1 and --img2 required for compare mode.")
            sys.exit(1)
        compare_two_images(args.img1, args.img2)

    elif args.mode == "identify":
        if not args.img:
            print("Error: --img required for identify mode.")
            sys.exit(1)
        identifier = FaceIdentifier(gallery_dir=args.gallery, model_path=args.model)
        img_bgr    = cv2.imread(args.img)
        if img_bgr is None:
            print(f"Error: could not read {args.img}")
            sys.exit(1)
        out_path = args.output or os.path.join(
            config.OUTPUT_DIR,
            "identified_" + os.path.basename(args.img),
        )
        identifier.identify_and_draw(img_bgr, output_path=out_path)

    elif args.mode == "webcam":
        if args.colab:
            run_webcam_colab(gallery_dir=args.gallery)
        else:
            run_webcam_opencv(gallery_dir=args.gallery)


if __name__ == "__main__":
    main()