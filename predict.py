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
        Load model, label encoder, and ArcFace target weights from disk.

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

        # 🔥 CRITICAL FIX: Load ArcFace weights to prevent dynamic batch size tracking exceptions
        w_path = os.path.join(config.MODEL_DIR, "arcface_weights.npy")
        if not os.path.exists(w_path):
            raise FileNotFoundError(f"Missing ArcFace target weights matrix at {w_path}")
        
        W_matrix = np.load(w_path)
        W_tensor = tf.convert_to_tensor(W_matrix, dtype=tf.float32)
        self.W_norm = tf.nn.l2_normalize(W_tensor, axis=0)
        self.scale = getattr(config, "ARC_SCALE", 64.0)
        
        logger.info(f"Ready — {self.num_classes} identities loaded.")

    # ── Low-level: numpy array → prediction ───

    def _preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Convert a BGR OpenCV image (any size) to a model-ready batch tensor.
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, config.IMAGE_SIZE[::-1])  # cv2: (W, H)
        tensor = resized.astype(np.float32)
        return np.expand_dims(tensor, axis=0)               # (1, H, W, 3)

    def _decode_prediction(self, probs: np.ndarray) -> dict:
        """
        Convert hyperspherical similarity probabilities to a structured result.
        """
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [(self.class_names[i], float(probs[i])) for i in top3_idx]
        top_conf = top3[0][1]
        top_name = top3[0][0]

        # Use an explicit default fallback check if config parameter doesn't exist
        conf_threshold = getattr(config, "CONFIDENCE_THRESHOLD", 0.45)
        if top_conf < conf_threshold:
            top_name = "Unknown"

        return {
            "identity":   top_name,
            "confidence": top_conf,
            "top3":       top3,
        }

    # ── Public: predict from a file path ──────

    def predict_image(self, image_path: str) -> dict:
        """
        Classify a face image stored on disk.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Could not decode image: {image_path}")

        return self.predict_frame(bgr)

    # ── Public: predict from a raw OpenCV frame ─

    def predict_frame(self, bgr_frame: np.ndarray) -> dict:
        """
        Classify a face region using mathematical hyperspherical transformations.
        """
        batch = self._preprocess(bgr_frame)
        
        # 🔥 FIX: Manually compute logits via metric weights to bypass Keras compilation layer variations
        embeddings = self.model(batch, training=False)
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        
        logits = tf.matmul(embeddings, self.W_norm) * self.scale
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        return self._decode_prediction(probs)


# ─────────────────────────────────────────────
# Webcam real-time recognition
# ─────────────────────────────────────────────

def run_webcam(predictor: VisiGuardPredictor) -> None:
    """
    Open the default webcam and run face detection + recognition in real time.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam. Check camera permissions.")
        return

    logger.info("Webcam started. Press Q to quit.")
    
    # Track persistent faces and properties to keep frame execution lightweight
    current_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 FIX: Update detection vectors periodically, but draw bounding boxes on EVERY frame 
        # to completely eliminate structural jitter and interface lag.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        
        if len(detected) > 0:
            current_faces = detected

        for (x, y, w, h) in current_faces:
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            result = predictor.predict_frame(face_crop)
            identity = result["identity"]
            confidence = result["confidence"]

            colour = (0, 200, 0) if identity != "Unknown" else (0, 0, 220)
            label = f"{identity.replace('_', ' ')} ({confidence*100:.0f}%)"

            cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
            cv2.putText(
                frame, label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2
            )

        cv2.imshow("VisiGuard – Real-time Recognition (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam closed successfully.")


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