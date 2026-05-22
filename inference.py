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
import base64

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
        print(f"[inference] Loading model from {path}...")
        
        # טעינה בטוחה עם האובייקטים המותאמים אישית של הארכיטקטורה שלך
        custom_objs = {
            "ArcFaceLayer": __import__("arcface").ArcFaceLayer,
            "_l2_norm": _l2_norm
        }
        _embedding_model = tf.keras.models.load_model(
            path, 
            compile=False, 
            custom_objects=custom_objs
        )
        print("[inference] Model loaded successfully.")
    return _embedding_model


def embed_face_crop(face_crop_bgr: np.ndarray, model) -> np.ndarray:
    """Extract 512-dim L2-normalised embedding from a face crop."""
    img = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, config.IMAGE_SIZE)
    img = img.astype(np.float32) / 255.0
    
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    
    embedding = model(img_tensor, training=False)
    return embedding.numpy()[0]


def compute_calibrated_confidence(cosine_sim: float) -> float:
    """
    מחשב את אחוז הביטחון (Confidence) על בסיס פונקציה סיגמואידית מיושרת לסף.
    מונע ממשחקי דמיון גבוליים (False Positives) לקפוץ ל-75% ביטחון ומאזן את ה-EER.
    """
    t = config.SAME_PERSON_THRESHOLD
    # מקדם שיפוע (k=12.0) המבטיח החלקה חדה ומדויקת סביב נקודת ההכרעה
    k = 12.0 
    conf = 1.0 / (1.0 + np.exp(-k * (cosine_sim - t)))
    return float(conf)


def compare_two_images(img1_path: str, img2_path: str, model_path: str = config.BEST_EMBEDDING_MODEL):
    """Compare two single images and print similarity/verdict."""
    model = get_embedding_model(model_path)
    detector = get_detector()

    # קריאה חסינת רווחים וסוגריים בנתיבי הקבצים בתוך סביבת ה-Inference
    def read_image_safe(p):
        try:
            with open(p, "rb") as f:
                arr = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    im1 = read_image_safe(img1_path)
    im2 = read_image_safe(img2_path)

    if im1 is None or im2 is None:
        print(f"Error: Could not read one or both images: {img1_path}, {img2_path}")
        sys.exit(1)

    dets1 = detector.detect_faces(im1)
    dets2 = detector.detect_faces(im2)

    if not dets1 or not dets2:
        print("Warning: Face detection failed in one of the images. Using full center crop fallback.")
        def fallback_crop(img):
            h, w = img.shape[:2]
            sz = min(h, w)
            crop = img[(h-sz)//2:(h+sz)//2, (w-sz)//2:(w+sz)//2]
            return cv2.resize(crop, config.IMAGE_SIZE)
        crop1 = dets1[0]["face_crop"] if dets1 else fallback_crop(im1)
        crop2 = dets2[0]["face_crop"] if dets2 else fallback_crop(im2)
    else:
        crop1 = dets1[0]["face_crop"]
        crop2 = dets2[0]["face_crop"]

    emb1 = embed_face_crop(crop1, model)
    emb2 = embed_face_crop(crop2, model)

    sim = float(np.dot(emb1, emb2))
    confidence = compute_calibrated_confidence(sim)
    is_match = sim >= config.SAME_PERSON_THRESHOLD

    print("\n" + "="*60)
    print(f" Verification Verdict : {'Same Person ✓' if is_match else 'Impostor Match ✗'}")
    print(f" Raw Cosine Similarity: {sim:.4f}")
    print(f" Calibrated Confidence : {confidence * 100:.2f}%")
    print("="*60 + "\n")

    return {"similarity": sim, "confidence": confidence, "match": is_match}


class FaceIdentifier:
    """Detects and identifies all faces in an image against a saved gallery."""
    def __init__(self, gallery_dir: str = config.GALLERY_DIR, model_path: str = config.BEST_EMBEDDING_MODEL):
        self.detector = get_detector()
        self.model = get_embedding_model(model_path)
        self.gallery_embeddings, self.class_names = build_gallery_embeddings(gallery_dir, self.model)
        print(f"[inference] FaceIdentifier ready with {len(self.class_names)} identities.")

    def identify_and_draw(self, img_bgr: np.ndarray, output_path: str = None) -> np.ndarray:
        if img_bgr is None:
            return np.zeros((*config.IMAGE_SIZE, 3), dtype=np.uint8)

        detections = self.detector.detect_faces(img_bgr)
        vis = img_bgr.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = det["face_crop"]
            if crop.size == 0: 
                continue

            emb = embed_face_crop(crop, self.model)

            if len(self.gallery_embeddings) == 0:
                label, conf_score = "Unknown", 0.0
            else:
                similarities = np.dot(self.gallery_embeddings, emb)
                best_idx = np.argmax(similarities)
                best_sim = similarities[best_idx]

                if best_sim >= config.SAME_PERSON_THRESHOLD:
                    label = self.class_names[best_idx]
                    conf_score = compute_calibrated_confidence(best_sim)
                else:
                    label = "Unknown"
                    conf_score = 1.0 - compute_calibrated_confidence(best_sim)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            caption = f"{label} ({conf_score*100:.1f}%)"
            cv2.putText(vis, caption, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, vis)
        return vis


def run_webcam_colab(gallery_dir: str):
    """Runs JavaScript-based live webcam acquisition inside Google Colab notebooks with real-time inference loop."""
    from IPython.display import display, Javascript, Image
    try:
        from google.colab.output import eval_js, clear_output
    except ImportError:
        print("Error: webcam mode requires a live Google Colab environment runtime.")
        return

    identifier = FaceIdentifier(gallery_dir=gallery_dir)

    js_code = """
    var video;
    var div;
    var stream;
    var canvas;

    async function startWebcam() {
      div = document.createElement('div');
      div.style.border = '2px solid #00E664';
      div.style.padding = '10px';
      div.style.width = '640px';
      div.style.margin = '0 auto';
      
      const title = document.createElement('h3');
      title.textContent = 'VisiGuard Live Webcam Feed';
      title.style.color = '#00E664';
      title.style.fontFamily = 'monospace';
      title.style.textAlign = 'center';
      div.appendChild(title);

      video = document.createElement('video');
      video.style.display = 'block';
      video.style.width = '100%';
      video.style.transform = 'scaleX(-1)';
      
      stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}});
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 480;

      document.body.appendChild(div);
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
    }

    async function captureFrame() {
      if (!video || !stream) return null;
      var ctx = canvas.getContext('2d');
      ctx.save();
      ctx.translate(640, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, 640, 480);
      ctx.restore();
      return canvas.toDataURL('image/jpeg', 0.85);
    }

    function stopWebcam() {
      if (stream) {
        stream.getVideoTracks()[0].stop();
      }
      if (div) {
        div.remove();
      }
    }
    """
    
    print("[webcam] Initializing live JavaScript webcam video feed hooks...")
    display(Javascript(js_code))
    eval_js('startWebcam()')
    
    print("[webcam] Live recognition active. Intercept or stop cell execution to terminate.")
    
    try:
        while True:
            frame_data = eval_js('captureFrame()')
            if not frame_data:
                break
                
            head, data = frame_data.split(',')
            img_bytes = base64.b64decode(data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                continue
                
            vis = identifier.identify_and_draw(img_bgr)
            
            _, encoded_img = cv2.imencode('.jpg', vis)
            encoded_base64 = base64.b64encode(encoded_img).decode('utf-8')
            
            clear_output(wait=True)
            display(Image(data=base64.b64decode(encoded_base64)))
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        print("\n[webcam] Live stream terminated by user.")
    finally:
        eval_js('stopWebcam()')


def parse_args():
    p = argparse.ArgumentParser(description="VisiGuard Evaluation & Inference Engine")
    p.add_argument("--mode",     choices=["compare", "identify", "webcam"], required=True)
    p.add_argument("--img1",     help="Path to first image (compare mode only)")
    p.add_argument("--img2",     help="Path to second image (compare mode only)")
    p.add_argument("--img",      help="Path to photo to identify (identify mode only)")
    p.add_argument("--gallery",  default=config.GALLERY_DIR, help="Path to gallery directory")
    p.add_argument("--model",    default=config.BEST_EMBEDDING_MODEL, help="Path to embedding model")
    p.add_argument("--output",   help="Optional explicit output path for results")
    p.add_argument("--colab",    action="store_true", help="Use Colab JS webcam capture instead of OpenCV.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "compare":
        if not args.img1 or not args.img2:
            print("Error: --img1 and --img2 required for compare mode.")
            sys.exit(1)
        compare_two_images(args.img1, args.img2, args.model)

    elif args.mode == "identify":
        if not args.img:
            print("Error: --img required for identify mode.")
            sys.exit(1)
            
        identifier = FaceIdentifier(gallery_dir=args.gallery, model_path=args.model)
        
        try:
            with open(args.img, "rb") as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[DEBUG] Failed to open/decode file at: {args.img}. Error: {e}")
            img_bgr = None

        if img_bgr is None:
            print(f"Error: could not read {args.img}")
            sys.exit(1)
            
        out_path = args.output or os.path.join(
            config.OUTPUT_DIR,
            "identified_" + os.path.basename(args.img),
        )
        identifier.identify_and_draw(img_bgr, out_path)
        print(f"[✓] Identification complete. Output saved to: {out_path}")

    elif args.mode == "webcam":
        run_webcam_colab(args.gallery)

if __name__ == "__main__":
    main()