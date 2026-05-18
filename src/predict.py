"""
predict.py
==========
Single-image deepfake prediction with full explainability output.

Usage (CLI):
    python predict.py --image path/to/face.jpg

Usage (Python):
    from predict import predict_and_explain
    result = predict_and_explain("path/to/face.jpg")

The result dict contains:
    label        : "REAL" or "FAKE"
    confidence   : float in [0, 1]
    raw_score    : raw sigmoid output (>= threshold → REAL)
    top_region   : most important anatomical face region
    region_scores: dict of all region importances
"""

import os
import argparse
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cswin_transformer import CUSTOM_OBJECTS, IMG_SIZE
from explainability import explain_image, build_figure

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — override with environment variables
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("outputs", "cswin_best.keras"))
OUT_DIR    = os.environ.get("OUT_DIR",    os.path.join("outputs", "predictions"))
THRESHOLD  = float(os.environ.get("THRESHOLD", "0.5"))
os.makedirs(OUT_DIR, exist_ok=True)

# ── OpenCV DNN face detector files ───────────────────────────────────────────
_DNN_PROTO = os.path.join(os.path.dirname(__file__), ".cache", "deploy.prototxt")
_DNN_MODEL = os.path.join(os.path.dirname(__file__), ".cache",
                           "res10_300x300_ssd_iter_140000.caffemodel")
_PROTO_URL = ("https://raw.githubusercontent.com/opencv/opencv/master/"
              "samples/dnn/face_detector/deploy.prototxt")
_MODEL_URL = ("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_"
              "face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")


# ══════════════════════════════════════════════════════════════════════════════
# FACE DETECTOR  (OpenCV DNN with Haar fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _download_dnn_files():
    os.makedirs(os.path.dirname(_DNN_PROTO), exist_ok=True)
    ok = True
    for path, url in [(_DNN_PROTO, _PROTO_URL), (_DNN_MODEL, _MODEL_URL)]:
        if not os.path.exists(path):
            try:
                print(f"  Downloading {os.path.basename(path)} ...")
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"  Download failed: {e}")
                ok = False
    return ok


class FaceDetectorCV:
    """OpenCV DNN face detector with Haar-cascade fallback."""

    def __init__(self):
        self._dnn_net = None
        self._haar    = None
        self._init_dnn()
        self._init_haar()

    def _init_dnn(self):
        if _download_dnn_files():
            try:
                self._dnn_net = cv2.dnn.readNetFromCaffe(_DNN_PROTO, _DNN_MODEL)
                print("  OpenCV DNN face detector loaded.")
            except Exception as e:
                print(f"  DNN load failed ({e}), using Haar fallback.")

    def _init_haar(self):
        p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(p)
        if self._haar.empty():
            self._haar = None

    def detect(self, img_rgb):
        """Return list of (x, y, w, h, confidence) sorted by area (largest first)."""
        faces = []
        if self._dnn_net is not None:
            h, w = img_rgb.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(img_rgb, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))
            self._dnn_net.setInput(blob)
            dets = self._dnn_net.forward()
            for i in range(dets.shape[2]):
                c = float(dets[0, 0, i, 2])
                if c < 0.5:
                    continue
                box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                fw, fh = x2 - x1, y2 - y1
                if fw > 0 and fh > 0:
                    faces.append((x1, y1, fw, fh, c))

        if not faces and self._haar is not None:
            gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            rects = self._haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, fw, fh) in (rects if len(rects) else []):
                faces.append((x, y, fw, fh, 0.8))

        faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        return faces


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path=MODEL_PATH):
    """Load CSWin checkpoint with all custom objects registered."""
    with keras.utils.custom_object_scope(CUSTOM_OBJECTS):
        m = keras.models.load_model(model_path, compile=False)
    # Warm-up: materialises weights so GradientTape can watch them
    _ = m(tf.zeros((1,) + IMG_SIZE + (3,), dtype=tf.float32), training=False)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image(image_path, face_detector):
    """
    Load image → detect face → resize → normalise to [0, 1].

    Returns:
        face_norm     : np.float32 (H, W, 3)
        orig_rgb      : np.uint8   (H, W, 3)
        bbox          : dict {x, y, w, h, conf} or None if fallback used
        used_fallback : bool
    """
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    orig_rgb     = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = orig_rgb.shape[:2]
    faces        = face_detector.detect(orig_rgb)
    bbox         = None
    used_fallback = False

    if faces:
        x, y, fw, fh, face_conf = faces[0]
        x2 = min(w_img, x + fw)
        y2 = min(h_img, y + fh)
        face_crop = orig_rgb[y:y2, x:x2]
        if face_crop.size == 0:
            faces = []

    if faces:
        face_norm = cv2.resize(face_crop, IMG_SIZE).astype(np.float32) / 255.0
        bbox      = {"x": x, "y": y, "w": fw, "h": fh, "conf": face_conf}
    else:
        face_norm     = cv2.resize(orig_rgb, IMG_SIZE).astype(np.float32) / 255.0
        used_fallback = True

    return face_norm, orig_rgb, bbox, used_fallback


# ══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_and_explain(image_path, model=None, face_detector=None,
                        threshold=THRESHOLD, save=True):
    """
    Full pipeline: preprocess → predict → explain → build figure → save.

    Args:
        image_path    : path to input image
        model         : pre-loaded Keras model (loaded fresh if None)
        face_detector : pre-loaded FaceDetectorCV (created fresh if None)
        threshold     : decision threshold (default 0.5)
        save          : save the explanation figure to OUT_DIR

    Returns dict with keys: label, confidence, raw_score, top_region, region_scores
    """
    if model is None:
        model = load_model()
    if face_detector is None:
        face_detector = FaceDetectorCV()

    # 1. Preprocess
    face_norm, orig_rgb, bbox, used_fallback = preprocess_image(
        image_path, face_detector)
    face_inp = np.expand_dims(face_norm, axis=0).astype(np.float32)

    # 2. Predict
    prob  = float(model(face_inp, training=False).numpy()[0, 0])
    label = "REAL" if prob >= threshold else "FAKE"
    conf  = prob if label == "REAL" else (1.0 - prob)
    print(f"\n  Verdict: {label}  ({conf*100:.1f}%)  [score={prob:.4f}]")

    # 3. Explain
    (ig_signed, ig_abs,
     smooth_signed, smooth_abs,
     occ_map, region_scores) = explain_image(model, face_inp)

    # 4. Build figure
    fig, top_region = build_figure(
        face_norm, orig_rgb, bbox, label, conf, prob,
        ig_signed, ig_abs, smooth_signed, smooth_abs,
        occ_map, region_scores, used_fallback, threshold)

    # 5. Save / display
    if save:
        basename  = os.path.splitext(os.path.basename(image_path))[0]
        out_path  = os.path.join(OUT_DIR, f"{basename}_explained.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Figure saved: {out_path}")
    plt.close(fig)

    # 6. Text summary
    print(f"\n{'='*60}")
    print(f"  PREDICTION")
    print(f"  Verdict        : {label}")
    print(f"  Confidence     : {conf*100:.1f}%")
    print(f"  Raw score      : {prob:.6f}  (>={threshold} → REAL)")
    print(f"{'─'*60}")
    print(f"  EXPLAINABILITY SUMMARY")
    print(f"  Method         : Integrated Gradients + SmoothGrad + Occlusion")
    print(f"  Most important : {top_region} ({region_scores[top_region]*100:.1f}%)")
    print(f"  Region breakdown:")
    for region, score in sorted(region_scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 30)
        print(f"    {region:<14s} {bar:<30s} {score*100:.1f}%")
    print(f"{'='*60}")

    return {"label": label, "confidence": conf, "raw_score": prob,
            "top_region": top_region, "region_scores": region_scores}


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSWin DeepFake Detector")
    parser.add_argument("--image",     required=True, help="Path to input image")
    parser.add_argument("--model",     default=MODEL_PATH, help="Path to .keras checkpoint")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Decision threshold (default 0.5)")
    parser.add_argument("--no-save",   action="store_true",
                        help="Skip saving the explanation figure")
    args = parser.parse_args()

    _model         = load_model(args.model)
    _face_detector = FaceDetectorCV()

    predict_and_explain(
        args.image,
        model=_model,
        face_detector=_face_detector,
        threshold=args.threshold,
        save=not args.no_save,
    )