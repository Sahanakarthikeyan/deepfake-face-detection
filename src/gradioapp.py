"""
gradio_app.py
=============
Gradio web UI for the CSWin DeepFake Detector.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT — rename this file to  gradio_app.py
Do NOT name it gradio.py — that shadows the library
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Quick start
-----------
1.  pip install -r requirements.txt
2.  Set the model path (one of):
      • put  cswin_best.keras  anywhere and set:
            Windows : set MODEL_PATH=C:\\full\\path\\to\\cswin_best.keras
            Linux   : export MODEL_PATH=/full/path/to/cswin_best.keras
      • OR place it at  outputs/cswin_best.keras  next to this file
3.  python gradio_app.py
4.  Open  http://127.0.0.1:7860  in your browser
"""

import os
import math
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gradio as gr

# ══════════════════════════════════════════════════════════════════════════════
# PORTABLE PATHS
# Every path is derived from this file's location — works on any machine.
# Override with env vars when needed.
# ══════════════════════════════════════════════════════════════════════════════
_HERE      = os.path.dirname(os.path.abspath(__file__))   # folder this file lives in
_ROOT      = os.path.dirname(_HERE)                        # one level up (repo root if in src/)

# If the file is NOT inside a src/ subfolder (e.g. placed at repo root),
# _ROOT == _HERE — the fallback paths still work fine.
_FALLBACK_MODEL = os.path.join(_ROOT,  "outputs", "cswin_best.keras")
_FALLBACK_OUT   = os.path.join(_ROOT,  "outputs", "gradio_out")
_FALLBACK_CACHE = os.path.join(_ROOT,  ".cache")

MODEL_PATH = os.environ.get("MODEL_PATH", _FALLBACK_MODEL)
OUT_DIR    = os.environ.get("OUT_DIR",    _FALLBACK_OUT)
CACHE_DIR  = os.environ.get("CACHE_DIR",  _FALLBACK_CACHE)

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DROP_PATH_RATE = 0.1
DROPOUT        = 0.1
IMG_SIZE       = (128, 128)
THRESHOLD      = 0.5
SHAP_N_BG      = 30
OCC_PATCH_SIZE = 8

# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE  (self-contained — identical to cswin_transformer.py)
# ══════════════════════════════════════════════════════════════════════════════

class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, min_lr, warmup_steps, total_steps, **kwargs):
        super().__init__(**kwargs)
        self.peak_lr = float(peak_lr); self.min_lr = float(min_lr)
        self.warmup_steps = float(warmup_steps); self.total_steps = float(total_steps)

    def __call__(self, step):
        step    = tf.cast(step, tf.float32)
        warmup  = self.peak_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        cos_arg = math.pi * (step - self.warmup_steps) / tf.maximum(
            self.total_steps - self.warmup_steps, 1.0)
        cosine  = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + tf.cos(cos_arg))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "min_lr": self.min_lr,
                "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}


class DropPath(layers.Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs); self.drop_prob = float(drop_prob)

    def call(self, x, training=False):
        if not training or self.drop_prob == 0.0:
            return x
        keep  = 1.0 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        return x * tf.math.floor(keep + tf.random.uniform(shape, dtype=x.dtype)) / keep

    def get_config(self):
        cfg = super().get_config(); cfg["drop_prob"] = self.drop_prob; return cfg


class PatchEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs); self.dim = dim
        self.proj = layers.Conv2D(dim, kernel_size=4, strides=4, padding="same", name="patch_proj")
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="patch_norm")

    def build(self, input_shape):
        H = int(input_shape[1]); W = int(input_shape[2])
        self.pos_embed = self.add_weight(
            name="pos_embed", shape=(1, H // 4, W // 4, self.dim),
            initializer="zeros", trainable=True, dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        return self.norm(self.proj(x) + tf.cast(self.pos_embed, x.dtype))

    def get_config(self):
        cfg = super().get_config(); cfg["dim"] = self.dim; return cfg


def h_split(x, s):
    B = tf.shape(x)[0]; H = tf.shape(x)[1]; W = tf.shape(x)[2]; C = x.shape[-1]
    return tf.reshape(tf.transpose(tf.reshape(x, [B, H//s, s, W, C]), [0,1,3,2,4]),
                      [B*(H//s), W*s, C])

def h_merge(x, B, H, W, C, s):
    return tf.reshape(tf.transpose(tf.reshape(x, [B, H//s, W, s, C]), [0,1,3,2,4]),
                      [B, H, W, C])

def v_split(x, s):
    B = tf.shape(x)[0]; H = tf.shape(x)[1]; W = tf.shape(x)[2]; C = x.shape[-1]
    return tf.reshape(tf.transpose(tf.reshape(x, [B, H, W//s, s, C]), [0,2,1,3,4]),
                      [B*(W//s), H*s, C])

def v_merge(x, B, H, W, C, s):
    return tf.reshape(tf.transpose(tf.reshape(x, [B, W//s, H, s, C]), [0,2,1,3,4]),
                      [B, H, W, C])


class StripeAttention(layers.Layer):
    def __init__(self, dim, num_heads, attn_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads; self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv   = layers.Dense(dim * 3, use_bias=True, name="qkv")
        self.proj  = layers.Dense(dim,     use_bias=True, name="proj")
        self.drop  = layers.Dropout(attn_drop)

    def call(self, x, training=False):
        B = tf.shape(x)[0]; N = tf.shape(x)[1]; C = x.shape[-1]
        qkv  = tf.transpose(
            tf.reshape(self.qkv(x), [B, N, 3, self.num_heads, self.head_dim]),
            [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.drop(
            tf.nn.softmax(tf.matmul(q, k, transpose_b=True) * self.scale, axis=-1),
            training=training)
        return self.proj(
            tf.reshape(tf.transpose(tf.matmul(attn, v), [0, 2, 1, 3]), [B, N, C]))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.num_heads * self.head_dim, "num_heads": self.num_heads})
        return cfg


class CSWinAttention(layers.Layer):
    def __init__(self, dim, num_heads, split_size, attn_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        assert dim % 2 == 0
        self.split_size = split_size; self.dim_half = dim // 2
        hh = max(1, num_heads // 2)
        self.attn_h = StripeAttention(self.dim_half, hh, attn_drop, name="attn_h")
        self.attn_v = StripeAttention(self.dim_half, hh, attn_drop, name="attn_v")
        self.lepe_h = layers.DepthwiseConv2D(3, padding="same", name="lepe_h")
        self.lepe_v = layers.DepthwiseConv2D(3, padding="same", name="lepe_v")

    def call(self, x, training=False):
        B = tf.shape(x)[0]; H = tf.shape(x)[1]; W = tf.shape(x)[2]; s = self.split_size
        x1, x2 = tf.split(x, 2, axis=-1)
        xh = h_merge(self.attn_h(h_split(x1, s), training=training),
                     B, H, W, self.dim_half, s) + self.lepe_h(x1)
        xv = v_merge(self.attn_v(v_split(x2, s), training=training),
                     B, H, W, self.dim_half, s) + self.lepe_v(x2)
        return tf.concat([xh, xv], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim_half * 2,
                    "num_heads": self.attn_h.num_heads * 2,
                    "split_size": self.split_size})
        return cfg


class CSWinBlock(layers.Layer):
    def __init__(self, dim, num_heads, split_size, mlp_ratio=4.0,
                 drop_path=0.0, proj_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.attn  = CSWinAttention(dim, num_heads, split_size, name="attn")
        self.dp1   = DropPath(drop_path, name="dp1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.mlp   = keras.Sequential([
            layers.Dense(int(dim * mlp_ratio), activation="gelu", name="fc1"),
            layers.Dropout(proj_drop),
            layers.Dense(dim, name="fc2"),
            layers.Dropout(proj_drop)], name="mlp")
        self.dp2   = DropPath(drop_path, name="dp2")

    def call(self, x, training=False):
        x = x + self.dp1(self.attn(self.norm1(x), training=training), training=training)
        x = x + self.dp2(self.mlp(self.norm2(x),  training=training), training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.attn.dim_half * 2,
                    "num_heads": self.attn.attn_h.num_heads * 2,
                    "split_size": self.attn.split_size})
        return cfg


class PatchMerging(layers.Layer):
    def __init__(self, out_dim, **kwargs):
        super().__init__(**kwargs); self.out_dim = out_dim
        self.conv = layers.Conv2D(out_dim, kernel_size=2, strides=2,
                                  padding="same", name="down_conv")
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="down_norm")

    def call(self, x):
        return self.norm(self.conv(x))

    def get_config(self):
        cfg = super().get_config(); cfg["out_dim"] = self.out_dim; return cfg


class CSWinTransformer(keras.Model):
    def __init__(self, embed_dim=64, depths=(2, 2, 6, 2), num_heads=(2, 4, 8, 16),
                 split_sizes=(2, 2, 4, 2), mlp_ratio=4.0,
                 drop_path_rate=DROP_PATH_RATE, proj_drop=DROPOUT,
                 num_classes=1, **kwargs):
        super().__init__(**kwargs)
        dpr = list(np.linspace(0, drop_path_rate, sum(depths)))
        bi = 0; dim = embed_dim
        self.patch_embed = PatchEmbedding(embed_dim, name="patch_embed")
        self._sc = []
        for si, (d, h, s) in enumerate(zip(depths, num_heads, split_sizes)):
            ns = []
            for b in range(d):
                n = f"s{si}_b{b}"
                setattr(self, n, CSWinBlock(dim, h, s, mlp_ratio, dpr[bi], proj_drop, name=n))
                ns.append(n); bi += 1
            dn = None
            if si < len(depths) - 1:
                dn = f"down_{si}"
                setattr(self, dn, PatchMerging(dim * 2, name=dn))
                dim *= 2
            self._sc.append((ns, dn))
        self.final_norm = layers.LayerNormalization(epsilon=1e-5, name="final_norm")
        self.gap        = layers.GlobalAveragePooling2D(name="gap")
        self.head_drop  = layers.Dropout(proj_drop, name="head_drop")
        self.head       = layers.Dense(num_classes, activation="sigmoid",
                                       name="head", dtype="float32")

    def call(self, x, training=False):
        x = self.patch_embed(x)
        for ns, dn in self._sc:
            for n in ns:
                x = getattr(self, n)(x, training=training)
            if dn:
                x = getattr(self, dn)(x)
        return self.head(self.head_drop(self.gap(self.final_norm(x)), training=training))

    def get_config(self):
        return {"embed_dim": 64, "depths": [2, 2, 6, 2], "num_heads": [2, 4, 8, 16],
                "split_sizes": [2, 2, 4, 2], "mlp_ratio": 4.0,
                "drop_path_rate": DROP_PATH_RATE, "proj_drop": DROPOUT, "num_classes": 1}


CUSTOM_OBJECTS = {
    "CSWinTransformer": CSWinTransformer, "CSWinBlock": CSWinBlock,
    "CSWinAttention":   CSWinAttention,   "StripeAttention": StripeAttention,
    "PatchEmbedding":   PatchEmbedding,   "PatchMerging": PatchMerging,
    "DropPath":         DropPath,         "WarmupCosineDecay": WarmupCosineDecay,
}

# ══════════════════════════════════════════════════════════════════════════════
# LAZY GLOBALS  — loaded once on first inference call, not at import time.
# This means  import gradio_app  never crashes even if the model is missing.
# ══════════════════════════════════════════════════════════════════════════════
_model    = None
_dnn_net  = None

def _load_model():
    """Load the CSWin model. Called once, result cached in _model."""
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n\n"
            f"Fix (choose one):\n"
            f"  1. Set env var before running:\n"
            f"       Windows : set MODEL_PATH=C:\\full\\path\\to\\cswin_best.keras\n"
            f"       Linux   : export MODEL_PATH=/full/path/to/cswin_best.keras\n"
            f"  2. Place cswin_best.keras at:\n"
            f"       {_FALLBACK_MODEL}")

    print("Loading model ...")
    with keras.utils.custom_object_scope(CUSTOM_OBJECTS):
        _model = keras.models.load_model(MODEL_PATH, compile=False)
    # Warm-up pass so GradientTape works immediately
    _ = _model(tf.zeros((1,) + IMG_SIZE + (3,), dtype=tf.float32), training=False)
    print("Model loaded ✅")
    return _model


def _load_detector():
    """Load OpenCV DNN face detector. Files are downloaded once to CACHE_DIR."""
    global _dnn_net
    if _dnn_net is not None:
        return _dnn_net

    proto = os.path.join(CACHE_DIR, "deploy.prototxt")
    caffemodel = os.path.join(CACHE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
    urls = {
        proto: ("https://raw.githubusercontent.com/opencv/opencv/master/"
                "samples/dnn/face_detector/deploy.prototxt"),
        caffemodel: ("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_"
                     "face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"),
    }
    for path, url in urls.items():
        if not os.path.exists(path):
            print(f"  Downloading {os.path.basename(path)} ...")
            urllib.request.urlretrieve(url, path)

    _dnn_net = cv2.dnn.readNetFromCaffe(proto, caffemodel)
    print("Face detector ready ✅")
    return _dnn_net


# ══════════════════════════════════════════════════════════════════════════════
# FACE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_and_crop_face(img_rgb):
    """
    Detect the largest face with OpenCV ResNet SSD.
    Falls back to full image resize if no face found.

    Returns: face_norm (H,W,3 float32 in [0,1]),
             bbox (dict {x,y,w,h,conf} or None),
             used_fallback (bool)
    """
    net   = _load_detector()
    h, w  = img_rgb.shape[:2]
    blob  = cv2.dnn.blobFromImage(cv2.resize(img_rgb, (300, 300)),
                                  1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets      = net.forward()
    best_conf = 0.0
    best_box  = None

    for i in range(dets.shape[2]):
        c = float(dets[0, 0, i, 2])
        if c > best_conf:
            best_conf = c
            best_box  = dets[0, 0, i, 3:7]

    if best_box is not None and best_conf > 0.5:
        x1, y1, x2, y2 = (best_box * np.array([w, h, w, h])).astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size > 0:
            face_norm = cv2.resize(crop, IMG_SIZE).astype(np.float32) / 255.0
            return face_norm, {"x": x1, "y": y1, "w": x2-x1, "h": y2-y1,
                               "conf": round(best_conf, 2)}, False

    face_norm = cv2.resize(img_rgb, IMG_SIZE).astype(np.float32) / 255.0
    return face_norm, None, True


# ══════════════════════════════════════════════════════════════════════════════
# XAI METHODS
# ══════════════════════════════════════════════════════════════════════════════

_SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_br",
    [(0.0, "#1565C0"), (0.35, "#90CAF9"), (0.5, "#F5F5F5"),
     (0.65, "#EF9A9A"), (1.0, "#B71C1C")])


def compute_integrated_gradients(model, inp, n_steps=50):
    baseline = tf.zeros_like(inp, dtype=tf.float32)
    inp_tf   = tf.cast(inp, tf.float32)
    grads    = []
    for alpha in tf.linspace(0.0, 1.0, n_steps):
        interp = baseline + alpha * (inp_tf - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interp)
            score = model(interp, training=False)[:, 0]
        grads.append(tape.gradient(score, interp).numpy()[0])
    ig     = (inp_tf.numpy()[0] - baseline.numpy()[0]) * np.mean(grads, axis=0)
    vmax   = np.abs(ig).max() + 1e-8
    ig_s   = ig / vmax
    ig_abs = gaussian_filter(np.abs(ig).mean(axis=-1), sigma=1.5)
    if ig_abs.max() > 0: ig_abs /= ig_abs.max()
    return ig_s, ig_abs


def compute_smoothgrad(model, inp, n_samples=20, noise_level=0.10):
    x_base = tf.cast(inp, tf.float32)
    acc    = np.zeros_like(inp[0])
    for _ in range(n_samples):
        xn = tf.clip_by_value(
            x_base + tf.random.normal(tf.shape(x_base), stddev=noise_level,
                                      dtype=tf.float32), 0.0, 1.0)
        with tf.GradientTape() as tape:
            tape.watch(xn)
            score = model(xn, training=False)[:, 0]
        acc += tape.gradient(score, xn).numpy()[0]
    acc  /= n_samples
    vmax  = np.abs(acc).max() + 1e-8
    ss    = acc / vmax
    sa    = gaussian_filter(np.abs(acc).mean(axis=-1), sigma=1.5)
    if sa.max() > 0: sa /= sa.max()
    return ss, sa


def compute_occlusion_map(model, inp):
    H, W     = IMG_SIZE
    base     = float(model(inp, training=False).numpy()[0, 0])
    sens     = np.zeros((H, W), dtype=np.float32)
    for r in range(0, H, OCC_PATCH_SIZE):
        for c in range(0, W, OCC_PATCH_SIZE):
            r2, c2 = min(H, r+OCC_PATCH_SIZE), min(W, c+OCC_PATCH_SIZE)
            p = inp.copy(); p[0, r:r2, c:c2, :] = 0.5
            sens[r:r2, c:c2] = abs(base - float(
                model(p, training=False).numpy()[0, 0]))
    if sens.max() > 0: sens /= sens.max()
    return gaussian_filter(sens, sigma=1.5)


def compute_region_scores(ig_abs, smooth_abs, occ_map):
    H, W = IMG_SIZE
    comb = (ig_abs + smooth_abs + occ_map) / 3.0
    def _m(r, c=None):
        s = comb[r[0]:r[1], :] if c is None else comb[r[0]:r[1], c[0]:c[1]]
        return float(s.mean()) if s.size > 0 else 0.0
    scores = {
        "Forehead":   _m([0,           int(H*.25)], [int(W*.15), int(W*.85)]),
        "Eyes/Brows": _m([int(H*.20),  int(H*.45)]),
        "Nose":       _m([int(H*.40),  int(H*.65)], [int(W*.30), int(W*.70)]),
        "Mouth":      _m([int(H*.60),  int(H*.85)], [int(W*.20), int(W*.80)]),
        "Cheeks/Jaw": _m([int(H*.45),  int(H*.90)]),
    }
    t = sum(scores.values()) + 1e-8
    return {k: v/t for k, v in scores.items()}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_xai_figure(face_norm, orig_rgb, bbox, label, conf, prob,
                     ig_signed, ig_abs, smooth_signed, smooth_abs,
                     occ_map, region_scores, used_fallback):
    ig_mean  = ig_signed.mean(axis=-1)
    ig_pos   = np.maximum( ig_mean, 0)
    ig_neg   = np.maximum(-ig_mean, 0)
    sm_mean  = smooth_signed.mean(axis=-1)
    combined = (ig_abs + smooth_abs + occ_map) / 3.0
    top      = max(region_scores, key=region_scores.get)
    col      = "#2ecc71" if label == "REAL" else "#e74c3c"

    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.30,
                            top=0.92, bottom=0.04)

    # ── Row 1: original / crop / verdict ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(orig_rgb); ax.axis("off"); ax.set_title("Input Image", fontsize=11)
    if bbox:
        ax.add_patch(patches.Rectangle(
            (bbox["x"], bbox["y"]), bbox["w"], bbox["h"],
            linewidth=3, edgecolor=col, facecolor="none"))
        ax.text(bbox["x"], max(0, bbox["y"]-8),
                f"Face {bbox['conf']:.2f}", color=col, fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.55, pad=2))

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(face_norm); ax.axis("off")
    ax.set_title("Extracted Face" if not used_fallback
                 else "Full Image (no face detected)", fontsize=11)

    ax = fig.add_subplot(gs[0, 2:])
    ax.set_facecolor(col); ax.axis("off")
    for txt, y, fs, alpha in [
        (label,                              0.64, 58, 1.0),
        (f"Confidence: {conf*100:.1f}%",     0.40, 24, 1.0),
        (f"REAL: {prob*100:.1f}%   |   FAKE: {(1-prob)*100:.1f}%",
                                             0.24, 14, 0.9),
        (f"Score: {prob:.4f}   |   Threshold: {THRESHOLD}",
                                             0.10, 11, 0.75),
    ]:
        kw = dict(ha="center", va="center", color="white",
                  alpha=alpha, transform=ax.transAxes)
        if fs >= 24: kw["fontweight"] = "bold"
        ax.text(0.5, y, txt, fontsize=fs, **kw)
    if used_fallback:
        ax.text(0.5, 0.02, "No face detected — full image used",
                ha="center", va="center", fontsize=9,
                color="yellow", transform=ax.transAxes)
    ax.set_title("Prediction", fontsize=11)

    # ── Row 2: attribution maps ───────────────────────────────────────────────
    def _heat(cell, data, cmap, title, symmetric=False):
        a = fig.add_subplot(cell); a.imshow(face_norm, alpha=0.45)
        vmax_ = max(abs(data).max(), 1e-8)
        kw = dict(cmap=cmap, alpha=0.65)
        kw.update({"vmin": -vmax_, "vmax": vmax_} if symmetric
                  else {"vmin": 0, "vmax": 1})
        im = a.imshow(data, **kw)
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        a.set_title(title, fontsize=10); a.axis("off")

    _heat(gs[1,0], ig_mean, _SHAP_CMAP,
          "Integrated Gradients\n(Red→REAL  Blue→FAKE)", symmetric=True)
    _heat(gs[1,1], ig_abs, "hot",
          "IG Absolute Saliency\n(Hot = high importance)")
    _heat(gs[1,2], sm_mean, _SHAP_CMAP,
          f"SmoothGrad (n={SHAP_N_BG})\n(Red→REAL  Blue→FAKE)", symmetric=True)
    _heat(gs[1,3], occ_map, "YlOrRd",
          f"Occlusion Sensitivity\n(patch={OCC_PATCH_SIZE}px)")

    # ── Row 3: IG decomposition + combined + region bar ───────────────────────
    for cell, data, cmap, title in [
        (gs[2,0], ig_pos, "Reds",  "IG Positive\n(Evidence FOR real)"),
        (gs[2,1], ig_neg, "Blues", "IG Negative\n(Evidence FOR fake)"),
    ]:
        a = fig.add_subplot(cell); a.imshow(face_norm, alpha=0.4)
        im = a.imshow(data, cmap=cmap, alpha=0.8, vmin=0, vmax=1)
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        a.set_title(title, fontsize=10); a.axis("off")

    ax = fig.add_subplot(gs[2,2]); ax.imshow(face_norm, alpha=0.35)
    im = ax.imshow(combined, cmap="magma", alpha=0.75, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    hy, hx = np.unravel_index(np.argsort(combined.ravel())[::-1][:5], combined.shape)
    ax.scatter(hx, hy, s=60, c="yellow", marker="*",
               edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_title("Combined Attribution\n(★ top-5 critical pixels)", fontsize=10)
    ax.axis("off")

    ax      = fig.add_subplot(gs[2,3])
    regions = list(region_scores.keys())
    pcts    = [region_scores[r]*100 for r in regions]
    bars    = ax.barh(regions, pcts,
                      color=["#E53935","#FB8C00","#43A047","#1E88E5","#8E24AA"],
                      edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, pcts):
        ax.text(val+0.3, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(pcts)*1.35)
    ax.set_xlabel("Relative Importance (%)", fontsize=10)
    ax.set_title("Face Region Importance\n(IG + SmoothGrad + Occlusion)", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.text(0.97, 0.04, f"Key: {top}", transform=ax.transAxes,
            ha="right", fontsize=9, style="italic", color="dimgray")

    verdict_col = "#1a7a3c" if label == "REAL" else "#c0392b"
    fig.suptitle(
        f"CSWin Transformer — DeepFake Detection\n"
        f"Verdict: {label}  ({conf*100:.1f}% confident)   "
        f"Score={prob:.4f}   Key region: {top}",
        fontsize=13, fontweight="bold", color=verdict_col, y=0.97)

    return fig, top


# ══════════════════════════════════════════════════════════════════════════════
# LOADING PLACEHOLDER
# ══════════════════════════════════════════════════════════════════════════════

def _make_loading_image():
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_facecolor("#f5f5f5"); ax.axis("off")
    ax.text(0.5, 0.55, "Computing XAI explanation...",
            ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(0.5, 0.35, "Please wait (30–60 seconds)",
            ha="center", va="center", fontsize=12, color="gray")
    path = os.path.join(OUT_DIR, "loading.png")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PREDICT FUNCTION  (streaming generator)
# ══════════════════════════════════════════════════════════════════════════════

def predict_and_explain(pil_image):
    """
    Streaming generator — yields twice:
      1st yield : prediction scores + loading placeholder  (instant)
      2nd yield : prediction scores + full XAI figure      (after ~30s)
    """
    if pil_image is None:
        yield {"REAL ✅": 0.0, "FAKE ⚠️": 0.0}, _make_loading_image()
        return

    try:
        model = _load_model()
    except FileNotFoundError as e:
        # Show the error inside Gradio rather than crashing
        err_img = os.path.join(OUT_DIR, "model_error.png")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_facecolor("#fff3cd"); ax.axis("off")
        ax.text(0.5, 0.6, "⚠  Model file not found", ha="center", va="center",
                fontsize=18, fontweight="bold", color="#856404")
        ax.text(0.5, 0.3, str(e)[:200], ha="center", va="center",
                fontsize=9, color="#856404", wrap=True)
        fig.savefig(err_img, bbox_inches="tight", dpi=100)
        plt.close(fig)
        yield {"REAL ✅": 0.0, "FAKE ⚠️": 0.0}, err_img
        return

    orig_rgb                   = np.array(pil_image.convert("RGB"))
    face_norm, bbox, fallback  = detect_and_crop_face(orig_rgb)
    inp                        = np.expand_dims(face_norm, 0)

    prob      = float(model(inp, training=False).numpy()[0, 0])
    real_prob = round(prob, 4)
    fake_prob = round(1.0 - prob, 4)
    label     = "REAL" if real_prob >= THRESHOLD else "FAKE"
    conf      = real_prob if label == "REAL" else fake_prob
    print(f"\n  Verdict: {label}  ({conf*100:.1f}%)")

    # First yield — instant result
    yield {"REAL ✅": real_prob, "FAKE ⚠️": fake_prob}, _make_loading_image()

    # XAI computation
    print("  [1/3] Integrated Gradients ...")
    ig_s, ig_a   = compute_integrated_gradients(model, inp)
    print("  [2/3] SmoothGrad ...")
    sm_s, sm_a   = compute_smoothgrad(model, inp, n_samples=SHAP_N_BG)
    print("  [3/3] Occlusion ...")
    occ          = compute_occlusion_map(model, inp)
    regions      = compute_region_scores(ig_a, sm_a, occ)

    fig, top = build_xai_figure(face_norm, orig_rgb, bbox, label, conf, prob,
                                ig_s, ig_a, sm_s, sm_a, occ, regions, fallback)
    out = os.path.join(OUT_DIR, "xai_output.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Top region: {top}")

    # Second yield — full XAI figure
    yield {"REAL ✅": real_prob, "FAKE ⚠️": fake_prob}, out


# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
demo = gr.Interface(
    fn=predict_and_explain,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Image(label="XAI Explanation"),
    ],
    title="Deepfake Detection — CSWin Transformer",
    description=(
        "Upload a face image to detect if it's **Real** or **Fake**.\n\n"
        "The verdict appears instantly. "
        "The full explanation (Integrated Gradients + SmoothGrad + "
        "Occlusion Sensitivity + Region Importance) follows in ~30 seconds."
    ),
    # allow_flagging="never",
    theme=gr.themes.Soft(),
)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n  Model path : {MODEL_PATH}")
    print(f"  Output dir : {OUT_DIR}")
    print(f"  Cache dir  : {CACHE_DIR}\n")

    demo.queue(max_size=5, default_concurrency_limit=1).launch(
        share=True,           # change to True for a public tunnel URL
        # server_name="0.0.0.0",
        # server_port=7860,
        show_error=True,
        allowed_paths=[os.path.abspath(OUT_DIR)]
    )
