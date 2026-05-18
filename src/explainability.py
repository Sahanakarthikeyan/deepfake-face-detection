"""
explainability.py
=================
Attribution methods for the CSWin deepfake detector.

All methods work on subclassed Keras models via tf.GradientTape — no
Functional-API requirement. Methods provided:

  1. Integrated Gradients (Sundararajan et al., 2017) — theoretically grounded,
     satisfies completeness axiom; closest to SHAP in motivation.
  2. SmoothGrad            (Smilkov et al., 2017)     — noise-averaged gradients
     for cleaner, less spiky saliency maps.
  3. Occlusion Sensitivity                             — model-agnostic; captures
     non-linear effects gradient methods can miss.
  4. Region importance scores — human-readable anatomical breakdown
     (Forehead / Eyes / Nose / Mouth / Cheeks).

Why GradientExplainer instead of DeepExplainer:
  DeepExplainer requires model.inputs (Functional API only).
  GradientExplainer uses tf.GradientTape internally and works on any
  callable TF model including subclassed models like CSWinTransformer.
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from cswin_transformer import IMG_SIZE

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SHAP_N_BG      = 30   # SmoothGrad: number of noisy samples
OCC_PATCH_SIZE = 8    # Occlusion: patch side length in pixels

# Shared colormap: blue = FAKE evidence, red = REAL evidence
_SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_br",
    [(0.0, "#1565C0"), (0.35, "#90CAF9"), (0.5, "#F5F5F5"),
     (0.65, "#EF9A9A"), (1.0, "#B71C1C")])


# ══════════════════════════════════════════════════════════════════════════════
# ATTRIBUTION METHODS
# ══════════════════════════════════════════════════════════════════════════════

def compute_input_gradient_map(model, inp):
    """
    Gradient of output score w.r.t. input pixels.

    For a binary deepfake detector:
      positive gradient → pixel pushes score toward REAL
      negative gradient → pixel pushes score toward FAKE

    Args:
        model : loaded Keras model
        inp   : np.float32 array of shape (1, H, W, 3)

    Returns:
        grad_signed : (H, W, 3) signed gradients, normalised to [-1, 1]
        grad_abs    : (H, W)    absolute-value saliency, normalised to [0, 1]
    """
    x = tf.cast(inp, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        score = model(x, training=False)[:, 0]
    grads = tape.gradient(score, x).numpy()[0]

    vmax        = np.abs(grads).max() + 1e-8
    grad_signed = grads / vmax

    grad_abs = np.abs(grads).mean(axis=-1)
    grad_abs = gaussian_filter(grad_abs, sigma=2)
    if grad_abs.max() > 0:
        grad_abs /= grad_abs.max()

    return grad_signed, grad_abs


def compute_smoothgrad(model, inp, n_samples=20, noise_level=0.10):
    """
    SmoothGrad: average gradients over n_samples noisy copies of the input.
    Reduces noise in gradient maps — produces cleaner SHAP-like attributions.
    (Hooker et al., 2019 / Smilkov et al., 2017)

    Args:
        noise_level : std of Gaussian noise as a fraction of input range [0, 1]

    Returns:
        smooth_signed : (H, W, 3) averaged signed gradients, in [-1, 1]
        smooth_abs    : (H, W)    smoothed absolute saliency, in [0, 1]
    """
    x_base      = tf.cast(inp, tf.float32)
    grads_accum = np.zeros_like(inp[0])

    for _ in range(n_samples):
        noise   = tf.random.normal(shape=tf.shape(x_base),
                                   stddev=noise_level, dtype=tf.float32)
        x_noisy = tf.clip_by_value(x_base + noise, 0.0, 1.0)
        with tf.GradientTape() as tape:
            tape.watch(x_noisy)
            score = model(x_noisy, training=False)[:, 0]
        grads_accum += tape.gradient(score, x_noisy).numpy()[0]

    grads_accum /= n_samples
    vmax          = np.abs(grads_accum).max() + 1e-8
    smooth_signed = grads_accum / vmax

    smooth_abs = np.abs(grads_accum).mean(axis=-1)
    smooth_abs = gaussian_filter(smooth_abs, sigma=1.5)
    if smooth_abs.max() > 0:
        smooth_abs /= smooth_abs.max()

    return smooth_signed, smooth_abs


def compute_integrated_gradients(model, inp, n_steps=50):
    """
    Integrated Gradients (Sundararajan et al., 2017).
    Theoretically grounded: satisfies completeness axiom.
    Integrates gradients along a straight path from black baseline to input.

    Returns:
        ig_signed : (H, W, 3) attribution in [-1, 1]
        ig_abs    : (H, W)    absolute attribution in [0, 1]
    """
    baseline  = tf.zeros_like(inp, dtype=tf.float32)
    inp_tf    = tf.cast(inp, tf.float32)
    grads_all = []

    for alpha in tf.linspace(0.0, 1.0, n_steps):
        interp = baseline + alpha * (inp_tf - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interp)
            score = model(interp, training=False)[:, 0]
        grads_all.append(tape.gradient(score, interp).numpy()[0])

    avg_grads = np.mean(grads_all, axis=0)
    ig        = (inp_tf.numpy()[0] - baseline.numpy()[0]) * avg_grads

    vmax      = np.abs(ig).max() + 1e-8
    ig_signed = ig / vmax

    ig_abs = np.abs(ig).mean(axis=-1)
    ig_abs = gaussian_filter(ig_abs, sigma=1.5)
    if ig_abs.max() > 0:
        ig_abs /= ig_abs.max()

    return ig_signed, ig_abs


def compute_occlusion_map(model, inp, patch_size=OCC_PATCH_SIZE):
    """
    Model-agnostic occlusion sensitivity.
    Greys out each region and measures the change in prediction score.

    Returns:
        sensitivity_map : (H, W) float32 in [0, 1]
    """
    H, W      = IMG_SIZE
    base_prob = float(model(inp, training=False).numpy()[0, 0])
    sens_map  = np.zeros((H, W), dtype=np.float32)

    for r in range(0, H, patch_size):
        for c in range(0, W, patch_size):
            r2 = min(H, r + patch_size)
            c2 = min(W, c + patch_size)
            patch                    = inp.copy()
            patch[0, r:r2, c:c2, :] = 0.5
            new_p                    = float(model(patch, training=False).numpy()[0, 0])
            sens_map[r:r2, c:c2]    = abs(base_prob - new_p)

    if sens_map.max() > 0:
        sens_map /= sens_map.max()
    return gaussian_filter(sens_map, sigma=1.5)


def compute_region_scores(ig_abs, smooth_abs, occ_map):
    """
    Average three attribution maps over 5 anatomical face regions.
    Returns a normalised importance dict (values sum to 1).
    """
    H, W     = IMG_SIZE
    combined = (ig_abs + smooth_abs + occ_map) / 3.0

    def _mean(rows, cols=None):
        r0, r1 = rows
        if cols is None:
            patch = combined[r0:r1, :]
        else:
            patch = combined[r0:r1, cols[0]:cols[1]]
        return float(patch.mean()) if patch.size > 0 else 0.0

    scores = {
        "Forehead":   _mean([0,           int(H * .25)], [int(W * .15), int(W * .85)]),
        "Eyes/Brows": _mean([int(H * .20), int(H * .45)]),
        "Nose":       _mean([int(H * .40), int(H * .65)], [int(W * .30), int(W * .70)]),
        "Mouth":      _mean([int(H * .60), int(H * .85)], [int(W * .20), int(W * .80)]),
        "Cheeks/Jaw": _mean([int(H * .45), int(H * .90)]),
    }
    total = sum(scores.values()) + 1e-8
    return {k: v / total for k, v in scores.items()}


def explain_image(model, face_inp):
    """
    Master explainability function. Runs all three attribution methods.

    Args:
        model    : loaded CSWinTransformer Keras model
        face_inp : np.float32 array of shape (1, H, W, 3)

    Returns:
        ig_signed, ig_abs        — Integrated Gradients
        smooth_signed, smooth_abs — SmoothGrad
        occ_map                  — Occlusion sensitivity map
        region_scores            — dict of anatomical region importances
    """
    print("  [1/3] Integrated Gradients ...")
    ig_signed, ig_abs     = compute_integrated_gradients(model, face_inp, n_steps=50)

    print("  [2/3] SmoothGrad ...")
    smooth_signed, smooth_abs = compute_smoothgrad(
        model, face_inp, n_samples=SHAP_N_BG, noise_level=0.10)

    print("  [3/3] Occlusion sensitivity ...")
    occ_map = compute_occlusion_map(model, face_inp, OCC_PATCH_SIZE)

    region_scores = compute_region_scores(ig_abs, smooth_abs, occ_map)
    return ig_signed, ig_abs, smooth_signed, smooth_abs, occ_map, region_scores


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _overlay_heatmap(ax, face_norm, heatmap, cmap, alpha=0.65,
                     title="", symmetric=False):
    """Helper: overlay a heatmap on the face image with a colorbar."""
    ax.imshow(face_norm)
    if symmetric:
        vmax = max(abs(heatmap).max(), 1e-8)
        im   = ax.imshow(heatmap, cmap=cmap, alpha=alpha, vmin=-vmax, vmax=vmax)
    else:
        im   = ax.imshow(heatmap, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def build_figure(face_norm, orig_rgb, bbox, label, conf, prob,
                 ig_signed, ig_abs, smooth_signed, smooth_abs,
                 occ_map, region_scores, used_fallback,
                 threshold=0.5):
    """
    Build the full 3-row explanation figure.

    Row 1 — Prediction (original image, face crop, verdict card)
    Row 2 — Attribution maps (IG signed, IG abs, SmoothGrad, Occlusion)
    Row 3 — Positive/negative decomposition + combined map + region bar chart

    Returns:
        fig        : matplotlib Figure
        top_region : str — name of the most important anatomical region
    """
    ig_mean = ig_signed.mean(axis=-1)
    ig_pos  = np.maximum( ig_mean, 0)
    ig_neg  = np.maximum(-ig_mean, 0)
    sm_mean = smooth_signed.mean(axis=-1)

    edge = "#2ecc71" if label == "REAL" else "#e74c3c"
    bg   = "#2ecc71" if label == "REAL" else "#e74c3c"

    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.42, wspace=0.30,
                            top=0.92, bottom=0.04)

    # ── Row 1: Prediction ────────────────────────────────────────────────────

    # [0,0] Original image + bounding box
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(orig_rgb); ax.axis("off")
    ax.set_title("Input Image", fontsize=11)
    if bbox:
        rect = patches.Rectangle(
            (bbox["x"], bbox["y"]), bbox["w"], bbox["h"],
            linewidth=3, edgecolor=edge, facecolor="none")
        ax.add_patch(rect)
        ax.text(bbox["x"], max(0, bbox["y"] - 8),
                f"Face {bbox['conf']:.2f}",
                color=edge, fontsize=9, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.55, pad=2))

    # [0,1] Extracted face crop
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(face_norm); ax.axis("off")
    ax.set_title("Extracted Face" if not used_fallback
                 else "Full Image (no face detected)", fontsize=11)

    # [0,2:4] Verdict card
    ax = fig.add_subplot(gs[0, 2:])
    ax.set_facecolor(bg); ax.axis("off")
    ax.text(0.5, 0.64, label, ha="center", va="center",
            fontsize=58, fontweight="bold", color="white",
            transform=ax.transAxes)
    ax.text(0.5, 0.39, f"Confidence: {conf*100:.1f}%",
            ha="center", va="center", fontsize=24, color="white",
            transform=ax.transAxes)
    ax.text(0.5, 0.22,
            f"Score: {prob:.4f}   |   Threshold: {threshold}",
            ha="center", va="center", fontsize=11,
            color="white", alpha=0.85, transform=ax.transAxes)
    if used_fallback:
        ax.text(0.5, 0.08, "No face detected — full image used",
                ha="center", va="center", fontsize=9,
                color="yellow", transform=ax.transAxes)
    ax.set_title("Prediction", fontsize=11)

    # ── Row 2: Attribution maps ───────────────────────────────────────────────

    ax = fig.add_subplot(gs[1, 0])
    _overlay_heatmap(ax, face_norm, ig_mean, _SHAP_CMAP, alpha=0.70,
                     title="Integrated Gradients\n(Red→REAL, Blue→FAKE)",
                     symmetric=True)

    ax = fig.add_subplot(gs[1, 1])
    _overlay_heatmap(ax, face_norm, ig_abs, "hot", alpha=0.65,
                     title="IG Absolute Saliency\n(Hot = high importance)")

    ax = fig.add_subplot(gs[1, 2])
    _overlay_heatmap(ax, face_norm, sm_mean, _SHAP_CMAP, alpha=0.70,
                     title=f"SmoothGrad (n={SHAP_N_BG})\n(Red→REAL, Blue→FAKE)",
                     symmetric=True)

    ax = fig.add_subplot(gs[1, 3])
    _overlay_heatmap(ax, face_norm, occ_map, "YlOrRd", alpha=0.65,
                     title=f"Occlusion Sensitivity\n"
                           f"(patch={OCC_PATCH_SIZE}px, bright=critical)")

    # ── Row 3: Positive/negative decomposition + region bar ───────────────────

    # [2,0] IG positive — evidence FOR real
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(face_norm, alpha=0.4)
    im = ax.imshow(ig_pos, cmap="Reds", alpha=0.8, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("IG Positive Contribution\n(Evidence FOR real)", fontsize=10)
    ax.axis("off")

    # [2,1] IG negative — evidence FOR fake
    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(face_norm, alpha=0.4)
    im = ax.imshow(ig_neg, cmap="Blues", alpha=0.8, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("IG Negative Contribution\n(Evidence FOR fake)", fontsize=10)
    ax.axis("off")

    # [2,2] Combined attribution (IG abs + SmoothGrad abs + Occlusion)
    combined = (ig_abs + np.abs(smooth_signed).mean(axis=-1) + occ_map) / 3.0
    ax = fig.add_subplot(gs[2, 2])
    ax.imshow(face_norm, alpha=0.35)
    im = ax.imshow(combined, cmap="magma", alpha=0.75, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    flat_idx = np.argsort(combined.ravel())[::-1][:5]
    hy, hx   = np.unravel_index(flat_idx, combined.shape)
    ax.scatter(hx, hy, s=60, c="yellow", marker="*",
               edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_title("Combined Attribution Map\n(★ = top-5 critical pixels)",
                 fontsize=10)
    ax.axis("off")

    # [2,3] Region importance bar chart
    ax = fig.add_subplot(gs[2, 3])
    regions    = list(region_scores.keys())
    scores_pct = [region_scores[r] * 100 for r in regions]
    bar_cols   = ["#E53935", "#FB8C00", "#43A047", "#1E88E5", "#8E24AA"]
    bars       = ax.barh(regions, scores_pct, color=bar_cols,
                         edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, scores_pct):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(scores_pct) * 1.35)
    ax.set_xlabel("Relative Importance (%)", fontsize=10)
    ax.set_title("Face Region Importance\n(IG + SmoothGrad + Occlusion)",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    top_region = max(region_scores, key=region_scores.get)
    ax.text(0.97, 0.04, f"Key: {top_region}",
            transform=ax.transAxes, ha="right",
            fontsize=9, style="italic", color="dimgray")

    # ── Supertitle ────────────────────────────────────────────────────────────
    verdict_col = "#1a7a3c" if label == "REAL" else "#c0392b"
    fig.suptitle(
        f"CSWin Transformer — DeepFake Detection Explanation\n"
        f"Verdict: {label}  ({conf*100:.1f}% confident)   "
        f"Score={prob:.4f}   Key region: {top_region}",
        fontsize=13, fontweight="bold", color=verdict_col, y=0.97)

    return fig, top_region