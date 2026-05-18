import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    classification_report, f1_score,
)

from cswin_transformer import CUSTOM_OBJECTS

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — override with environment variables if needed
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
DATA_DIR   = os.environ.get("DATA_DIR",   os.path.join("data", "mtcnn_output"))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("outputs", "cswin_best.keras"))
OUT_DIR    = os.environ.get("OUT_DIR",    os.path.join("outputs", "eval_plots"))
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_class_names():
    train_path = os.path.join(DATA_DIR, "train")
    folders = sorted([d for d in os.listdir(train_path)
                      if os.path.isdir(os.path.join(train_path, d))])
    fake = next((f for f in folders if "fake" in f.lower()), folders[0])
    real = next((f for f in folders if "real" in f.lower()), folders[-1])
    print(f"[LABELS]  0='{fake}' (FAKE)   1='{real}' (REAL)")
    return [fake, real]


def load_test_ds(class_names):
    ds = keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", class_names=class_names, shuffle=False)
    return ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)


def load_model():
    with keras.utils.custom_object_scope(CUSTOM_OBJECTS):
        m = keras.models.load_model(MODEL_PATH, compile=False)
    m.compile(loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc"),
                       keras.metrics.Precision(name="precision"),
                       keras.metrics.Recall(name="recall")])
    return m


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION + PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_and_plot(model, test_ds):
    print("\nCollecting predictions ...")
    y_prob = model.predict(test_ds, verbose=1).flatten()
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).flatten()

    auc  = roc_auc_score(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)
    fpr, tpr, thresh_roc = roc_curve(y_true, y_prob)
    prec, rec, _         = precision_recall_curve(y_true, y_prob)

    # Optimal threshold via Youden-J
    best_i = int(np.argmax(tpr - fpr))
    best_t = float(thresh_roc[best_i])
    y_pred = (y_prob >= best_t).astype(int)

    print(f"\n{'='*60}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Avg Prec : {ap:.4f}")
    print(f"  Optimal threshold (Youden-J): {best_t:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred,
                                 target_names=["FAKE", "REAL"], digits=4))

    # F1 vs threshold sweep
    t_range   = np.linspace(0.01, 0.99, 200)
    f1s       = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
                 for t in t_range]
    best_f1_t = t_range[int(np.argmax(f1s))]

    # Confusion matrix
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    tn, fp, fn, tp = cm.ravel()

    # ── 6-panel figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # Panel 1 — Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    im  = ax1.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax1.text(j, i,
                     f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                     ha="center", va="center", fontsize=12,
                     color="white" if cm_norm[i,j] > 0.5 else "black")
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["FAKE", "REAL"])
    ax1.set_yticklabels(["FAKE", "REAL"])
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    ax1.set_title(f"Confusion Matrix\n(thresh={best_t:.3f})", fontsize=11)
    plt.colorbar(im, ax=ax1, fraction=0.046)

    # Panel 2 — ROC curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fpr, tpr, lw=2, color="#2196F3", label=f"AUC={auc:.4f}")
    ax2.scatter(fpr[best_i], tpr[best_i], s=100, color="red", zorder=5,
                label=f"Optimal t={best_t:.3f}")
    ax2.fill_between(fpr, tpr, alpha=0.08, color="#2196F3")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    ax2.set_title("ROC Curve", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.01])

    # Panel 3 — Precision-Recall
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(rec, prec, lw=2, color="#4CAF50", label=f"AP={ap:.4f}")
    ax3.fill_between(rec, prec, alpha=0.08, color="#4CAF50")
    ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
    ax3.set_title("Precision-Recall Curve", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.set_xlim([0, 1]); ax3.set_ylim([0, 1.01])

    # Panel 4 — Score distribution
    ax4 = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 1, 51)
    ax4.hist(y_prob[y_true==0], bins=bins, alpha=0.65,
             color="#F44336", label="FAKE", edgecolor="white", linewidth=0.3)
    ax4.hist(y_prob[y_true==1], bins=bins, alpha=0.65,
             color="#2196F3", label="REAL", edgecolor="white", linewidth=0.3)
    ax4.axvline(best_t, color="black", lw=2, linestyle="--",
                label=f"thresh={best_t:.3f}")
    ax4.set_xlabel("Predicted probability"); ax4.set_ylabel("Count")
    ax4.set_title("Score Distribution", fontsize=11)
    ax4.legend(fontsize=9)

    # Panel 5 — F1 vs threshold
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t_range, f1s, lw=2, color="#9C27B0")
    ax5.axvline(best_f1_t, color="red", lw=2, linestyle="--",
                label=f"Best F1={max(f1s):.4f} @ t={best_f1_t:.3f}")
    ax5.set_xlabel("Threshold"); ax5.set_ylabel("F1-Score")
    ax5.set_title("F1 vs Threshold", fontsize=11)
    ax5.legend(fontsize=9)

    # Panel 6 — Per-class recall bar
    ax6 = fig.add_subplot(gs[1, 2])
    fake_acc = tn / (tn + fp) * 100
    real_acc = tp / (tp + fn) * 100
    overall  = (tn + tp) / cm.sum() * 100
    bars = ax6.bar(["FAKE\nrecall", "REAL\nrecall", "Overall\nacc"],
                   [fake_acc, real_acc, overall],
                   color=["#F44336", "#2196F3", "#FF9800"],
                   edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, [fake_acc, real_acc, overall]):
        ax6.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.8,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax6.set_ylim([0, 115]); ax6.set_ylabel("(%)")
    ax6.set_title("Per-Class Accuracy", fontsize=11)
    ax6.axhline(90, color="green", linestyle="--", lw=1,
                alpha=0.6, label="90% line")
    ax6.legend(fontsize=8)

    fig.suptitle(
        f"CSWin Transformer  DeepFake Detection\n"
        f"ROC-AUC={auc:.4f}   Best-F1={max(f1s):.4f}   "
        f"Best-thresh={best_f1_t:.3f}",
        fontsize=13, fontweight="bold", y=0.98)

    out = os.path.join(OUT_DIR, "cswin_evaluation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlots saved to {out}")
    plt.close()

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  ROC-AUC        : {auc:.4f}")
    print(f"  Average Prec   : {ap:.4f}")
    print(f"  Best F1-Score  : {max(f1s):.4f}  @ threshold {best_f1_t:.3f}")
    print(f"  FAKE recall    : {fake_acc:.2f}%")
    print(f"  REAL recall    : {real_acc:.2f}%")
    print(f"  Overall acc    : {overall:.2f}%")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"{'='*60}")


def main():
    print("CSWin Transformer  Evaluation")
    class_names = get_class_names()
    test_ds     = load_test_ds(class_names)
    model       = load_model()
    evaluate_and_plot(model, test_ds)


if __name__ == "__main__":
    main()