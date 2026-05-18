import os
import json
import shutil

from tensorflow import keras

from cswin_transformer import (
    create_datasets,
    build_model,
    build_callbacks,
    overfit_probe,
    CUSTOM_OBJECTS,
    WarmupCosineDecay,
    PEAK_LR, MIN_LR, WARMUP_EPOCHS, EPOCHS,
    WEIGHT_DECAY,
    MODEL_PATH, EPOCH_FILE,
)


def main():
    print("\n" + "=" * 70)
    print("  CSWin Transformer  Research-Grade Training")
    print("=" * 70)

    train_ds, val_ds, test_ds, class_weight, class_names = create_datasets()
    steps = sum(1 for _ in train_ds)
    print(f"\n  Steps/epoch={steps}  Total={steps*EPOCHS}"
          f"  Warmup={steps*WARMUP_EPOCHS}")

    # ── Checkpoint resume logic ───────────────────────────────────────────────
    # Priority 1: working directory checkpoint (most recent)
    # Priority 2: BACKUP_CKPT env var (e.g. a previously saved checkpoint)
    # Priority 3: start from scratch
    initial_epoch = 0

    def _try_load(path):
        with keras.utils.custom_object_scope(CUSTOM_OBJECTS):
            m = keras.models.load_model(path, compile=False)
        total  = steps * EPOCHS
        warmup = steps * WARMUP_EPOCHS
        sched  = WarmupCosineDecay(PEAK_LR, MIN_LR, warmup, total)
        opt    = keras.optimizers.AdamW(
            learning_rate=sched, weight_decay=WEIGHT_DECAY, clipnorm=1.0)
        m.compile(
            optimizer=opt,
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
            metrics=["accuracy",
                     keras.metrics.AUC(name="auc"),
                     keras.metrics.Precision(name="precision"),
                     keras.metrics.Recall(name="recall")],
        )
        return m

    # Optional: point BACKUP_CKPT to a previously saved checkpoint via env var
    BACKUP_CKPT = os.environ.get("BACKUP_CKPT", "")

    if os.path.exists(MODEL_PATH):
        print(f"  Resuming from working checkpoint: {MODEL_PATH}")
        model = _try_load(MODEL_PATH)
    elif BACKUP_CKPT and os.path.exists(BACKUP_CKPT):
        print(f"  Restoring from backup: {BACKUP_CKPT}")
        shutil.copy(BACKUP_CKPT, MODEL_PATH)
        backup_epoch = BACKUP_CKPT.replace("cswin_best.keras", "last_epoch.json")
        if os.path.exists(backup_epoch):
            shutil.copy(backup_epoch, EPOCH_FILE)
        model = _try_load(MODEL_PATH)
    else:
        print("  No checkpoint found — starting from scratch")
        model = build_model(steps)

    if os.path.exists(EPOCH_FILE):
        with open(EPOCH_FILE) as f:
            initial_epoch = json.load(f).get("epoch", 0)
        print(f"  Resuming from epoch {initial_epoch + 1}")

    # Uncomment to verify before full run:
    # probe_acc = overfit_probe(train_ds)
    # if probe_acc < 0.70:
    #     raise RuntimeError("Architecture failed overfit probe")

    print(f"\n  Training from epoch {initial_epoch + 1} ...\n")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=build_callbacks(),
        class_weight=class_weight,
    )

    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)
    results = model.evaluate(test_ds, return_dict=True)
    for k, v in results.items():
        print(f"  {k:20s}: {v:.4f}")


if __name__ == "__main__":
    main()