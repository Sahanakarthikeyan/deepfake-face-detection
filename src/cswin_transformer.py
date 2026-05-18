import os, json, math, shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — override any of these with environment variables
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE       = (128, 128)
BATCH_SIZE     = 32
EPOCHS         = 20
PEAK_LR        = 1e-4
WARMUP_EPOCHS  = 10
MIN_LR         = 1e-6
WEIGHT_DECAY   = 0.05
DROP_PATH_RATE = 0.1
DROPOUT        = 0.1

# Paths: use env vars when set, else fall back to sensible local defaults
DATA_DIR    = os.environ.get("DATA_DIR",    os.path.join("data", "mtcnn_output"))
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR",  os.path.join("outputs"))
MODEL_PATH  = os.environ.get("MODEL_PATH",  os.path.join(OUTPUT_DIR, "cswin_best.keras"))
EPOCH_FILE  = os.environ.get("EPOCH_FILE",  os.path.join(OUTPUT_DIR, "last_epoch.json"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# LR SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, min_lr, warmup_steps, total_steps, **kwargs):
        super().__init__(**kwargs)
        self.peak_lr      = float(peak_lr)
        self.min_lr       = float(min_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)

    def __call__(self, step):
        step    = tf.cast(step, tf.float32)
        warmup  = self.peak_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        cos_arg = math.pi * (step - self.warmup_steps) / \
                  tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        cosine  = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * \
                  (1.0 + tf.cos(cos_arg))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "min_lr": self.min_lr,
                "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

def _get_class_names(data_dir):
    train_path = os.path.join(data_dir, "train")
    folders = sorted([d for d in os.listdir(train_path)
                      if os.path.isdir(os.path.join(train_path, d))])
    fake_folder = next((f for f in folders if "fake" in f.lower()), folders[0])
    real_folder = next((f for f in folders if "real" in f.lower()), folders[-1])
    print(f"[LABELS]  0='{fake_folder}' (FAKE)   1='{real_folder}' (REAL)")
    return [fake_folder, real_folder]


def create_datasets():
    class_names = _get_class_names(DATA_DIR)

    def _load(split, augment=False):
        ds = keras.preprocessing.image_dataset_from_directory(
            os.path.join(DATA_DIR, split),
            image_size=IMG_SIZE, batch_size=BATCH_SIZE,
            label_mode="binary", class_names=class_names,
            shuffle=(split == "train"), seed=42,
        )
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(
                lambda x, y: (tf.image.random_flip_left_right(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = _load("train",      augment=True)
    val_ds   = _load("validation", augment=False)
    test_ds  = _load("test",       augment=False)

    class_weight = None
    try:
        train_path = os.path.join(DATA_DIR, "train")
        counts = {cls: len([f for f in os.listdir(os.path.join(train_path, cls))
                             if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                  for cls in class_names}
        n0, n1 = counts[class_names[0]], counts[class_names[1]]
        total  = n0 + n1
        ratio  = max(n0, n1) / (min(n0, n1) + 1e-6)
        print(f"[DATA]  FAKE:{n0}  REAL:{n1}  ratio:{ratio:.2f}")
        if ratio > 1.2:
            class_weight = {0: total/(2*n0), 1: total/(2*n1)}
            print(f"[DATA]  class_weight={class_weight}")
    except Exception as e:
        print(f"[DATA]  class weight error: {e}")

    return train_ds, val_ds, test_ds, class_weight, class_names


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class DropPath(layers.Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prob)

    def call(self, x, training=False):
        if not training or self.drop_prob == 0.0:
            return x
        keep  = 1.0 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        noise = keep + tf.random.uniform(shape, dtype=x.dtype)
        return x * tf.math.floor(noise) / keep

    def get_config(self):
        cfg = super().get_config()
        cfg["drop_prob"] = self.drop_prob
        return cfg


class PatchEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim       = dim
        self.proj      = layers.Conv2D(dim, kernel_size=4, strides=4,
                                       padding="same", name="patch_proj")
        self.norm      = layers.LayerNormalization(epsilon=1e-5, name="patch_norm")
        self._pe_shape = None

    def build(self, input_shape):
        H = int(input_shape[1])
        W = int(input_shape[2])
        pH, pW = H // 4, W // 4
        self._pe_shape = (1, pH, pW, self.dim)
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=self._pe_shape,
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(self, x):
        x = self.proj(x)
        x = x + tf.cast(self.pos_embed, x.dtype)
        return self.norm(x)

    def get_config(self):
        cfg = super().get_config()
        cfg["dim"] = self.dim
        return cfg


# ── Stripe helpers ────────────────────────────────────────────────────────────

def h_split(x, s):
    B = tf.shape(x)[0]; H = tf.shape(x)[1]; W = tf.shape(x)[2]; C = x.shape[-1]
    x = tf.reshape(x, [B, H // s, s, W, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4])
    return tf.reshape(x, [B * (H // s), W * s, C])

def h_merge(x, B, H, W, C, s):
    x = tf.reshape(x, [B, H // s, W, s, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4])
    return tf.reshape(x, [B, H, W, C])

def v_split(x, s):
    B = tf.shape(x)[0]; H = tf.shape(x)[1]; W = tf.shape(x)[2]; C = x.shape[-1]
    x = tf.reshape(x, [B, H, W // s, s, C])
    x = tf.transpose(x, [0, 2, 1, 3, 4])
    return tf.reshape(x, [B * (W // s), H * s, C])

def v_merge(x, B, H, W, C, s):
    x = tf.reshape(x, [B, W // s, H, s, C])
    x = tf.transpose(x, [0, 2, 1, 3, 4])
    return tf.reshape(x, [B, H, W, C])


class StripeAttention(layers.Layer):
    def __init__(self, dim, num_heads, attn_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = layers.Dense(dim * 3, use_bias=True, name="qkv")
        self.proj = layers.Dense(dim,      use_bias=True, name="proj")
        self.drop = layers.Dropout(attn_drop)

    def call(self, x, training=False):
        B = tf.shape(x)[0]; N = tf.shape(x)[1]; C = x.shape[-1]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.drop(attn, training=training)
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B, N, C])
        return self.proj(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.num_heads * self.head_dim,
                    "num_heads": self.num_heads})
        return cfg


class CSWinAttention(layers.Layer):
    def __init__(self, dim, num_heads, split_size, attn_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        assert dim % 2 == 0
        self.split_size = split_size
        self.dim_half   = dim // 2
        hh = max(1, num_heads // 2)
        self.attn_h = StripeAttention(self.dim_half, hh, attn_drop, name="attn_h")
        self.attn_v = StripeAttention(self.dim_half, hh, attn_drop, name="attn_v")
        self.lepe_h = layers.DepthwiseConv2D(3, padding="same", name="lepe_h")
        self.lepe_v = layers.DepthwiseConv2D(3, padding="same", name="lepe_v")

    def call(self, x, training=False):
        B = tf.shape(x)[0]; H = tf.shape(x)[1]; W = tf.shape(x)[2]
        s = self.split_size
        x1, x2 = tf.split(x, 2, axis=-1)
        xh = h_merge(self.attn_h(h_split(x1, s), training=training),
                     B, H, W, self.dim_half, s) + self.lepe_h(x1)
        xv = v_merge(self.attn_v(v_split(x2, s), training=training),
                     B, H, W, self.dim_half, s) + self.lepe_v(x2)
        return tf.concat([xh, xv], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim":        self.dim_half * 2,
                    "num_heads":  self.attn_h.num_heads * 2,
                    "split_size": self.split_size})
        return cfg


class CSWinBlock(layers.Layer):
    def __init__(self, dim, num_heads, split_size,
                 mlp_ratio=4.0, drop_path=0.0, proj_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.attn  = CSWinAttention(dim, num_heads, split_size, name="attn")
        self.dp1   = DropPath(drop_path, name="dp1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.mlp   = keras.Sequential([
            layers.Dense(int(dim * mlp_ratio), activation="gelu", name="fc1"),
            layers.Dropout(proj_drop),
            layers.Dense(dim, name="fc2"),
            layers.Dropout(proj_drop),
        ], name="mlp")
        self.dp2 = DropPath(drop_path, name="dp2")

    def call(self, x, training=False):
        x = x + self.dp1(self.attn(self.norm1(x), training=training),
                         training=training)
        x = x + self.dp2(self.mlp(self.norm2(x),  training=training),
                         training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim":        self.attn.dim_half * 2,
                    "num_heads":  self.attn.attn_h.num_heads * 2,
                    "split_size": self.attn.split_size})
        return cfg


class PatchMerging(layers.Layer):
    def __init__(self, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.conv = layers.Conv2D(out_dim, kernel_size=2, strides=2,
                                   padding="same", name="down_conv")
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="down_norm")

    def call(self, x):
        return self.norm(self.conv(x))

    def get_config(self):
        cfg = super().get_config()
        cfg["out_dim"] = self.out_dim
        return cfg


class CSWinTransformer(keras.Model):
    def __init__(self,
                 embed_dim=64,
                 depths=(2, 2, 6, 2),
                 num_heads=(2, 4, 8, 16),
                 split_sizes=(2, 2, 4, 2),
                 mlp_ratio=4.0,
                 drop_path_rate=DROP_PATH_RATE,
                 proj_drop=DROPOUT,
                 num_classes=1,
                 **kwargs):
        super().__init__(**kwargs)

        dpr     = list(np.linspace(0, drop_path_rate, sum(depths)))
        blk_idx = 0
        dim     = embed_dim

        self.patch_embed    = PatchEmbedding(embed_dim, name="patch_embed")
        self._stage_configs = []

        for si, (depth, heads, split) in enumerate(
                zip(depths, num_heads, split_sizes)):
            names = []
            for b in range(depth):
                n   = f"s{si}_b{b}"
                blk = CSWinBlock(dim, heads, split, mlp_ratio,
                                 dpr[blk_idx], proj_drop, name=n)
                setattr(self, n, blk)
                names.append(n)
                blk_idx += 1
            dn = None
            if si < len(depths) - 1:
                dn = f"down_{si}"
                setattr(self, dn, PatchMerging(dim * 2, name=dn))
                dim *= 2
            self._stage_configs.append((names, dn))

        self.final_norm = layers.LayerNormalization(epsilon=1e-5, name="final_norm")
        self.gap        = layers.GlobalAveragePooling2D(name="gap")
        self.head_drop  = layers.Dropout(proj_drop, name="head_drop")
        self.head       = layers.Dense(num_classes, activation="sigmoid",
                                       name="head", dtype="float32")

    def call(self, x, training=False):
        x = self.patch_embed(x)
        for names, dn in self._stage_configs:
            for n in names:
                x = getattr(self, n)(x, training=training)
            if dn is not None:
                x = getattr(self, dn)(x)
        x = self.final_norm(x)
        x = self.gap(x)
        x = self.head_drop(x, training=training)
        return self.head(x)

    def get_config(self):
        return {
            "embed_dim":      64,
            "depths":         [2, 2, 6, 2],
            "num_heads":      [2, 4, 8, 16],
            "split_sizes":    [2, 2, 4, 2],
            "mlp_ratio":      4.0,
            "drop_path_rate": DROP_PATH_RATE,
            "proj_drop":      DROPOUT,
            "num_classes":    1,
        }


CUSTOM_OBJECTS = {
    "CSWinTransformer":  CSWinTransformer,
    "CSWinBlock":        CSWinBlock,
    "CSWinAttention":    CSWinAttention,
    "StripeAttention":   StripeAttention,
    "PatchEmbedding":    PatchEmbedding,
    "PatchMerging":      PatchMerging,
    "DropPath":          DropPath,
    "WarmupCosineDecay": WarmupCosineDecay,
}


# ══════════════════════════════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════════════════════════════

def build_model(steps_per_epoch):
    model = CSWinTransformer()
    model(tf.zeros((1,) + IMG_SIZE + (3,), dtype=tf.float32), training=False)

    total  = steps_per_epoch * EPOCHS
    warmup = steps_per_epoch * WARMUP_EPOCHS
    sched  = WarmupCosineDecay(PEAK_LR, MIN_LR, warmup, total)
    opt    = keras.optimizers.AdamW(
        learning_rate=sched,
        weight_decay=WEIGHT_DECAY,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=opt,
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# OVERFIT PROBE
# ══════════════════════════════════════════════════════════════════════════════

def overfit_probe(train_ds, n=128, epochs=60):
    print("\n" + "=" * 60)
    print("  OVERFIT PROBE — must reach >= 0.95 accuracy")
    print("=" * 60)
    imgs, lbls = [], []
    for x, y in train_ds:
        imgs.append(x.numpy()); lbls.append(y.numpy())
        if sum(len(a) for a in imgs) >= n:
            break
    imgs = np.concatenate(imgs)[:n]
    lbls = np.concatenate(lbls)[:n]
    print(f"  Subset  FAKE:{int((lbls==0).sum())}  REAL:{int((lbls==1).sum())}")

    m = CSWinTransformer()
    m(tf.zeros((1,) + IMG_SIZE + (3,), dtype=tf.float32), training=False)
    m.compile(
        optimizer=keras.optimizers.AdamW(1e-4, weight_decay=0.0, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    h   = m.fit(imgs, lbls, epochs=epochs, batch_size=16, verbose=0)
    acc = h.history["accuracy"][-1]
    print(f"  Final accuracy: {acc:.4f}")
    if acc >= 0.95:
        print("  PASS  architecture healthy.")
    elif acc >= 0.70:
        print("  PARTIAL  may still train; check data pipeline.")
    else:
        print("  FAIL  architecture broken; do not proceed.")
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

class EpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open(EPOCH_FILE, "w") as f:
            json.dump({"epoch": epoch + 1}, f)


def build_callbacks():
    return [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
        EpochLogger(),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=20, restore_best_weights=True, verbose=1,
        ),
    ]