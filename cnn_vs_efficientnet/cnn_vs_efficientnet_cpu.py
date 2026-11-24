"""
CNN (from scratch) vs EfficientNetB0 (pretrained, transfer learning + fine-tuning)
Dataset: CIFAR-10
LIGHT VERSION for CPU (no GPU): smaller data subset, 64x64 images, fewer epochs
Outputs: training_curves.png, comparison_bar.png, results.json, results.md
"""
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
BATCH = 128
IMG_SIZE = 64            # EfficientNet input size (CIFAR images are upscaled from 32x32)
EPOCHS_CNN = 8
EPOCHS_EFF_HEAD = 3      # phase 1: train only the new classifier head
EPOCHS_EFF_FINE = 2      # phase 2: fine-tune top layers of EfficientNet
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

tf.keras.utils.set_random_seed(SEED)

# ---------------------------------------------------------------- data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()

x_val, y_val = x_train[-5000:], y_train[-5000:]
x_train, y_train = x_train[:-5000], y_train[:-5000]
# light version: use a subset so it runs on CPU in ~15 min
x_train, y_train = x_train[:8000], y_train[:8000]
x_val, y_val = x_val[:1000], y_val[:1000]
x_test, y_test = x_test[:2000], y_test[:2000]
print("train/val/test:", len(x_train), len(x_val), len(x_test))


def make_ds(x, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x.astype("float32"), y))
    if training:
        ds = ds.shuffle(10_000, seed=SEED)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(x_train, y_train, training=True)
val_ds = make_ds(x_val, y_val)
test_ds = make_ds(x_test, y_test)

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.1, 0.1),
], name="augment")


# ---------------------------------------------------------------- models
def build_cnn():
    """Simple CNN trained from scratch."""
    inp = layers.Input((32, 32, 3))
    x = augment(inp)
    x = layers.Rescaling(1 / 255.0)(x)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(10, activation="softmax")(x)
    return models.Model(inp, out, name="simple_cnn")


def build_efficientnet():
    """EfficientNetB0 pretrained on ImageNet + new classification head."""
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inp = layers.Input((32, 32, 3))
    x = augment(inp)
    x = layers.Resizing(IMG_SIZE, IMG_SIZE)(x)
    x = base(x, training=False)          # EfficientNet rescales inputs internally (0-255)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(10, activation="softmax")(x)
    return models.Model(inp, out, name="efficientnetb0"), base


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])


def merge(h1, h2=None):
    out = {k: list(v) for k, v in h1.items()}
    if h2:
        for k, v in h2.items():
            out[k] += list(v)
    return out


# ---------------------------------------------------------------- train CNN
print("\n===== Training CNN from scratch =====")
cnn = build_cnn()
compile_model(cnn, 1e-3)
t0 = time.time()
h_cnn = cnn.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_CNN, verbose=2).history
cnn_time = time.time() - t0
cnn_loss, cnn_acc = cnn.evaluate(test_ds, verbose=0)
print(f"CNN test acc: {cnn_acc:.4f} | loss: {cnn_loss:.4f} | time: {cnn_time:.0f}s")

# ---------------------------------------------------------------- train EfficientNet
print("\n===== Training EfficientNetB0 (phase 1: head only) =====")
eff, base = build_efficientnet()
compile_model(eff, 1e-3)
t0 = time.time()
h1 = eff.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_EFF_HEAD, verbose=2).history

print("\n===== Training EfficientNetB0 (phase 2: fine-tuning top layers) =====")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
for layer in base.layers:                 # keep BatchNorm frozen for stable fine-tuning
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
compile_model(eff, 1e-4)
h2 = eff.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_EFF_FINE, verbose=2).history
eff_time = time.time() - t0
h_eff = merge(h1, h2)
eff_loss, eff_acc = eff.evaluate(test_ds, verbose=0)
print(f"EfficientNet test acc: {eff_acc:.4f} | loss: {eff_loss:.4f} | time: {eff_time:.0f}s")

# ---------------------------------------------------------------- results
results = {
    "CNN (from scratch)": {
        "test_accuracy": float(cnn_acc), "test_loss": float(cnn_loss),
        "train_time_sec": round(cnn_time), "params": int(cnn.count_params()),
        "epochs": EPOCHS_CNN},
    "EfficientNetB0 (pretrained)": {
        "test_accuracy": float(eff_acc), "test_loss": float(eff_loss),
        "train_time_sec": round(eff_time), "params": int(eff.count_params()),
        "epochs": EPOCHS_EFF_HEAD + EPOCHS_EFF_FINE},
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("results.md", "w") as f:
    f.write("| Model | Test Acc | Test Loss | Train Time (s) | Params | Epochs |\n")
    f.write("|---|---|---|---|---|---|\n")
    for name, r in results.items():
        f.write(f"| {name} | {r['test_accuracy']:.4f} | {r['test_loss']:.4f} | "
                f"{r['train_time_sec']} | {r['params']:,} | {r['epochs']} |\n")
print("\n" + open("results.md").read())

# training curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, h in (("CNN", h_cnn), ("EfficientNetB0", h_eff)):
    ax[0].plot(h["accuracy"], label=f"{name} train")
    ax[0].plot(h["val_accuracy"], "--", label=f"{name} val")
    ax[1].plot(h["loss"], label=f"{name} train")
    ax[1].plot(h["val_loss"], "--", label=f"{name} val")
ax[0].set_title("Accuracy"); ax[1].set_title("Loss")
for a in ax:
    a.set_xlabel("epoch"); a.legend(); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("training_curves.png", dpi=150); plt.close()

# bar comparison
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
names = ["CNN", "EfficientNetB0"]
ax[0].bar(names, [cnn_acc, eff_acc], color=["#4C72B0", "#DD8452"])
ax[0].set_title("Test accuracy"); ax[0].set_ylim(0, 1)
ax[1].bar(names, [cnn_time, eff_time], color=["#4C72B0", "#DD8452"])
ax[1].set_title("Training time (s)")
plt.tight_layout(); plt.savefig("comparison_bar.png", dpi=150); plt.close()

print("Saved: results.json, results.md, training_curves.png, comparison_bar.png")
