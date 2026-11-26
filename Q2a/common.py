"""Shared utilities for Q2a (CNN image classification on CIFAR-10)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
RESULTS_DIR = "results"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def set_seed(seed=SEED):
    keras.utils.set_random_seed(seed)


def load_data(n_train=10000, n_val=5000):
    """CIFAR-10 with a small training subset so overfitting is easy to observe.
    Train / val are disjoint subsets of the official train set; test = official test set."""
    (x, y), (x_te, y_te) = keras.datasets.cifar10.load_data()
    y, y_te = y.ravel(), y_te.ravel()
    idx = np.random.default_rng(SEED).permutation(len(x))
    val_idx, tr_idx = idx[:n_val], idx[n_val:n_val + n_train]
    f = lambda a: a.astype("float32") / 255.0
    return {"x_train": f(x[tr_idx]), "y_train": y[tr_idx],
            "x_val": f(x[val_idx]), "y_val": y[val_idx],
            "x_test": f(x_te), "y_test": y_te}


def build_cnn(dropout=0.0, augment=False, filters=(32, 64, 128),
              dense_units=256, lr=1e-3):
    """VGG-style CNN: [Conv-Conv-MaxPool(-Dropout)] x len(filters) -> Flatten -> Dense -> Softmax.
    - augment: random flip / translation / rotation layers (active only during training)
    - dropout: p after the Dense layer, p/2 after each pooling block (0 = disabled)
    """
    inputs = keras.Input(shape=(32, 32, 3))
    x = inputs
    if augment:
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomTranslation(0.1, 0.1)(x)
        x = layers.RandomRotation(0.05)(x)
    for f in filters:
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
        if dropout > 0:
            x = layers.Dropout(dropout / 2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit(model, data, epochs, batch_size, callbacks=None):
    return model.fit(data["x_train"], data["y_train"],
                     validation_data=(data["x_val"], data["y_val"]),
                     epochs=epochs, batch_size=batch_size,
                     callbacks=callbacks or [], verbose=2)


def evaluate_test(model, data):
    loss, acc = model.evaluate(data["x_test"], data["y_test"], verbose=0)
    y_pred = model.predict(data["x_test"], verbose=0).argmax(axis=1)
    return {"test_loss": float(loss), "test_acc": float(acc)}, y_pred


def summarize_history(hist):
    """Key numbers for the report: final train/val acc, generalization gap, best val epoch."""
    h = hist.history
    best = int(np.argmax(h["val_accuracy"]))
    return {"epochs_run": len(h["loss"]),
            "final_train_acc": float(h["accuracy"][-1]),
            "final_val_acc": float(h["val_accuracy"][-1]),
            "final_train_loss": float(h["loss"][-1]),
            "final_val_loss": float(h["val_loss"][-1]),
            "acc_gap_train_minus_val": float(h["accuracy"][-1] - h["val_accuracy"][-1]),
            "best_val_acc": float(h["val_accuracy"][best]),
            "best_val_epoch": best + 1}


def plot_curves(histories, path):
    """histories: {name: history_dict}. Left = loss, right = accuracy (solid=train, dashed=val)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (name, h) in enumerate(histories.items()):
        c = f"C{i}"
        axes[0].plot(h["loss"], c=c, label=f"{name} (train)")
        axes[0].plot(h["val_loss"], c=c, ls="--", label=f"{name} (val)")
        axes[1].plot(h["accuracy"], c=c, label=f"{name} (train)")
        axes[1].plot(h["val_accuracy"], c=c, ls="--", label=f"{name} (val)")
    for ax, t in zip(axes, ["Loss", "Accuracy"]):
        ax.set_title(t); ax.set_xlabel("Epoch"); ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
