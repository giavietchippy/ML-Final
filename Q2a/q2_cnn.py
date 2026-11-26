"""Q2a - CNN on CIFAR-10: baseline vs. two overfitting-reduction techniques.
Configs: (i) baseline  (ii) Dropout  (iii) Data Augmentation  (iv) Dropout + Data Augmentation
Usage:  python q2_cnn.py [--epochs 40] [--batch 64]
Outputs (./results): results.json, curves_all.png, confusion_<config>.png
"""
import argparse
import os

import numpy as np
from sklearn.metrics import classification_report

from Q2a.common import (CLASS_NAMES, RESULTS_DIR, build_cnn, evaluate_test, fit, load_data,
                    plot_confusion, plot_curves, save_json, set_seed, summarize_history)

CONFIGS = {
    "baseline":      dict(dropout=0.0, augment=False),
    "dropout":       dict(dropout=0.4, augment=False),
    "augmentation":  dict(dropout=0.0, augment=True),
    "dropout+aug":   dict(dropout=0.4, augment=True),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data = load_data()
    print({k: v.shape for k, v in data.items()})
    results, histories = {}, {}

    for name, cfg in CONFIGS.items():
        print(f"\n===== {name} {cfg} =====")
        set_seed()                                   # same init seed for every config
        model = build_cnn(**cfg)
        if name == "baseline":
            model.summary()
        hist = fit(model, data, args.epochs, args.batch)
        test_metrics, y_pred = evaluate_test(model, data)
        results[name] = {**cfg, **summarize_history(hist), **test_metrics,
                         "n_params": int(model.count_params())}
        histories[name] = hist.history
        plot_confusion(data["y_test"], y_pred, f"Confusion matrix - {name}",
                       os.path.join(RESULTS_DIR, f"confusion_{name.replace('+', '_')}.png"))
        print(classification_report(data["y_test"], y_pred, target_names=CLASS_NAMES, digits=3))
        print(results[name])

    plot_curves(histories, os.path.join(RESULTS_DIR, "curves_all.png"))
    for name, h in histories.items():                # one figure per config is easier to read
        plot_curves({name: h}, os.path.join(RESULTS_DIR, f"curves_{name.replace('+', '_')}.png"))
    save_json(results, os.path.join(RESULTS_DIR, "results.json"))

    print("\n%-14s %9s %9s %9s %9s" % ("config", "train", "val", "gap", "TEST"))
    for n, r in results.items():
        print("%-14s %9.4f %9.4f %9.4f %9.4f" % (n, r["final_train_acc"], r["final_val_acc"],
                                                 r["acc_gap_train_minus_val"], r["test_acc"]))
