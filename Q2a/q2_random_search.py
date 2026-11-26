"""Q2a extension - Hyperparameter optimization by Random Search (+ Early Stopping per trial).
Searches lr, dropout, dense_units, batch size, network width on top of the
'dropout + augmentation' setting. Model selection uses the VALIDATION set only;
the TEST set is touched once, for the best trial.
Usage:  python q2_random_search.py [--trials 12] [--epochs 30]
Run q2_cnn.py first (for the baseline numbers used in the comparison).
"""
import argparse
import json
import os

import numpy as np
from tensorflow import keras

from Q2a.common import (RESULTS_DIR, build_cnn, evaluate_test, fit, load_data, plot_confusion,
                    plot_curves, save_json, set_seed, summarize_history)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data = load_data()
    rng = np.random.default_rng(0)
    trials, best = [], {"val": -1}

    for t in range(args.trials):
        cfg = dict(lr=float(10 ** rng.uniform(-4, np.log10(3e-3))),
                   dropout=float(rng.choice([0.2, 0.3, 0.4, 0.5])),
                   dense_units=int(rng.choice([128, 256, 512])),
                   filters=[(32, 64, 128), (64, 128, 256)][int(rng.integers(2))],
                   augment=True)
        batch = int(rng.choice([32, 64, 128]))
        print(f"\n=== trial {t + 1}/{args.trials}: {cfg}, batch={batch} ===")
        set_seed()
        model = build_cnn(**cfg)
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                           restore_best_weights=True)
        hist = fit(model, data, args.epochs, batch, callbacks=[es])
        s = summarize_history(hist)
        val_acc = float(model.evaluate(data["x_val"], data["y_val"], verbose=0)[1])
        trials.append({**cfg, "filters": list(cfg["filters"]), "batch": batch,
                       "epochs_run": s["epochs_run"], "val_acc": val_acc})
        if val_acc > best["val"]:
            best = {"val": val_acc, "cfg": trials[-1], "model": model, "hist": hist.history}

    test_metrics, y_pred = evaluate_test(best["model"], data)
    plot_confusion(data["y_test"], y_pred, "Confusion matrix - best tuned model",
                   os.path.join(RESULTS_DIR, "confusion_tuned.png"))
    plot_curves({"tuned": best["hist"]}, os.path.join(RESULTS_DIR, "curves_tuned.png"))

    base_path = os.path.join(RESULTS_DIR, "results.json")
    base = json.load(open(base_path)) if os.path.exists(base_path) else {}
    out = {"trials": sorted(trials, key=lambda r: -r["val_acc"]),
           "best_config": best["cfg"], "best_val_acc": best["val"], **test_metrics,
           "reference_test_acc": {k: v["test_acc"] for k, v in base.items()}}
    save_json(out, os.path.join(RESULTS_DIR, "random_search.json"))

    print("\nTop trials by val acc:")
    for r in out["trials"][:5]:
        print(r)
    print("\nBest:", best["cfg"], "\nTEST:", test_metrics)
    for k, v in out["reference_test_acc"].items():
        print(f"  reference {k:14s} test_acc = {v:.4f}")
