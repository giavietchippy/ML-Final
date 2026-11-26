"""Exact confusion analysis for the report (baseline vs. best tuned model).

Retrains ONLY the baseline and the best random-search config, using the same data split,
seed and settings as q2_cnn.py / q2_random_search.py, saves their test predictions, and
computes per-class accuracy and the most confused class pairs from the real counts.

- Does not modify any existing script or any existing file in results/.
- Writes only to results/exact/.
- Compares the retrained test accuracy with the original results.json / random_search.json
  and tells you whether the run was reproduced.

Usage:
    python reproduce_confusion.py                  # retrain both, save, analyze
    python reproduce_confusion.py --analyze-only   # reuse saved .npy files, no training
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix

from Q2a.common import CLASS_NAMES, RESULTS_DIR

N = len(CLASS_NAMES)


# ----------------------------------------------------------------- analysis (pure numpy)
def analyze(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N)))
    per_class = cm.diagonal() / cm.sum(axis=1)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    ordered = [(CLASS_NAMES[i], CLASS_NAMES[j], int(off[i, j]))
               for i, j in (np.unravel_index(k, off.shape)
                            for k in np.argsort(off, axis=None)[::-1][:5])]
    sym = off + off.T                                   # unordered pairs: i->j plus j->i
    pairs = {(i, j): int(sym[i, j]) for i in range(N) for j in range(i + 1, N)}
    unordered = [(CLASS_NAMES[i], CLASS_NAMES[j], c)
                 for (i, j), c in sorted(pairs.items(), key=lambda kv: -kv[1])[:5]]
    return {"cm": cm, "per_class": per_class, "accuracy": float(cm.trace() / cm.sum()),
            "ordered": ordered, "unordered": unordered}


def write_report(y_test, preds, outdir, refs=None):
    """preds: {'baseline': y_pred, 'tuned': y_pred}. Returns markdown text."""
    res = {k: analyze(y_test, p) for k, p in preds.items()}
    names = list(res)
    md = ["# Exact confusion analysis (from saved test predictions)\n"]
    md.append("## Test accuracy recomputed from predictions\n")
    md.append("| model | test acc (this run) | test acc (original) |")
    md.append("|---|---:|---:|")
    for k in names:
        ref = f"{refs[k]:.4f}" if refs and k in refs else "n/a"
        md.append(f"| {k} | {res[k]['accuracy']:.4f} | {ref} |")
    md.append("\n## Per-class test accuracy (recall)\n")
    md.append("| class | " + " | ".join(names) + (" | change |" if len(names) == 2 else " |"))
    md.append("|---|" + "---:|" * (len(names) + (1 if len(names) == 2 else 0)))
    for i, c in enumerate(CLASS_NAMES):
        row = f"| {c} | " + " | ".join(f"{res[k]['per_class'][i]:.3f}" for k in names)
        if len(names) == 2:
            row += f" | {res[names[1]]['per_class'][i] - res[names[0]]['per_class'][i]:+.3f}"
        md.append(row + " |")
    for k in names:
        md.append(f"\n## {k}: most confused ordered pairs (true -> predicted)\n")
        md.append("| true | predicted | count |\n|---|---|---:|")
        md += [f"| {a} | {b} | {c} |" for a, b, c in res[k]["ordered"]]
        md.append(f"\n## {k}: most confused unordered pairs (both directions summed)\n")
        md.append("| class A | class B | count |\n|---|---|---:|")
        md += [f"| {a} | {b} | {c} |" for a, b, c in res[k]["unordered"]]
    text = "\n".join(md) + "\n"
    os.makedirs(outdir, exist_ok=True)
    open(os.path.join(outdir, "confusion_analysis.md"), "w", encoding="utf-8").write(text)
    json.dump({k: {"accuracy": v["accuracy"],
                   "per_class": dict(zip(CLASS_NAMES, map(float, v["per_class"]))),
                   "ordered": v["ordered"], "unordered": v["unordered"],
                   "confusion_matrix": v["cm"].tolist()} for k, v in res.items()},
              open(os.path.join(outdir, "confusion_analysis.json"), "w"), indent=2)
    return text


# ----------------------------------------------------------------- retraining
def retrain(data, results_dir, outdir, epochs_base, epochs_tuned):
    from tensorflow import keras
    from Q2a.common import build_cnn, evaluate_test, fit, set_seed

    rs = json.load(open(os.path.join(results_dir, "random_search.json")))
    cfg = rs["best_config"]
    print("Best tuned config from random_search.json:", cfg)

    # baseline: identical to q2_cnn.py (seed, no dropout, no augmentation, 40 epochs, batch 64)
    set_seed()
    m = build_cnn(dropout=0.0, augment=False)
    fit(m, data, epochs_base, 64)
    met, y_base = evaluate_test(m, data)
    print("baseline test:", met)
    np.save(os.path.join(outdir, "y_pred_baseline.npy"), y_base)
    m.save(os.path.join(outdir, "model_baseline.keras"))

    # tuned: identical to the best trial in q2_random_search.py (early stopping on val_loss)
    set_seed()
    m = build_cnn(lr=cfg["lr"], dropout=cfg["dropout"], dense_units=cfg["dense_units"],
                  filters=tuple(cfg["filters"]), augment=cfg["augment"])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    fit(m, data, epochs_tuned, cfg["batch"], [es])
    met, y_tuned = evaluate_test(m, data)
    print("tuned test:", met)
    np.save(os.path.join(outdir, "y_pred_tuned.npy"), y_tuned)
    m.save(os.path.join(outdir, "model_tuned.keras"))
    np.save(os.path.join(outdir, "y_test.npy"), data["y_test"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--epochs-base", type=int, default=40)
    ap.add_argument("--epochs-tuned", type=int, default=30)
    a = ap.parse_args()
    outdir = os.path.join(a.results_dir, "exact")
    os.makedirs(outdir, exist_ok=True)

    if not a.analyze_only:
        from Q2a.common import load_data
        retrain(load_data(), a.results_dir, outdir, a.epochs_base, a.epochs_tuned)

    y_test = np.load(os.path.join(outdir, "y_test.npy"))
    preds = {k: np.load(os.path.join(outdir, f"y_pred_{k}.npy")) for k in ["baseline", "tuned"]}
    refs = {"baseline": json.load(open(os.path.join(a.results_dir, "results.json")))["baseline"]["test_acc"],
            "tuned": json.load(open(os.path.join(a.results_dir, "random_search.json")))["test_acc"]}
    text = write_report(y_test, preds, outdir, refs)
    print(text)
    for k in preds:
        diff = float((preds[k] == y_test).mean()) - refs[k]
        status = "REPRODUCED (identical)" if abs(diff) < 1e-4 else f"DIFFERS from original by {diff:+.4f}"
        print(f"[check] {k}: {status}")
    print(f"Saved to {outdir}/ (confusion_analysis.md / .json, y_pred_*.npy, model_*.keras)")
