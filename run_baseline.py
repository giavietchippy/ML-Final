"""run_baseline.py - Q1b: preprocessing, Train/Val/Test split, gradient check, baseline FNN training."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fnn import FNN, gradient_check

SEED = 42
df = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv")
print("Original shape:", df.shape, "| missing:", int(df.isna().sum().sum()),
      "| duplicates:", int(df.duplicated().sum()))
df = df.drop_duplicates().reset_index(drop=True)   # avoid duplicate samples leaking between train and test
print("After dropping duplicates:", df.shape)
df.describe().T.to_csv("data_summary.csv")

X = df.drop(columns="strength").values
y = df[["strength"]].values

# 70 / 15 / 15
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)
print("Train/Val/Test:", len(X_tr), len(X_va), len(X_te))

# Standardization: fit on the training set ONLY
xs, ys = StandardScaler().fit(X_tr), StandardScaler().fit(y_tr)
X_tr, X_va, X_te = xs.transform(X_tr), xs.transform(X_va), xs.transform(X_te)
y_tr, y_va, y_te = ys.transform(y_tr), ys.transform(y_va), ys.transform(y_te)

# ---- Gradient checking (with and without L2) ----
chk = {}
for act in ["tanh", "sigmoid", "relu"]:
    net = FNN([8, 16, 8, 1], activation=act, seed=1)
    chk[act] = {"no_L2": gradient_check(net, X_tr[:64], y_tr[:64]),
                "L2=0.5": gradient_check(net, X_tr[:64], y_tr[:64], l2_lambda=0.5)}
print("Gradient check (max relative error):")
for k, v in chk.items():
    print(f"  {k:8s} {v}")

# ---- Baseline: 8-64-32-1, ReLU, no regularization ----
net = FNN([8, 64, 32, 1], activation="relu", seed=0)
net.fit(X_tr, y_tr, X_va, y_va, epochs=600, lr=0.01, batch_size=32, verbose=100)

res = {s: net.evaluate(X_, y_, ys) for s, X_, y_ in
       [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]}
print(json.dumps(res, indent=2))
json.dump({"gradient_check": chk, "baseline": res,
           "best_val_epoch_in_history": int(np.argmin(net.val_loss_history)) + 1},
          open("baseline_results.json", "w"), indent=2)

plt.figure(figsize=(7, 4.2))
plt.plot(net.train_loss_history, label="Train loss")
plt.plot(net.val_loss_history, label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("Loss J = MSE/2 (standardized scale)")
plt.title("Baseline FNN 8-64-32-1 (ReLU), no regularization")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("baseline_loss.png", dpi=200)
