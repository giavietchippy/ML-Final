"""
fnn.py - Feedforward Neural Network for regression, implemented from scratch with NumPy only.

Notation follows the lecture slides "Introduction to Neural Networks and Backpropagation":
    z^l = W^l a^(l-1) + b^l ,  a^l = f(z^l)
    delta^l = dJ/dz^l
    BP1: delta^L = (a^L - y) ⊙ f'(z^L)            (linear output layer: f' = 1)
    BP2: delta^l = (W^(l+1)^T delta^(l+1)) ⊙ f'(z^l)
    BP3: dJ/db^l = delta^l
    BP4: dJ/dW^l = delta^l (a^(l-1))^T
Batch implementation: each sample is one ROW (A has shape (m, n_l)), so W has shape (n_(l-1), n_l).

Loss: J = (1/2m) * sum_i ||yhat_i - y_i||^2  [+ (lam/2m) * sum_l ||W^l||_F^2]
No PyTorch/TensorFlow/autograd: gradients are computed by hand with the Chain Rule.
"""
import numpy as np


# ----------------------------- Activation functions ------------------------
def _relu(z):
    return np.maximum(0.0, z)


def _relu_d(z, a):
    return (z > 0).astype(z.dtype)


def _tanh(z):
    return np.tanh(z)


def _tanh_d(z, a):
    return 1.0 - a ** 2


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _sigmoid_d(z, a):
    return a * (1.0 - a)


ACTIVATIONS = {
    "relu": (_relu, _relu_d),
    "tanh": (_tanh, _tanh_d),
    "sigmoid": (_sigmoid, _sigmoid_d),
}


# ------------------------------ Evaluation metrics -------------------------
def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


class FNN:
    """FNN with several hidden layers and a linear output layer (regression).

    layer_sizes: e.g. [8, 64, 32, 1] -> 2 hidden layers.
    """

    def __init__(self, layer_sizes, activation="relu", seed=0):
        assert len(layer_sizes) >= 4, "At least 2 hidden layers are required by the assignment."
        self.sizes = list(layer_sizes)
        self.act, self.act_d = ACTIVATIONS[activation]
        self.activation = activation
        self.rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            if activation == "relu":          # He initialization
                std = np.sqrt(2.0 / n_in)
            else:                              # Xavier initialization
                std = np.sqrt(1.0 / n_in)
            self.W.append(self.rng.normal(0.0, std, size=(n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))
        self.train_loss_history = []
        self.val_loss_history = []

    # ------------------------------ Forward -------------------------------
    def forward(self, X):
        """Return the lists of z and a for every layer (A[0] = X)."""
        A, Z = [X], []
        L = len(self.W)
        for l in range(L):
            z = A[-1] @ self.W[l] + self.b[l]
            Z.append(z)
            A.append(z if l == L - 1 else self.act(z))   # linear output layer
        return Z, A

    def predict(self, X):
        return self.forward(X)[1][-1]

    # ------------------------------- Loss ---------------------------------
    def loss(self, X, y, l2_lambda=0.0):
        """J = (1/2m) sum ||yhat - y||^2 + (lam/2m) sum ||W||^2."""
        m = X.shape[0]
        yhat = self.predict(X)
        J = np.sum((yhat - y) ** 2) / (2.0 * m)
        if l2_lambda > 0:
            J += l2_lambda / (2.0 * m) * sum(np.sum(W ** 2) for W in self.W)
        return float(J)

    # ------------------------------ Backward ------------------------------
    def backward(self, X, y, l2_lambda=0.0):
        """Backpropagation via the Chain Rule (BP1-BP4). Returns (dW, db)."""
        m = X.shape[0]
        Z, A = self.forward(X)
        L = len(self.W)
        dW, db = [None] * L, [None] * L
        delta = (A[-1] - y)                       # BP1 (f' = 1 at the output layer)
        for l in range(L - 1, -1, -1):
            dW[l] = A[l].T @ delta / m            # BP4
            db[l] = delta.sum(axis=0, keepdims=True) / m   # BP3
            if l2_lambda > 0:                     # L2 hook: add (lam/m) W
                dW[l] = dW[l] + (l2_lambda / m) * self.W[l]
            if l > 0:                             # BP2
                delta = (delta @ self.W[l].T) * self.act_d(Z[l - 1], A[l])
        return dW, db

    # ----------------------------- Weights --------------------------------
    def get_weights(self):
        return [W.copy() for W in self.W], [b.copy() for b in self.b]

    def set_weights(self, weights):
        Ws, bs = weights
        self.W = [W.copy() for W in Ws]
        self.b = [b.copy() for b in bs]

    # ------------------------------- Train --------------------------------
    def fit(self, X, y, X_val=None, y_val=None, epochs=500, lr=0.01,
            batch_size=32, l2_lambda=0.0, early_stopping=False,
            patience=20, min_delta=0.0, verbose=0, shuffle_seed=0):
        """Mini-batch Gradient Descent.

        The stored loss history is the pure data MSE (1/2m)sum||yhat-y||^2 (no L2 term),
        so configurations with/without L2 are compared fairly.
        Early stopping: monitor the validation loss, stop when it has not improved by
        at least `min_delta` for `patience` epochs, and restore the best weights.
        """
        rng = np.random.default_rng(shuffle_seed)
        m = X.shape[0]
        self.train_loss_history, self.val_loss_history = [], []
        best, best_w, best_epoch, wait = np.inf, None, 0, 0
        self.stopped_epoch = epochs
        for ep in range(1, epochs + 1):
            idx = rng.permutation(m)
            for s in range(0, m, batch_size):
                bi = idx[s:s + batch_size]
                dW, db = self._batch_grad(X[bi], y[bi], l2_lambda, m)
                for l in range(len(self.W)):
                    self.W[l] -= lr * dW[l]
                    self.b[l] -= lr * db[l]
            self.train_loss_history.append(self.loss(X, y))
            if X_val is not None:
                vl = self.loss(X_val, y_val)
                self.val_loss_history.append(vl)
                if early_stopping:
                    if vl < best - min_delta:
                        best, best_w, best_epoch, wait = vl, self.get_weights(), ep, 0
                    else:
                        wait += 1
                        if wait >= patience:
                            self.stopped_epoch = ep
                            break
            if verbose and ep % verbose == 0:
                v = self.val_loss_history[-1] if self.val_loss_history else float("nan")
                print(f"epoch {ep:4d}  train {self.train_loss_history[-1]:.5f}  val {v:.5f}")
        if early_stopping and best_w is not None:
            self.set_weights(best_w)
            self.best_epoch = best_epoch
        return self

    def _batch_grad(self, Xb, yb, l2_lambda, m_total):
        """Mini-batch gradient. The L2 penalty is scaled by m_total (size of the whole
        training set), i.e. each step adds (lam/m_total) W, to match the full objective J."""
        dW, db = self.backward(Xb, yb, 0.0)
        if l2_lambda > 0:
            dW = [g + (l2_lambda / m_total) * W for g, W in zip(dW, self.W)]
        return dW, db

    # ------------------------------ Evaluation ----------------------------
    def evaluate(self, X, y, y_scaler=None):
        """Return a dict of MSE/RMSE/MAE. If y_scaler is given, convert back to the original scale."""
        yp = self.predict(X)
        if y_scaler is not None:
            yp, y = y_scaler.inverse_transform(yp), y_scaler.inverse_transform(y)
        return {"MSE": mse(y, yp), "RMSE": rmse(y, yp), "MAE": mae(y, yp)}


# --------------------------- Gradient checking ------------------------------
def gradient_check(net, X, y, l2_lambda=0.0, eps=1e-5, n_checks=40, seed=0):
    """Compare backprop gradients with central finite differences.
    Returns the largest relative error (float64)."""
    rng = np.random.default_rng(seed)
    dW, db = net.backward(X, y, l2_lambda)
    worst = 0.0
    params = [(net.W, dW, l) for l in range(len(net.W))] + \
             [(net.b, db, l) for l in range(len(net.b))]
    for _ in range(n_checks):
        plist, glist, l = params[rng.integers(len(params))]
        arr = plist[l]
        i, j = rng.integers(arr.shape[0]), rng.integers(arr.shape[1])
        old = arr[i, j]
        arr[i, j] = old + eps
        Jp = net.loss(X, y, l2_lambda)
        arr[i, j] = old - eps
        Jm = net.loss(X, y, l2_lambda)
        arr[i, j] = old
        num, ana = (Jp - Jm) / (2 * eps), glist[l][i, j]
        rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
        worst = max(worst, rel)
    return worst
