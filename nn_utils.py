"""Helper module for Neural Network Lab: datasets, preprocessing and figures."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASETS = ["Two Blobs", "Rings", "Checkerboard", "Spiral", "Half Moons"]
ACTIVATIONS = ["relu", "tanh", "sigmoid", "elu", "swish"]

POS, NEG = "#1B9E77", "#D95F02"       # class 1 = green, class 0 = orange


# ---------------------------------------------------------------- data
def make_dataset(name, n, noise, seed):
    rng = np.random.RandomState(seed)
    if name == "Two Blobs":
        X, y = make_blobs(n, centers=[(-1.5, 0), (1.5, 0)], cluster_std=0.7 + 2 * noise,
                          random_state=seed)
    elif name == "Rings":
        r = np.where(rng.rand(n) < 0.5, 1.0, 2.2)
        a = rng.uniform(0, 2 * np.pi, n)
        X = np.c_[r * np.cos(a), r * np.sin(a)] + rng.normal(0, noise * 0.6, (n, 2))
        y = (r > 1.5).astype(int)
    elif name == "Checkerboard":
        X = rng.uniform(-2, 2, (n, 2))
        y = ((np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2).astype(int)
        X = X + rng.normal(0, noise * 0.3, X.shape)
    elif name == "Spiral":
        m = n // 2
        t = np.sqrt(rng.uniform(0.04, 1, m)) * 3 * np.pi
        arm = np.c_[t * np.cos(t), t * np.sin(t)] / 3
        X = np.vstack([arm, -arm]) + rng.normal(0, noise * 0.5, (2 * m, 2))
        y = np.r_[np.zeros(m), np.ones(m)].astype(int)
    else:  # Half Moons
        X, y = make_moons(n, noise=noise * 0.8, random_state=seed)
    return X.astype("float32"), y.astype("float32")


def split_and_scale(X, y, test_size, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)                      # fit on training data only
    return sc.transform(Xtr).astype("float32"), sc.transform(Xte).astype("float32"), ytr, yte


def make_grid(X, res=90, pad=0.4):
    xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, res)
    ys = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, res)
    xx, yy = np.meshgrid(xs, ys)
    return {"xx": xx, "yy": yy, "pts": np.c_[xx.ravel(), yy.ravel()].astype("float32")}


# ---------------------------------------------------------------- figures
def _points(ax, Xtr, ytr, Xte, yte):
    for cls, col in ((0, NEG), (1, POS)):
        m = ytr == cls
        ax.scatter(Xtr[m, 0], Xtr[m, 1], s=16, c=col, edgecolor="white", linewidth=0.3)
        m = yte == cls
        ax.scatter(Xte[m, 0], Xte[m, 1], s=30, c=col, marker="s", edgecolor="black", linewidth=0.6)


def fig_boundary(grid, probs, Xtr, ytr, Xte, yte, title="", point=None):
    """probs: flat array of P(class 1) for every grid point (None = data only)."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    if probs is not None:
        p = np.asarray(probs).reshape(grid["xx"].shape)
        ax.contourf(grid["xx"], grid["yy"], p, levels=np.linspace(0, 1, 11),
                    cmap="PuOr", alpha=0.6, vmin=0, vmax=1)
        ax.contour(grid["xx"], grid["yy"], p, levels=[0.5], colors="k", linewidths=1.6)
    _points(ax, Xtr, ytr, Xte, yte)
    if point is not None:
        ax.scatter(*point, marker="X", s=200, c="gold", edgecolor="black", zorder=5)
    ax.set_xlim(grid["xx"].min(), grid["xx"].max()); ax.set_ylim(grid["yy"].min(), grid["yy"].max())
    ax.set_title(title, fontsize=10); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.text(0.02, 0.02, "● train   ■ test", transform=ax.transAxes, fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.tight_layout()
    return fig


def fig_history(hist, upto=None):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    n = len(hist["loss"]) if upto is None else upto
    ep = np.arange(1, n + 1)
    for ax, k, lab in ((a, "loss", "Loss"), (b, "accuracy", "Accuracy")):
        ax.plot(ep, hist[k][:n], color=POS, label="train")
        ax.plot(ep, hist["val_" + k][:n], color=NEG, label="validation")
        ax.set_xlabel("epoch"); ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    b.set_ylim(0.3, 1.02)
    fig.tight_layout()
    return fig


def _act(name, x):
    if name == "relu": return np.maximum(0, x)
    if name == "tanh": return np.tanh(x)
    if name == "sigmoid": return 1 / (1 + np.exp(-x))
    if name == "elu": return np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1)
    return x / (1 + np.exp(-x))                         # swish


def fig_activations(highlight=None):
    x = np.linspace(-5, 5, 400)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name in ACTIVATIONS:
        on = name == highlight
        ax.plot(x, _act(name, x), label=name, lw=3 if on else 1.4, alpha=1 if on else 0.7)
    ax.axhline(0, c="grey", lw=0.6); ax.axvline(0, c="grey", lw=0.6)
    ax.set_ylim(-1.5, 3); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=5, loc="upper left")
    ax.set_title("Activation functions", fontsize=10)
    fig.tight_layout()
    return fig


def fig_weight_hist(weights):
    fig, axes = plt.subplots(1, len(weights), figsize=(3 * len(weights), 2.6), squeeze=False)
    for i, (ax, w) in enumerate(zip(axes[0], weights)):
        ax.hist(w.ravel(), bins=20, color=POS, edgecolor="white")
        ax.set_title(f"Layer {i + 1}  {w.shape[0]}→{w.shape[1]}", fontsize=9)
    fig.tight_layout()
    return fig


def fig_confusion(cm):
    fig, ax = plt.subplots(figsize=(3.4, 3))
    ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    fig.tight_layout()
    return fig
