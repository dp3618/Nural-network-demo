"""
Neural Network Lab - an interactive TensorFlow/Keras demo built with Streamlit.
Run:  streamlit run app.py
"""
import time

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from nn_utils import (DATASETS, ACTIVATIONS, make_dataset, split_and_scale, make_grid,
                      fig_boundary, fig_history, fig_activations, fig_weight_hist, fig_confusion)

# ------------------ EDIT THESE ------------------
AUTHOR = "YOUR NAME"
REG_NO = "YOUR REG NO"
CLASS_INFO = "B.Tech CSE (Data Science), 3rd year"
# ------------------------------------------------

st.set_page_config(page_title="Neural Network Lab", page_icon="🔬", layout="wide")


def show(fig, slot=None):
    (slot or st).pyplot(fig)
    plt.close(fig)


def parse_layers(text):
    try:
        sizes = [int(t) for t in text.replace(" ", "").split(",") if t]
    except ValueError:
        return None
    return sizes if 1 <= len(sizes) <= 6 and all(1 <= s <= 64 for s in sizes) else None


def build_model(c):
    tf.keras.utils.set_random_seed(c["seed"])
    layers = [tf.keras.Input(shape=(2,))]
    for u in c["layers"]:
        layers.append(tf.keras.layers.Dense(u, activation=c["act"]))
        if c["dropout"] > 0:
            layers.append(tf.keras.layers.Dropout(c["dropout"]))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)
    opts = {"Adam": tf.keras.optimizers.Adam, "RMSprop": tf.keras.optimizers.RMSprop,
            "SGD + momentum": lambda lr: tf.keras.optimizers.SGD(lr, momentum=0.9)}
    model.compile(optimizer=opts[c["opt"]](c["lr"]), loss="binary_crossentropy", metrics=["accuracy"])
    return model


class Recorder(tf.keras.callbacks.Callback):
    """Records history + decision-boundary snapshots and updates the live UI."""

    def __init__(self, epochs, grid_pts, every, bar, live_slot):
        super().__init__()
        self.epochs, self.pts, self.every = epochs, grid_pts, every
        self.bar, self.live = bar, live_slot
        self.hist = {k: [] for k in ("loss", "accuracy", "val_loss", "val_accuracy")}
        self.snaps = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.hist:
            self.hist[k].append(float(logs[k]))
        e = epoch + 1
        self.bar.progress(e / self.epochs, text=f"epoch {e}/{self.epochs} · "
                          f"val acc {logs['val_accuracy']:.1%} · loss {logs['loss']:.3f}")
        if e % self.every == 0 or e == self.epochs or e == 1:
            self.snaps[e] = self.model(self.pts, training=False).numpy().ravel()
            if e % (self.every * 2) == 0 or e == self.epochs:
                show(fig_history(self.hist), self.live)


# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.title("🔬 Lab controls")
sb.subheader("Data")
dname = sb.selectbox("Dataset", DATASETS, index=3)
n = sb.slider("Samples", 200, 1500, 600, 100)
noise = sb.slider("Noise", 0.0, 0.6, 0.15, 0.05)
tsize = sb.slider("Validation share", 0.1, 0.5, 0.25, 0.05)

sb.subheader("Architecture")
ltxt = sb.text_input("Hidden layer sizes (comma separated)", "16, 16, 8",
                     help="e.g. '8' = one hidden layer of 8 neurons, '16,16,8' = three layers")
layers = parse_layers(ltxt)
if layers is None:
    sb.error("Enter 1-6 whole numbers between 1 and 64, e.g. 16, 8")
    st.stop()
act = sb.selectbox("Activation", ACTIVATIONS, index=1)
dropout = sb.select_slider("Dropout", [0.0, 0.1, 0.2, 0.3, 0.5], value=0.0)

sb.subheader("Training")
opt = sb.selectbox("Optimizer", ["Adam", "RMSprop", "SGD + momentum"])
lr = sb.select_slider("Learning rate", [0.0005, 0.001, 0.003, 0.01, 0.03, 0.1], value=0.01)
epochs = sb.slider("Epochs", 20, 400, 120, 10)
batch = sb.select_slider("Batch size", [16, 32, 64, 128], value=32)
seed = int(sb.number_input("Seed", 0, 9999, 7))

cfg = dict(data=dname, n=n, noise=noise, tsize=tsize, layers=layers, act=act, dropout=dropout,
           opt=opt, lr=lr, epochs=epochs, batch=batch, seed=seed)

X, y = make_dataset(dname, n, noise, seed)
Xtr, Xte, ytr, yte = split_and_scale(X, y, tsize, seed)
grid = make_grid(np.vstack([Xtr, Xte]))

# ------------------------------------------------------------------ page
st.title("🔬 Neural Network Lab")
st.caption("Design a TensorFlow/Keras network, train it, then replay how its decision boundary evolved.")
tab_train, tab_act, tab_about = st.tabs(["🧪 Train & replay", "📈 Activation functions", "ℹ️ About"])

with tab_train:
    go = st.button("▶ Train model", type="primary")
    bar_slot = st.empty()
    left, right = st.columns([1, 1.4])
    left_slot, right_slot = left.empty(), right.empty()

    if go:
        model = build_model(cfg)
        bar = bar_slot.progress(0.0, text="starting…")
        rec = Recorder(epochs, grid["pts"], max(1, epochs // 12), bar, right_slot)
        t0 = time.time()
        model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=epochs, batch_size=batch,
                  verbose=0, callbacks=[rec])
        st.session_state["run"] = dict(cfg=dict(cfg), model=model, hist=rec.hist, snaps=rec.snaps,
                                       secs=time.time() - t0, data=(Xtr, ytr, Xte, yte), grid=grid)
        bar_slot.success(f"Done in {time.time() - t0:.1f}s")

    run = st.session_state.get("run")
    if run is None:
        show(fig_boundary(grid, None, Xtr, ytr, Xte, yte, f"{dname} · {len(Xtr)} train / {len(Xte)} validation"),
             left_slot)
        right_slot.info("Pick a dataset and architecture in the sidebar, then press **Train model**.")
    else:
        if run["cfg"] != cfg and not go:
            st.warning("Sidebar settings changed — press **Train model** to apply them. Showing the previous run.")
        rXtr, rytr, rXte, ryte = run["data"]
        model, hist, snaps = run["model"], run["hist"], run["snaps"]
        eps = sorted(snaps)

        st.subheader("Training replay")
        ep = st.select_slider("Scrub through training (epoch)", options=eps, value=eps[-1])
        c1, c2 = st.columns([1, 1.4])
        with c1:
            show(fig_boundary(run["grid"], snaps[ep], rXtr, rytr, rXte, ryte, f"Decision boundary · epoch {ep}"))
        with c2:
            show(fig_history(hist, upto=ep))

        loss_tr, acc_tr = model.evaluate(rXtr, rytr, verbose=0)
        loss_te, acc_te = model.evaluate(rXte, ryte, verbose=0)
        m = st.columns(4)
        m[0].metric("Train accuracy", f"{acc_tr:.1%}")
        m[1].metric("Validation accuracy", f"{acc_te:.1%}", f"{acc_te - acc_tr:+.1%} vs train")
        m[2].metric("Validation loss", f"{loss_te:.3f}")
        m[3].metric("Parameters", f"{model.count_params():,}")
        if acc_tr - acc_te > 0.08:
            st.warning("Large train/validation gap → **overfitting**. Try dropout, fewer neurons or more samples.")
        elif acc_tr < 0.8:
            st.warning("Low accuracy → **underfitting**. Try more layers/neurons, tanh/relu or more epochs.")
        else:
            st.success("Healthy fit: train and validation accuracy are high and close.")

        t1, t2, t3 = st.tabs(["Confusion matrix", "Weight distributions", "Probe a point"])
        with t1:
            pred = (model.predict(rXte, verbose=0).ravel() > 0.5).astype(int)
            show(fig_confusion(confusion_matrix(ryte.astype(int), pred, labels=[0, 1])))
        with t2:
            ws = [l.get_weights()[0] for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
            show(fig_weight_hist(ws))
        with t3:
            a, b = st.columns(2)
            px = a.slider("x1 (scaled)", -3.0, 3.0, 0.0, 0.1)
            py = b.slider("x2 (scaled)", -3.0, 3.0, 0.0, 0.1)
            p = float(model(np.array([[px, py]], dtype="float32"), training=False).numpy()[0, 0])
            st.metric("Predicted class", int(p > 0.5), f"P(class 1) = {p:.1%}")
            show(fig_boundary(run["grid"], snaps[eps[-1]], rXtr, rytr, rXte, ryte, "Your probe point", point=(px, py)))

with tab_act:
    st.markdown("Activation functions add the non-linearity that lets a network bend its decision boundary. "
                "The one you selected in the sidebar is drawn thicker.")
    show(fig_activations(act))
    st.markdown("""
| Function | Notes |
|---|---|
| **relu** | Fast, simple; units can "die" if they only output 0 |
| **tanh** | Smooth, zero-centred; great for small 2-D problems |
| **sigmoid** | Squashes to (0, 1); can learn slowly in deep stacks |
| **elu** | Like relu but smooth for negative inputs |
| **swish** | Smooth, slightly non-monotonic; often works well |
""")

with tab_about:
    st.markdown(f"""
### About
**Neural Network Lab** – web-based interactive neural network learning demo (TensorFlow + Streamlit).

**Author:** {AUTHOR} · **Reg. No.:** {REG_NO} · {CLASS_INFO}

**Try:** *Spiral* with `16,16,8` + tanh → solved; with `2` → underfits. *Half Moons*, noise 0.5,
`32,32,32`, dropout 0 vs 0.3 → watch the train/validation gap.
""")
