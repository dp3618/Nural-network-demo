# 🔬 Neural Network Lab

**ANN – Activity 2:** Interactive neural network learning demo (web-based, TensorFlow).

A Streamlit app to design a **TensorFlow/Keras** network, train it on a 2-D classification problem, and **replay** how its decision boundary evolved epoch by epoch.

**Author:** YOUR NAME · Reg. No. YOUR REG NO · B.Tech CSE (Data Science), 3rd year

🔗 **Live demo:** _add your Streamlit Cloud link here_

## Features
- **5 datasets:** Two Blobs, Rings, Checkerboard, Spiral, Half Moons (adjustable samples, noise, validation share)
- **Free-form architecture:** type hidden layer sizes such as `16, 16, 8` (1–6 layers, up to 64 neurons each)
- **Activations:** relu, tanh, sigmoid, elu, swish · **Dropout** 0–0.5
- **Training controls:** Adam / RMSprop / SGD + momentum, learning rate, epochs, batch size, seed
- **Live training view:** progress bar with loss/accuracy curves updating as the model learns
- **Training replay slider:** scrub through epochs and watch the decision boundary change
- **Analysis:** metrics, over/underfitting diagnosis, confusion matrix, weight histograms, "probe a point" predictor
- **Activation functions tab:** plots and notes for every activation

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, Keras model builder, training recorder callback |
| `nn_utils.py` | Dataset generation, preprocessing, all Matplotlib figures |
| `requirements.txt` | Dependencies |

## Run locally
```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Use Python 3.10–3.12.

## Deploy on Streamlit Community Cloud
1. Push the 4 files to a public GitHub repo (in the root).
2. <https://share.streamlit.io> → **New app** → pick repo, branch `main`, main file `app.py`.
3. Advanced settings → Python 3.11 → **Deploy**.

## Experiments
- **Spiral**, `16,16,8`, tanh → solved; hidden size `2` → underfits.
- **Half Moons**, noise 0.5, `32,32,32`: dropout 0 vs 0.3 → compare the train/validation gap.
- **Checkerboard** with relu vs sigmoid; learning rate 0.1 vs 0.001.

## Tech stack
TensorFlow / Keras · Streamlit · scikit-learn · Matplotlib · NumPy
