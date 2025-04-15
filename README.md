# 📈 Financial ML — Mid-Price Prediction from Limit Order Book (HFT Data)

This project applies Machine Learning and Deep Learning models to **predict mid-price movements** using high-frequency limit order book (LOB) data. Built during MSc coursework, it replicates real-world conditions faced by market makers and HFT systems.

---

## 📦 Project Structure

```
financial-ml-midprice-prediction/
│
├── B246085.ipynb            <- Main notebook (ML pipeline: CNN, LSTM, RBF, MLP, etc.)
├── B246085.pdf              <- Report explaining full methodology and results
├── requirements.txt         <- Python dependencies
├── README.md                <- You're here!
├── /images/                 <- Charts & visualizations
└── Quant-Journal.md         <- Personal learning log / reflection
```

---

## 📊 Dataset

- **Source:** Provided during Financial Machine Learning coursework (Amazon stock)
- **Type:** High-frequency Limit Order Book (LOB)
- **Size:** 562,650 entries (5:00–21:00, single trading day)
- **Features:** 46 columns including 10 levels of bid/ask prices & volumes

> Mid-price is computed as the average of best bid and best ask (Level 1).

---

## ⚙️ Workflow Overview

1. **Preprocessing**
   - Remove missing values (NaN, 0, inf)
   - Filter transactions to NYSE hours (9:30–16:00)
   
2. **Feature Engineering**
   - Lagged features (1 timestep)
   - Rolling mean (window size 4)
   - Log returns

3. **Normalization**
   - MinMax Scaling to improve training and prevent bias

4. **Modeling**
   - CNN (2 conv layers + dense)
   - CNN Autoencoder (feature extraction)
   - LSTM (256 units)
   - Multi-layer Perceptrons (shallow to deep)
   - RBFNN (KMeans + RBFSampler + Ridge)
   - Ensemble Models: Random Forest, Gradient Boosting
   - Classical Models: Bayesian Ridge, ARIMA

5. **Evaluation**
   - Metrics: RMSE, MPE
   - TimeSeriesSplit cross-validation

---

## 📈 Results Summary

| Model                  | RMSE ↓   | MPE (%) ↓   |
|------------------------|----------|-------------|
| Bayesian Regression    | Lowest   | Lowest      |
| CNN Autoencoder        | Highest  | -           |
| Very Deep MLP          | Strong   | Slightly overpredicts |

> Detailed visualizations and full metrics in PDF report & notebook.

---

## 🔍 Key Visuals

Include saved charts in `/images/`, e.g.:
- Actual vs Predicted Mid-Price (MLP, CNN, LSTM)
- RMSE comparisons
- Mid-price time series

---

## 🧠 Learnings

- Handcrafted features matter significantly in LOB settings
- Bayesian Ridge regression worked surprisingly well
- Deep models need proper tuning + normalization to compete

---

## 🚀 Future Work

- Try transformer-based time series models
- Build real-time prediction dashboard
- Integrate with simulated trading environment

---

## 🛠️ Requirements

See `requirements.txt` for exact dependencies. Core libraries include:
- `pandas`, `numpy`, `matplotlib`
- `sklearn`, `tensorflow`, `keras`
- `statsmodels`, `scipy`

---

## 👤 Author
Project by [Your Name] during MSc in Financial Machine Learning (2023–24).

---

## 📚 References
- Ntakaris et al., 2019 — Feature Engineering for Mid-Price Prediction
- Dash & Dash, 2016 — Hybrid Stock Framework
- McKerahan et al., 2023 — AI in Stock Trading

See full list in the PDF report.

