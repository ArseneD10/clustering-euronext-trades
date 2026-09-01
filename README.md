# Trade Flow Market Regime Analysis
> **Project Status:** 🛠️ *Active Research / Training Phase (updated September 2026)*
> **Target:** Unsupervised discovery of market regimes through order-flow embeddings.

---

## Overview

This project investigates whether the **structural profile of trades** — independently of price movements — carries enough information to characterise market regimes and anticipate volatility.

The central hypothesis is the following:

$$\sigma = f(\Omega) \mid \Omega \not\ni R_t$$

where $\sigma$ is realised volatility and $\Omega$ is a feature set that contains **no direct price-return signal** (volume, trade timing, trade type, …).

To test this, a **multi-task transformer encoder** is trained on several complementary objectives. The shared latent representations it produces are then used for clustering, regime detection, and cross-asset correlation analysis.

---

## Data

| Field | Value |
|---|---|
| Source | Euronext trade tape (CSV) |
| Universe | 624 small-cap constituents (< EUR 1B market cap) |
| Coverage | 10 March 2026 → 20 March 2026 |
| Processing library | [Polars](https://pola.rs/) — chosen for its performance on large tabular datasets |
| Train/test split | Non-overlapping, chronological blocks per asset, with a purge margin around block boundaries to remove leakage from overlapping trade-sequence windows |

The raw tape is read, filtered, and engineered into sequences of trades per asset before being fed into the model.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  (1)  INPUT ENCODING                                        │
│                                                             │
│   ┌──────────────────────┐  ┌──────────────────────────┐   │
│   │  Categorical Encoder │  │    Feature Encoder       │   │
│   │  (Ticker, Trade type)│  │  (Log-volume, Δt, …)     │   │
│   └──────────┬───────────┘  └────────────┬─────────────┘   │
│              └──────────────┬────────────┘                  │
│                             ▼                               │
│  (2)  ENCODER BLOCK                                         │
│       Transformer — Residual Attention + MLP                │
│                             │                               │
│  (3)  TASK HEADS  ◄─────────┘                               │
│       (A) VAE   (B) Volatility Est.*  │
└─────────────────────────────────────────────────────────────┘
```

### (1) Input Encoding

Two parallel branches encode heterogeneous inputs before fusion:

- **Categorical Encoder** — embeds discrete fields (company ticker, trade type, …)
- **Feature Encoder** — projects continuous fields (log-volume, inter-trade duration, …)

### (2) Shared Encoder Block

A standard transformer block with **residual attention + MLP sublayers**.

### (3) Task Heads

| ID | Task | Description |
|---|---|---|
| **(A)** | Variational Auto-Encoder | The model receives trade sequences, compresses them into a latent code, and reconstructs them. Encourages the encoder to learn a *normal* representation of per-asset trade dynamics, conditioned on the company embedding. |
| **(B)** | Volatility Estimator (contemporaneous) | Using only price-agnostic features ($\Omega$), the model estimates the **realised standard deviation of returns over the same window** the features are drawn from: $\hat\sigma = f(\Omega)$. This is a nowcasting task, not a forecast, it asks what volatility the observed order-flow mechanics alone would imply, so that departures from that estimate (Application B) can be read as structural anomalies rather than noise. |


> **Note on tasks B** — A probabilistic output of the form $V = \mu + \sigma \cdot \varepsilon$ was tested but proved unstable during training; the final model uses a direct regression head.

---

## Applications

| # | Application |
|---|---|
| **A** | **Asset clustering** via company-ticker embeddings (k-means or equivalent on the categorical embeddings) |
| **B** | **Residual–volatility study**: correlation analysis between Task B's estimation residuals ($\sigma_{\mathrm{realised}} - \hat\sigma$) and next-period volatility $\sigma_{t+1}$, as an early signal of structural regime shifts |
| **C** | **Trade clustering and impact on price**: via umap decomposition + kmeans clustering, analysis which trade induce most volatility in the market |
| **D** | **Activation concentration analysis**: examination of where activations concentrate across all layers after fine-tuning steps |

---

## Project Structure

```
.
├── training.ipynb          # Data loading, processing, and multi-task model training
├── analysis.ipynb          # Loads trained models to produce embeddings, clustering & result analysis
└── src/
    ├── analysis.py         # Residual analysis & activation magnitude inspection
    ├── base_encoder.py     # Shared feature encoding backbone
    ├── clustering.py       # Embedding clustering & visualisation
    ├── config.py           # Model & optimiser hyperparameters
    ├── directional_nn.py   # Directional prediction head (task C)
    ├── processing.py       # Data pipeline
    ├── trainer.py          # Trainer class
    ├── vae.py              # Variational Auto-Encoder head (task A)
    └── volatility_nn.py    # Volatility estimation head (task B)
```

---

## Workflow

Experiments are split across two notebooks:

**`training.ipynb`** — data to trained models:
1. Data loading and preprocessing via Polars
2. Feature engineering and sequence construction
3. Multi-task model training (tasks A, B, and C), with temporal train/validation/test splits per asset to prevent leakage from overlapping sequence windows

**`analysis.ipynb`** — trained models to results:
4. Embedding extraction and clustering
5. Residual, gradient and activation analyses

---

## Research Insights & Challenges (In Progress)

Current observations, to be read as preliminary given training is still ongoing:

- **Latent Space:** The VAE shows strong convergence on reconstruction, suggesting the encoder effectively captures the "syntax" of trade flows.
