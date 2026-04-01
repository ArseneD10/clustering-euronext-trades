# Trade Flow Market Regime Analysis
> **Project Status:** 🛠️ *Active Research / Training Phase (as of April 2026)* > **Target:** Unsupervised discovery of market regimes through order-flow embeddings.

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
| Coverage | 10 March 2026 → 20 March 2026 |
| Processing library | [Polars](https://pola.rs/) — chosen for its performance on large tabular datasets |

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
│       (A) VAE   (B) Volatility Est.*   (C) Volatility t+1   │
└─────────────────────────────────────────────────────────────┘
```

### (1) Input Encoding

Two parallel branches encode heterogeneous inputs before fusion:

- **Categorical Encoder** — embeds discrete fields (company ticker, trade type, …)
- **Feature Encoder** — projects continuous fields (log-volume, inter-trade duration, …)

### (2) Shared Encoder Block

A standard transformer block with **residual attention + MLP sublayers**

### (3) Task Heads

| ID | Task | Description |
|---|---|---|
| **(A)** | Variational Auto-Encoder | The model receives trade sequences, compresses them into a latent code, and reconstructs them. Encourages the encoder to learn a *normal* representation of per-asset trade dynamics, conditioned on the company embedding. |
| **(B)** | Volatility Estimator | Using only price-agnostic features ($\Omega$), the model predicts the **realised standard deviation of returns** over the selected window: $V = f(\Omega)$. |
| **(C)** *(experimental)* | Volatility Predictor | Predicts the **Volatility estimated** over the next period. Tested in two settings: (i) using frozen encoders from tasks A & B to measure transfer value, (ii) using a randomly, not frozen, initialised  encoder as a baseline. |

> **Note on task B and C** — A probabilistic output of the form $V = \mu + \sigma \cdot \varepsilon$ was tested but proved unstable during training; the final model uses a direct regression head.

---

## Applications

| # | Application |
|---|---|
| **A** | **Asset clustering** via company-ticker embeddings (k-means or equivalent on the categorical embeddings) |
| **B** | **Residual–volatility study**: correlation analysis between model residuals and next-period volatility $\sigma_{t+1}$ |
| **C** | **Gradient concentration analysis**: examination of where gradients focus in the early layers after fine-tuning steps |
| **D** | **Activation concentration analysis**: examination of where activation focus all layers after fine-tuning steps |

---

## Project Structure

```
.
├── notebook.ipynb          # End-to-end research pipeline (data → training → analysis)
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

All experiments are orchestrated from **`notebook.ipynb`**, which covers:

1. Data loading and preprocessing via Polars
2. Feature engineering and sequence construction
3. Multi-task model training (tasks A, B, and C)
4. Embedding extraction and clustering
5. Residual, gradient and activation analyses

---


## Research Insights & Challenges (In Progress)
Current Observations:

Latent Space: The VAE shows strong convergence on reconstruction, suggesting the encoder effectively captures the "syntax" of trade flows.

The Volatility Gap: Preliminary results for Task B show the model currently struggles to outperform a naive mean baseline (MSE ~1.48 vs 1.45). This suggests a high noise-to-signal ratio in price-agnostic features or a potential distribution shift between training and test sets.

