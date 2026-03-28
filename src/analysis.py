from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import mlx.core as mx
from mlx.nn.losses import cross_entropy, mse_loss
import polars as pl
from typing import List
import matplotlib.pyplot as plt

from .trainer import unflatten

def mx_corr(a: mx.ArrayLike, b: mx.ArrayLike):
    a_mean = a.mean()
    b_mean = b.mean()
    cov = mx.mean((a - a_mean) * (b - b_mean))
    return cov / (a.std() * b.std())

def vae_loss(model, c_input, categorical_input):
    c_pred, categorical_pred = model(features=  c_input, categorical_features = categorical_input)
    c_mse = mx.mean(mx.sum((c_input - c_pred)**2, axis=-1), axis=-1)
    categorical_loss = mx.array([0.0])
    for name in list(categorical_input.keys()):
        categorical_loss += cross_entropy(categorical_input[name], categorical_pred[name], reduction="mean")
    return c_mse + categorical_loss

def volatility_loss(model, c_input, categorical_input, target):
    pred = model(features=  c_input, categorical_features = categorical_input)
    return mse_loss(pred, targets=target, reduction="mean")
    
def get_vae_residuals(model, buffer, categorical_col):
    model.train(False)
    residuals = []
    for batch in buffer:
        c_input = mx.array(batch["data"], mx.float32)
        categorical_input = unflatten(col=categorical_col, array=batch["categorical"])
        loss_residual = vae_loss(model, c_input, categorical_input)
        residuals.append(loss_residual.item())
    return residuals

def get_volatility_residuals(model, buffer, categorical_col):
    model.train(False)
    residuals = []
    for batch in buffer:
        target = mx.array(batch["target"], mx.float32)
        c_input = mx.array(batch["data"], mx.float32)
        categorical_input = unflatten(col=categorical_col, array=batch["categorical"])
        loss_residual = volatility_loss(model, c_input, categorical_input, target=target)
        residuals.append(loss_residual.item())
    return residuals

def get_vae_residuals_and_grad(model, buffer, categorical_col):
    pass

def get_volatility_residuals_and_grad(model, buffer, categorical_col):
    pass

def analyse_vae_residuals(model, buffer, volatility_sequence: mx.ArrayLike, categorical_col: List[str], save_fig: bool = True, name="vae"):
    residuals = get_vae_residuals(model, buffer, categorical_col)
    _analyse(volatility_sequence=volatility_sequence, residuals=residuals, model_name=name, save_fig=save_fig)

def analyse_volatility_residuals(model, buffer, volatility_sequence: mx.ArrayLike, categorical_col: List[str], save_fig: bool = True, name="volatility_clustering"):
    residuals = get_volatility_residuals(model, buffer, categorical_col)
    _analyse(volatility_sequence=volatility_sequence, residuals=residuals, model_name=name, save_fig=save_fig)

def _analyse(volatility_sequence, residuals, model_name, save_fig = True):
    residuals = mx.array(residuals, dtype=mx.float32)
    assert(len(residuals) == len(volatility_sequence)), f"Volatility && Residuals doesnt have same length, {len(residuals)} | {len(volatility_sequence)}"
    autocorr_lag = 10
    lookback = 5
    lookforward = 5
    corrcoeffs = []

    for ids in mx.arange(0, lookback+1):
        vol = volatility_sequence[: -ids]
        res = residuals[ids :]
        corr = mx_corr(vol, res)
        corrcoeffs.append(corr)

    for ids in mx.arange(1, lookforward+1):
        vol = volatility_sequence[ids :]
        res = residuals[: -ids]
        corr = mx_corr(vol, res)
        corrcoeffs.append(corr)

    fig = plt.figure(figsize=(9, 21))
    ax1 = fig.add_subplot(131)
    ax1.set_title(f"Corr volatility <-> {model_name} residuals")
    ax1.plot(mx.arange(-(lookback+1), lookforward+1), corrcoeffs)
    ax1.grid()

    ax2 = fig.add_subplot(132)
    ax2.set_title(f"{model_name} residuals auto-corr")
    plot_acf(residuals, ax=ax2, lags=autocorr_lag)

    ax3 = fig.add_subplot(133)
    ax3.set_title(f"{model_name} residuals partial auto-corr")
    plot_pacf(residuals, ax=ax3, lags=autocorr_lag)
    
    if save_fig:
        fig.savefig(f"{model_name}_residuals")
    
    plt.show()