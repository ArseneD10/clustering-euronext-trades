from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import mlx.core as mx
from mlx.nn.losses import cross_entropy, mse_loss
import polars as pl
from typing import List
import matplotlib.pyplot as plt

from .trainer import unflatten
from tqdm import tqdm
import numpy as np


def mx_corr(a: mx.ArrayLike, b: mx.ArrayLike):
    a_mean = a.mean()
    b_mean = b.mean()
    cov = mx.mean((a - a_mean) * (b - b_mean), axis=-1)
    return cov / (a.std() * b.std())

def vae_loss(model, c_input, categorical_input):
    LAMBDA_CAT = 0.01
    BETA = 0.1
    c_pred, categorical_pred, params = model(features=c_input, categorical_features=categorical_input)
    c_mse = mse_loss(c_pred, c_input, reduction="mean")
    categorical_loss = mx.array(0.0, dtype=mx.float32)
    for name in list(categorical_input.keys()):
        categorical_loss += cross_entropy(categorical_input[name], categorical_pred[name], reduction="mean") 
    mu, logvar = params
    kl_loss = mx.mean(-0.5 * mx.sum(1. + logvar - mu**2 - mx.exp(logvar)))

    return c_mse + (categorical_loss*LAMBDA_CAT) + kl_loss * BETA

def volatility_loss(model, c_input, categorical_input, target):
    pred = model(features=  c_input, categorical_features = categorical_input)
    return mse_loss(pred[0][0], targets=target, reduction="mean")
    
def get_vae_residuals(model, buffer, categorical_col):
    model.train(False)
    residuals = {}
    isin_ids = [ids for ids, c in enumerate(categorical_col) if c == "MifidInstrumentID"][0]
    for batch in tqdm(buffer):
        c_input = mx.array(batch["data"], mx.float32)[None, :, :]
        categorical_input = unflatten(col=categorical_col, array=batch["categorical"], unsqueeze=True)
        isin_id = int(batch["categorical"][0, isin_ids])
        loss_residual = vae_loss(model, c_input, categorical_input)
        
        if residuals.get(isin_id):
            residuals[isin_id].append(loss_residual.item())
        else:
            residuals[isin_id] = [loss_residual.item()]
    
    return residuals

def get_volatility_residuals(model, buffer, categorical_col, scale):
    model.train(False)
    residuals = {}
    isin_ids = [ids for ids, c in enumerate(categorical_col) if c == "MifidInstrumentID"][0]
    for batch in tqdm(buffer):
        target = mx.array(batch["target"], mx.float32) * scale
        c_input = mx.array(batch["data"], mx.float32)[None, :, :]
        categorical_input = unflatten(col=categorical_col, array=batch["categorical"], unsqueeze=True)
        loss_residual = volatility_loss(model, c_input, categorical_input, target=target)
        isin_id = int(batch["categorical"][0, isin_ids])
        
        if residuals.get(isin_id):
            residuals[isin_id].append(loss_residual.item() / (scale**2))
        else:
            residuals[isin_id] = [loss_residual.item() / (scale**2)]
    
    return residuals

def get_vae_residuals_and_grad(model, buffer, categorical_col):
    pass

def get_volatility_residuals_and_grad(model, buffer, categorical_col):
    pass


def get_volatility_model_activation(model, buffer, categorical_col, scale=10, max_isin = 10):
    model.train(False)
    tree = {}
    count = {}
    isin_ids = [ids for ids, c in enumerate(categorical_col) if c == "MifidInstrumentID"][0]

    for batch in tqdm(buffer):
        c_input = mx.array(batch["data"], mx.float32)[None, :]
        categorical_input = unflatten(col=categorical_col, array=batch["categorical"], unsqueeze=True)
        _, activation = model.forward_with_activation(features=  c_input, categorical_features = categorical_input)
        isin_id = int(batch["categorical"][0, isin_ids])
        
        if tree.get(isin_id):
            isin_tree = tree[isin_id]
            for (k1, v1), (k2, v2) in zip(isin_tree.items(), activation.items()):
                tree[isin_id][k1] += v2 
            count[isin_id] += 1
            
        else:
            if len(tree.keys()) > max_isin:
                continue
            tree[isin_id] = activation
            count[isin_id] = 1
    
    for isin in list(count.keys()):
        isin_tree = tree[isin]
        for k in list(isin_tree.keys()):
            tree[isin][k] /= count[isin_id]
    
    return tree

def isin_to_company_name(isin: int):
    pass

def get_detokenizer(tokenizer, sub):
    return {v: k for k, v in tokenizer[sub].items()}

def _indexing_mlx(array, isin):
    # mlx doesnt implement boolean indexing since the array's size must be previsible 
    np_array = np.array(array)
    mask = np_array[:, 1] == isin
    np_array = np_array[mask] # [SEQUENCE, 2] (Vol, isin ids)
    return mx.array(np_array[:, 0]) 

def analyse_vae_residuals(model, buffer, volatility_sequence: mx.ArrayLike, categorical_col: List[str], save_fig: bool = True, name="vae", top_pct=0.5):
    residuals = get_vae_residuals(model, buffer, categorical_col)
    mean_corr = get_most_significant_analyse(volatility_sequence, residuals)
    top_k = int(len(mean_corr)*top_pct)
    print(top_k, '<_top k')
    best_corr = mean_corr.top_k(k=top_k, by="score", reverse=False)
    print("View of topk ->", best_corr)
    for row in best_corr.iter_rows(named=True):
        isin = row["isin_id"]
        res = residuals[isin]
        vol_seq = _indexing_mlx(volatility_sequence, isin)
        _analyse(volatility_sequence=vol_seq, residuals=res, model_name=name, save_fig=save_fig, isin=isin)

def analyse_volatility_residuals(model, buffer, volatility_sequence: mx.ArrayLike, categorical_col: List[str], scale, save_fig: bool = True, name="volatility_clustering", top_pct: float=0.3):
    residuals = get_volatility_residuals(model, buffer, categorical_col, scale)
    mean_corr = get_most_significant_analyse(volatility_sequence, residuals)
    top_k = int(len(mean_corr)*top_pct)
    print(top_k, '<_top k')
    best_corr = mean_corr.top_k(k=top_k, by="score", reverse=False)
    print("View of topk ->", best_corr)
    for row in best_corr.iter_rows(named=True):
        isin = row["isin_id"]
        res = residuals[isin]
        vol_seq = _indexing_mlx(volatility_sequence, isin)
        _analyse(volatility_sequence=vol_seq, residuals=res, model_name=name, save_fig=save_fig, isin=isin)

def _analyse(volatility_sequence, residuals, model_name, save_fig = True, isin: str=""):
    residuals = mx.array(residuals, dtype=mx.float32)
    assert(len(residuals) == len(volatility_sequence)), f"Volatility && Residuals doesnt have same length, {len(residuals)} | {len(volatility_sequence)}"
    autocorr_lag = 30
    lookback = 15
    lookforward = 15
    corrcoeffs = []

    if len(residuals) < autocorr_lag+1:
        return []

    for ids in range(1, lookback+1):
        vol = volatility_sequence[: -ids]
        res = residuals[ids :]
        corr = mx_corr(vol, res)
        corrcoeffs.append(corr)

    corr = mx_corr(volatility_sequence, residuals)
    corrcoeffs.append(corr)

    for ids in range(1, lookforward+1):
        vol = volatility_sequence[ids :]
        res = residuals[: -ids]
        corr = mx_corr(vol, res)
        corrcoeffs.append(corr)
    index = np.arange(len(corrcoeffs)) - lookforward

    
    if save_fig:
        fig = plt.figure(figsize=(21, 9))
        ax1 = fig.add_subplot(611)
        ax1.set_title(f"Corr volatility <-> {model_name} residuals")
        ax1.plot(index, corrcoeffs)
        ax1.set_xlabel("Residuals t")
        ax1.set_ylabel("Corr")
        ax1.grid()

        ax2 = fig.add_subplot(612)
        plot_acf(residuals, ax=ax2, lags=autocorr_lag, zero=False, auto_ylims=True)
        ax2.set_title(f"{model_name} residuals auto-corr")
                
        ax3 = fig.add_subplot(613)
        plot_pacf(residuals, ax=ax3, lags=autocorr_lag, zero=False, auto_ylims=True)  
        ax3.set_title(f"{model_name} residuals partial auto-corr")
        
        ax4 = fig.add_subplot(614)
        plot_acf(volatility_sequence, ax=ax4, lags=autocorr_lag, zero=False, auto_ylims=True)
        ax4.set_title(f"{model_name} volatility auto-corr")
        
        ax5 = fig.add_subplot(615)
        plot_pacf(volatility_sequence, ax=ax5, lags=autocorr_lag, zero=False, auto_ylims=True)  
        ax5.set_title(f"{model_name} volatility partial auto-corr")
        

        ax6 = fig.add_subplot(616)
        ax6.set_title(f"{model_name} residuals")
        ax6.plot(range(len(residuals)), residuals)
        ax6.set_xlabel("t")
        ax6.set_ylabel("résiduals")
        ax6.grid()


        fig.savefig(f"plot/{model_name}_residual_{isin}")
        
        plt.show()
    
    return corrcoeffs

def get_most_significant_analyse(volatility_sequence, residuals):
    min_lenght = 10 # forward+backward
    isin_id = list(residuals.keys())
    score = {"isin_id": [], "score": []}
    for isin in isin_id:
        res = residuals[isin]
        if len(res) < min_lenght:
            score["isin_id"].append(isin)
            score["score"].append(0.0)
            continue
        vol_seq = _indexing_mlx(volatility_sequence, isin)
        corr = _analyse(volatility_sequence=vol_seq, residuals=res, model_name="", save_fig=False)
        mean_corr = np.mean(corr)
        score["isin_id"].append(isin)
        score["score"].append(mean_corr)
    return pl.from_dict(score, schema={"isin_id": pl.Int64, "score": pl.Float64})


def plot_hook(hook):
    fig = plt.figure(figsize=(10, 10))
    ax=[]
    size = len(hook)
    for idx, (name, data) in enumerate(hook.items()):
        ax.append(fig.add_subplot(size, 1, idx+1))
        ax[idx].set_title(name)
        ax[idx].imshow(np.array(data))
    plt.show()

def get_hook_path(hook, top_layer=0.1):
    path = []
    max_lenght = 0.
    for data in hook.values():
        data = np.array(data).flatten()
        max_lenght = max(len(data), max_lenght)
    
    for name, data in hook.items():
        data = np.array(data).flatten()
        padd = np.zeros(shape=(max_lenght, ))
        top = np.quantile(data, q=1-top_layer)
        mask = data > top
        padd_left = int((max_lenght - len(mask)) / 2) 
        padd_right = max_lenght - len(mask) - padd_left

        if padd_right == 0:
            padd = padd[padd_left :] = mask
        else:
            padd[padd_left :-padd_right] = mask
        path.append(padd) 
    path = np.stack(path)
    return path

def plot_hook_path(hook, top_layer = 0.1):
    path = get_hook_path(hook=hook,top_layer=top_layer)
    pct = top_layer*100
    plt.figure(figsize=(70, 20))
    plt.title(f"TOP {pct}% magnitude")
    plt.imshow(path)
    plt.ylabel(list(hook.keys()))
    plt.colorbar()
    plt.show()


def get_hook_similiarity(tree: dict, top_layer=0.1):
    def sim_fun(a, b):
        assert a.shape == b.shape
        intersection = ((a == 1) & (b == 1)).sum()
        union = ((a == 1) | (b == 1)).sum() 
        return intersection / union
        
    n = len(tree.keys())
    score = np.zeros((n, n))   
    avg_score = 0
    all_hook_path = []
    
    for isin in list(tree.keys()):
        hook = tree[isin]
        hook_path = get_hook_path(hook, top_layer=top_layer)  
        all_hook_path.append(hook_path)
    
    for i, hook_path1 in enumerate(all_hook_path):
        for j, hook_path2 in enumerate(all_hook_path):
            sim = sim_fun(hook_path1, hook_path2)
            score[i, j] = sim
            if i != j:
                avg_score += sim
    
    avg_score /= (n*n - n)
    return score, avg_score

