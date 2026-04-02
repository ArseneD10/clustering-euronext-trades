from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
from copy import deepcopy
import umap
from typing import List, Dict
import matplotlib.pyplot as plt
import polars as pl
import mlx.core as mx

SEED = 22
np.random.seed(SEED)


def get_best_cluster(embedding: np.ndarray, max_cluster: int = 100):
    best = {"score": -1.0, "cluster": 0, "model": None}  
    for c in range(2, max_cluster+1):
        km = KMeans(n_clusters=c, init="k-means++", random_state=SEED)
        pred = km.fit_predict(embedding)
        score = silhouette_score(embedding, pred)
        if score > best["score"]:
            best["score"] = score
            best["cluster"] = c
            best["model"] = deepcopy(km)
    return best

def get_dbscan_cluster(embedding: np.ndarray):
    dbs = DBSCAN(eps=1.0, min_samples=1, p=2)
    pred = dbs.fit_predict(embedding)
    if len(np.unique(pred)) <= 1:
        return {"score": -1, "pred": pred, "model": dbs}
    mask = pred != -1
    if mask.sum() == 0:
        score = 0
    else:
        score = silhouette_score(embedding[mask], pred[mask], metric="euclidean")
    return {"score": score, "pred": pred, "model": dbs}

def plot_umap(embedding: np.ndarray, cluster: np.ndarray, save_fig = True):
    um = umap.UMAP(n_neighbors=15) # @todo optimize n_neighbors
    reduced_embedding = um.fit_transform(embedding)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    ax.set_title("Reduced embedding Umap")
    ax.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], c=cluster)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid()
    if save_fig:
        fig.savefig("pca_cluster")
    plt.show()

def plot_pca_cluster(embedding: np.ndarray, cluster: np.ndarray, save_fig = True):
    assert(len(embedding) == len(cluster))
    pca = PCA(n_components=2)
    reduced_embedding = pca.fit_transform(embedding)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    ax.set_title("Reduced embedding PCA")
    ax.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], c=cluster)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid()
    if save_fig:
        fig.savefig("pca_cluster")
    plt.show()

def get_group(cluster_pred ,pred_labels: List[int], sub_tokenizer: Dict[str, int]):
    detokenizer = {v: k for k, v in sub_tokenizer.items()}
    n_clusters = np.unique(cluster_pred)
    group = {str(n): [] for n in n_clusters}
    for p, label in zip(cluster_pred, pred_labels):
        group[str(p)].append(detokenizer[label])
    
    return group

def get_cluster_intersection(a: Dict[int, List[str]], b: Dict[int, List[str]], keep_treshold = 1):
    intersection = {}
    for ak, av in a.items():
        for bk, bv in b.items():
            b_intersection = [b for b in bv if b in av]
            if len(b_intersection) > keep_treshold:
                intersection[f"{ak}-{bk}"] = b_intersection

    return intersection



def run_analyse(df: pl.DataFrame, col, model, tokenizer): # @Arsene to improve

    data = df.explode(pl.all().exclude(pl.Int32)).select(col).unique()
    subtokenizer = tokenizer[col]
    data = data.to_numpy()
    data = mx.array(data, dtype=mx.int32)
    print(data.shape)
    model_inpt = {col: data}
    pred = model.feature_encoder.categorical_encoder({col: data})[col]
    mx.eval(pred)
    if len(pred.shape) > 2:
        pred = pred[:,0,:]
    pred = np.array(pred)
    best_km = get_best_cluster(embedding=pred, max_cluster=min(100, np.max(data.shape) - 1))

    print("Best k-mean -->", best_km)
    km_pred = best_km["model"].predict(pred)
    data_np = np.array(data[:, 0])
    cluster = get_group(cluster_pred=km_pred, pred_labels=data_np, sub_tokenizer=subtokenizer)

    best_dbs = get_dbscan_cluster(embedding=pred)
    dbs_pred = best_dbs["pred"]
    dbs_cluster = get_group(cluster_pred=dbs_pred, pred_labels=data_np, sub_tokenizer=subtokenizer)
    print("Nb dbs cluster-->", len(dbs_cluster.keys()))
    print(dbs_cluster)
    plot_pca_cluster(embedding=pred, cluster=km_pred, save_fig=False)
    plot_umap(embedding=pred, cluster=km_pred, save_fig=False)
    return cluster