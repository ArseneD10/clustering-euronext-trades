from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
from copy import deepcopy
import umap
from typing import List, Dict
import matplotlib.pyplot as plt

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
    mask = pred != -1
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
    group = {n: [] for n in n_clusters}
    for p, label in zip(cluster_pred, pred_labels):
        group[p].append(detokenizer[label])
    
    return group

def get_cluster_intersection(a: Dict[int, List[str]], b: Dict[int, List[str]], keep_treshold = 1):
    intersection = {}
    for ak, av in a.items():
        for bk, bv in b.items():
            b_intersection = [b for b in bv if b in av]
            if len(b_intersection) > keep_treshold:
                intersection[f"{ak}-{bk}"] = b_intersection

    return intersection