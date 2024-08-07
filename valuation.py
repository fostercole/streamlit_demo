"""
Main implmentation for diversity and relevance measures for data valuation
"""
import collections
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.linalg import cosm
from scipy.stats import gmean, pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              WeightedRandomSampler)
from torchvision import models, transforms
from vendi_score import vendi


def covariance(X, normalize=True):
    """
    Computes covariance matrix
    """
    if normalize:
        X = X - X.mean(0)
    n = X.shape[0]
    norm = n - 1
    gram = np.dot(X.T, X)
    return gram / norm


def svd(covariance_matrix):
    """
    Eigendecomposition of covariance matrix
    """
    eig_val, eig_vec = np.linalg.eig(covariance_matrix)
    eig_val = eig_val.real
    return eig_val, eig_vec


def compute_volume(cov, epsilon=1e-8):
    """
    Compute volume of covariance matrix
    """
    return np.sqrt(np.linalg.det(cov) + epsilon)


def compute_volumes(datasets, d=1):
    """
    Compute volume of data matrix
    """
    volumes = np.zeros(len(datasets))
    for i, dataset in enumerate(datasets):
        volumes[i] = np.sqrt(np.linalg.det(dataset.T @ dataset) + 1e-8)
    return volumes


def compute_projected_volumes(datasets, projection, d=1):
    """
    Compute projected volume of data matrix
    """
    volumes = np.zeros(len(datasets))
    for i, dataset in enumerate(datasets):
        volumes[i] = np.sqrt(np.linalg.det(dataset.T @ dataset) + 1e-8)
    return volumes


def compute_X_tilde_and_counts(
    X: torch.Tensor, omega: float = 0.1
) -> Tuple[np.ndarray, Dict]:
    """
    https://github.com/opendataval/opendataval/blob/main/opendataval/dataval/volume/rvs.py

    Compresses the original feature matrix X to  X_tilde with the specified omega.

    Returns
    -------
    np.ndarray
       compressed form of X as a d-cube
    dict[tuple, int]
       cubes: a dictionary of cubes with the respective counts in each dcube
    """
    assert 0 < omega <= 1, "omega must be within range [0,1]."

    # Dictionary to store frequency for each cube
    cubes = collections.Counter()
    omega_dict = collections.defaultdict(list)
    min_ds = np.min(X, axis=0)

    # a dictionary to store cubes of not full size
    for entry in X:
        cube_key = tuple(math.floor(ent.item() / omega) for ent in entry - min_ds)
        cubes[cube_key] += 1
        omega_dict[cube_key].append(entry)

    X_tilde = np.stack([np.mean(value, axis=0) for value in omega_dict.values()])
    return X_tilde, cubes


def compute_robust_volumes(X_tilde: np.ndarray, hypercubes: dict[tuple, int]):
    """
    https://github.com/opendataval/opendataval/blob/main/opendataval/dataval/volume/rvs.py
    """
    alpha = 1.0 / (10 * len(X_tilde))  # it means we set beta = 10

    flat_data = X_tilde.reshape(-1, X_tilde.shape[1])
    (sign, volume) = np.linalg.slogdet(np.dot(flat_data.T, flat_data))
    robustness_factor = 1.0

    for freq_count in hypercubes.values():
        robustness_factor *= (1 - alpha ** (freq_count + 1)) / (1 - alpha)

    return sign, volume, robustness_factor


def get_volume(X, omega=0.1, norm=False):
    """
    From https://github.com/ZhaoxuanWu/VolumeBased-DataValuation/blob/main/volume.py
    """
    X_tilde, cubes = compute_X_tilde_and_counts(X, omega=omega)
    sign, vol, robustness_factor = compute_robust_volumes(X_tilde, cubes)
    robust_vol = robustness_factor * vol
    return dict(
        robust_vol=robust_vol, sign=sign, vol=vol, robustness_factor=robustness_factor
    )


def cluster_valuation(
    buyer_data, seller_data, k_means=None, n_clusters=10, n_components=25
):
    if k_means is None:
        k_means = KMeans(n_clusters=n_clusters, n_init="auto")
        k_means.fit(buyer_data)
    buyer_clusters = {
        k: buyer_data[k_means.predict(buyer_data) == k] for k in range(n_clusters)
    }
    seller_clusters = {
        k: seller_data[k_means.predict(seller_data) == k] for k in range(n_clusters)
    }
    cluster_rel = {}
    cluster_vol = {}
    # for j in tqdm(range(n_clusters)):
    for j in range(n_clusters):
        cluster_pca = PCA(
            n_components=n_components, svd_solver="randomized", whiten=False
        )
        cluster_pca.fit(buyer_clusters[j])
        ws = []
        rs = []
        vs = []
        for i in range(n_clusters):
            if seller_clusters[i].shape[0] == 0 or seller_clusters[i].shape[0] == 1:
                ws.append(0)
                rs.append(0)
                vs.append(0)
            else:
                ws.append(seller_clusters[i].shape[0] / seller_data.shape[0])
                rs.append(valuation.get_relevance(cluster_pca, seller_clusters[i]))
                # vs.append(valuation.get_volume(np.cov(cluster_pca.transform(seller_clusters[i]).T)))
                vs.append(
                    valuation.get_volume(cluster_pca.transform(seller_clusters[i]))
                )
        cluster_rel[j] = np.average(rs, weights=ws)
        cluster_vol[j] = np.average(vs, weights=ws)
    buyer_weights = [v.shape[0] / buyer_data.shape[0] for v in buyer_clusters.values()]
    # print(buyer_weights)
    rel = np.average(list(cluster_rel.values()), weights=buyer_weights)
    vol = np.average(list(cluster_vol.values()), weights=buyer_weights)
    return rel, vol


def compute_eigen_rel_div(buyer_values, seller_values, threshold=0.1):
    # only include directions with value above this threshold
    keep_mask = buyer_values >= threshold

    C = np.maximum(buyer_values, seller_values)
    div_components = np.abs(buyer_values - seller_values) / C
    rel_components = np.minimum(buyer_values, seller_values) / C
    rel = np.prod(np.where(keep_mask, rel_components, 1)) ** (
        1 / max(1, keep_mask.sum())
    )
    div = np.prod(np.where(keep_mask, div_components, 1)) ** (
        1 / max(1, keep_mask.sum())
    )
    return rel, div


def get_measurements(
    buyer_data,
    seller_data,
    threshold=0.1,
    n_components=10,
    verbose=False,
    normalize=False,
    omega=0.1,
    dtype=np.float32,
    decomp=None,
    decomp_kwargs={},
    # use_smallest_components = False,
    use_rbf_kernel=False,
    use_neg_components=False,
    # neg_weight=0.2,
    num_neg=10,
    use_dp=False,
    dp_epsilon=0.1,
    dp_delta=None,
    return_components=False,
):
    """
    Main valuation function
    """
    start_time = time.perf_counter()
    buyer_data = np.array(buyer_data, dtype=dtype)
    seller_data = np.array(seller_data, dtype=dtype)
    seller_cov = np.cov(seller_data, rowvar=False)
    buyer_cov = np.cov(buyer_data, rowvar=False)
    buyer_val, buyer_vec = np.linalg.eig(buyer_cov)
    order = np.argsort(buyer_val)[::-1]
    sorted_buyer_val = buyer_val[order]
    sorted_buyer_vec = buyer_vec[:, order]

    slice_index = np.s_[:n_components]
    buyer_values = sorted_buyer_val.real[slice_index]
    buyer_components = sorted_buyer_vec.real[:, slice_index]

    if decomp is not None:
        Decomp = decomp(n_components=n_components, **decomp_kwargs)
        Decomp.fit(buyer_data)
        Decomp.mean_ = np.zeros(seller_data.shape[1])  # dummy mean
        proj_buyer_cov = Decomp.transform(buyer_cov)
        proj_seller_cov = Decomp.transform(seller_cov)
        # seller_values = np.linalg.norm(proj_seller_cov, axis=0)
    else:
        proj_buyer_cov = buyer_cov @ buyer_components
        proj_seller_cov = seller_cov @ buyer_components

    seller_values = np.linalg.norm(proj_seller_cov, axis=0)
    rel, div = compute_eigen_rel_div(buyer_values, seller_values, threshold=threshold)
    M, D = seller_data.shape

    if decomp is not None:
        # project seller data onto buyer's components
        X_sell = Decomp.transform(seller_data)
    else:
        X_sell = seller_data @ buyer_components

    # Entropy based diversity https://arxiv.org/abs/2210.02410
    K = lambda a, b: np.exp(-np.linalg.norm(a - b))
    if use_rbf_kernel:
        vs = vendi.score(X_sell, K, normalize=True)
    else:
        vs = vendi.score_dual(X_sell, normalize=True)

    if normalize:
        Norm = Normalizer(norm="l2")
        X_sell = Norm.fit_transform(X_sell)

    # Dispersion based diversity https://arxiv.org/abs/2003.08529
    dis = gmean(np.std(X_sell, axis=0))

    # Volume based diversity https://proceedings.neurips.cc/paper/2021/file/59a3adea76fadcb6dd9e54c96fc155d1-Paper.pdf
    vol = get_volume(X_sell, omega=omega)["robust_vol"]

    # Compute the cosine similarity and L2 Distance
    # buyer_mean = np.mean(buyer_cov, axis=0)
    # seller_mean = np.mean(seller_cov, axis=0)
    # cos = np.dot(buyer_mean, seller_mean) / (np.linalg.norm(buyer_mean) * np.linalg.norm(seller_mean))
    # l2 = - np.linalg.norm(buyer_mean - seller_mean) # negative since we want the ordering to match
    buyer_mean = np.mean(proj_buyer_cov, axis=0)
    seller_mean = np.mean(proj_seller_cov, axis=0)

    if use_dp:
        noise = gaussian_mechanism(proj_seller_cov, epsilon=dp_epsilon, delta=dp_delta)
        seller_mean += noise
    
    cos = np.dot(buyer_mean, seller_mean) / (
        np.linalg.norm(buyer_mean) * np.linalg.norm(seller_mean)
    )
    l2 = -np.linalg.norm(
        buyer_mean - seller_mean
    )  # negative since we want the ordering to match

    corr = pearsonr(buyer_mean, seller_mean).statistic
    
    ret = dict(
        correlation=corr,
        overlap=rel,
        l2=l2,
        cosine=cos,
        difference=div,
        volume=vol,
        vendi=vs,
        dispersion=dis,
    )
    if return_components:
        ret['buyer_components'] = buyer_values
        ret['seller_components'] = seller_values
        

    if use_neg_components:
        neg_slice_index = np.random.choice(np.arange(round(D * 0.8), D), num_neg)
        neg_buyer_values = sorted_buyer_val.real[neg_slice_index]
        neg_buyer_components = sorted_buyer_vec.real[:, neg_slice_index]
        neg_seller_values = np.linalg.norm(seller_cov @ neg_buyer_components, axis=0)
        neg_rel, neg_div = compute_eigen_rel_div(
            neg_buyer_values, neg_seller_values, threshold=threshold
        )
        neg_X_sell = seller_data @ neg_buyer_components
        neg_dis = gmean(np.std(neg_X_sell, axis=0))
        if use_rbf_kernel:
            neg_vs = vendi.score(neg_X_sell, K, normalize=True)
        else:
            neg_vs = vendi.score_dual(neg_X_sell, normalize=True)
        neg_vol = get_volume(neg_X_sell, omega=omega)["robust_vol"]

        neg_proj_buyer_cov = buyer_cov @ neg_buyer_components
        neg_proj_seller_cov = seller_cov @ neg_buyer_components

        neg_buyer_mean = np.mean(neg_proj_buyer_cov, axis=0)
        neg_seller_mean = np.mean(neg_proj_seller_cov, axis=0)
        neg_cos = np.dot(neg_buyer_mean, neg_seller_mean) / (
            np.linalg.norm(neg_buyer_mean) * np.linalg.norm(neg_seller_mean)
        )
        neg_l2 = -np.linalg.norm(neg_buyer_mean - neg_seller_mean)
        ret["neg_relevance"] = neg_rel
        ret["neg_l2"] = neg_l2
        ret["neg_cosine"] = neg_cos
        ret["neg_diversity"] = neg_div
        ret["neg_dispersion"] = neg_div
        ret["neg_volume"] = neg_vol
        ret["neg_vendi"] = neg_vs

    end_time = time.perf_counter()

    if verbose:
        print(f"{slice_index=}")
        print(f"{buyer_values=}")
        print(buyer_components.shape)
        print(seller_values.shape)
        print(f"{seller_values=}")
        # print(f"{keep_mask.nonzero()[0].shape[0]=}")
        print(np.prod(seller_values))
        print("time", end_time - start_time)

    return ret



def gaussian_mechanism(data, sensitivity=None, delta=None, epsilon=0.1):
    """
    Applies the Gaussian mechanism for differential privacy.

    data: The original output of the function/query before adding noise.
    sensitivity: The sensitivity of the function/query. It measures the maximum change in the output
                        that any single individual's data can cause.
                    Defaults to absolute difference between max and min values divided by sample count
    delta: The parameter for the Gaussian mechanism that allows for a small probability of the
                  privacy guarantee not holding. This should be smaller than the inverse of any imaginable
                  dataset size. 
                  Defaults to 1/(dataset size)^2
    epsilon: The privacy loss parameter. Smaller values mean better privacy.
    
    return: The differentially private result of the function/query after adding Gaussian noise.

    """
    data = np.array(data)
    assert data.ndim >= 2, 'Check data shape. Assumes first dimension is number samples'
    
    # Calculate the sigma (standard deviation) for the Gaussian distribution
    N = data.shape[1]
    
    if sensitivity is None:
        sensitivity = np.abs(data.max(0) - data.min(0)) / N

    if delta is None:
        delta = 1 / N**2
        
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * (sensitivity / epsilon)

    # Generate the noise to add to the data
    noise = np.random.normal(0, sigma, size=N)

    # Add the noise to the original data and return
    return noise