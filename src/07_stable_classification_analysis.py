import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture

import time
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
# 导入Path
from pathlib import Path
import os



# =========================
# 1. 基础工具
# =========================

@dataclass
class PipelineConfig:
    intermediate_dim: int = 2500
    final_dim: int = 300
    random_state: int = 42
    n_clusters: int = 2


def safe_pca_dims(X: np.ndarray, intermediate_dim: int, final_dim: int) -> Tuple[int, int]:
    """
    自动调整 PCA 维度，避免超过样本数/特征数限制
    """
    n_samples, n_features = X.shape
    max_dim = min(n_samples - 1, n_features - 1)
    inter_dim = min(intermediate_dim, max_dim)
    final_dim = min(final_dim, inter_dim - 1) if inter_dim > 1 else 1
    final_dim = max(final_dim, 1)
    return inter_dim, final_dim


def two_step_pca(
        X: np.ndarray,
        intermediate_dim: int = 2500,
        final_dim: int = 300,
        random_state: int = 42,
        fit_on_input: bool = True,
        pca1: Optional[PCA] = None,
        pca2: Optional[PCA] = None
) -> Tuple[np.ndarray, PCA, PCA]:
    """
    两步 PCA：
    原始 embedding -> intermediate_dim -> final_dim
    """
    inter_dim, fin_dim = safe_pca_dims(X, intermediate_dim, final_dim)

    if fit_on_input:
        pca1 = PCA(n_components=inter_dim, svd_solver="randomized", random_state=random_state)
        X_inter = pca1.fit_transform(X)

        pca2 = PCA(n_components=fin_dim, svd_solver="randomized", random_state=random_state)
        X_final = pca2.fit_transform(X_inter)
    else:
        if pca1 is None or pca2 is None:
            raise ValueError("When fit_on_input=False, pca1 and pca2 must be provided.")
        X_inter = pca1.transform(X)
        X_final = pca2.transform(X_inter)

    return X_final, pca1, pca2


def cluster_labels(
        X_red: np.ndarray,
        method: str = "kmeans",
        n_clusters: int = 2,
        random_state: int = 42,
        **kwargs
) -> np.ndarray:
    """
    支持几种常见聚类方法
    """
    method = method.lower()

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=50, random_state=random_state)
        labels = model.fit_predict(X_red)

    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(X_red)

    elif method == "spectral":
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            assign_labels="kmeans",
            random_state=random_state
        )
        labels = model.fit_predict(X_red)

    elif method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        labels = model.fit_predict(X_red)

    elif method == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_red)

    elif method == "hdbscan":
        try:
            import hdbscan
        except ImportError:
            raise ImportError("Please install hdbscan first: pip install hdbscan")
        min_cluster_size = kwargs.get("min_cluster_size", 20)
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = model.fit_predict(X_red)

    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    return labels


def valid_silhouette(X_red: np.ndarray, labels: np.ndarray) -> float:
    """
    处理 DBSCAN/HDBSCAN 可能产生噪声点或单簇情况
    """
    valid_mask = labels >= 0
    unique_labels = np.unique(labels[valid_mask])

    if valid_mask.sum() < 3 or len(unique_labels) < 2:
        return np.nan

    return silhouette_score(X_red[valid_mask], labels[valid_mask])


# =========================
# 2. 原始聚类 + 多方法比较
# =========================

def compare_clustering_methods(
        X: np.ndarray,
        config: PipelineConfig,
        methods: List[str] = ["kmeans", "hierarchical", "spectral", "gmm"]
) -> pd.DataFrame:
    """
    比较不同聚类方法的 silhouette
    """
    X_red, pca1, pca2 = two_step_pca(
        X,
        intermediate_dim=config.intermediate_dim,
        final_dim=config.final_dim,
        random_state=config.random_state,
        fit_on_input=True
    )

    rows = []
    for method in methods:
        try:
            labels = cluster_labels(
                X_red,
                method=method,
                n_clusters=config.n_clusters,
                random_state=config.random_state
            )
            sil = valid_silhouette(X_red, labels)
            n_found = len(np.unique(labels[labels >= 0])) if np.any(labels >= 0) else 0

            rows.append({
                "method": method,
                "silhouette": sil,
                "n_clusters_found": n_found
            })
        except Exception as e:
            rows.append({
                "method": method,
                "silhouette": np.nan,
                "n_clusters_found": np.nan
            })
            print(f"[WARN] {method} failed: {e}")

    df = pd.DataFrame(rows).sort_values("silhouette", ascending=False)
    return df


# =========================
# 3. 重采样稳定性：subsampling + consensus + ARI/NMI
# =========================

def subsample_consensus_stability(
        X: np.ndarray,
        method: str = "kmeans",
        n_clusters: int = 2,
        n_iter: int = 200,
        subsample_frac: float = 0.8,
        intermediate_dim: int = 2500,
        final_dim: int = 300,
        random_state: int = 42,
        show_progress: bool = True
) -> dict:
    """
    每次随机抽取一部分样本，重新做 PCA + clustering
    计算：
    1) 与全数据参考标签的一致性（ARI / NMI）
    2) consensus matrix
    3) PAC (proportion of ambiguous clustering)
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]

    # 全数据参考聚类
    t0 = time.perf_counter()
    X_red_full, _, _ = two_step_pca(
        X,
        intermediate_dim=intermediate_dim,
        final_dim=final_dim,
        random_state=random_state,
        fit_on_input=True
    )
    ref_labels = cluster_labels(
        X_red_full,
        method=method,
        n_clusters=n_clusters,
        random_state=random_state
    )
    t_ref = time.perf_counter() - t0
    print(f"[INFO] Reference PCA + clustering done in {t_ref:.1f}s")

    # 记录 co-occurrence 和 same-cluster
    cooccur = np.zeros((n_samples, n_samples), dtype=np.float64)
    same_cluster = np.zeros((n_samples, n_samples), dtype=np.float64)

    ari_list = []
    nmi_list = []

    subsample_size = int(np.floor(subsample_frac * n_samples))

    iterator = range(n_iter)
    if show_progress:
        iterator = tqdm(iterator, total=n_iter, desc="Subsample stability", ncols=100)

    global_start = time.perf_counter()

    for b in iterator:
        idx = np.sort(rng.choice(n_samples, size=subsample_size, replace=False))
        X_sub = X[idx]

        X_red_sub, _, _ = two_step_pca(
            X_sub,
            intermediate_dim=intermediate_dim,
            final_dim=final_dim,
            random_state=random_state + b + 1,
            fit_on_input=True
        )

        sub_labels = cluster_labels(
            X_red_sub,
            method=method,
            n_clusters=n_clusters,
            random_state=random_state + b + 1
        )

        # 与参考标签对比
        ref_sub = ref_labels[idx]
        valid_mask = (sub_labels >= 0) & (ref_sub >= 0)

        if (
            valid_mask.sum() > 2
            and len(np.unique(sub_labels[valid_mask])) > 1
            and len(np.unique(ref_sub[valid_mask])) > 1
        ):
            ari = adjusted_rand_score(ref_sub[valid_mask], sub_labels[valid_mask])
            nmi = normalized_mutual_info_score(ref_sub[valid_mask], sub_labels[valid_mask])
            ari_list.append(ari)
            nmi_list.append(nmi)

        # 更新 consensus
        valid_idx_local = np.where(sub_labels >= 0)[0]
        valid_idx_global = idx[valid_idx_local]
        valid_labels = sub_labels[valid_idx_local]

        if len(valid_idx_global) >= 2:
            iu, ju = np.triu_indices(len(valid_idx_global), k=1)
            gi = valid_idx_global[iu]
            gj = valid_idx_global[ju]

            cooccur[gi, gj] += 1
            same_mask = (valid_labels[iu] == valid_labels[ju])
            same_cluster[gi[same_mask], gj[same_mask]] += 1

        if show_progress:
            elapsed = time.perf_counter() - global_start
            avg_per_iter = elapsed / (b + 1)
            eta = avg_per_iter * (n_iter - b - 1)

            iterator.set_postfix({
                "avg_s/iter": f"{avg_per_iter:.2f}",
                "ETA_s": f"{eta:.0f}",
                "mean_ARI": f"{np.mean(ari_list):.3f}" if ari_list else "NA",
                "mean_NMI": f"{np.mean(nmi_list):.3f}" if nmi_list else "NA",
            })

    # 对称化
    cooccur = cooccur + cooccur.T
    same_cluster = same_cluster + same_cluster.T

    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.divide(
            same_cluster,
            cooccur,
            out=np.zeros_like(same_cluster),
            where=cooccur > 0
        )

    # PAC
    triu_mask = np.triu(np.ones_like(consensus, dtype=bool), k=1)
    valid_pairs = triu_mask & (cooccur > 0)
    consensus_vals = consensus[valid_pairs]

    pac_low, pac_high = 0.1, 0.9
    pac = np.mean((consensus_vals > pac_low) & (consensus_vals < pac_high))

    within_vals = []
    between_vals = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if cooccur[i, j] == 0:
                continue
            if ref_labels[i] == ref_labels[j]:
                within_vals.append(consensus[i, j])
            else:
                between_vals.append(consensus[i, j])

    total_time = time.perf_counter() - global_start
    print(f"[INFO] Subsample stability finished in {total_time:.1f}s ({total_time / 60:.1f} min)")

    result = {
        "ref_labels": ref_labels,
        "consensus_matrix": consensus,
        "ari_values_vs_reference": ari_list,
        "nmi_values_vs_reference": nmi_list,
        "mean_ari_vs_reference": float(np.mean(ari_list)) if len(ari_list) else np.nan,
        "std_ari_vs_reference": float(np.std(ari_list)) if len(ari_list) else np.nan,
        "mean_nmi_vs_reference": float(np.mean(nmi_list)) if len(nmi_list) else np.nan,
        "std_nmi_vs_reference": float(np.std(nmi_list)) if len(nmi_list) else np.nan,
        "mean_within_cluster_consensus": float(np.mean(within_vals)) if len(within_vals) else np.nan,
        "mean_between_cluster_consensus": float(np.mean(between_vals)) if len(between_vals) else np.nan,
        "PAC_0.1_0.9": float(pac),
        "n_iter": n_iter,
        "subsample_frac": subsample_frac,
        "total_runtime_sec": total_time,
    }
    return result


# =========================
# 4. 轻微扰动稳定性：加噪声后重跑完整流程
# =========================

def perturbation_stability(
        X: np.ndarray,
        method: str = "kmeans",
        n_clusters: int = 2,
        n_iter: int = 100,
        noise_std_ratio: float = 0.01,
        intermediate_dim: int = 2500,
        final_dim: int = 300,
        random_state: int = 42
) -> Dict:
    """
    给原始 embedding 加轻微高斯噪声，每次重跑完整 PCA + clustering
    看与原始参考标签的 ARI/NMI
    """
    rng = np.random.default_rng(random_state)

    X_red_full, _, _ = two_step_pca(
        X,
        intermediate_dim=intermediate_dim,
        final_dim=final_dim,
        random_state=random_state,
        fit_on_input=True
    )
    ref_labels = cluster_labels(
        X_red_full,
        method=method,
        n_clusters=n_clusters,
        random_state=random_state
    )

    feature_std = np.std(X, axis=0, ddof=1)
    noise_scale = np.where(feature_std == 0, 1e-8, feature_std) * noise_std_ratio

    ari_list = []
    nmi_list = []

    for b in range(n_iter):
        noise = rng.normal(loc=0.0, scale=noise_scale, size=X.shape)
        X_noisy = X + noise

        X_red_noisy, _, _ = two_step_pca(
            X_noisy,
            intermediate_dim=intermediate_dim,
            final_dim=final_dim,
            random_state=random_state + 1000 + b,
            fit_on_input=True
        )
        noisy_labels = cluster_labels(
            X_red_noisy,
            method=method,
            n_clusters=n_clusters,
            random_state=random_state + 1000 + b
        )

        valid_mask = (ref_labels >= 0) & (noisy_labels >= 0)
        if (
            valid_mask.sum() > 2
            and len(np.unique(ref_labels[valid_mask])) > 1
            and len(np.unique(noisy_labels[valid_mask])) > 1
        ):
            ari_list.append(adjusted_rand_score(ref_labels[valid_mask], noisy_labels[valid_mask]))
            nmi_list.append(normalized_mutual_info_score(ref_labels[valid_mask], noisy_labels[valid_mask]))

    return {
        "ref_labels": ref_labels,
        "noise_std_ratio": noise_std_ratio,
        "n_iter": n_iter,
        "ari_values_vs_reference": ari_list,
        "nmi_values_vs_reference": nmi_list,
        "mean_ari_vs_reference": float(np.mean(ari_list)) if len(ari_list) else np.nan,
        "std_ari_vs_reference": float(np.std(ari_list)) if len(ari_list) else np.nan,
        "mean_nmi_vs_reference": float(np.mean(nmi_list)) if len(nmi_list) else np.nan,
        "std_nmi_vs_reference": float(np.std(nmi_list)) if len(nmi_list) else np.nan,
    }


# =========================
# 5. 可选：不同算法之间的一致性
# =========================

def algorithm_agreement(
        X: np.ndarray,
        config: PipelineConfig,
        methods: List[str] = ["kmeans", "hierarchical", "spectral", "gmm"]
) -> pd.DataFrame:
    """
    比较不同聚类方法之间的标签一致性（ARI/NMI）
    """
    X_red, _, _ = two_step_pca(
        X,
        intermediate_dim=config.intermediate_dim,
        final_dim=config.final_dim,
        random_state=config.random_state,
        fit_on_input=True
    )

    label_dict = {}
    for m in methods:
        try:
            label_dict[m] = cluster_labels(
                X_red,
                method=m,
                n_clusters=config.n_clusters,
                random_state=config.random_state
            )
        except Exception as e:
            print(f"[WARN] {m} failed: {e}")

    rows = []
    valid_methods = list(label_dict.keys())
    for i in range(len(valid_methods)):
        for j in range(i + 1, len(valid_methods)):
            m1, m2 = valid_methods[i], valid_methods[j]
            l1, l2 = label_dict[m1], label_dict[m2]

            valid_mask = (l1 >= 0) & (l2 >= 0)
            if valid_mask.sum() > 2 and len(np.unique(l1[valid_mask])) > 1 and len(np.unique(l2[valid_mask])) > 1:
                ari = adjusted_rand_score(l1[valid_mask], l2[valid_mask])
                nmi = normalized_mutual_info_score(l1[valid_mask], l2[valid_mask])
            else:
                ari, nmi = np.nan, np.nan

            rows.append({
                "method_1": m1,
                "method_2": m2,
                "ARI": ari,
                "NMI": nmi
            })

    return pd.DataFrame(rows).sort_values(["ARI", "NMI"], ascending=False)


# =========================
# 6. 导出表格 + 作图
# =========================

def pretty_method_name(method: str) -> str:
    mapping = {
        "kmeans": "K-means",
        "gmm": "Gaussian mixture modeling",
        "spectral": "Spectral clustering",
        "hierarchical": "Hierarchical clustering",
        "dbscan": "DBSCAN",
        "hdbscan": "HDBSCAN",
    }
    return mapping.get(str(method).lower(), method)


def export_tables(
        method_df: pd.DataFrame,
        sub_res: dict,
        noise_res: dict,
        agree_df: pd.DataFrame,
        out_dir: str
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # eTable 1: clustering method comparison
    method_out = method_df.copy()
    method_out["method"] = method_out["method"].map(pretty_method_name)
    method_out["silhouette"] = method_out["silhouette"].round(3)
    method_out = method_out.rename(columns={
        "method": "Clustering algorithm",
        "silhouette": "Silhouette score",
        "n_clusters_found": "Number of clusters identified"
    })
    method_out.to_csv(out_dir / "eTable_clustering_method_comparison.csv", index=False)

    # eTable 2: robustness summary
    robust_out = pd.DataFrame([
        {
            "Analysis": "Subsampling consensus stability",
            "n_iter": sub_res["n_iter"],
            "Setting": f"subsample_frac={sub_res['subsample_frac']}",
            "Mean ARI vs reference": round(sub_res["mean_ari_vs_reference"], 3),
            "SD ARI": round(sub_res["std_ari_vs_reference"], 3),
            "Mean NMI vs reference": round(sub_res["mean_nmi_vs_reference"], 3),
            "SD NMI": round(sub_res["std_nmi_vs_reference"], 3),
            "Within-cluster consensus": round(sub_res["mean_within_cluster_consensus"], 3),
            "Between-cluster consensus": round(sub_res["mean_between_cluster_consensus"], 4),
            "PAC (0.1-0.9)": round(sub_res["PAC_0.1_0.9"], 3),
        },
        {
            "Analysis": "Perturbation stability",
            "n_iter": noise_res["n_iter"],
            "Setting": f"noise_std_ratio={noise_res['noise_std_ratio']}",
            "Mean ARI vs reference": round(noise_res["mean_ari_vs_reference"], 3),
            "SD ARI": round(noise_res["std_ari_vs_reference"], 3),
            "Mean NMI vs reference": round(noise_res["mean_nmi_vs_reference"], 3),
            "SD NMI": round(noise_res["std_nmi_vs_reference"], 3),
            "Within-cluster consensus": np.nan,
            "Between-cluster consensus": np.nan,
            "PAC (0.1-0.9)": np.nan,
        }
    ])
    robust_out.to_csv(out_dir / "eTable_clustering_robustness_summary.csv", index=False)

    # eTable 3: algorithm agreement
    agree_out = agree_df.copy()
    agree_out["method_1"] = agree_out["method_1"].map(pretty_method_name)
    agree_out["method_2"] = agree_out["method_2"].map(pretty_method_name)
    agree_out["ARI"] = agree_out["ARI"].round(3)
    agree_out["NMI"] = agree_out["NMI"].round(3)
    agree_out = agree_out.rename(columns={
        "method_1": "Method 1",
        "method_2": "Method 2"
    })
    agree_out.to_csv(out_dir / "eTable_clustering_algorithm_agreement.csv", index=False)

    # consensus matrix 原始数值
    np.save(out_dir / "consensus_matrix.npy", sub_res["consensus_matrix"])


def plot_method_comparison(method_df: pd.DataFrame, out_png: str) -> None:
    df = method_df.copy().dropna(subset=["silhouette"])
    df["method"] = df["method"].map(pretty_method_name)
    df = df.sort_values("silhouette", ascending=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(df["method"], df["silhouette"])
    ax.set_ylabel("Silhouette score")
    ax.set_xlabel("")
    ax.set_title("Comparison of candidate clustering methods")
    ax.set_ylim(0, max(df["silhouette"].max() * 1.25, 0.1))
    ax.tick_params(axis="x", rotation=20)

    for bar, val in zip(bars, df["silhouette"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_consensus_heatmap(consensus_matrix: np.ndarray, ref_labels: np.ndarray, out_png: str) -> None:
    order = np.argsort(ref_labels)
    cm = consensus_matrix[order][:, order]
    ordered_labels = ref_labels[order]

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    im = ax.imshow(cm, aspect="auto", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Consensus")

    boundaries = np.where(np.diff(ordered_labels) != 0)[0] + 0.5
    for b in boundaries:
        ax.axhline(b, linewidth=1)
        ax.axvline(b, linewidth=1)

    ax.set_title("Consensus matrix of repeated subsampling")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _single_boxplot(ax, values, title, ylabel):
    if values is None or len(values) == 0:
        ax.text(0.5, 0.5, "No valid values", ha="center", va="center")
        ax.set_title(title)
        return

    ax.boxplot(values, widths=0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks([1])
    ax.set_xticklabels([""])
    vmin = min(values)
    vmax = max(values)
    margin = max((vmax - vmin) * 0.2, 0.002)
    ax.set_ylim(max(0, vmin - margin), min(1.01, vmax + margin))


def plot_stability_distributions(sub_res: dict, noise_res: dict, out_png: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.5))

    _single_boxplot(
        axes[0, 0],
        sub_res.get("ari_values_vs_reference", []),
        "Subsampling stability: ARI",
        "ARI vs reference"
    )
    _single_boxplot(
        axes[0, 1],
        sub_res.get("nmi_values_vs_reference", []),
        "Subsampling stability: NMI",
        "NMI vs reference"
    )
    _single_boxplot(
        axes[1, 0],
        noise_res.get("ari_values_vs_reference", []),
        "Perturbation stability: ARI",
        "ARI vs reference"
    )
    _single_boxplot(
        axes[1, 1],
        noise_res.get("nmi_values_vs_reference", []),
        "Perturbation stability: NMI",
        "NMI vs reference"
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _agreement_matrix(agree_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    methods = sorted(
        set(agree_df["method_1"].tolist()) | set(agree_df["method_2"].tolist())
    )
    pretty_methods = [pretty_method_name(m) for m in methods]
    mat = pd.DataFrame(np.nan, index=pretty_methods, columns=pretty_methods)

    for m in pretty_methods:
        mat.loc[m, m] = 1.0

    for _, row in agree_df.iterrows():
        m1 = pretty_method_name(row["method_1"])
        m2 = pretty_method_name(row["method_2"])
        mat.loc[m1, m2] = row[metric]
        mat.loc[m2, m1] = row[metric]

    return mat


def plot_algorithm_agreement(agree_df: pd.DataFrame, out_png: str) -> None:
    ari_mat = _agreement_matrix(agree_df, "ARI")
    nmi_mat = _agreement_matrix(agree_df, "NMI")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for ax, mat, title in zip(
        axes,
        [ari_mat, nmi_mat],
        ["Algorithm agreement (ARI)", "Algorithm agreement (NMI)"]
    ):
        im = ax.imshow(mat.values, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_title(title)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Agreement")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def get_base_dir() -> str:
    """获取当前脚本所在目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# 6. 示例主程序
# =========================

if __name__ == "__main__":
    # -------------------------
    # 读取 embedding 矩阵
    # -------------------------

    base_dir = get_base_dir()
    df = pd.read_csv(os.path.join(base_dir, "demo_data", "pathology_ollama_embed_deepseek.csv"))
    embed_cols = [col for col in df.columns if col.startswith("emb_")]
    X = df[embed_cols].values.astype(np.float32)

    config = PipelineConfig(
        intermediate_dim=2500,
        final_dim=300,
        random_state=42,
        n_clusters=2
    )

    out_dir = Path(os.path.join(base_dir,r"results\stability"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 比较不同聚类方法
    print("\n=== Compare clustering methods ===")
    method_df = compare_clustering_methods(
        X,
        config=config,
        methods=["kmeans", "hierarchical", "spectral", "gmm"]
    )
    print(method_df)

    # 最终主方法
    final_method = "kmeans"

    # 2) 重采样稳定性
    print("\n=== Subsample consensus stability ===")
    sub_res = subsample_consensus_stability(
        X,
        method=final_method,
        n_clusters=2,
        n_iter=30,
        subsample_frac=0.8,
        intermediate_dim=2500,
        final_dim=300,
        random_state=42,
        show_progress=True
    )
    for k, v in sub_res.items():
        if k not in ["ref_labels", "consensus_matrix", "ari_values_vs_reference", "nmi_values_vs_reference"]:
            print(f"{k}: {v}")

    # 3) 轻微扰动稳定性
    print("\n=== Perturbation stability ===")
    noise_res = perturbation_stability(
        X,
        method=final_method,
        n_clusters=2,
        n_iter=30,
        noise_std_ratio=0.01,
        intermediate_dim=2500,
        final_dim=300,
        random_state=42
    )
    for k, v in noise_res.items():
        if k not in ["ref_labels", "ari_values_vs_reference", "nmi_values_vs_reference"]:
            print(f"{k}: {v}")

    # 4) 不同算法之间一致性
    print("\n=== Algorithm agreement ===")
    agree_df = algorithm_agreement(
        X,
        config=config,
        methods=["kmeans", "hierarchical", "spectral", "gmm"]
    )
    print(agree_df)

    # 5) 导出表格
    export_tables(method_df, sub_res, noise_res, agree_df, str(out_dir))

    # 6) 画图
    plot_method_comparison(
        method_df,
        str(out_dir / "eFig_method_comparison.png")
    )

    plot_consensus_heatmap(
        sub_res["consensus_matrix"],
        sub_res["ref_labels"],
        str(out_dir / "eFig_consensus_matrix.png")
    )

    plot_stability_distributions(
        sub_res,
        noise_res,
        str(out_dir / "eFig_stability_distributions.png")
    )

    plot_algorithm_agreement(
        agree_df,
        str(out_dir / "eFig_algorithm_agreement.png")
    )

    print("\n[INFO] All tables and figures have been saved to:")
    print(out_dir)
