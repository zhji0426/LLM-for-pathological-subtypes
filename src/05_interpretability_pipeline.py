
import os
import json
import glob
import warnings
import re
from collections import defaultdict, Counter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score, silhouette_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import shap
import statsmodels.api as sm
from lifelines import CoxPHFitter, KaplanMeierFitter

from embed_ollama_03 import SectionProcessor, EmbeddingGenerator, ReportConfig as BaseReportConfig

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
@dataclass
class PipelineConfig(BaseReportConfig):
    """端到端流程配置"""
    # 输入输出
    json_dir: Optional[str] = None
    flattened_csv: Optional[str] = None
    embedding_cache_dir: str = "embedding_cache"
    output_dir: str = "analysis_output"

    # 标签与结局
    label_csv: Optional[str] = None
    label_col: str = "best_cluster_label"
    outcome_csv: Optional[str] = None

    # 分析参数
    n_clusters: int = 2
    random_state: int = 42
    pca_components: int = 50
    bootstrap_iterations: int = 200
    bootstrap_sample_ratio: float = 0.8

    # 可解释性参数
    rare_category_min_count: int = 5
    top_shap_n: int = 30
    counterfactual_top_features: int = 20
    prototype_top_features: int = 20
    block_prefix_tokens: int = 2
    counterfactual_max_cases: Optional[int] = None

    # 结局关联
    binary_outcomes: List[str] = field(default_factory=list)
    continuous_outcomes: List[str] = field(default_factory=list)
    survival_specs: List[Tuple[str, str]] = field(default_factory=list)
    adjust_covariates: List[str] = field(default_factory=list)

    # ID 列
    id_cols_priority: Tuple[str, ...] = ("phID", "bio_id", "source_file")

    # 图像
    fig_dpi: int = 150

    def __post_init__(self):
        super().__post_init__()
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "tables"), exist_ok=True)


# -----------------------------------------------------------------------------
# 通用工具
# -----------------------------------------------------------------------------
def save_df(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def l2_normalize(x: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def align_labels(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """二分类标签方向对齐"""
    ref_u = np.unique(ref)
    pred_u = np.unique(pred)
    if len(ref_u) != 2 or len(pred_u) != 2:
        return pred
    flipped = 1 - pred
    return flipped if adjusted_rand_score(ref, flipped) > adjusted_rand_score(ref, pred) else pred


def collapse_rare_categories(series: pd.Series, min_count: int = 5) -> pd.Series:
    s = series.copy()
    s = s.where(~pd.isna(s), other="missing").astype(str)
    vc = s.value_counts(dropna=False)
    rare = vc[vc < min_count].index
    return s.where(~s.isin(rare), other="__RARE__")


def clean_feature_name_for_json_match(feature_name: str) -> str:
    """
    清理 SHAP / one-hot 聚合后可能残留的类别占位后缀，避免影响后续映射到 JSON 键。
    """
    x = str(feature_name).strip()
    x = re.sub(r'(?:__RARE__|__RARE_|_RARE__|_RARE_)$', '', x)
    x = re.sub(r'__+', '_', x).strip('_')
    return x


def safe_close() -> None:
    plt.tight_layout()
    plt.close()


def infer_merge_keys(left: pd.DataFrame, right: pd.DataFrame, priority: Tuple[str, ...]) -> List[str]:
    keys = [c for c in priority if c in left.columns and c in right.columns]
    if not keys:
        raise ValueError(f"无法找到共同 ID 列，可用优先级为: {priority}")
    return keys


def build_json_feature_universe(json_dir: str, sections: List[str]) -> Set[str]:
    """收集 JSON 中真实存在的 feature 全键名: section_key"""
    universe: Set[str] = set()
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    for fpath in tqdm(json_files, desc="扫描 JSON 特征宇宙"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sec in sections:
                flat = SectionProcessor.flatten_dict(data.get(sec, {}))
                for k in flat.keys():
                    universe.add(f"{sec}_{k}")
        except Exception:
            continue
    return universe


def map_raw_features_to_json_keys(raw_features: List[str], json_feature_universe: Set[str]) -> Tuple[Set[str], pd.DataFrame]:
    """
    将 SHAP 的原始特征名映射到 JSON 键名。
    匹配规则：
      1) 精确匹配
      2) suffix 匹配：json_key.endswith("_" + raw_feature)
      3) 规范化后匹配（去掉空格/重复下划线）
      4) 先清理可能残留的 one-hot 类别占位后缀（如 __RARE__）
    """
    rows = []
    matched: Set[str] = set()

    def norm(x: str) -> str:
        x = clean_feature_name_for_json_match(x)
        return "_".join(str(x).strip().split())

    universe_norm = {norm(k): k for k in json_feature_universe}

    for feat in raw_features:
        feat = str(feat)
        feat_clean = clean_feature_name_for_json_match(feat)
        feat_norm = norm(feat_clean)
        local_hits = set()

        candidates = {feat, feat_clean, feat_norm}

        # 1) exact
        for cand in candidates:
            if cand in json_feature_universe:
                local_hits.add(cand)

        # 2) normalized exact
        for cand in candidates:
            cand_norm = norm(cand)
            if cand_norm in universe_norm:
                local_hits.add(universe_norm[cand_norm])

        # 3) suffix
        suffix_hits = {
            k for k in json_feature_universe
            if any(k.endswith(f"_{cand}") for cand in candidates if cand)
        }
        local_hits.update(suffix_hits)

        status = "matched" if local_hits else "unmatched"
        if local_hits:
            matched.update(local_hits)

        rows.append({
            "raw_feature": feat,
            "raw_feature_clean": feat_clean,
            "matched_json_keys": " | ".join(sorted(local_hits)) if local_hits else "",
            "n_matches": len(local_hits),
            "status": status
        })

    return matched, pd.DataFrame(rows)


def pick_existing_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


# -----------------------------------------------------------------------------
# 嵌入缓存
# -----------------------------------------------------------------------------
class EmbeddingCache:
    """管理各部分嵌入的生成与缓存"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.section_names = SectionProcessor.SECTION_NAMES
        self.section_dim = config.single_embedding_dim
        self.generator = EmbeddingGenerator(config)

    def _get_cache_path(self, source_file: str) -> str:
        base = os.path.splitext(os.path.basename(source_file))[0]
        return os.path.join(self.config.embedding_cache_dir, f"{base}.npz")

    @staticmethod
    def _extract_ids(filename: str) -> Tuple[str, str]:
        try:
            basename = os.path.basename(filename)
            name = os.path.splitext(basename)[0].replace("pathology_features_", "")
            parts = name.split("_")
            phID = parts[0] if parts else ""
            bio_id = parts[1] if len(parts) > 1 else ""
            return phID, bio_id
        except Exception:
            return "", ""

    def process_and_cache(self, force_recompute: bool = False) -> pd.DataFrame:
        json_files = sorted(glob.glob(os.path.join(self.config.json_dir, "*.json")))
        if not json_files:
            raise FileNotFoundError(f"在 {self.config.json_dir} 中未找到 JSON 文件")

        metadata_rows = []
        all_sec_embs = {sec: [] for sec in self.section_names}
        all_full_embs = []

        for row_id, fpath in enumerate(tqdm(json_files, desc="处理 JSON 文件")):
            cache_path = self._get_cache_path(fpath)

            if (not force_recompute) and os.path.exists(cache_path):
                data = np.load(cache_path, allow_pickle=True)
                sec_embs = {sec: data[sec] for sec in self.section_names}
                full_emb = data["full_embedding"]
            else:
                with open(fpath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                sec_texts = SectionProcessor.extract_all_section_texts(raw)
                sec_embs = self.generator.generate_section_embeddings(sec_texts)
                full_emb = self.generator.concatenate_embeddings(sec_embs)

                np.savez(
                    cache_path,
                    **{sec: np.asarray(sec_embs[sec], dtype=np.float32) for sec in self.section_names},
                    full_embedding=np.asarray(full_emb, dtype=np.float32),
                )

            phID, bio_id = self._extract_ids(fpath)
            metadata_rows.append({
                "row_id": row_id,
                "phID": phID,
                "bio_id": bio_id,
                "source_file": os.path.basename(fpath),
            })

            for sec in self.section_names:
                all_sec_embs[sec].append(np.asarray(sec_embs[sec], dtype=np.float32))
            all_full_embs.append(np.asarray(full_emb, dtype=np.float32))

        for sec in self.section_names:
            np.save(os.path.join(self.config.embedding_cache_dir, f"{sec}_embeddings.npy"), np.asarray(all_sec_embs[sec], dtype=np.float32))
        np.save(os.path.join(self.config.embedding_cache_dir, "full_embeddings.npy"), np.asarray(all_full_embs, dtype=np.float32))

        metadata_df = pd.DataFrame(metadata_rows)
        metadata_df.to_csv(os.path.join(self.config.embedding_cache_dir, "metadata.csv"), index=False)
        return metadata_df

    def load_embeddings(self) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]:
        metadata = pd.read_csv(os.path.join(self.config.embedding_cache_dir, "metadata.csv"))
        sec_embs = {}
        for sec in self.section_names:
            path = os.path.join(self.config.embedding_cache_dir, f"{sec}_embeddings.npy")
            sec_embs[sec] = np.load(path)
        full_emb = np.load(os.path.join(self.config.embedding_cache_dir, "full_embeddings.npy"))
        return metadata, sec_embs, full_emb


# -----------------------------------------------------------------------------
# 1. 稳定性分析
# -----------------------------------------------------------------------------
class StabilityAnalyzer:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def run(self, X: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
        n_comp = min(self.cfg.pca_components, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=self.cfg.random_state)
        Xr = pca.fit_transform(X)

        gmm = GaussianMixture(
            n_components=self.cfg.n_clusters,
            covariance_type="full",
            random_state=self.cfg.random_state,
            n_init=20,
            reg_covar=1e-5,
        )
        labels = gmm.fit_predict(Xr)
        sil = silhouette_score(Xr, labels) if len(np.unique(labels)) > 1 else np.nan
        print(f"参考聚类轮廓系数: {sil:.4f}")

        n = Xr.shape[0]
        rng = np.random.default_rng(self.cfg.random_state)
        ari_list = []

        for _ in tqdm(range(self.cfg.bootstrap_iterations), desc="Bootstrap"):
            bs_n = max(10, int(n * self.cfg.bootstrap_sample_ratio))
            idx = rng.choice(n, bs_n, replace=True)

            gmm_boot = GaussianMixture(
                n_components=self.cfg.n_clusters,
                covariance_type="full",
                random_state=self.cfg.random_state,
                n_init=10,
                reg_covar=1e-5,
            )
            gmm_boot.fit(Xr[idx])
            pred = gmm_boot.predict(Xr)
            pred = align_labels(labels, pred)
            ari_list.append(adjusted_rand_score(labels, pred))

        df = pd.DataFrame({"bootstrap_ari": ari_list})
        save_df(df, os.path.join(self.cfg.output_dir, "tables", "bootstrap_ari.csv"))

        plt.figure(figsize=(6, 4), dpi=self.cfg.fig_dpi)
        plt.hist(ari_list, bins=30, edgecolor="k")
        plt.axvline(np.mean(ari_list), linestyle="--")
        plt.title(f"Bootstrap ARI (mean={np.mean(ari_list):.3f})")
        plt.xlabel("ARI vs reference")
        plt.savefig(os.path.join(self.cfg.output_dir, "figures", "stability_bootstrap_ari.png"))
        safe_close()

        print(f"Bootstrap ARI: {np.mean(ari_list):.4f} ± {np.std(ari_list):.4f}")
        return labels, df


# -----------------------------------------------------------------------------
# 2. Section attribution
# -----------------------------------------------------------------------------
class SectionAttributor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.sections = SectionProcessor.SECTION_NAMES

    def run(self, full_emb: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        n_sec = len(self.sections)
        dim = full_emb.shape[1] // n_sec
        if full_emb.shape[1] != n_sec * dim:
            raise ValueError("full_emb 维度无法被 section 数整除，请检查 embedding 拼接过程。")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.cfg.random_state)
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")

        auc_full = np.mean(cross_val_score(clf, full_emb, y, cv=cv, scoring="roc_auc"))
        rows = []

        for i, sec in enumerate(self.sections):
            slc = slice(i * dim, (i + 1) * dim)
            X_sec = full_emb[:, slc]
            auc_sec = np.mean(cross_val_score(clf, X_sec, y, cv=cv, scoring="roc_auc"))

            keep_idx = np.setdiff1d(np.arange(full_emb.shape[1]), np.arange(slc.start, slc.stop))
            X_loo = full_emb[:, keep_idx]
            auc_loo = np.mean(cross_val_score(clf, X_loo, y, cv=cv, scoring="roc_auc"))

            mu0 = np.mean(X_sec[y == 0], axis=0)
            mu1 = np.mean(X_sec[y == 1], axis=0)
            centroid_dist = np.linalg.norm(mu1 - mu0)

            rows.append({
                "section": sec,
                "auc_section_only": auc_sec,
                "auc_leave_out": auc_loo,
                "auc_drop": auc_full - auc_loo,
                "centroid_distance": centroid_dist,
            })

        df = pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)
        save_df(df, os.path.join(self.cfg.output_dir, "tables", "section_attribution.csv"))

        plt.figure(figsize=(8, 5), dpi=self.cfg.fig_dpi)
        sns.barplot(data=df, x="section", y="auc_drop")
        plt.title("Section importance (AUC drop when removed)")
        plt.ylabel("AUC drop")
        plt.xticks(rotation=45, ha="right")
        plt.savefig(os.path.join(self.cfg.output_dir, "figures", "section_attribution.png"))
        safe_close()
        return df


# -----------------------------------------------------------------------------
# 3. Teacher-student SHAP
# -----------------------------------------------------------------------------
class TeacherStudentAnalyzer:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    @staticmethod
    def _fallback_encoded_to_raw_feature(encoded_name: str) -> str:
        """
        仅作兜底使用。真正的 raw_feature 映射优先使用 OneHotEncoder 拟合后的列顺序，
        避免类别值中包含下划线或 __RARE__ 时误截断原始列名。
        """
        encoded_name = str(encoded_name)
        if encoded_name.startswith("num__"):
            return encoded_name[len("num__"):]
        if encoded_name.startswith("cat__"):
            tail = encoded_name[len("cat__"):]
            tail = clean_feature_name_for_json_match(tail)
            parts = tail.split("_")
            if len(parts) >= 2:
                return "_".join(parts[:-1])
            return tail
        return clean_feature_name_for_json_match(encoded_name)

    @staticmethod
    def _build_encoded_raw_mapping(
        preprocessor: ColumnTransformer,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> pd.DataFrame:
        """
        根据拟合后的 preprocessor 直接重建“编码列 -> 原始列”映射。
        这样即使类别值为 __RARE__、missing 或包含多个下划线，也不会污染 raw_feature。
        """
        encoded_feature_names = preprocessor.get_feature_names_out().tolist()
        raw_feature_names: List[str] = []

        # 数值特征顺序与 ColumnTransformer 输出顺序一致
        raw_feature_names.extend([str(c) for c in numeric_cols])

        if categorical_cols:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            for col, cats in zip(categorical_cols, ohe.categories_):
                raw_feature_names.extend([str(col)] * len(cats))

        if len(raw_feature_names) != len(encoded_feature_names):
            raw_feature_names = [
                TeacherStudentAnalyzer._fallback_encoded_to_raw_feature(x)
                for x in encoded_feature_names
            ]

        raw_feature_names = [clean_feature_name_for_json_match(x) for x in raw_feature_names]

        return pd.DataFrame({
            "encoded_feature": encoded_feature_names,
            "raw_feature": raw_feature_names,
        })

    def run(self, flat_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        y = flat_df[label_col].astype(int).values

        exclude = {label_col, "timestamp", "total_embedding_dim", "source_file", "phID", "bio_id", "row_id"} | set(self.cfg.id_cols_priority)
        feature_cols = [c for c in flat_df.columns if c not in exclude]
        X_raw = flat_df[feature_cols].copy()

        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_raw[c])]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]

        for c in categorical_cols:
            X_raw[c] = collapse_rare_categories(X_raw[c], self.cfg.rare_category_min_count)

        numeric_trans = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        categorical_trans = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_trans, numeric_cols),
            ("cat", categorical_trans, categorical_cols),
        ])

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=self.cfg.random_state,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.2, stratify=y, random_state=self.cfg.random_state
        )

        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        model.fit(X_train_proc, y_train)

        y_pred_prob = model.predict_proba(X_test_proc)[:, 1]
        auc = roc_auc_score(y_test, y_pred_prob)
        print(f"Teacher-student 模型 AUC: {auc:.4f}")

        feature_map = self._build_encoded_raw_mapping(preprocessor, numeric_cols, categorical_cols)
        encoded_feature_names = feature_map["encoded_feature"].tolist()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_proc)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if len(np.asarray(shap_values).shape) == 3:
            shap_values = np.asarray(shap_values)[:, :, -1]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim > 2:
            shap_values = shap_values.reshape(shap_values.shape[0], -1)

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0).ravel()

        if len(mean_abs_shap) != len(encoded_feature_names):
            min_len = min(len(mean_abs_shap), len(encoded_feature_names))
            mean_abs_shap = mean_abs_shap[:min_len]
            encoded_feature_names = encoded_feature_names[:min_len]

        encoded_imp_df = feature_map.copy()
        encoded_imp_df["importance"] = mean_abs_shap
        encoded_imp_df = encoded_imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

        raw_imp_df = (
            encoded_imp_df.groupby("raw_feature", as_index=False)["importance"]
            .sum()
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        save_df(encoded_imp_df, os.path.join(self.cfg.output_dir, "tables", "shap_top_features_encoded.csv"))
        save_df(raw_imp_df, os.path.join(self.cfg.output_dir, "tables", "shap_top_features_raw.csv"))

        top_n = min(self.cfg.top_shap_n, len(raw_imp_df))
        plot_df = raw_imp_df.head(top_n).copy()

        plt.figure(figsize=(8, max(6, 0.25 * len(plot_df))), dpi=self.cfg.fig_dpi)
        plt.barh(plot_df["raw_feature"][::-1], plot_df["importance"][::-1])
        plt.xlabel("Aggregated mean |SHAP|")
        plt.title("Top raw SHAP features")
        plt.savefig(os.path.join(self.cfg.output_dir, "figures", "shap_top_features_raw.png"))
        safe_close()

        return raw_imp_df


# -----------------------------------------------------------------------------
# 4. Counterfactual
# -----------------------------------------------------------------------------
class CounterfactualEngine:
    def __init__(self, cfg: PipelineConfig, cache: EmbeddingCache):
        self.cfg = cfg
        self.cache = cache
        self.sections = SectionProcessor.SECTION_NAMES
        self.section_dim = cfg.single_embedding_dim

    @staticmethod
    def _dict_to_text(d: Dict[str, Any]) -> str:
        return "; ".join(f"{k}: {v}" for k, v in d.items() if v not in [None, "", []])

    def _axis_from_labels(self, full_emb: np.ndarray, y: np.ndarray) -> np.ndarray:
        mu0 = l2_normalize(full_emb[y == 0].mean(axis=0))
        mu1 = l2_normalize(full_emb[y == 1].mean(axis=0))
        return l2_normalize(mu1 - mu0)

    def build_prototypes(self, metadata: pd.DataFrame, y: np.ndarray) -> Dict[int, Dict[Tuple[str, str], str]]:
        protos = {0: defaultdict(list), 1: defaultdict(list)}
        for pos, row in metadata.reset_index(drop=True).iterrows():
            label = int(y[pos])
            source = row["source_file"]
            fpath = os.path.join(self.cfg.json_dir, source)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            for sec in self.sections:
                flat = SectionProcessor.flatten_dict(data.get(sec, {}))
                for k, v in flat.items():
                    if v not in (None, "", []):
                        protos[label][(sec, k)].append(str(v))

        out = {0: {}, 1: {}}
        for lbl in [0, 1]:
            for key, vals in protos[lbl].items():
                if vals:
                    out[lbl][key] = Counter(vals).most_common(1)[0][0]
        return out

    def _safe_generate_section_embedding(self, text: str, sec: str) -> np.ndarray:
        emb = self.cache.generator.generate_embedding_with_retry(text, sec)
        if emb is None:
            emb = np.zeros(self.section_dim, dtype=float)
        return np.asarray(emb, dtype=float)

    def _get_original_sec_embs(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        sec_texts = SectionProcessor.extract_all_section_texts(data)
        return {sec: self._safe_generate_section_embedding(sec_texts[sec], sec) for sec in self.sections}

    def delete_experiment(self, metadata: pd.DataFrame, y: np.ndarray, axis: np.ndarray, top_features: Set[str]) -> pd.DataFrame:
        rows = []
        sample_indices = np.arange(len(metadata))
        if self.cfg.counterfactual_max_cases:
            sample_indices = sample_indices[: self.cfg.counterfactual_max_cases]

        total_features_processed = 0
        total_skipped = 0

        for idx in tqdm(sample_indices, desc="Delete counterfactual"):
            row = metadata.iloc[idx]
            source = row["source_file"]
            label = int(y[idx])
            fpath = os.path.join(self.cfg.json_dir, source)

            if not os.path.exists(fpath):
                continue

            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            sec_embs_orig = self._get_original_sec_embs(data)
            full_orig = l2_normalize(self.cache.generator.concatenate_embeddings(sec_embs_orig))

            for sec in self.sections:
                flat = SectionProcessor.flatten_dict(data.get(sec, {}))
                for k, v in flat.items():
                    full_key = f"{sec}_{k}"
                    total_features_processed += 1
                    if full_key not in top_features:
                        total_skipped += 1
                        continue

                    new_flat = flat.copy()
                    del new_flat[k]
                    new_text = self._dict_to_text(new_flat)
                    new_emb = self._safe_generate_section_embedding(new_text, sec)

                    new_full = np.concatenate([
                        new_emb if s == sec else sec_embs_orig[s] for s in self.sections
                    ])
                    new_full = l2_normalize(new_full)
                    delta = float((new_full - full_orig) @ axis)

                    rows.append({
                        "source_file": source,
                        "label": label,
                        "section": sec,
                        "feature": full_key,
                        "old_value": str(v)[:100] if v is not None else "",
                        "delta_proj": delta,
                        "support_score": float(-delta if label == 0 else delta),
                    })

        print(f"特征处理统计: 总计 {total_features_processed}, 跳过 {total_skipped}, 保留 {len(rows)}")
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["source_file", "label", "section", "feature", "old_value", "delta_proj", "support_score"])
        save_df(df, os.path.join(self.cfg.output_dir, "tables", "counterfactual_delete.csv"))
        return df

    def swap_experiment(self, metadata: pd.DataFrame, y: np.ndarray, axis: np.ndarray, prototypes: Dict[int, Dict], top_features: Set[str]) -> pd.DataFrame:
        rows = []
        sample_indices = np.arange(len(metadata))
        if self.cfg.counterfactual_max_cases:
            sample_indices = sample_indices[: self.cfg.counterfactual_max_cases]

        for idx in tqdm(sample_indices, desc="Swap counterfactual"):
            row = metadata.iloc[idx]
            source = row["source_file"]
            label = int(y[idx])
            opp = 1 - label
            fpath = os.path.join(self.cfg.json_dir, source)

            if not os.path.exists(fpath):
                continue

            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            sec_embs_orig = self._get_original_sec_embs(data)
            full_orig = l2_normalize(self.cache.generator.concatenate_embeddings(sec_embs_orig))

            for sec in self.sections:
                flat = SectionProcessor.flatten_dict(data.get(sec, {}))
                for k, v in flat.items():
                    full_key = f"{sec}_{k}"
                    if full_key not in top_features:
                        continue
                    new_val = prototypes[opp].get((sec, k))
                    if new_val is None or str(new_val) == str(v):
                        continue

                    new_flat = flat.copy()
                    new_flat[k] = new_val
                    new_text = self._dict_to_text(new_flat)
                    new_emb = self._safe_generate_section_embedding(new_text, sec)

                    new_full = np.concatenate([
                        new_emb if s == sec else sec_embs_orig[s] for s in self.sections
                    ])
                    new_full = l2_normalize(new_full)
                    delta = float((new_full - full_orig) @ axis)

                    rows.append({
                        "source_file": source,
                        "label": label,
                        "section": sec,
                        "feature": full_key,
                        "old_value": str(v),
                        "new_value": str(new_val),
                        "delta_proj": delta,
                        "support_score": float(-delta if label == 0 else delta),
                    })

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["source_file", "label", "section", "feature", "old_value", "new_value", "delta_proj", "support_score"])
        save_df(df, os.path.join(self.cfg.output_dir, "tables", "counterfactual_swap.csv"))
        return df

    def block_replace_experiment(self, metadata: pd.DataFrame, y: np.ndarray, axis: np.ndarray, prototypes: Dict[int, Dict], top_features: Set[str]) -> pd.DataFrame:
        rows = []
        sample_indices = np.arange(len(metadata))
        if self.cfg.counterfactual_max_cases:
            sample_indices = sample_indices[: self.cfg.counterfactual_max_cases]

        for idx in tqdm(sample_indices, desc="Block replace"):
            row = metadata.iloc[idx]
            source = row["source_file"]
            label = int(y[idx])
            opp = 1 - label
            fpath = os.path.join(self.cfg.json_dir, source)

            if not os.path.exists(fpath):
                continue

            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            sec_embs_orig = self._get_original_sec_embs(data)
            full_orig = l2_normalize(self.cache.generator.concatenate_embeddings(sec_embs_orig))

            for sec in self.sections:
                flat = SectionProcessor.flatten_dict(data.get(sec, {}))
                blocks = defaultdict(list)

                for k in flat.keys():
                    full_key = f"{sec}_{k}"
                    if full_key not in top_features:
                        continue
                    prefix = "_".join(k.split("_")[: self.cfg.block_prefix_tokens])
                    blocks[prefix].append(k)

                for prefix, keys in blocks.items():
                    new_flat = flat.copy()
                    changed = 0
                    for k in keys:
                        new_val = prototypes[opp].get((sec, k))
                        if new_val is not None and str(new_val) != str(new_flat.get(k)):
                            new_flat[k] = new_val
                            changed += 1

                    if changed == 0:
                        continue

                    new_text = self._dict_to_text(new_flat)
                    new_emb = self._safe_generate_section_embedding(new_text, sec)
                    new_full = np.concatenate([
                        new_emb if s == sec else sec_embs_orig[s] for s in self.sections
                    ])
                    new_full = l2_normalize(new_full)
                    delta = float((new_full - full_orig) @ axis)

                    rows.append({
                        "source_file": source,
                        "label": label,
                        "section": sec,
                        "block": prefix,
                        "n_changed": changed,
                        "delta_proj": delta,
                        "support_score": float(-delta if label == 0 else delta),
                    })

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["source_file", "label", "section", "block", "n_changed", "delta_proj", "support_score"])
        save_df(df, os.path.join(self.cfg.output_dir, "tables", "counterfactual_block.csv"))
        return df

    def prototype_trajectory(self, full_emb: np.ndarray, y: np.ndarray, prototypes: Dict[int, Dict]) -> pd.DataFrame:
        """使用当前对齐后的 full_emb，而不是重新读取全部缓存 embedding"""
        axis = self._axis_from_labels(full_emb, y)
        rows = []

        for start in [0, 1]:
            target = 1 - start
            base_dict = {sec: {} for sec in self.sections}
            target_dict = {sec: {} for sec in self.sections}

            for (sec, k), v in prototypes[start].items():
                base_dict[sec][k] = v
            for (sec, k), v in prototypes[target].items():
                target_dict[sec][k] = v

            step = 0
            rows.append({
                "start": start,
                "step": step,
                "description": "baseline",
                "position": self._position_of_dict(base_dict, axis),
            })

            for sec in self.sections:
                all_keys = sorted(set(base_dict[sec].keys()) | set(target_dict[sec].keys()))
                groups = defaultdict(list)
                for k in all_keys:
                    prefix = "_".join(k.split("_")[: self.cfg.block_prefix_tokens])
                    groups[prefix].append(k)

                for prefix, ks in groups.items():
                    for k in ks:
                        if k in target_dict[sec]:
                            base_dict[sec][k] = target_dict[sec][k]
                    step += 1
                    rows.append({
                        "start": start,
                        "step": step,
                        "description": f"{sec}:{prefix}",
                        "position": self._position_of_dict(base_dict, axis),
                    })

        df = pd.DataFrame(rows)
        save_df(df, os.path.join(self.cfg.output_dir, "tables", "prototype_trajectory.csv"))

        plt.figure(figsize=(10, 6), dpi=self.cfg.fig_dpi)
        for start in [0, 1]:
            sub = df[df["start"] == start]
            plt.plot(sub["step"], sub["position"], marker="o", label=f"Start class {start}")
        plt.xlabel("Recombination step")
        plt.ylabel("Position on class-separation axis")
        plt.legend()
        plt.savefig(os.path.join(self.cfg.output_dir, "figures", "prototype_trajectory.png"))
        safe_close()
        return df

    def _position_of_dict(self, sec_dicts: Dict[str, Dict], axis: np.ndarray) -> float:
        full = []
        for sec in self.sections:
            d = sec_dicts[sec]
            text = self._dict_to_text(d)
            emb = self._safe_generate_section_embedding(text, sec)
            full.extend(emb)
        full = l2_normalize(np.asarray(full, dtype=float))
        return float(full @ axis)


# -----------------------------------------------------------------------------
# 5. 结局分析
# -----------------------------------------------------------------------------
class OutcomeAnalyzer:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def run_binary(self, df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
        cols = [self.cfg.label_col, outcome_col] + self.cfg.adjust_covariates
        cols = pick_existing_columns(df, cols)
        work = df[cols].dropna().copy()
        X = pd.get_dummies(work[[c for c in cols if c != outcome_col]], drop_first=True)
        X = sm.add_constant(X)
        y = work[outcome_col].astype(int)
        model = sm.Logit(y, X).fit(disp=False)
        res = model.summary2().tables[1].reset_index().rename(columns={"index": "variable"})
        res["outcome"] = outcome_col
        return res

    def run_continuous(self, df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
        cols = [self.cfg.label_col, outcome_col] + self.cfg.adjust_covariates
        cols = pick_existing_columns(df, cols)
        work = df[cols].dropna().copy()
        X = pd.get_dummies(work[[c for c in cols if c != outcome_col]], drop_first=True)
        X = sm.add_constant(X)
        y = work[outcome_col]
        model = sm.OLS(y, X).fit()
        res = model.summary2().tables[1].reset_index().rename(columns={"index": "variable"})
        res["outcome"] = outcome_col
        return res

    def run_survival(self, df: pd.DataFrame, time_col: str, event_col: str) -> pd.DataFrame:
        cols = [time_col, event_col, self.cfg.label_col] + self.cfg.adjust_covariates
        cols = pick_existing_columns(df, cols)
        work = df[cols].dropna().copy()

        cat_cols = [c for c in work.columns if c not in [time_col, event_col] and not pd.api.types.is_numeric_dtype(work[c])]
        cox_df = pd.get_dummies(work, columns=cat_cols, drop_first=True)

        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col=time_col, event_col=event_col)
        res = cph.summary.reset_index().rename(columns={"index": "variable"})
        res["time_col"] = time_col
        res["event_col"] = event_col

        kmf = KaplanMeierFitter()
        plt.figure(figsize=(8, 6), dpi=self.cfg.fig_dpi)
        for grp, sub in work.groupby(self.cfg.label_col):
            kmf.fit(sub[time_col], sub[event_col], label=f"cluster {grp}")
            kmf.plot_survival_function()
        plt.title(f"Kaplan-Meier: {event_col}")
        plt.savefig(os.path.join(self.cfg.output_dir, "figures", f"km_{event_col}.png"))
        safe_close()
        return res


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def main(cfg: PipelineConfig):
    try:
        print("=" * 60)
        print("步骤1: 处理嵌入向量")
        print("=" * 60)
        cache = EmbeddingCache(cfg)
        cache.process_and_cache(force_recompute=False)
        metadata_df, sec_embs, full_emb = cache.load_embeddings()
        print(f"样本数: {len(metadata_df)}")
        print(f"嵌入维度: {full_emb.shape[1]}")

        print("\n" + "=" * 60)
        print("步骤2: 标签处理")
        print("=" * 60)

        if cfg.label_csv and os.path.exists(cfg.label_csv):
            print(f"使用外部标签文件: {cfg.label_csv}")
            label_df = pd.read_csv(cfg.label_csv)
            print(f"标签文件列: {label_df.columns.tolist()}")

            base_meta = metadata_df.copy()
            keys = infer_merge_keys(base_meta, label_df, cfg.id_cols_priority)
            print(f"使用 ID 列进行合并: {keys}")

            label_use = label_df[keys + [cfg.label_col]].copy()
            merged = base_meta.merge(label_use, on=keys, how="inner")

            if merged.empty:
                raise ValueError("合并后数据为空，请检查 ID 列是否一致。")

            if merged.duplicated(subset=["row_id"]).any():
                dup_n = int(merged.duplicated(subset=["row_id"]).sum())
                print(f"警告: 合并后存在重复 row_id，已去重。重复数={dup_n}")
                merged = merged.drop_duplicates(subset=["row_id"]).copy()

            merged = merged.sort_values("row_id").reset_index(drop=True)
            row_idx = merged["row_id"].astype(int).values

            full_emb = full_emb[row_idx]
            for sec in sec_embs:
                sec_embs[sec] = sec_embs[sec][row_idx]

            metadata_df = merged.copy()
            y = metadata_df[cfg.label_col].astype(int).values

            print(f"使用外部标签，样本数: {len(y)}")
            print(f"类别分布: {np.bincount(y)}")
        else:
            print("未提供外部标签，进行自动聚类")
            stab = StabilityAnalyzer(cfg)
            y, _ = stab.run(full_emb)
            metadata_df = metadata_df.copy()
            metadata_df[cfg.label_col] = y
            print(f"自动聚类完成。类别分布: {np.bincount(y)}")

        label_out = metadata_df[["row_id", "phID", "bio_id", "source_file"]].copy()
        label_out[cfg.label_col] = y
        save_df(label_out, os.path.join(cfg.output_dir, "tables", "final_labels.csv"))
        print(f"标签已保存至: {os.path.join(cfg.output_dir, 'tables', 'final_labels.csv')}")

        print("\n" + "=" * 60)
        print("步骤3: Section-level attribution")
        print("=" * 60)
        sec_attr = SectionAttributor(cfg)
        sec_df = sec_attr.run(full_emb, y)
        print("\n各部分重要性排序:")
        for _, row in sec_df.iterrows():
            print(f"  {row['section']}: AUC下降 = {row['auc_drop']:.4f}")

        print("\n" + "=" * 60)
        print("步骤4: Teacher-student SHAP分析")
        print("=" * 60)
        shap_imp = None

        if cfg.flattened_csv and os.path.exists(cfg.flattened_csv):
            print(f"加载扁平化数据: {cfg.flattened_csv}")
            flat_df = pd.read_csv(cfg.flattened_csv)

            cols_to_remove = [
                "mest_c_score_m", "mest_c_score_e", "mest_c_score_s",
                "mest_c_score_t", "mest_c_score_c", "key_pathology_terms",
                "inflammation_activity_summary_glomerular_activity_score",
                "inflammation_activity_summary_tubulointerstitial_activity_score",
                "inflammation_activity_summary_vascular_activity_score",
                "inflammation_activity_summary_overall_inflammatory_burden"
            ]
            for col in cols_to_remove:
                if col in flat_df.columns:
                    flat_df.drop(columns=[col], inplace=True)

            merge_keys = infer_merge_keys(flat_df, label_out, cfg.id_cols_priority)
            print(f"SHAP 模块合并键: {merge_keys}")
            flat_merged = flat_df.merge(label_out[merge_keys + [cfg.label_col]], on=merge_keys, how="inner")
            print(f"合并后数据形状: {flat_merged.shape}")

            teacher = TeacherStudentAnalyzer(cfg)
            shap_imp = teacher.run(flat_merged, cfg.label_col)
            print("SHAP 分析完成")
        else:
            print("未提供 flattened_csv 或文件不存在，跳过 SHAP。")

        print("\n" + "=" * 60)
        print("步骤5: 反事实实验")
        print("=" * 60)

        json_feature_universe = build_json_feature_universe(cfg.json_dir, SectionProcessor.SECTION_NAMES)
        save_df(
            pd.DataFrame({"json_feature_key": sorted(json_feature_universe)}),
            os.path.join(cfg.output_dir, "tables", "json_feature_universe.csv"),
        )

        top_features_set: Set[str] = set()
        feature_map_df = pd.DataFrame(columns=["raw_feature", "matched_json_keys", "n_matches", "status"])

        if shap_imp is not None and not shap_imp.empty:
            raw_top = shap_imp.head(cfg.counterfactual_top_features)["raw_feature"].astype(str).tolist()
            top_features_set, feature_map_df = map_raw_features_to_json_keys(raw_top, json_feature_universe)
            save_df(feature_map_df, os.path.join(cfg.output_dir, "tables", "shap_to_json_feature_mapping.csv"))
            print(f"SHAP 原始 top 特征数: {len(raw_top)}")
            print(f"成功映射到 JSON 特征键数: {len(top_features_set)}")

        if not top_features_set:
            print("警告: SHAP 特征未能成功映射，退回为使用全部 JSON 特征键。")
            top_features_set = set(json_feature_universe)

        cf_engine = CounterfactualEngine(cfg, cache)
        axis = cf_engine._axis_from_labels(full_emb, y)
        print(f"类分离轴构建完成，轴范数: {np.linalg.norm(axis):.4f}")

        print("构建类原型...")
        prototypes = cf_engine.build_prototypes(metadata_df, y)
        print(f"原型构建完成: 类0有{len(prototypes[0])}个特征，类1有{len(prototypes[1])}个特征")

        print("\n[反事实删除实验]")
        cf_delete = cf_engine.delete_experiment(metadata_df, y, axis, top_features_set)
        print(f"删除实验记录数: {len(cf_delete)}")

        print("\n[反事实替换为原型]")
        cf_swap = cf_engine.swap_experiment(metadata_df, y, axis, prototypes, top_features_set)
        print(f"替换实验记录数: {len(cf_swap)}")

        print("\n[反事实块替换]")
        cf_block = cf_engine.block_replace_experiment(metadata_df, y, axis, prototypes, top_features_set)
        print(f"块替换实验记录数: {len(cf_block)}")

        print("\n" + "=" * 60)
        print("步骤6: 原型重组轨迹")
        print("=" * 60)
        proto_df = cf_engine.prototype_trajectory(full_emb, y, prototypes)
        print(f"原型轨迹生成完成，包含 {len(proto_df)} 个步骤")

        # if cfg.outcome_csv and os.path.exists(cfg.outcome_csv):
        #     print("\n" + "=" * 60)
        #     print("步骤7: 临床结局关联分析")
        #     print("=" * 60)
        #
        #     outcome_df = pd.read_csv(cfg.outcome_csv)
        #     merge_keys = infer_merge_keys(label_out, outcome_df, cfg.id_cols_priority)
        #     print(f"结局分析合并键: {merge_keys}")
        #
        #     merged_outcome = label_out.merge(outcome_df, on=merge_keys, how="inner")
        #     print(f"合并后数据形状: {merged_outcome.shape}")
        #
        #     outcome_analyzer = OutcomeAnalyzer(cfg)
        #
        #     for b in cfg.binary_outcomes:
        #         if b in merged_outcome.columns:
        #             res = outcome_analyzer.run_binary(merged_outcome, b)
        #             save_df(res, os.path.join(cfg.output_dir, "tables", f"outcome_binary_{b}.csv"))
        #
        #     for c in cfg.continuous_outcomes:
        #         if c in merged_outcome.columns:
        #             res = outcome_analyzer.run_continuous(merged_outcome, c)
        #             save_df(res, os.path.join(cfg.output_dir, "tables", f"outcome_cont_{c}.csv"))
        #
        #     for t, e in cfg.survival_specs:
        #         if t in merged_outcome.columns and e in merged_outcome.columns:
        #             res = outcome_analyzer.run_survival(merged_outcome, t, e)
        #             save_df(res, os.path.join(cfg.output_dir, "tables", f"outcome_surv_{e}.csv"))
        # else:
        #     print("未提供 outcome_csv 或文件不存在，跳过临床结局关联分析。")

        print("\n" + "=" * 60)
        print("所有分析完成！结果保存在:", cfg.output_dir)
        print("=" * 60)

    except Exception as e:
        print(f"\n主流程执行失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def get_base_dir() -> str:
    """获取当前脚本所在目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    base_dir = get_base_dir()
    cfg = PipelineConfig(
        json_dir=os.path.join(base_dir, "pathology_feature_deepseek_cleaned"),
        flattened_csv=os.path.join(base_dir, "demo_data","pathology_lower_flattened_deepseek.csv"),
        embedding_cache_dir=os.path.join(base_dir, "embedding_cache"),
        output_dir= os.path.join(base_dir, "import_feature_results_v2"),
        label_csv=os.path.join(base_dir, "deepseek_clustering_results/20260329_021547/clustered_data_simplified.csv"),
        model_name="qwen3-embedding:latest",
        max_workers=36,
        bootstrap_iterations=200,
        binary_outcomes=["endpoint"],
        survival_specs=[("end_follow", "endpoint")]
    )
    main(cfg)
