"""
Robust Clustering Evaluator
For high-dimensional data clustering evaluation and comparison
Includes complete clustering analysis pipeline with PDF export
"""

import numpy as np
import warnings
import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import pandas as pd

# JAMA compliant color palette (simplified)
JAMA_COLORS = {
    'blue': '#1F77B4',  # Dark blue
    'orange': '#FF7F0E',  # Orange
    'green': '#2CA02C',  # Green
    'red': '#D62728',  # Red
    'purple': '#9467BD',  # Purple
    'brown': '#8C564B',  # Brown
    'pink': '#E377C2',  # Pink
    'gray': '#7F7F7F',  # Gray
    'olive': '#BCBD22',  # Olive
    'cyan': '#17BECF',  # Cyan
    'light_blue': '#AEC7E8',  # Light blue
    'light_orange': '#FFBB78',  # Light orange
    'light_gray': '#C7C7C7'  # Light gray
}

# Set matplotlib style for JAMA with PDF optimization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=list(JAMA_COLORS.values())[:8])
plt.rcParams['pdf.fonttype'] = 42  # Ensure TrueType fonts in PDF
plt.rcParams['ps.fonttype'] = 42  # Ensure TrueType fonts in PS
plt.rcParams['figure.dpi'] = 300  # High resolution for PDF
plt.rcParams['savefig.dpi'] = 300  # High resolution for PDF

# Check scikit-learn version
import sklearn

sklearn_version = sklearn.__version__
print(f"scikit-learn version: {sklearn_version}")


# Parse version number
def parse_version(version_str):
    parts = version_str.split('.')
    major = int(parts[0]) if parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return major, minor, patch


sklearn_major, sklearn_minor, sklearn_patch = parse_version(sklearn_version)

# Determine whether to use n_jobs parameter
USE_N_JOBS_IN_KMEANS = sklearn_major < 1  # Version 1.0+ removed n_jobs parameter

# Check if ensure_all_finite parameter is supported
SUPPORTS_ENSURE_ALL_FINITE = sklearn_major > 1 or (sklearn_major == 1 and sklearn_minor >= 6)

# Check TSNE parameter name
TSNE_PARAM_ITER = 'max_iter' if sklearn_major >= 1 else 'n_iter'

# Try importing optional dependencies
try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN not available. Install with: pip install hdbscan")

try:
    from umap import UMAP

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from kneed import KneeLocator

    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False

try:
    from sklearn.manifold import TSNE

    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

warnings.filterwarnings('ignore', category=UserWarning)


class RobustClusteringEvaluator:
    """
    Robust high-dimensional clustering evaluator

    Supports multiple clustering algorithms and evaluation metrics,
    automatically selects the best clustering solution
    """

    def __init__(self,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 verbose: bool = True):
        """
        Initialize evaluator

        Parameters:
        -----------
        random_state : int, default=42
            Random seed
        n_jobs : int, default=-1
            Number of parallel jobs
        verbose : bool, default=True
            Whether to display detailed output
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.results = {}
        self.best_result = None
        self.best_labels = None
        self.reduced_embeddings = None

        # Set random seed
        np.random.seed(random_state)

        self._log(f"scikit-learn version: {sklearn_version}")
        self._log(f"TSNE iteration parameter: {TSNE_PARAM_ITER}")
        if USE_N_JOBS_IN_KMEANS:
            self._log("KMeans supports n_jobs parameter")
        else:
            self._log("KMeans does not support n_jobs parameter (scikit-learn >= 1.0)")

        if TSNE_AVAILABLE:
            self._log("TSNE dimensionality reduction available")
        else:
            self._log("TSNE dimensionality reduction not available")

        if UMAP_AVAILABLE:
            self._log("UMAP dimensionality reduction available")
        else:
            self._log("UMAP dimensionality reduction not available")

    def _log(self, message: str):
        """Log message"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    def normalize_embeddings(self,
                             embeddings: np.ndarray,
                             norm_type: str = 'l2') -> np.ndarray:
        """
        Normalize embedding vectors

        Parameters:
        -----------
        embeddings : np.ndarray
            Original embedding vectors
        norm_type : str, default='l2'
            Normalization type ('l1', 'l2', 'max')

        Returns:
        --------
        np.ndarray
            Normalized embedding vectors
        """
        self._log(f"Normalizing embedding vectors (norm={norm_type})")

        # Handle NaN values
        embeddings_copy = embeddings.copy()
        embeddings_copy[np.isnan(embeddings_copy)] = 0

        # Directly call normalize without additional parameters
        return normalize(embeddings_copy, norm=norm_type)

    def optimize_tsne_params(self, n_samples: int, n_features: int) -> Dict[str, Any]:
        """
        Optimize t-SNE parameters based on data characteristics

        Parameters:
        -----------
        n_samples : int
            Number of samples
        n_features : int
            Number of features

        Returns:
        --------
        Dict[str, Any]
            Optimized t-SNE parameters
        """
        params = {}

        # Adjust perplexity based on sample count
        if n_samples < 50:
            params['perplexity'] = max(5, n_samples // 10)
        elif n_samples < 200:
            params['perplexity'] = min(30, max(10, n_samples // 20))
        elif n_samples < 1000:
            params['perplexity'] = min(50, max(15, n_samples // 50))
        elif n_samples < 10000:
            params['perplexity'] = min(100, max(30, n_samples // 100))
        else:
            params['perplexity'] = min(200, n_samples // 200)

        # Adjust learning rate based on data size
        if n_samples > 10000:
            params['learning_rate'] = 200
        elif n_samples > 5000:
            params['learning_rate'] = 150
        elif n_samples > 1000:
            params['learning_rate'] = 100
        elif n_samples > 500:
            params['learning_rate'] = 50
        else:
            params['learning_rate'] = 'auto'  # Auto selection

        # Adjust iteration count
        if n_features > 5000:
            params[TSNE_PARAM_ITER] = 2000
            if TSNE_PARAM_ITER == 'max_iter':
                params['n_iter_without_progress'] = 500
        elif n_features > 1000:
            params[TSNE_PARAM_ITER] = 1500
            if TSNE_PARAM_ITER == 'max_iter':
                params['n_iter_without_progress'] = 450
        else:
            params[TSNE_PARAM_ITER] = 1000
            if TSNE_PARAM_ITER == 'max_iter':
                params['n_iter_without_progress'] = 300

        # Initialization method selection
        if n_samples > 1000:
            params['init'] = 'pca'
        else:
            # Small datasets can try random initialization for better local minima
            params['init'] = 'random'

        # Method selection
        if n_samples > 5000:
            params['method'] = 'barnes_hut'
            params['angle'] = 0.5
        else:
            # Small datasets can use exact method
            params['method'] = 'exact'

        # Early exaggeration phase (better separation)
        params['early_exaggeration'] = 12.0

        # Random seed
        params['random_state'] = self.random_state

        return params

    def reduce_dimension(self,
                         embeddings: np.ndarray,
                         target_dim: int = 200,
                         method: str = 'two_stage_pca',
                         pca_intermediate_dim: int = 200,
                         **kwargs) -> np.ndarray:
        """
        Dimensionality reduction - simplified version

        Parameters:
        -----------
        embeddings : np.ndarray
            Original embedding vectors
        target_dim : int, default=200
            Target dimension
        method : str, default='two_stage_pca'
            Dimensionality reduction method ('pca', 'svd', 'two_stage_pca', 'umap',
                                           'pca_then_tsne', 'pca_then_umap')
        pca_intermediate_dim : int, default=200
            PCA intermediate dimension (for pca_then_tsne, etc.)
        **kwargs : dict
            Additional parameters for dimensionality reduction algorithms

        Returns:
        --------
        np.ndarray
            Dimensionally reduced embedding vectors
        """
        self._log(f"Dimensionality reduction: {embeddings.shape[1]} -> {target_dim} dimensions (method={method})")

        n_samples, n_features = embeddings.shape
        actual_target_dim = min(target_dim, n_features, n_samples - 1)

        if actual_target_dim < target_dim:
            self._log(f"Warning: Target dimension adjusted to {actual_target_dim} (limited by data shape)")

        if method == 'pca':
            reducer = PCA(n_components=actual_target_dim,
                          random_state=self.random_state,
                          **kwargs)
            reduced = reducer.fit_transform(embeddings)
            explained_variance = sum(reducer.explained_variance_ratio_)
            self._log(f"PCA explained variance ratio: {explained_variance:.4f}")

        elif method == 'svd':
            reducer = TruncatedSVD(n_components=actual_target_dim,
                                   random_state=self.random_state,
                                   **kwargs)
            reduced = reducer.fit_transform(embeddings)

        elif method == 'two_stage_pca':
            # Stage 1: Reduce to intermediate dimension
            stage1_dim = min(pca_intermediate_dim, n_features, n_samples - 1)
            pca1 = PCA(n_components=stage1_dim,
                       random_state=self.random_state)
            intermediate = pca1.fit_transform(embeddings)
            self._log(f"Stage 1 PCA: {n_features} -> {stage1_dim} dimensions")
            # Display stage 1 explained variance ratio
            stage1_variance = sum(pca1.explained_variance_ratio_)
            self._log(f"Stage 1 total explained variance ratio: {stage1_variance:.4f}")

            # Stage 2: Reduce to target dimension
            stage2_dim = min(actual_target_dim, stage1_dim)
            pca2 = PCA(n_components=stage2_dim,
                       random_state=self.random_state)
            reduced = pca2.fit_transform(intermediate)
            self._log(f"Stage 2 PCA: {stage1_dim} -> {stage2_dim} dimensions")
            # Display stage 2 explained variance ratio
            stage2_variance = sum(pca2.explained_variance_ratio_)
            self._log(f"Stage 2 total explained variance ratio: {stage2_variance:.4f}")
            total_variance = stage1_variance * stage2_variance
            self._log(f"Total explained variance ratio: {total_variance:.4f}")

        elif method == 'umap' and UMAP_AVAILABLE:
            # UMAP parameter optimization
            umap_params = {
                'n_components': actual_target_dim,
                'random_state': self.random_state,
                'n_neighbors': kwargs.get('n_neighbors', min(15, n_samples - 1)),
                'min_dist': kwargs.get('min_dist', 0.1),
                'metric': kwargs.get('metric', 'cosine'),
                'n_epochs': kwargs.get('n_epochs', None),
                'learning_rate': kwargs.get('learning_rate', 1.0),
                'spread': kwargs.get('spread', 1.0),
                'low_memory': kwargs.get('low_memory', False),
                'verbose': self.verbose
            }

            # Adjust parameters based on data size
            if n_samples > 10000:
                umap_params['n_neighbors'] = min(30, n_samples - 1)
                umap_params['min_dist'] = 0.3
                if 'low_memory' not in kwargs:
                    umap_params['low_memory'] = True

            reducer = UMAP(**umap_params)
            reduced = reducer.fit_transform(embeddings)
            self._log(f"UMAP dimensionality reduction completed")

        elif method == 'umap' and not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")

        elif method == 'pca_then_umap' and UMAP_AVAILABLE:
            # Stage 1: PCA to intermediate dimension
            stage1_dim = min(pca_intermediate_dim, n_features, n_samples - 1)

            self._log(f"Stage 1 PCA: {n_features} -> {stage1_dim} dimensions")

            pca1 = PCA(n_components=stage1_dim,
                       random_state=self.random_state)
            intermediate = pca1.fit_transform(embeddings)

            pca_variance = sum(pca1.explained_variance_ratio_)
            self._log(f"PCA explained variance ratio: {pca_variance:.4f}")

            # Stage 2: UMAP to target dimension
            self._log(f"Stage 2 UMAP: {pca_intermediate_dim} -> {actual_target_dim} dimensions")

            # UMAP parameter settings
            umap_params = {
                'n_components': actual_target_dim,
                'random_state': self.random_state,
                'n_neighbors': kwargs.get('n_neighbors', min(15, n_samples - 1)),
                'min_dist': kwargs.get('min_dist', 0.1),
                'metric': kwargs.get('metric', 'euclidean'),
                'n_epochs': kwargs.get('n_epochs', None),
                'learning_rate': kwargs.get('learning_rate', 1.0),
                'spread': kwargs.get('spread', 1.0),
                'low_memory': kwargs.get('low_memory', False),
                'verbose': self.verbose
            }

            try:
                umap_reducer = UMAP(**umap_params)
                reduced = umap_reducer.fit_transform(intermediate)
                self._log(f"PCA+UMAP dimensionality reduction completed")
            except Exception as e:
                self._log(f"UMAP failed: {e}")
                self._log("Falling back to PCA")
                # If UMAP fails, use PCA
                pca_fallback = PCA(n_components=actual_target_dim,
                                   random_state=self.random_state)
                reduced = pca_fallback.fit_transform(intermediate)
                self._log(f"PCA completed, explained variance ratio: {sum(pca_fallback.explained_variance_ratio_):.4f}")

        elif method == 'pca_then_tsne':
            if not TSNE_AVAILABLE:
                raise ImportError("TSNE not available. Install scikit-learn with t-SNE support")

            # Stage 1: PCA to specified intermediate dimension
            pca_intermediate_dim = min(pca_intermediate_dim, n_features, n_samples - 1)
            self._log(f"Stage 1 PCA: {n_features} -> {pca_intermediate_dim} dimensions")

            pca1 = PCA(n_components=pca_intermediate_dim,
                       random_state=self.random_state)
            intermediate = pca1.fit_transform(embeddings)

            pca_variance = sum(pca1.explained_variance_ratio_)
            self._log(f"PCA explained variance ratio: {pca_variance:.4f}")
            self._log(
                f"First 10 principal components explained variance: {sum(pca1.explained_variance_ratio_[:10]):.4f}")

            # Stage 2: t-SNE to target dimension
            self._log(f"Stage 2 t-SNE: {pca_intermediate_dim} -> {actual_target_dim} dimensions")

            # Get optimized t-SNE parameters
            tsne_params = self.optimize_tsne_params(n_samples, pca_intermediate_dim)

            # Override with user-provided parameters
            tsne_params.update({k: v for k, v in kwargs.items() if k in [
                'perplexity', 'learning_rate', TSNE_PARAM_ITER, 'n_iter_without_progress',
                'init', 'method', 'angle', 'early_exaggeration'
            ]})

            # Ensure n_components is correct
            tsne_params['n_components'] = actual_target_dim

            # Adjust parameters based on scikit-learn version
            if sklearn_major < 1:
                tsne_params['n_jobs'] = self.n_jobs

            # Set verbose
            tsne_params['verbose'] = 1 if self.verbose else 0

            # Remove None values to avoid warnings
            tsne_params = {k: v for k, v in tsne_params.items() if v is not None}

            self._log(f"t-SNE parameters: perplexity={tsne_params.get('perplexity', 'default')}, "
                      f"learning_rate={tsne_params.get('learning_rate', 'default')}, "
                      f"{TSNE_PARAM_ITER}={tsne_params.get(TSNE_PARAM_ITER, 'default')}, "
                      f"init={tsne_params.get('init', 'default')}")

            try:
                tsne = TSNE(**tsne_params)
                reduced = tsne.fit_transform(intermediate)

                # Check KL divergence - lower is better, typically < 1.0 is good
                if hasattr(tsne, 'kl_divergence_'):
                    kl_divergence = tsne.kl_divergence_
                    self._log(f"t-SNE completed, final KL divergence: {kl_divergence:.4f}")
                else:
                    self._log("t-SNE completed")

            except Exception as e:
                self._log(f"t-SNE failed: {e}")
                self._log("Falling back to PCA")
                # If t-SNE fails, use PCA
                pca_fallback = PCA(n_components=actual_target_dim,
                                   random_state=self.random_state)
                reduced = pca_fallback.fit_transform(intermediate)
                self._log(f"PCA completed, explained variance ratio: {sum(pca_fallback.explained_variance_ratio_):.4f}")

        elif method == 'tsne_direct':
            """Direct t-SNE (without PCA preprocessing)"""
            if not TSNE_AVAILABLE:
                raise ImportError("TSNE not available. Install scikit-learn with t-SNE support")

            self._log(f"Direct t-SNE: {n_features} -> {actual_target_dim} dimensions")

            # Only suitable for small datasets
            if n_samples > 3000:
                self._log("Warning: Dataset is large, recommend using pca_then_tsne method")

            # Get optimized t-SNE parameters
            tsne_params = self.optimize_tsne_params(n_samples, n_features)

            # Override with user-provided parameters
            tsne_params.update({k: v for k, v in kwargs.items() if k in [
                'perplexity', 'learning_rate', TSNE_PARAM_ITER, 'n_iter_without_progress',
                'init', 'method', 'angle', 'early_exaggeration'
            ]})

            # Ensure n_components is correct
            tsne_params['n_components'] = actual_target_dim

            # Adjust parameters based on scikit-learn version
            if sklearn_major < 1:
                tsne_params['n_jobs'] = self.n_jobs

            # Set verbose
            tsne_params['verbose'] = 1 if self.verbose else 0

            self._log(f"Direct t-SNE parameters: perplexity={tsne_params.get('perplexity', 'default')}, "
                      f"{TSNE_PARAM_ITER}={tsne_params.get(TSNE_PARAM_ITER, 'default')}")

            try:
                tsne = TSNE(**tsne_params)
                reduced = tsne.fit_transform(embeddings)

                if hasattr(tsne, 'kl_divergence_'):
                    kl_divergence = tsne.kl_divergence_
                    self._log(f"Direct t-SNE completed, KL divergence: {kl_divergence:.4f}")

                    # If KL divergence is high, try optimization
                    if kl_divergence > 2.0 and n_samples < 5000:
                        self._log(
                            f"Direct t-SNE KL divergence is high ({kl_divergence:.4f}), trying PCA preprocessing...")
                        # Fall back to pca_then_tsne method
                        self._log("Switching to pca_then_tsne method...")
                        return self.reduce_dimension(
                            embeddings, target_dim, 'pca_then_tsne', pca_intermediate_dim, **kwargs
                        )
                else:
                    self._log("Direct t-SNE completed")

            except Exception as e:
                self._log(f"Direct t-SNE failed: {e}")
                self._log("Falling back to pca_then_tsne method")
                # If direct t-SNE fails, use pca_then_tsne
                return self.reduce_dimension(
                    embeddings, target_dim, 'pca_then_tsne', pca_intermediate_dim, **kwargs
                )

        elif method == 'pca_high_variance':
            """PCA with high explained variance ratio"""
            self._log(f"High variance PCA: {n_features} -> {actual_target_dim} dimensions")

            # Calculate dimensions needed for specified explained variance ratio
            target_variance = kwargs.get('target_variance', 0.95)

            # Use all dimensions to calculate cumulative variance
            pca_test = PCA(n_components=min(n_features, n_samples - 1),
                           random_state=self.random_state)
            pca_test.fit(embeddings)
            cumulative_variance = np.cumsum(pca_test.explained_variance_ratio_)

            # Find dimensions needed to reach target explained variance ratio
            required_dim = np.argmax(cumulative_variance >= target_variance) + 1
            if required_dim == 0:  # If not found, use max dimension
                required_dim = min(n_features, n_samples - 1)

            # If required dimension is less than target dimension, use target dimension
            pca_dim = max(required_dim, actual_target_dim)
            pca_dim = min(pca_dim, n_features, n_samples - 1)

            self._log(
                f"To achieve {target_variance:.1%} explained variance, {required_dim} principal components are needed")
            self._log(f"Using {pca_dim} principal components for PCA")

            pca = PCA(n_components=pca_dim, random_state=self.random_state)
            reduced = pca.fit_transform(embeddings)

            actual_variance = sum(pca.explained_variance_ratio_)
            self._log(f"PCA explained variance ratio: {actual_variance:.4f}")

            # If further reduction to target dimension is needed
            if pca_dim > actual_target_dim:
                self._log(f"Further reduction from {pca_dim} to {actual_target_dim} dimensions")
                pca2 = PCA(n_components=actual_target_dim, random_state=self.random_state)
                reduced = pca2.fit_transform(reduced)
                variance2 = sum(pca2.explained_variance_ratio_)
                total_variance = actual_variance * variance2
                self._log(f"Stage 2 explained variance ratio: {variance2:.4f}")
                self._log(f"Total explained variance ratio: {total_variance:.4f}")

        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")

        return reduced

    def estimate_optimal_k(self,
                           embeddings: np.ndarray,
                           k_range: Tuple[int, int] = (2, 20),
                           method: str = 'silhouette') -> int:
        """
        Estimate optimal number of clusters

        Parameters:
        -----------
        embeddings : np.ndarray
            Embedding vectors
        k_range : Tuple[int, int], default=(2, 20)
            K value range
        method : str, default='silhouette'
            Estimation method ('silhouette', 'elbow', 'gap')

        Returns:
        --------
        int
            Estimated optimal K value
        """
        min_k, max_k = k_range

        if method == 'silhouette':
            silhouette_scores = []
            valid_ks = []

            for k in range(min_k, min(max_k + 1, len(embeddings) - 1)):
                try:
                    kmeans = KMeans(n_clusters=k,
                                    random_state=self.random_state,
                                    n_init=10)
                    labels = kmeans.fit_predict(embeddings)

                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(embeddings, labels)
                        silhouette_scores.append(score)
                        valid_ks.append(k)
                except:
                    continue

            if silhouette_scores:
                optimal_k = valid_ks[np.argmax(silhouette_scores)]
                self._log(f"Silhouette method estimated optimal K: {optimal_k}")
                return optimal_k

        elif method == 'elbow' and KNEED_AVAILABLE:
            inertias = []
            valid_ks = []

            for k in range(min_k, min(max_k + 1, len(embeddings) - 1)):
                try:
                    kmeans = KMeans(n_clusters=k,
                                    random_state=self.random_state,
                                    n_init=10)
                    kmeans.fit(embeddings)
                    inertias.append(kmeans.inertia_)
                    valid_ks.append(k)
                except:
                    continue

            if len(inertias) >= 4:
                try:
                    kneedle = KneeLocator(valid_ks, inertias,
                                          curve='convex',
                                          direction='decreasing')
                    optimal_k = kneedle.knee
                    if optimal_k is not None:
                        self._log(f"Elbow method estimated optimal K: {optimal_k}")
                        return optimal_k
                except:
                    pass

        # Default return middle value
        default_k = min(max(min_k + 2, (min_k + max_k) // 2), len(embeddings) - 2)
        self._log(f"Using default K value: {default_k}")
        return default_k

    def estimate_dbscan_eps(self,
                            embeddings: np.ndarray,
                            min_samples: int = 5,
                            percentile: float = 90) -> float:
        """
        Estimate optimal eps parameter for DBSCAN

        Parameters:
        -----------
        embeddings : np.ndarray
            Embedding vectors
        min_samples : int, default=5
            min_samples parameter for DBSCAN
        percentile : float, default=90
            Percentile for eps estimation

        Returns:
        --------
        float
            Estimated eps value
        """
        # Use k-nearest neighbor distance to estimate eps
        nbrs = NearestNeighbors(n_neighbors=min_samples,
                                metric='cosine',
                                n_jobs=self.n_jobs)
        nbrs.fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)

        # Calculate k-th nearest neighbor distance
        k_distances = distances[:, -1]
        eps = np.percentile(k_distances, percentile)

        self._log(f"Estimated DBSCAN eps (min_samples={min_samples}, percentile={percentile}): {eps:.4f}")
        return float(eps)

    def evaluate_clustering(self,
                            X: np.ndarray,
                            labels: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Evaluate clustering results

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        labels : np.ndarray
            Cluster labels

        Returns:
        --------
        Optional[Dict[str, Any]]
            Evaluation result dictionary, returns None if evaluation fails
        """
        # Filter noise points (label = -1)
        mask = labels != -1
        n_noise = np.sum(~mask)
        n_clusters = len(np.unique(labels[mask]))

        # Check if there's enough data for evaluation
        if n_clusters < 2 or np.sum(mask) < 4:
            return None

        X_filtered = X[mask]
        labels_filtered = labels[mask]

        try:
            # Calculate evaluation metrics
            sil_score = silhouette_score(X_filtered, labels_filtered, metric='cosine')
            ch_score = calinski_harabasz_score(X_filtered, labels_filtered)
            db_score = davies_bouldin_score(X_filtered, labels_filtered)

            # Calculate cluster size statistics
            cluster_sizes = np.bincount(labels_filtered + 1)  # +1 to make labels start from 1
            cluster_stats = {
                'min_size': np.min(cluster_sizes),
                'max_size': np.max(cluster_sizes),
                'mean_size': np.mean(cluster_sizes),
                'std_size': np.std(cluster_sizes),
                'median_size': np.median(cluster_sizes)
            }

            return {
                'silhouette': float(sil_score),
                'calinski_harabasz': float(ch_score),
                'davies_bouldin': float(db_score),
                'n_clusters': int(n_clusters),
                'n_noise': int(n_noise),
                'noise_ratio': float(n_noise / len(labels)),
                'cluster_stats': cluster_stats,
                'valid_points': int(np.sum(mask))
            }
        except Exception as e:
            if self.verbose:
                self._log(f"Error evaluating clustering: {e}")
            return None

    def run_dbscan(self,
                   X: np.ndarray,
                   eps_values: Optional[List[float]] = None,
                   min_samples_list: Optional[List[int]] = None,
                   auto_estimate_eps: bool = True) -> Dict[str, Any]:
        """
        Run DBSCAN clustering

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        eps_values : List[float], optional
            List of eps parameter values
        min_samples_list : List[int], optional
            List of min_samples parameter values
        auto_estimate_eps : bool, default=True
            Whether to automatically estimate eps

        Returns:
        --------
        Dict[str, Any]
            DBSCAN clustering results
        """
        self._log("Running DBSCAN clustering...")

        if eps_values is None:
            eps_values = [0.3, 0.4, 0.5, 0.6, 0.7]

        if min_samples_list is None:
            min_samples_list = [3, 5, 7]

        results = {}

        for min_samples in min_samples_list:
            # Automatically estimate eps
            if auto_estimate_eps:
                try:
                    eps_estimated = self.estimate_dbscan_eps(X, min_samples)
                    current_eps_values = list(set([eps_estimated] + eps_values))
                except:
                    current_eps_values = eps_values
            else:
                current_eps_values = eps_values

            for eps in current_eps_values:
                try:
                    dbscan = DBSCAN(eps=eps,
                                    min_samples=min_samples,
                                    metric='cosine',
                                    n_jobs=self.n_jobs)
                    labels = dbscan.fit_predict(X)

                    # Evaluate results
                    eval_result = self.evaluate_clustering(X, labels)

                    if eval_result:
                        key = f"DBSCAN_eps={eps:.2f}_min_samples={min_samples}"
                        results[key] = {
                            'labels': labels.copy(),
                            'algorithm': 'DBSCAN',
                            'params': {'eps': eps, 'min_samples': min_samples},
                            **eval_result
                        }
                        self._log(f"  {key}: Silhouette={eval_result['silhouette']:.4f}, "
                                  f"Clusters={eval_result['n_clusters']}")

                except Exception as e:
                    if self.verbose:
                        self._log(f"  DBSCAN failed: eps={eps}, min_samples={min_samples}, error: {e}")

        return results

    def run_hdbscan(self,
                    X: np.ndarray,
                    min_cluster_sizes: Optional[List[int]] = None,
                    min_samples_list: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run HDBSCAN clustering - fixed version

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        min_cluster_sizes : List[int], optional
            List of minimum cluster sizes
        min_samples_list : List[int], optional
            List of min_samples parameter values

        Returns:
        --------
        Dict[str, Any]
            HDBSCAN clustering results
        """
        if not HDBSCAN_AVAILABLE:
            self._log("HDBSCAN not available, skipping...")
            return {}

        self._log("Running HDBSCAN clustering...")

        if min_cluster_sizes is None:
            min_cluster_sizes = [5, 10, 15, 20]

        if min_samples_list is None:
            min_samples_list = [5, 10]

        results = {}

        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_samples_list:
                try:
                    # Method 1: Try directly using feature matrix
                    # Note: HDBSCAN may be sensitive to data type, ensure using double precision
                    X_double = X.astype(np.float64) if X.dtype != np.float64 else X

                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric='euclidean',
                        cluster_selection_method='eom',
                        core_dist_n_jobs=self.n_jobs,
                        algorithm='generic',
                        prediction_data=False  # Reduce memory usage
                    )
                    labels = clusterer.fit_predict(X_double)

                    # Evaluate results
                    eval_result = self.evaluate_clustering(X, labels)

                    if eval_result:
                        key = f"HDBSCAN_min_cluster_size={min_cluster_size}_min_samples={min_samples}"
                        results[key] = {
                            'labels': labels.copy(),
                            'algorithm': 'HDBSCAN',
                            'params': {'min_cluster_size': min_cluster_size,
                                       'min_samples': min_samples,
                                       'metric': 'euclidean'},
                            **eval_result
                        }
                        self._log(f"  {key}: Silhouette={eval_result['silhouette']:.4f}, "
                                  f"Clusters={eval_result['n_clusters']}")

                except Exception as e:
                    if self.verbose:
                        self._log(f"  HDBSCAN failed: min_cluster_size={min_cluster_size}, "
                                  f"min_samples={min_samples}, error: {e}")

                    # Method 2: Try using cosine distance
                    try:
                        from sklearn.metrics.pairwise import cosine_distances

                        self._log(f"  Trying cosine distance matrix...")
                        # Calculate cosine distance matrix
                        distance_matrix = cosine_distances(X)

                        # Convert to double precision
                        distance_matrix = distance_matrix.astype(np.float64)

                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            metric='precomputed',
                            cluster_selection_method='eom',
                            core_dist_n_jobs=self.n_jobs
                        )
                        labels = clusterer.fit_predict(distance_matrix)

                        # Evaluate results
                        eval_result = self.evaluate_clustering(X, labels)

                        if eval_result:
                            key = f"HDBSCAN_cosine_min_cluster_size={min_cluster_size}_min_samples={min_samples}"
                            results[key] = {
                                'labels': labels.copy(),
                                'algorithm': 'HDBSCAN',
                                'params': {'min_cluster_size': min_cluster_size,
                                           'min_samples': min_samples,
                                           'metric': 'precomputed_cosine'},
                                **eval_result
                            }
                            self._log(f"  {key}: Silhouette={eval_result['silhouette']:.4f}, "
                                      f"Clusters={eval_result['n_clusters']}")
                    except Exception as e2:
                        if self.verbose:
                            self._log(f"  HDBSCAN cosine distance also failed: {e2}")

        return results

    def run_kmeans(self,
                   X: np.ndarray,
                   n_clusters_range: Optional[Union[range, List[int]]] = None,
                   n_init: int = 10,
                   estimate_k: bool = True) -> Dict[str, Any]:
        """
        Run KMeans clustering - fixed version

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        n_clusters_range : Union[range, List[int]], optional
            Range of cluster counts
        n_init : int, default=10
            Number of initializations
        estimate_k : bool, default=True
            Whether to automatically estimate optimal K

        Returns:
        --------
        Dict[str, Any]
            KMeans clustering results
        """
        self._log("Running KMeans clustering...")

        if n_clusters_range is None:
            max_clusters = min(20, len(X) // 10)
            n_clusters_range = range(2, max(3, max_clusters + 1))

        # Automatically estimate optimal K
        if estimate_k:
            try:
                optimal_k = self.estimate_optimal_k(X, method='silhouette')
                if optimal_k:
                    n_clusters_range = list(set([optimal_k] + list(n_clusters_range)))
            except:
                pass

        results = {}

        for n_clusters in n_clusters_range:
            try:
                # Determine whether to include n_jobs parameter based on scikit-learn version
                if USE_N_JOBS_IN_KMEANS:
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        init='k-means++',
                        n_init=n_init,
                        max_iter=300,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs  # Old versions use
                    )
                else:
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        init='k-means++',
                        n_init=n_init,
                        max_iter=300,
                        random_state=self.random_state
                        # New versions removed n_jobs parameter
                    )

                labels = kmeans.fit_predict(X)

                # Evaluate results
                eval_result = self.evaluate_clustering(X, labels)

                if eval_result:
                    key = f"KMeans_n_clusters={n_clusters}"
                    results[key] = {
                        'labels': labels.copy(),
                        'algorithm': 'KMeans',
                        'params': {'n_clusters': n_clusters},
                        **eval_result,
                        'inertia': float(kmeans.inertia_)
                    }
                    self._log(f"  {key}: Silhouette={eval_result['silhouette']:.4f}")

            except Exception as e:
                if self.verbose:
                    self._log(f"  KMeans failed: n_clusters={n_clusters}, error: {e}")

        return results

    def run_spectral(self,
                     X: np.ndarray,
                     n_clusters_range: Optional[Union[range, List[int]]] = None,
                     estimate_k: bool = True) -> Dict[str, Any]:
        """
        Run spectral clustering

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        n_clusters_range : Union[range, List[int]], optional
            Range of cluster counts
        estimate_k : bool, default=True
            Whether to automatically estimate optimal K

        Returns:
        --------
        Dict[str, Any]
            Spectral clustering results
        """
        self._log("Running spectral clustering...")

        if n_clusters_range is None:
            max_clusters = min(20, len(X) // 10)
            n_clusters_range = range(2, max(3, max_clusters + 1))

        # Automatically estimate optimal K
        if estimate_k:
            try:
                optimal_k = self.estimate_optimal_k(X, method='silhouette')
                if optimal_k:
                    n_clusters_range = list(set([optimal_k] + list(n_clusters_range)))
            except:
                pass

        results = {}

        for n_clusters in n_clusters_range:
            try:
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='nearest_neighbors',
                    n_neighbors=min(10, len(X) - 1),
                    random_state=self.random_state,
                    n_jobs=self.n_jobs)
                labels = spectral.fit_predict(X)

                # Evaluate results
                eval_result = self.evaluate_clustering(X, labels)

                if eval_result:
                    key = f"Spectral_n_clusters={n_clusters}"
                    results[key] = {
                        'labels': labels.copy(),
                        'algorithm': 'Spectral',
                        'params': {'n_clusters': n_clusters},
                        **eval_result
                    }
                    self._log(f"  {key}: Silhouette={eval_result['silhouette']:.4f}")

            except Exception as e:
                if self.verbose:
                    self._log(f"  Spectral clustering failed: n_clusters={n_clusters}, error: {e}")

        return results

    def run_hierarchical(self,
                         X: np.ndarray,
                         n_clusters_range: Optional[Union[range, List[int]]] = None,
                         linkage_methods: Optional[List[str]] = None,
                         estimate_k: bool = True) -> Dict[str, Any]:
        """
        Run hierarchical clustering

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        n_clusters_range : Union[range, List[int]], optional
            Range of cluster counts
        linkage_methods : List[str], optional
            List of linkage methods
        estimate_k : bool, default=True
            Whether to automatically estimate optimal K

        Returns:
        --------
        Dict[str, Any]
            Hierarchical clustering results
        """
        self._log("Running hierarchical clustering...")

        if n_clusters_range is None:
            max_clusters = min(20, len(X) // 10)
            n_clusters_range = range(2, max(3, max_clusters + 1))

        if linkage_methods is None:
            linkage_methods = ['ward', 'complete', 'average']

        # Automatically estimate optimal K
        if estimate_k:
            try:
                optimal_k = self.estimate_optimal_k(X, method='silhouette')
                if optimal_k:
                    n_clusters_range = list(set([optimal_k] + list(n_clusters_range)))
            except:
                pass

        results = {}

        for n_clusters in n_clusters_range:
            for method in linkage_methods:
                try:
                    # Calculate distance matrix
                    if method == 'ward':
                        # Ward method requires Euclidean distance
                        distance_matrix = pdist(X, metric='euclidean')
                    else:
                        distance_matrix = pdist(X, metric='cosine')

                    # Hierarchical clustering
                    Z = linkage(distance_matrix, method=method)
                    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

                    # Adjust labels to start from 0
                    labels = labels - 1

                    # Evaluate results
                    eval_result = self.evaluate_clustering(X, labels)

                    if eval_result:
                        key = f"Hierarchical_{method}_n_clusters={n_clusters}"
                        results[key] = {
                            'labels': labels.copy(),
                            'algorithm': 'Hierarchical',
                            'params': {'n_clusters': n_clusters, 'method': method},
                            **eval_result
                        }
                        self._log(f"  {key}: Silhouette={eval_result['silhouette']:.4f}")

                except Exception as e:
                    if self.verbose:
                        self._log(f"  Hierarchical clustering failed: method={method}, "
                                  f"n_clusters={n_clusters}, error: {e}")

        return results

    def run_all_clustering(self,
                           X: np.ndarray,
                           algorithms: Optional[List[str]] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Run all clustering algorithms

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        algorithms : List[str], optional
            List of algorithms to run
        **kwargs : dict
            Additional parameters for each algorithm

        Returns:
        --------
        Dict[str, Any]
            All clustering results
        """
        if algorithms is None:
            algorithms = ['dbscan', 'hdbscan', 'kmeans', 'spectral', 'hierarchical']

        all_results = {}

        # Run each algorithm
        if 'dbscan' in algorithms:
            dbscan_results = self.run_dbscan(X, **kwargs.get('dbscan', {}))
            all_results.update(dbscan_results)

        if 'hdbscan' in algorithms and HDBSCAN_AVAILABLE:
            hdbscan_results = self.run_hdbscan(X, **kwargs.get('hdbscan', {}))
            all_results.update(hdbscan_results)

        if 'kmeans' in algorithms:
            kmeans_results = self.run_kmeans(X, **kwargs.get('kmeans', {}))
            all_results.update(kmeans_results)

        if 'spectral' in algorithms:
            spectral_results = self.run_spectral(X, **kwargs.get('spectral', {}))
            all_results.update(spectral_results)

        if 'hierarchical' in algorithms:
            hierarchical_results = self.run_hierarchical(X, **kwargs.get('hierarchical', {}))
            all_results.update(hierarchical_results)

        # Save results
        self.results.update(all_results)

        return all_results

    def get_best_results(self,
                         metric: str = 'silhouette',
                         top_k: int = 5,
                         min_clusters: int = 2,
                         max_noise_ratio: float = 0.5) -> List[Tuple[str, Dict]]:
        """
        Get best clustering results

        Parameters:
        -----------
        metric : str, default='silhouette'
            Evaluation metric ('silhouette', 'calinski_harabasz', 'davies_bouldin')
        top_k : int, default=5
            Number of best results to return
        min_clusters : int, default=2
            Minimum number of clusters
        max_noise_ratio : float, default=0.5
            Maximum noise ratio

        Returns:
        --------
        List[Tuple[str, Dict]]
            List of best results
        """
        if not self.results:
            return []

        # Filter results
        filtered_results = {}
        for key, result in self.results.items():
            if (result['n_clusters'] >= min_clusters and
                    result['noise_ratio'] <= max_noise_ratio):
                filtered_results[key] = result

        if not filtered_results:
            return []

        # Sort by metric
        if metric == 'davies_bouldin':
            # davies_bouldin: lower is better
            sorted_results = sorted(filtered_results.items(),
                                    key=lambda x: x[1][metric])
        else:
            # Other metrics: higher is better
            sorted_results = sorted(filtered_results.items(),
                                    key=lambda x: x[1][metric], reverse=True)

        return sorted_results[:top_k]

    def print_summary(self, top_k: int = 10):
        """Print clustering results summary"""
        if not self.results:
            print("No clustering results available")
            return

        print("\n" + "=" * 80)
        print("Clustering Algorithm Comparison Summary")
        print("=" * 80)

        # Group by algorithm
        algorithm_groups = {}
        for key, result in self.results.items():
            algo = result['algorithm']
            if algo not in algorithm_groups:
                algorithm_groups[algo] = []
            algorithm_groups[algo].append((key, result))

        # Print top results for each algorithm
        for algo, results in algorithm_groups.items():
            print(f"\n{algo} Algorithm:")
            print("-" * 40)

            # Sort by silhouette score
            sorted_results = sorted(results, key=lambda x: x[1]['silhouette'], reverse=True)

            for i, (key, result) in enumerate(sorted_results[:3]):
                noise_percent = result['noise_ratio'] * 100
                print(f"  {i + 1}. {key}")
                print(f"     Clusters: {result['n_clusters']:2d}, "
                      f"Noise: {noise_percent:5.1f}%, "
                      f"Silhouette: {result['silhouette']:.4f}, "
                      f"Calinski-Harabasz: {result['calinski_harabasz']:.0f}")

        # Print overall best results
        print("\n" + "=" * 80)
        print("Best Clustering Results:")
        print("=" * 80)

        best_results = self.get_best_results(metric='silhouette', top_k=top_k)

        for i, (key, result) in enumerate(best_results):
            noise_percent = result['noise_ratio'] * 100
            print(f"{i + 1}. {key}")
            print(f"   Algorithm: {result['algorithm']}")
            print(f"   Clusters: {result['n_clusters']}, Noise Ratio: {noise_percent:.1f}%")
            print(f"   Silhouette Score: {result['silhouette']:.4f}")
            print(f"   Calinski-Harabasz Index: {result['calinski_harabasz']:.2f}")
            print(f"   Davies-Bouldin Index: {result['davies_bouldin']:.4f}")
            print()

    def robust_high_dim_clustering(self,
                                   embeddings: np.ndarray,
                                   target_dim: int = 200,
                                   reduction_method: str = 'two_stage_pca',
                                   norm_type: str = 'l2',
                                   pca_intermediate_dim: int = 200,
                                   run_all_algorithms: bool = True,
                                   **kwargs) -> Tuple[Optional[np.ndarray],
    Optional[Dict],
    np.ndarray]:
        """
        Robust high-dimensional clustering pipeline

        Parameters:
        -----------
        embeddings : np.ndarray
            Original embedding vectors
        target_dim : int, default=200
            Target dimensionality reduction dimension
        reduction_method : str, default='two_stage_pca'
            Dimensionality reduction method
        norm_type : str, default='l2'
            Normalization type
        pca_intermediate_dim : int, default=200
            PCA intermediate dimension
        run_all_algorithms : bool, default=True
            Whether to run all clustering algorithms
        **kwargs : dict
            Other parameters

        Returns:
        --------
        Tuple[Optional[np.ndarray], Optional[Dict], np.ndarray]
            Cluster labels, best result, dimensionally reduced embeddings
        """
        # 1. Normalization
        self._log("Step 1: Normalizing embedding vectors")
        embeddings_norm = self.normalize_embeddings(embeddings, norm_type)

        # 2. Dimensionality reduction
        self._log("Step 2: Dimensionality reduction")
        reduced_embeddings = self.reduce_dimension(
            embeddings_norm, target_dim, reduction_method, pca_intermediate_dim, **kwargs
        )
        self.reduced_embeddings = reduced_embeddings

        # 3. Run clustering algorithms
        self._log("Step 3: Running clustering algorithms")
        if run_all_algorithms:
            self.run_all_clustering(reduced_embeddings, **kwargs)
        else:
            # Only run DBSCAN and KMeans
            dbscan_results = self.run_dbscan(reduced_embeddings)
            kmeans_results = self.run_kmeans(reduced_embeddings)
            self.results.update(dbscan_results)
            self.results.update(kmeans_results)

        # 4. Select best result
        self._log("Step 4: Selecting best result")
        best_results = self.get_best_results(metric='silhouette', top_k=1)

        if best_results:
            best_key, best_result = best_results[0]
            self.best_result = best_result
            self.best_labels = best_result['labels']

            self._log(f"Best clustering result: {best_key}")
            self._log(f"  Algorithm: {best_result['algorithm']}")
            self._log(f"  Silhouette Score: {best_result['silhouette']:.4f}")
            self._log(f"  Number of Clusters: {best_result['n_clusters']}")
            self._log(f"  Noise Ratio: {best_result['noise_ratio']:.2%}")

            # Print summary
            self.print_summary(top_k=5)

            return self.best_labels, best_result, reduced_embeddings
        else:
            self._log("Warning: No valid clustering results found")
            return None, None, reduced_embeddings

    def save_results(self,
                     filepath: str = "clustering_results.npz",
                     save_embeddings: bool = False):
        """
        Save clustering results

        Parameters:
        -----------
        filepath : str, default="clustering_results.npz"
            Save path
        save_embeddings : bool, default=False
            Whether to save dimensionally reduced embeddings
        """
        import pickle

        # Prepare data to save
        save_data = {
            'results': self.results,
            'best_result': self.best_result,
            'best_labels': self.best_labels,
            'random_state': self.random_state
        }

        if save_embeddings and self.reduced_embeddings is not None:
            save_data['reduced_embeddings'] = self.reduced_embeddings

        # Save as pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        self._log(f"Clustering results saved to: {filepath}")

    def load_results(self, filepath: str):
        """
        Load clustering results

        Parameters:
        -----------
        filepath : str
            Load path
        """
        import pickle

        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)

        self.results = loaded_data.get('results', {})
        self.best_result = loaded_data.get('best_result')
        self.best_labels = loaded_data.get('best_labels')
        self.reduced_embeddings = loaded_data.get('reduced_embeddings')

        self._log(f"Clustering results loaded from {filepath}")

    def plot_clustering_results(self,
                                reduced_embeddings: np.ndarray,
                                labels: np.ndarray,
                                save_path: Optional[str] = None,
                                algorithm_name: str = "Clustering",
                                show_plots: bool = True) -> plt.Figure:
        """
        Plot clustering results

        Parameters:
        -----------
        reduced_embeddings : np.ndarray
            Dimensionally reduced embeddings (typically 2D or 3D)
        labels : np.ndarray
            Cluster labels
        save_path : str, optional
            Path to save the figure (PDF format)
        algorithm_name : str
            Algorithm name for figure title
        show_plots : bool
            Whether to display the figure

        Returns:
        --------
        plt.Figure
            The created figure
        """
        if reduced_embeddings.shape[1] > 3:
            # If dimension > 3, use PCA to reduce to 2D
            pca = PCA(n_components=2, random_state=self.random_state)
            plot_embeddings = pca.fit_transform(reduced_embeddings)
            explained_variance = sum(pca.explained_variance_ratio_)
            self._log(f"PCA visualization: Explained variance = {explained_variance:.4f}")
        elif reduced_embeddings.shape[1] == 3:
            plot_embeddings = reduced_embeddings
        else:
            plot_embeddings = reduced_embeddings

        # Create figure with JAMA-compliant style
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{algorithm_name} Clustering Results',
                     fontsize=16, fontweight='bold',
                     fontname='Arial')

        # 1. Main clustering plot
        ax1 = axes[0, 0]
        if plot_embeddings.shape[1] >= 2:
            scatter = ax1.scatter(plot_embeddings[:, 0], plot_embeddings[:, 1],
                                  c=labels, cmap='tab20', s=30, alpha=0.7,
                                  edgecolors='w', linewidth=0.5)
            ax1.set_xlabel('Dimension 1' if plot_embeddings.shape[1] > 2 else 'Principal Component 1',
                           fontname='Arial')
            ax1.set_ylabel('Dimension 2' if plot_embeddings.shape[1] > 2 else 'Principal Component 2',
                           fontname='Arial')
        ax1.set_title('Cluster Distribution', fontname='Arial', fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Cluster Label', fontname='Arial')

        # 2. Cluster size histogram
        ax2 = axes[0, 1]
        unique_labels, counts = np.unique(labels, return_counts=True)
        # Filter noise points (label = -1)
        valid_mask = unique_labels != -1
        valid_labels = unique_labels[valid_mask]
        valid_counts = counts[valid_mask]

        # Use JAMA colors for bars
        colors = [JAMA_COLORS['blue'] if i % 2 == 0 else JAMA_COLORS['orange']
                  for i in range(len(valid_labels))]

        bars = ax2.bar(valid_labels.astype(str), valid_counts,
                       color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_title(f'Cluster Size Distribution ({len(valid_labels)} clusters)',
                      fontname='Arial', fontweight='bold')
        ax2.set_xlabel('Cluster Label', fontname='Arial')
        ax2.set_ylabel('Number of Samples', fontname='Arial')
        ax2.tick_params(axis='x', rotation=45)

        # Add count labels on bars
        for bar, count in zip(bars, valid_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + max(valid_counts) * 0.01,
                     f'{count}', ha='center', va='bottom', fontsize=9, fontname='Arial')

        # 3. Noise point analysis
        ax3 = axes[0, 2]
        noise_count = np.sum(labels == -1)
        total_count = len(labels)
        noise_ratio = noise_count / total_count if total_count > 0 else 0

        pie_data = [total_count - noise_count, noise_count]
        pie_labels = ['Clustered Points', f'Noise Points\n({noise_ratio:.1%})']
        colors = [JAMA_COLORS['green'], JAMA_COLORS['red']]

        wedges, texts, autotexts = ax3.pie(pie_data, labels=pie_labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90,
                                           explode=(0.05, 0))

        # Style pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontname('Arial')

        ax3.set_title('Noise Point Analysis', fontname='Arial', fontweight='bold')
        ax3.axis('equal')

        # 4. Evaluation metrics radar chart (if available)
        ax4 = axes[1, 0]
        if hasattr(self, 'best_result') and self.best_result:
            metrics = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
            values = [
                self.best_result.get('silhouette', 0),
                self.best_result.get('calinski_harabasz', 0),
                -self.best_result.get('davies_bouldin', 0)  # DBI: lower is better, so take negative
            ]

            # Normalize to 0-1 range
            norm_values = []
            for i, val in enumerate(values):
                if metrics[i] == 'Silhouette':
                    # Silhouette score range is -1 to 1
                    norm_val = (val + 1) / 2
                elif metrics[i] == 'Calinski-Harabasz':
                    # CH index typically positive, normalize to 0-1
                    norm_val = min(val / 1000, 1) if val > 0 else 0
                else:
                    # DBI (negative) normalization
                    norm_val = (val + 2) / 4 if val > -2 else 0  # Assuming DBI range ~0-2

                norm_values.append(max(0, min(1, norm_val)))

            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            norm_values += norm_values[:1]
            angles += angles[:1]

            ax4 = plt.subplot(2, 3, 4, polar=True)
            ax4.plot(angles, norm_values, 'o-', linewidth=2, color=JAMA_COLORS['purple'])
            ax4.fill(angles, norm_values, color=JAMA_COLORS['purple'], alpha=0.25)
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics, fontname='Arial')
            ax4.set_ylim(0, 1)
            ax4.set_title('Evaluation Metrics Radar Chart', fontname='Arial', fontweight='bold', pad=20)
            ax4.grid(True, alpha=0.3)
            ax4.set_facecolor('white')

        # 5. Intra vs inter cluster distances
        ax5 = axes[1, 1]
        if plot_embeddings.shape[1] >= 2:
            # Calculate sample distances
            from sklearn.metrics.pairwise import pairwise_distances
            if len(labels) > 1000:
                # Sample large datasets
                sample_idx = np.random.choice(len(labels), min(1000, len(labels)), replace=False)
                sample_embeddings = plot_embeddings[sample_idx]
                sample_labels = labels[sample_idx]
            else:
                sample_embeddings = plot_embeddings
                sample_labels = labels

            # Calculate intra and inter cluster distances
            intra_distances = []
            inter_distances = []

            unique_labels = np.unique(sample_labels)
            valid_labels = unique_labels[unique_labels != -1]

            if len(valid_labels) > 1:
                for label in valid_labels:
                    mask = sample_labels == label
                    if np.sum(mask) > 1:
                        # Intra-cluster distances
                        intra_dist = pairwise_distances(sample_embeddings[mask], metric='euclidean')
                        intra_distances.append(np.mean(intra_dist))

                    # Inter-cluster distances
                    for other_label in valid_labels:
                        if other_label > label:
                            other_mask = sample_labels == other_label
                            inter_dist = pairwise_distances(sample_embeddings[mask],
                                                            sample_embeddings[other_mask],
                                                            metric='euclidean')
                            inter_distances.append(np.mean(inter_dist))

                if intra_distances and inter_distances:
                    categories = ['Intra-cluster', 'Inter-cluster']
                    means = [np.mean(intra_distances), np.mean(inter_distances)]
                    stds = [np.std(intra_distances), np.std(inter_distances)]

                    # Use JAMA colors
                    bar_colors = [JAMA_COLORS['blue'], JAMA_COLORS['orange']]
                    bars = ax5.bar(categories, means, yerr=stds,
                                   capsize=10, alpha=0.7, color=bar_colors,
                                   edgecolor='black', linewidth=0.5)
                    ax5.set_title('Cluster Quality Analysis',
                                  fontname='Arial', fontweight='bold')
                    ax5.set_ylabel('Mean Distance', fontname='Arial')
                    ax5.grid(True, alpha=0.3, axis='y')

                    # Calculate separation ratio
                    separation = means[1] / means[0] if means[0] > 0 else 0
                    ax5.text(0.5, -0.15, f'Separation Ratio = {separation:.2f}',
                             transform=ax5.transAxes, ha='center',
                             bbox=dict(boxstyle="round,pad=0.3",
                                       facecolor=JAMA_COLORS['light_gray'],
                                       alpha=0.7),
                             fontname='Arial')

        # 6. Algorithm comparison chart
        ax6 = axes[1, 2]
        if hasattr(self, 'results') and self.results:
            algorithms = []
            silhouette_scores = []

            for key, result in self.results.items():
                if 'silhouette' in result:
                    algorithms.append(result.get('algorithm', 'Unknown'))
                    silhouette_scores.append(result['silhouette'])

            if algorithms:
                # Group by algorithm, take highest score for each
                algo_scores = {}
                for algo, score in zip(algorithms, silhouette_scores):
                    if algo not in algo_scores or score > algo_scores[algo]:
                        algo_scores[algo] = score

                sorted_algorithms = sorted(algo_scores.items(), key=lambda x: x[1], reverse=True)
                algo_names = [f"{algo}" for algo, _ in sorted_algorithms[:10]]
                algo_scores_list = [score for _, score in sorted_algorithms[:10]]

                # Use JAMA color palette
                colors = list(JAMA_COLORS.values())[:len(algo_names)]
                bars = ax6.barh(algo_names, algo_scores_list,
                                color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                ax6.set_xlabel('Silhouette Score', fontname='Arial')
                ax6.set_title('Algorithm Performance Comparison',
                              fontname='Arial', fontweight='bold')
                ax6.grid(True, alpha=0.3, axis='x')

                # Mark best algorithm
                if algo_scores_list:
                    best_idx = np.argmax(algo_scores_list)
                    bars[best_idx].set_edgecolor('red')
                    bars[best_idx].set_linewidth(2)
                    bars[best_idx].set_hatch('//')

        plt.tight_layout()

        # Save figure as PDF
        if save_path:
            if not save_path.endswith('.pdf'):
                save_path = save_path.replace('.png', '.pdf').replace('.jpg', '.pdf')

            # Save as high-quality PDF
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none',
                        format='pdf', transparent=False)
            self._log(f"Clustering results plot saved to PDF: {save_path}")

        # Show or close figure
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_silhouette_rank_lineplot(self,
                                      output_dir: str = ".",
                                      save_as_pdf: bool = True,
                                      show_plot: bool = False) -> plt.Figure:
        """
        Plot Rank vs Silhouette Score line plot for all clustering methods

        Parameters:
        -----------
        output_dir : str
            Output directory
        save_as_pdf : bool
            Whether to save as PDF
        show_plot : bool
            Whether to display the plot

        Returns:
        --------
        plt.Figure
            The created figure
        """
        if not hasattr(self, 'results') or not self.results:
            self._log("No results available for plotting")
            return None

        # Prepare data
        algorithm_data = []

        for key, result in self.results.items():
            algorithm = result.get('algorithm', 'Unknown')
            params = result.get('params', {})

            # Extract algorithm parameters for labeling
            param_str = ""
            if algorithm == 'DBSCAN':
                param_str = f"eps={params.get('eps', 0):.2f}, min_samples={params.get('min_samples', 0)}"
            elif algorithm == 'HDBSCAN':
                param_str = f"min_cluster_size={params.get('min_cluster_size', 0)}"
            elif algorithm == 'KMeans':
                param_str = f"k={params.get('n_clusters', 0)}"
            elif algorithm == 'Spectral':
                param_str = f"k={params.get('n_clusters', 0)}"
            elif algorithm == 'Hierarchical':
                param_str = f"k={params.get('n_clusters', 0)}, method={params.get('method', '')}"

            algorithm_data.append({
                'algorithm': algorithm,
                'parameters': param_str,
                'silhouette': result.get('silhouette', 0),
                'calinski_harabasz': result.get('calinski_harabasz', 0),
                'davies_bouldin': result.get('davies_bouldin', 0),
                'n_clusters': result.get('n_clusters', 0),
                'noise_ratio': result.get('noise_ratio', 0),
                'full_name': key
            })

        if not algorithm_data:
            return None

        # Create DataFrame and sort by silhouette score
        import pandas as pd
        df_algorithms = pd.DataFrame(algorithm_data)
        df_algorithms = df_algorithms.sort_values('silhouette', ascending=False).reset_index(drop=True)
        df_algorithms['rank'] = df_algorithms.index + 1

        # Create figure with JAMA-compliant style
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Clustering Algorithm Performance: Rank vs Silhouette Score',
                     fontsize=16, fontweight='bold', fontname='Arial')

        # 1. Main line plot: Rank vs Silhouette Score
        ax1 = axes[0, 0]

        # Group by algorithm type for coloring
        algorithms = df_algorithms['algorithm'].unique()
        color_map = {}
        for i, algo in enumerate(algorithms):
            color_map[algo] = list(JAMA_COLORS.values())[i % len(JAMA_COLORS)]

        # Plot each algorithm type
        for algo in algorithms:
            algo_data = df_algorithms[df_algorithms['algorithm'] == algo]
            if len(algo_data) > 0:
                ax1.plot(algo_data['rank'], algo_data['silhouette'],
                         'o-', linewidth=2, markersize=6,
                         color=color_map[algo], label=algo, alpha=0.8)

        ax1.set_xlabel('Rank (sorted by Silhouette Score)', fontname='Arial', fontsize=11)
        ax1.set_ylabel('Silhouette Score', fontname='Arial', fontsize=11)
        ax1.set_title('Rank vs Silhouette Score for All Clustering Methods',
                      fontname='Arial', fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=10, framealpha=0.7)

        # Add horizontal line for good silhouette threshold (0.5)
        ax1.axhline(y=0.5, color=JAMA_COLORS['red'], linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(0.02, 0.52, 'Good clustering threshold (0.5)',
                 transform=ax1.transAxes, fontsize=9, color=JAMA_COLORS['red'],
                 fontname='Arial')

        # Add data point labels for top 5
        top_5 = df_algorithms.head(5)
        for _, row in top_5.iterrows():
            ax1.annotate(f"{row['algorithm']}\n(Rank {row['rank']})",
                         xy=(row['rank'], row['silhouette']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=9, fontname='Arial',
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor=color_map[row['algorithm']],
                                   alpha=0.2))

        # 2. Silhouette score distribution by algorithm type
        ax2 = axes[0, 1]

        # Prepare data for boxplot
        boxplot_data = []
        boxplot_labels = []
        for algo in algorithms:
            algo_scores = df_algorithms[df_algorithms['algorithm'] == algo]['silhouette'].values
            if len(algo_scores) > 0:
                boxplot_data.append(algo_scores)
                boxplot_labels.append(algo)

        # Create boxplot with JAMA colors
        box_colors = [color_map[algo] for algo in boxplot_labels]
        boxplot = ax2.boxplot(boxplot_data, labels=boxplot_labels,
                              patch_artist=True, medianprops=dict(color='black', linewidth=2))

        # Apply colors to boxes
        for patch, color in zip(boxplot['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax2.set_xlabel('Algorithm Type', fontname='Arial', fontsize=11)
        ax2.set_ylabel('Silhouette Score Distribution', fontname='Arial', fontsize=11)
        ax2.set_title('Silhouette Score Distribution by Algorithm Type',
                      fontname='Arial', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Number of clusters vs silhouette score
        ax3 = axes[1, 0]

        # Group data for scatter plot
        scatter_data = {}
        for algo in algorithms:
            algo_data = df_algorithms[df_algorithms['algorithm'] == algo]
            if len(algo_data) > 0:
                scatter_data[algo] = {
                    'clusters': algo_data['n_clusters'].values,
                    'silhouette': algo_data['silhouette'].values,
                    'color': color_map[algo]
                }

        # Plot scatter for each algorithm
        for algo, data in scatter_data.items():
            scatter = ax3.scatter(data['clusters'], data['silhouette'],
                                  c=data['color'], s=60, alpha=0.7,
                                  edgecolor='white', linewidth=0.5,
                                  label=algo)

        ax3.set_xlabel('Number of Clusters', fontname='Arial', fontsize=11)
        ax3.set_ylabel('Silhouette Score', fontname='Arial', fontsize=11)
        ax3.set_title('Number of Clusters vs Silhouette Score',
                      fontname='Arial', fontweight='bold', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=9, framealpha=0.7)

        # Add trend line
        if len(df_algorithms) > 1:
            z = np.polyfit(df_algorithms['n_clusters'], df_algorithms['silhouette'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(df_algorithms['n_clusters'].min(),
                                  df_algorithms['n_clusters'].max(), 100)
            ax3.plot(x_range, p(x_range), color=JAMA_COLORS['gray'],
                     linestyle='--', alpha=0.7, linewidth=1.5,
                     label=f'Trend (slope={z[0]:.4f})')

        # 4. Performance summary table for top algorithms
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Create summary table for top 10 algorithms
        top_10 = df_algorithms.head(10)
        if len(top_10) > 0:
            table_data = []
            for _, row in top_10.iterrows():
                table_data.append([
                    row['rank'],
                    row['algorithm'][:15],  # Truncate if too long
                    f"{row['silhouette']:.4f}",
                    row['n_clusters'],
                    f"{row['noise_ratio']:.2%}",
                    f"{row['davies_bouldin']:.3f}"
                ])

            # Create table
            table = ax4.table(cellText=table_data,
                              colLabels=['Rank', 'Algorithm', 'Silhouette',
                                         'Clusters', 'Noise %', 'DBI'],
                              cellLoc='center',
                              loc='center',
                              colWidths=[0.1, 0.2, 0.15, 0.1, 0.15, 0.1])

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            # Style table with JAMA colors
            for (row, col), cell in table.get_celld().items():
                if row == 0:  # Header row
                    cell.set_text_props(fontweight='bold', fontname='Arial')
                    cell.set_facecolor(JAMA_COLORS['light_blue'])
                else:
                    # Color code silhouette scores
                    silhouette_score = float(table_data[row - 1][2])
                    if silhouette_score >= 0.7:
                        cell.set_facecolor(JAMA_COLORS['green'])
                        cell.set_alpha(0.3)
                    elif silhouette_score >= 0.5:
                        cell.set_facecolor(JAMA_COLORS['orange'])
                        cell.set_alpha(0.3)
                    else:
                        cell.set_facecolor(JAMA_COLORS['red'])
                        cell.set_alpha(0.2)

            ax4.set_title('Top 10 Clustering Algorithms Performance Summary',
                          fontname='Arial', fontweight='bold', fontsize=12, pad=20)

        plt.tight_layout()

        # Save figure
        if save_as_pdf:
            output_path = os.path.join(output_dir, "silhouette_rank_lineplot.pdf")
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none',
                        format='pdf', transparent=False)
            self._log(f"Silhouette rank line plot saved to: {output_path}")
        else:
            output_path = os.path.join(output_dir, "silhouette_rank_lineplot.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self._log(f"Silhouette rank line plot saved to: {output_path}")

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_silhouette_progression_lineplot(self,
                                             output_dir: str = ".",
                                             save_as_pdf: bool = True,
                                             show_plot: bool = False) -> plt.Figure:
        """
        Plot Silhouette Score progression line plot focusing on parameter variations

        Parameters:
        -----------
        output_dir : str
            Output directory
        save_as_pdf : bool
            Whether to save as PDF
        show_plot : bool
            Whether to display the plot

        Returns:
        --------
        plt.Figure
            The created figure
        """
        if not hasattr(self, 'results') or not self.results:
            self._log("No results available for plotting")
            return None

        # Prepare data grouped by algorithm and parameter
        import pandas as pd
        algorithm_data = []

        for key, result in self.results.items():
            algorithm = result.get('algorithm', 'Unknown')
            params = result.get('params', {})

            # Create parameter identifier
            param_id = ""
            if algorithm == 'DBSCAN':
                param_id = f"eps={params.get('eps', 0):.3f}"
            elif algorithm == 'HDBSCAN':
                param_id = f"min_cluster_size={params.get('min_cluster_size', 0)}"
            elif algorithm == 'KMeans':
                param_id = f"k={params.get('n_clusters', 0)}"
            elif algorithm == 'Spectral':
                param_id = f"k={params.get('n_clusters', 0)}"
            elif algorithm == 'Hierarchical':
                param_id = f"{params.get('method', '')}_k={params.get('n_clusters', 0)}"

            algorithm_data.append({
                'algorithm': algorithm,
                'param_id': param_id,
                'silhouette': result.get('silhouette', 0),
                'n_clusters': result.get('n_clusters', 0),
                'noise_ratio': result.get('noise_ratio', 0),
                'params': params,
                'full_name': key
            })

        if not algorithm_data:
            return None

        df = pd.DataFrame(algorithm_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Group by algorithm
        algorithms = df['algorithm'].unique()
        color_map = {}
        for i, algo in enumerate(algorithms):
            color_map[algo] = list(JAMA_COLORS.values())[i % len(JAMA_COLORS)]

        # Plot each algorithm's parameter progression
        for algo in algorithms:
            algo_data = df[df['algorithm'] == algo].sort_values('silhouette', ascending=False)

            if len(algo_data) > 1:
                # Plot line connecting parameter variations
                ax.plot(range(len(algo_data)), algo_data['silhouette'].values,
                        'o-', linewidth=2, markersize=8,
                        color=color_map[algo], label=algo, alpha=0.8)

                # Annotate key points
                best_idx = algo_data['silhouette'].idxmax()
                best_row = algo_data.loc[best_idx]

                ax.annotate(f"Best: {best_row['param_id']}\n{best_row['silhouette']:.4f}",
                            xy=(algo_data.index.get_loc(best_idx), best_row['silhouette']),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=9, fontname='Arial',
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor=color_map[algo],
                                      alpha=0.2))

        # Customize plot
        ax.set_xlabel('Parameter Configuration Index', fontname='Arial', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontname='Arial', fontsize=12)
        ax.set_title('Silhouette Score Progression by Algorithm Parameter Configuration',
                     fontname='Arial', fontweight='bold', fontsize=14)

        # Add grid and reference lines
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0.5, color=JAMA_COLORS['red'], linestyle='--', alpha=0.5, linewidth=1)
        ax.text(0.02, 0.52, 'Good clustering threshold (0.5)',
                transform=ax.transAxes, fontsize=10, color=JAMA_COLORS['red'],
                fontname='Arial')

        # Add legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.8)

        # Add statistics box
        stats_text = []
        stats_text.append(f"Total configurations: {len(df)}")
        stats_text.append(f"Algorithms tested: {len(algorithms)}")
        stats_text.append(f"Best silhouette: {df['silhouette'].max():.4f}")
        stats_text.append(f"Mean silhouette: {df['silhouette'].mean():.4f}")

        stats_str = "\n".join(stats_text)
        ax.text(0.02, 0.98, stats_str, transform=ax.transAxes,
                fontsize=10, fontname='Arial', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor=JAMA_COLORS['light_gray'],
                          alpha=0.7))

        plt.tight_layout()

        # Save figure
        if save_as_pdf:
            output_path = os.path.join(output_dir, "silhouette_progression_lineplot.pdf")
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none',
                        format='pdf', transparent=False)
            self._log(f"Silhouette progression line plot saved to: {output_path}")
        else:
            output_path = os.path.join(output_dir, "silhouette_progression_lineplot.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self._log(f"Silhouette progression line plot saved to: {output_path}")

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def create_algorithm_comparison_chart(self, output_dir: str, save_as_pdf: bool = True):
        """
        Create algorithm comparison summary chart

        Parameters:
        -----------
        output_dir : str
            Output directory
        save_as_pdf : bool, default=True
            Whether to save as PDF
        """
        if not hasattr(self, 'results') or not self.results:
            return

        # Extract evaluation metrics for all algorithms
        algorithm_data = []

        for key, result in self.results.items():
            algorithm = result.get('algorithm', 'Unknown')
            params = result.get('params', {})

            # Extract main parameters
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])

            algorithm_data.append({
                'algorithm': algorithm,
                'parameters': param_str,
                'silhouette': result.get('silhouette', 0),
                'calinski_harabasz': result.get('calinski_harabasz', 0),
                'davies_bouldin': result.get('davies_bouldin', 0),
                'n_clusters': result.get('n_clusters', 0),
                'noise_ratio': result.get('noise_ratio', 0),
                'key': key
            })

        # Create DataFrame
        df_algorithms = pd.DataFrame(algorithm_data)

        if len(df_algorithms) == 0:
            return

        # Sort by silhouette score
        df_algorithms = df_algorithms.sort_values('silhouette', ascending=False)

        # Create comparison figure with JAMA style
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Clustering Algorithm Performance Comparison',
                     fontsize=16, fontweight='bold', fontname='Arial')

        # 1. Silhouette score comparison
        ax1 = axes[0, 0]
        algorithms_short = [f"{row['algorithm'][:15]}... ({i + 1})"
                            for i, (_, row) in enumerate(df_algorithms.iterrows())]

        # Use JAMA color gradient
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(algorithms_short)))

        bars1 = ax1.barh(range(len(algorithms_short)),
                         df_algorithms['silhouette'],
                         color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_yticks(range(len(algorithms_short)))
        ax1.set_yticklabels(algorithms_short, fontname='Arial')
        ax1.set_xlabel('Silhouette Score', fontname='Arial')
        ax1.set_title('Algorithm Ranking by Silhouette Score',
                      fontname='Arial', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Add score labels on bars
        for i, (bar, score) in enumerate(zip(bars1, df_algorithms['silhouette'])):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height() / 2.,
                     f'{score:.4f}', ha='left', va='center',
                     fontsize=9, fontname='Arial', fontweight='bold')

        # 2. Cluster count vs silhouette score
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df_algorithms['n_clusters'],
                              df_algorithms['silhouette'],
                              s=df_algorithms['calinski_harabasz'] / 10,  # Point size represents CH index
                              c=df_algorithms['davies_bouldin'],  # Color represents DBI
                              cmap='RdYlGn_r', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Number of Clusters', fontname='Arial')
        ax2.set_ylabel('Silhouette Score', fontname='Arial')
        ax2.set_title('Clusters vs Silhouette Score',
                      fontname='Arial', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add color bar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Davies-Bouldin Index (lower is better)', fontname='Arial')

        # 3. Noise ratio analysis
        ax3 = axes[1, 0]
        noise_data = df_algorithms[['algorithm', 'noise_ratio']].copy()
        noise_data = noise_data.groupby('algorithm')['noise_ratio'].mean().reset_index()

        if len(noise_data) > 0:
            # Use JAMA colors
            colors = [JAMA_COLORS['red'] if ratio > 0.3 else
                      JAMA_COLORS['orange'] if ratio > 0.1 else
                      JAMA_COLORS['green']
                      for ratio in noise_data['noise_ratio']]

            bars3 = ax3.bar(noise_data['algorithm'],
                            noise_data['noise_ratio'] * 100,
                            color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax3.set_xlabel('Algorithm', fontname='Arial')
            ax3.set_ylabel('Noise Ratio (%)', fontname='Arial')
            ax3.set_title('Noise Ratio by Algorithm',
                          fontname='Arial', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')

        # 4. Comprehensive evaluation radar chart (top 5 algorithms)
        ax4 = axes[1, 1]
        top_5 = df_algorithms.head(5)

        if len(top_5) > 1:
            metrics = ['Silhouette', 'CH Index', 'DBI Index', 'Noise Ratio']

            # Normalize metrics
            norm_scores = []
            algorithm_names = []
            algo_colors = list(JAMA_COLORS.values())[:len(top_5)]

            for idx, (_, row) in enumerate(top_5.iterrows()):
                # Silhouette: -1 to 1, normalize to 0-1
                sil_norm = (row['silhouette'] + 1) / 2

                # CH index: normalize to 0-1 (assuming max 5000)
                ch_norm = min(row['calinski_harabasz'] / 5000, 1)

                # DBI index: lower is better, normalize
                dbi_norm = 1 - min(row['davies_bouldin'] / 2, 1)  # Assuming max 2

                # Noise ratio: lower is better
                noise_norm = 1 - row['noise_ratio']

                norm_scores.append([sil_norm, ch_norm, dbi_norm, noise_norm])
                algorithm_names.append(f"{row['algorithm']} #{idx + 1}")

            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]

            ax4 = plt.subplot(2, 2, 4, polar=True)

            for i, scores in enumerate(norm_scores):
                scores += scores[:1]
                ax4.plot(angles, scores, 'o-', linewidth=2, color=algo_colors[i],
                         label=algorithm_names[i], alpha=0.7)
                ax4.fill(angles, scores, color=algo_colors[i], alpha=0.1)

            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics, fontname='Arial')
            ax4.set_ylim(0, 1)
            ax4.set_title('Top 5 Algorithms Comprehensive Evaluation',
                          fontname='Arial', fontweight='bold', pad=20)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), prop={'family': 'Arial'})
            ax4.grid(True, alpha=0.3)
            ax4.set_facecolor('white')

        plt.tight_layout()

        # Save chart
        if save_as_pdf:
            comparison_path = os.path.join(output_dir, "algorithm_comparison_summary.pdf")
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none',
                        format='pdf', transparent=False)
        else:
            comparison_path = os.path.join(output_dir, "algorithm_comparison_summary.png")
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')

        plt.close(fig)

        print(f"Algorithm comparison summary chart saved to: {comparison_path}")

        # Save detailed algorithm comparison data to CSV
        csv_path = os.path.join(output_dir, "algorithm_comparison_details.csv")
        df_algorithms.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Detailed algorithm comparison data saved to: {csv_path}")

        return df_algorithms

    def create_silhouette_rank_analysis(self, output_dir: str, save_as_pdf: bool = True):
        """
        Create silhouette rank analysis charts

        Parameters:
        -----------
        output_dir : str
            Output directory
        save_as_pdf : bool, default=True
            Whether to save as PDF
        """
        if not hasattr(self, 'results') or not self.results:
            self._log("No results available for silhouette rank analysis")
            return

        # Create both silhouette rank plots
        fig1 = self.plot_silhouette_rank_lineplot(output_dir, save_as_pdf, show_plot=False)
        fig2 = self.plot_silhouette_progression_lineplot(output_dir, save_as_pdf, show_plot=False)

        # Create a summary combined figure
        self._log("Creating combined silhouette analysis report...")
        self._create_silhouette_summary_report(output_dir, save_as_pdf)

        return fig1, fig2

    def _create_silhouette_summary_report(self, output_dir: str, save_as_pdf: bool = True):
        """
        Create comprehensive silhouette analysis summary report

        Parameters:
        -----------
        output_dir : str
            Output directory
        save_as_pdf : bool
            Whether to save as PDF
        """
        if not hasattr(self, 'results') or not self.results:
            return

        import pandas as pd
        import numpy as np

        # Prepare data
        algorithm_data = []
        for key, result in self.results.items():
            algorithm = result.get('algorithm', 'Unknown')
            algorithm_data.append({
                'algorithm': algorithm,
                'silhouette': result.get('silhouette', 0),
                'n_clusters': result.get('n_clusters', 0),
                'noise_ratio': result.get('noise_ratio', 0),
                'config_name': key
            })

        df = pd.DataFrame(algorithm_data)

        # Create summary figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Silhouette Score Comprehensive Analysis',
                     fontsize=16, fontweight='bold', fontname='Arial')

        # Left: Rank vs Silhouette (simplified)
        ax1 = axes[0]
        df_sorted = df.sort_values('silhouette', ascending=False).reset_index(drop=True)
        df_sorted['rank'] = df_sorted.index + 1

        algorithms = df_sorted['algorithm'].unique()
        color_map = {}
        for i, algo in enumerate(algorithms):
            color_map[algo] = list(JAMA_COLORS.values())[i % len(JAMA_COLORS)]

        for algo in algorithms:
            algo_data = df_sorted[df_sorted['algorithm'] == algo]
            if len(algo_data) > 0:
                ax1.plot(algo_data['rank'], algo_data['silhouette'],
                         'o-', linewidth=1.5, markersize=5,
                         color=color_map[algo], label=algo, alpha=0.7)

        ax1.set_xlabel('Rank (sorted by Silhouette Score)', fontname='Arial', fontsize=11)
        ax1.set_ylabel('Silhouette Score', fontname='Arial', fontsize=11)
        ax1.set_title('Silhouette Score Ranking Across All Methods',
                      fontname='Arial', fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.2, linestyle='--')
        ax1.legend(loc='lower left', fontsize=9)

        # Add reference lines
        ax1.axhline(y=0.7, color=JAMA_COLORS['green'], linestyle='--', alpha=0.5, linewidth=1)
        ax1.axhline(y=0.5, color=JAMA_COLORS['orange'], linestyle='--', alpha=0.5, linewidth=1)
        ax1.axhline(y=0.25, color=JAMA_COLORS['red'], linestyle='--', alpha=0.5, linewidth=1)

        # Right: Algorithm performance summary
        ax2 = axes[1]
        ax2.axis('off')

        # Calculate algorithm statistics
        algo_stats = df.groupby('algorithm').agg({
            'silhouette': ['mean', 'max', 'std', 'count'],
            'n_clusters': ['mean', 'std']
        }).round(3)

        # Create table
        table_data = []
        for algo in algorithms:
            stats = algo_stats.loc[algo]
            table_data.append([
                algo,
                f"{stats[('silhouette', 'max')]:.4f}",
                f"{stats[('silhouette', 'mean')]:.4f}",
                f"{stats[('silhouette', 'std')]:.4f}",
                int(stats[('silhouette', 'count')]),
                f"{stats[('n_clusters', 'mean')]:.1f}"
            ])

        # Sort by max silhouette
        table_data.sort(key=lambda x: float(x[1]), reverse=True)

        # Add header
        table_data.insert(0, ['Algorithm', 'Best', 'Mean', 'Std', 'N', 'Avg Clusters'])

        # Create table
        table = ax2.table(cellText=table_data,
                          cellLoc='center',
                          loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)

        # Style table
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # Header
                cell.set_text_props(fontweight='bold', fontname='Arial')
                cell.set_facecolor(JAMA_COLORS['light_blue'])
            elif col == 1:  # Best silhouette column
                score = float(table_data[row][1])
                if score >= 0.7:
                    cell.set_facecolor(JAMA_COLORS['green'])
                elif score >= 0.5:
                    cell.set_facecolor(JAMA_COLORS['orange'])
                else:
                    cell.set_facecolor(JAMA_COLORS['red'])
                cell.set_alpha(0.3)

        ax2.set_title('Algorithm Performance Statistics',
                      fontname='Arial', fontweight='bold', fontsize=12, pad=20)

        plt.tight_layout()

        # Save figure
        if save_as_pdf:
            output_path = os.path.join(output_dir, "silhouette_summary_report.pdf")
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none',
                        format='pdf', transparent=False)
        else:
            output_path = os.path.join(output_dir, "silhouette_summary_report.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
        self._log(f"Silhouette summary report saved to: {output_path}")


    def save_all_plots_as_pdf(self, output_dir: str, filename: str = "clustering_analysis_report.pdf"):
        """
        Save all clustering plots as a single PDF report

        Parameters:
        -----------
        output_dir : str
            Output directory
        filename : str, default="clustering_analysis_report.pdf"
            PDF filename
        """
        if not hasattr(self, 'results') or not self.results:
            self._log("No results available to save as PDF")
            return

        pdf_path = os.path.join(output_dir, filename)

        # Create PDF document
        with PdfPages(pdf_path) as pdf:
            self._log(f"Creating PDF report: {pdf_path}")

            # 1. Cover page
            fig_cover = plt.figure(figsize=(11, 8.5))
            plt.text(0.5, 0.7, 'Clustering Analysis Report',
                     fontsize=24, fontweight='bold', fontname='Arial',
                     ha='center', va='center', transform=fig_cover.transFigure)
            plt.text(0.5, 0.6, 'High-Dimensional Data Clustering Evaluation',
                     fontsize=16, fontname='Arial',
                     ha='center', va='center', transform=fig_cover.transFigure)

            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            plt.text(0.5, 0.4, f'Generated: {timestamp}',
                     fontsize=12, fontname='Arial',
                     ha='center', va='center', transform=fig_cover.transFigure)

            # Add algorithm summary
            if self.best_result:
                best_algo = self.best_result['algorithm']
                best_silhouette = self.best_result['silhouette']
                best_clusters = self.best_result['n_clusters']

                summary_text = (f"Best Algorithm: {best_algo}\n"
                                f"Silhouette Score: {best_silhouette:.4f}\n"
                                f"Number of Clusters: {best_clusters}")

                plt.text(0.5, 0.3, summary_text,
                         fontsize=14, fontname='Arial',
                         ha='center', va='center', transform=fig_cover.transFigure,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor=JAMA_COLORS['light_gray'], alpha=0.7))

            plt.axis('off')
            pdf.savefig(fig_cover, bbox_inches='tight')
            plt.close(fig_cover)

            # 2. Best clustering results
            if self.best_labels is not None and self.reduced_embeddings is not None:
                self._log("Adding best clustering results to PDF...")
                fig_best = self.plot_clustering_results(
                    reduced_embeddings=self.reduced_embeddings,
                    labels=self.best_labels,
                    algorithm_name=f"Best Algorithm: {self.best_result['algorithm'] if self.best_result else 'Unknown'}",
                    show_plots=False
                )
                pdf.savefig(fig_best, bbox_inches='tight')
                plt.close(fig_best)

            # 3. Algorithm comparison chart
            self._log("Adding algorithm comparison chart to PDF...")
            df_algorithms = self.create_algorithm_comparison_chart(output_dir, save_as_pdf=False)

            # Recreate the figure for PDF
            fig_comparison = plt.figure(figsize=(15, 12))
            self._create_pdf_algorithm_comparison(fig_comparison, df_algorithms)
            pdf.savefig(fig_comparison, bbox_inches='tight')
            plt.close(fig_comparison)

            # 4. Individual algorithm results (top 5)
            self._log("Adding individual algorithm results to PDF...")
            top_results = self.get_best_results(metric='silhouette', top_k=5)

            for i, (key, result) in enumerate(top_results, 1):
                if i > 1:  # Skip first (already in best results)
                    algorithm_name = result['algorithm']
                    algo_labels = result['labels']

                    fig_algo = self.plot_clustering_results(
                        reduced_embeddings=self.reduced_embeddings,
                        labels=algo_labels,
                        algorithm_name=f"{algorithm_name} (Rank {i})",
                        show_plots=False
                    )

                    # Add algorithm metadata
                    plt.figtext(0.02, 0.02,
                                f"Silhouette: {result['silhouette']:.4f} | "
                                f"Clusters: {result['n_clusters']} | "
                                f"Noise Ratio: {result['noise_ratio']:.2%}",
                                fontsize=10, fontname='Arial',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor=JAMA_COLORS['light_gray']))

                    pdf.savefig(fig_algo, bbox_inches='tight')
                    plt.close(fig_algo)

            # 5. Summary statistics
            self._log("Adding summary statistics to PDF...")
            fig_summary = self._create_summary_page()
            pdf.savefig(fig_summary, bbox_inches='tight')
            plt.close(fig_summary)

            # 6. Method details
            self._log("Adding method details to PDF...")
            fig_methods = self._create_methods_page()
            pdf.savefig(fig_methods, bbox_inches='tight')
            plt.close(fig_methods)

            # 7. Silhouette Rank Line Plot
            self._log("Adding silhouette rank line plot to PDF...")
            fig_silhouette_rank = self.plot_silhouette_rank_lineplot(
                output_dir=output_dir,
                save_as_pdf=False,
                show_plot=False
            )
            if fig_silhouette_rank:
                pdf.savefig(fig_silhouette_rank, bbox_inches='tight')
                plt.close(fig_silhouette_rank)

            # 8. Silhouette Progression Line Plot
            self._log("Adding silhouette progression line plot to PDF...")
            fig_silhouette_prog = self.plot_silhouette_progression_lineplot(
                output_dir=output_dir,
                save_as_pdf=False,
                show_plot=False
            )
            if fig_silhouette_prog:
                pdf.savefig(fig_silhouette_prog, bbox_inches='tight')
                plt.close(fig_silhouette_prog)

        self._log(f"Complete PDF report saved to: {pdf_path}")
        return pdf_path

    def _create_pdf_algorithm_comparison(self, fig, df_algorithms):
        """
        Create algorithm comparison chart for PDF

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure object
        df_algorithms : pd.DataFrame
            Algorithm comparison data
        """
        axes = fig.subplots(2, 2)
        fig.suptitle('Clustering Algorithm Performance Comparison',
                     fontsize=16, fontweight='bold', fontname='Arial')

        # 1. Silhouette score comparison
        ax1 = axes[0, 0]
        algorithms_short = [f"{row['algorithm'][:15]}... ({i + 1})"
                            for i, (_, row) in enumerate(df_algorithms.iterrows())]

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(algorithms_short)))

        bars1 = ax1.barh(range(len(algorithms_short)),
                         df_algorithms['silhouette'],
                         color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_yticks(range(len(algorithms_short)))
        ax1.set_yticklabels(algorithms_short, fontname='Arial')
        ax1.set_xlabel('Silhouette Score', fontname='Arial')
        ax1.set_title('Algorithm Ranking by Silhouette Score',
                      fontname='Arial', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        for i, (bar, score) in enumerate(zip(bars1, df_algorithms['silhouette'])):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height() / 2.,
                     f'{score:.4f}', ha='left', va='center',
                     fontsize=9, fontname='Arial', fontweight='bold')

        # 2. Cluster count vs silhouette score
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df_algorithms['n_clusters'],
                              df_algorithms['silhouette'],
                              s=df_algorithms['calinski_harabasz'] / 10,
                              c=df_algorithms['davies_bouldin'],
                              cmap='RdYlGn_r', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Number of Clusters', fontname='Arial')
        ax2.set_ylabel('Silhouette Score', fontname='Arial')
        ax2.set_title('Clusters vs Silhouette Score',
                      fontname='Arial', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Davies-Bouldin Index (lower is better)', fontname='Arial')

        # 3. Noise ratio analysis
        ax3 = axes[1, 0]
        noise_data = df_algorithms[['algorithm', 'noise_ratio']].copy()
        noise_data = noise_data.groupby('algorithm')['noise_ratio'].mean().reset_index()

        if len(noise_data) > 0:
            colors = [JAMA_COLORS['red'] if ratio > 0.3 else
                      JAMA_COLORS['orange'] if ratio > 0.1 else
                      JAMA_COLORS['green']
                      for ratio in noise_data['noise_ratio']]

            bars3 = ax3.bar(noise_data['algorithm'],
                            noise_data['noise_ratio'] * 100,
                            color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax3.set_xlabel('Algorithm', fontname='Arial')
            ax3.set_ylabel('Noise Ratio (%)', fontname='Arial')
            ax3.set_title('Noise Ratio by Algorithm',
                          fontname='Arial', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')

        # 4. Top algorithms table
        ax4 = axes[1, 1]
        ax4.axis('off')

        top_5 = df_algorithms.head(5)
        if len(top_5) > 0:
            table_data = []
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                table_data.append([
                    i,
                    row['algorithm'],
                    f"{row['silhouette']:.4f}",
                    row['n_clusters'],
                    f"{row['noise_ratio']:.2%}",
                    f"{row['davies_bouldin']:.3f}"
                ])

            # Create table
            table = ax4.table(cellText=table_data,
                              colLabels=['Rank', 'Algorithm', 'Silhouette', 'Clusters', 'Noise %', 'DBI'],
                              cellLoc='center',
                              loc='center')

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

            # Style table
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(fontweight='bold', fontname='Arial')
                    cell.set_facecolor(JAMA_COLORS['light_blue'])
                elif row % 2 == 0:
                    cell.set_facecolor(JAMA_COLORS['light_gray'])

            ax4.set_title('Top 5 Clustering Algorithms',
                          fontname='Arial', fontweight='bold', pad=20)

        plt.tight_layout()

    def _create_summary_page(self):
        """Create summary statistics page for PDF"""
        fig = plt.figure(figsize=(11, 8.5))

        # Title
        plt.text(0.5, 0.95, 'Clustering Analysis Summary',
                 fontsize=20, fontweight='bold', fontname='Arial',
                 ha='center', va='center', transform=fig.transFigure)

        # Overall statistics
        if self.best_result:
            stats_text = []
            stats_text.append(f"Best Algorithm: {self.best_result['algorithm']}")
            stats_text.append(f"Silhouette Score: {self.best_result['silhouette']:.4f}")
            stats_text.append(f"Number of Clusters: {self.best_result['n_clusters']}")
            stats_text.append(f"Noise Ratio: {self.best_result['noise_ratio']:.2%}")
            stats_text.append(f"Calinski-Harabasz Index: {self.best_result['calinski_harabasz']:.2f}")
            stats_text.append(f"Davies-Bouldin Index: {self.best_result['davies_bouldin']:.4f}")

            if 'cluster_stats' in self.best_result:
                stats = self.best_result['cluster_stats']
                stats_text.append(f"\nCluster Statistics:")
                stats_text.append(f"  Min Cluster Size: {stats['min_size']}")
                stats_text.append(f"  Max Cluster Size: {stats['max_size']}")
                stats_text.append(f"  Mean Cluster Size: {stats['mean_size']:.1f}")
                stats_text.append(f"  Std Cluster Size: {stats['std_size']:.1f}")

            stats_str = "\n".join(stats_text)
            plt.text(0.1, 0.7, stats_str,
                     fontsize=12, fontname='Arial',
                     va='top', transform=fig.transFigure,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor=JAMA_COLORS['light_gray'], alpha=0.7))

        # Algorithm count
        if self.results:
            algo_counts = {}
            for result in self.results.values():
                algo = result.get('algorithm', 'Unknown')
                algo_counts[algo] = algo_counts.get(algo, 0) + 1

            algo_text = ["Algorithms Tested:"]
            for algo, count in sorted(algo_counts.items()):
                algo_text.append(f"  {algo}: {count} configurations")

            algo_str = "\n".join(algo_text)
            plt.text(0.6, 0.7, algo_str,
                     fontsize=12, fontname='Arial',
                     va='top', transform=fig.transFigure,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor=JAMA_COLORS['light_blue'], alpha=0.7))

        # Data information
        if self.reduced_embeddings is not None:
            data_text = [
                "Data Information:",
                f"Original Dimensions: {getattr(self, 'original_dim', 'Unknown')}",
                f"Reduced Dimensions: {self.reduced_embeddings.shape[1]}",
                f"Number of Samples: {self.reduced_embeddings.shape[0]}"
            ]

            data_str = "\n".join(data_text)
            plt.text(0.1, 0.4, data_str,
                     fontsize=12, fontname='Arial',
                     va='top', transform=fig.transFigure,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor=JAMA_COLORS['light_orange'], alpha=0.7))

        # Evaluation metrics explanation
        metrics_text = [
            "Evaluation Metrics:",
            "• Silhouette Score: Measures how similar an object is to its own cluster",
            "  compared to other clusters. Range: -1 to 1 (higher is better).",
            "• Calinski-Harabasz Index: Ratio of between-cluster dispersion",
            "  to within-cluster dispersion (higher is better).",
            "• Davies-Bouldin Index: Average similarity between each cluster",
            "  and its most similar cluster (lower is better).",
            "• Noise Ratio: Percentage of points not assigned to any cluster."
        ]

        metrics_str = "\n".join(metrics_text)
        plt.text(0.1, 0.2, metrics_str,
                 fontsize=10, fontname='Arial',
                 va='top', transform=fig.transFigure)

        plt.axis('off')
        return fig

    def _create_methods_page(self):
        """Create methods description page for PDF"""
        fig = plt.figure(figsize=(11, 8.5))

        # Title
        plt.text(0.5, 0.95, 'Clustering Methods Description',
                 fontsize=20, fontweight='bold', fontname='Arial',
                 ha='center', va='center', transform=fig.transFigure)

        # Methods description
        methods_text = [
            "Clustering Algorithms Used:",
            "",
            "1. KMeans:",
            "   • Partition-based clustering algorithm",
            "   • Requires specifying number of clusters (k)",
            "   • Minimizes within-cluster variance",
            "   • Uses k-means++ initialization for better convergence",
            "",
            "2. DBSCAN:",
            "   • Density-based spatial clustering",
            "   • Discovers clusters of arbitrary shape",
            "   • Identifies noise points",
            "   • Parameters: eps (neighborhood radius), min_samples",
            "",
            "3. HDBSCAN:",
            "   • Hierarchical density-based clustering",
            "   • Extension of DBSCAN with variable density clusters",
            "   • Automatically determines number of clusters",
            "   • More robust to parameter settings",
            "",
            "4. Spectral Clustering:",
            "   • Uses eigenvalues of similarity matrix",
            "   • Effective for non-convex clusters",
            "   • Transforms data to new space then applies KMeans",
            "   • Uses nearest neighbors affinity",
            "",
            "5. Hierarchical Clustering:",
            "   • Builds hierarchy of clusters",
            "   • Agglomerative approach (bottom-up)",
            "   • Linkage methods: ward, complete, average",
            "   • Uses cosine or Euclidean distance"
        ]

        methods_str = "\n".join(methods_text)
        plt.text(0.1, 0.85, methods_str,
                 fontsize=10, fontname='Arial',
                 va='top', transform=fig.transFigure)

        # Evaluation methods
        eval_text = [
            "",
            "Dimensionality Reduction Methods:",
            "• PCA: Principal Component Analysis for linear reduction",
            "• Two-stage PCA: PCA to intermediate dimension, then to target",
            "• PCA + t-SNE: PCA preprocessing followed by t-SNE",
            "• UMAP: Uniform Manifold Approximation and Projection",
            "• High Variance PCA: Keeps components explaining 95% variance"
        ]

        eval_str = "\n".join(eval_text)
        plt.text(0.1, 0.4, eval_str,
                 fontsize=10, fontname='Arial',
                 va='top', transform=fig.transFigure)

        # Parameter optimization
        param_text = [
            "",
            "Parameter Optimization:",
            "• Automatic K estimation using silhouette, elbow methods",
            "• DBSCAN eps estimation using k-nearest neighbors",
            "• t-SNE parameters optimized based on data size",
            "• Multiple parameter combinations tested for each algorithm"
        ]

        param_str = "\n".join(param_text)
        plt.text(0.1, 0.2, param_str,
                 fontsize=10, fontname='Arial',
                 va='top', transform=fig.transFigure,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=JAMA_COLORS['light_gray'], alpha=0.7))

        plt.axis('off')
        return fig


# Helper functions
def analyze_cluster_composition(labels: np.ndarray,
                                metadata: pd.DataFrame,
                                cluster_id: Optional[int] = None) -> pd.DataFrame:
    """
    Analyze cluster composition

    Parameters:
    -----------
    labels : np.ndarray
        Cluster labels
    metadata : pd.DataFrame
        Metadata
    cluster_id : int, optional
        Specific cluster ID to analyze, if None analyze all clusters

    Returns:
    --------
    pd.DataFrame
        Cluster composition analysis results
    """
    import pandas as pd

    # Add labels to metadata
    data_with_labels = metadata.copy()
    data_with_labels['cluster'] = labels

    if cluster_id is not None:
        # Analyze specific cluster
        cluster_data = data_with_labels[data_with_labels['cluster'] == cluster_id]
    else:
        # Analyze all clusters
        cluster_data = data_with_labels

    # For numerical columns, calculate statistics
    numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'cluster']

    results = {}

    if cluster_id is not None:
        # Single cluster analysis
        for col in numeric_cols:
            results[col] = {
                'mean': cluster_data[col].mean(),
                'std': cluster_data[col].std(),
                'min': cluster_data[col].min(),
                'max': cluster_data[col].max(),
                'median': cluster_data[col].median()
            }
    else:
        # All clusters analysis
        for col in numeric_cols:
            cluster_stats = cluster_data.groupby('cluster')[col].agg(['mean', 'std', 'count']).round(3)
            results[col] = cluster_stats

    return pd.DataFrame(results)


# Quick use function
def quick_clustering(embeddings: np.ndarray,
                     target_dim: int = 200,
                     algorithm: str = 'auto',
                     **kwargs) -> Tuple[np.ndarray, Dict]:
    """
    Quick clustering function

    Parameters:
    -----------
    embeddings : np.ndarray
        Embedding vectors
    target_dim : int, default=200
        Target dimension
    algorithm : str, default='auto'
        Clustering algorithm ('auto', 'dbscan', 'kmeans', 'hdbscan')
    **kwargs : dict
        Other parameters

    Returns:
    --------
    Tuple[np.ndarray, Dict]
        Cluster labels and result information
    """
    evaluator = RobustClusteringEvaluator(verbose=False)

    # Normalization
    embeddings_norm = evaluator.normalize_embeddings(embeddings)

    # Dimensionality reduction
    reduced_embeddings = evaluator.reduce_dimension(embeddings_norm, target_dim)

    # Run specified algorithm
    if algorithm == 'auto' or algorithm == 'dbscan':
        results = evaluator.run_dbscan(reduced_embeddings, **kwargs)
    elif algorithm == 'kmeans':
        results = evaluator.run_kmeans(reduced_embeddings, **kwargs)
    elif algorithm == 'hdbscan' and HDBSCAN_AVAILABLE:
        results = evaluator.run_hdbscan(reduced_embeddings, **kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Get best result
    best_results = evaluator.get_best_results(top_k=1)

    if best_results:
        best_key, best_result = best_results[0]
        return best_result['labels'], best_result
    else:
        raise RuntimeError("Clustering failed, no valid results found")


class ClusteringPipeline:
    """
    Complete clustering analysis pipeline
    """

    def __init__(self, random_state: int = 42, output_dir: str = "./clustering_results"):
        """
        Initialize clustering pipeline

        Parameters:
        -----------
        random_state : int
            Random seed
        output_dir : str
            Results save directory
        """
        self.random_state = random_state
        self.output_dir = output_dir
        self.evaluator = None
        self.results = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data file

        Parameters:
        -----------
        file_path : str
            Data file path
        **kwargs : dict
            Additional parameters for pandas.read_csv()

        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        print(f"Loading data: {file_path}")

        # Select loading method based on file extension
        if file_path.endswith('.csv'):
            # Copy kwargs to avoid modifying caller's dictionary
            read_kwargs = kwargs.copy()

            # Merge dtype: prioritize user-provided dtype settings
            user_dtype = read_kwargs.get('dtype', {}) or {}
            if not isinstance(user_dtype, dict):
                # If user provided non-dict (e.g., numpy dtype), ignore and use empty dict
                user_dtype = {}

            # Ensure phID is read as string
            if 'phID' not in user_dtype:
                user_dtype['phID'] = str

            read_kwargs['dtype'] = user_dtype

            # Optional: disable low_memory mode to avoid chunked type inference
            if 'low_memory' not in read_kwargs:
                read_kwargs['low_memory'] = False

            df = pd.read_csv(file_path, **read_kwargs)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path, **kwargs)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path, **kwargs)
        elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
            df = pd.read_pickle(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        return df

    def extract_embeddings(self, df: pd.DataFrame,
                           exclude_columns: Optional[List[str]] = None,
                           embedding_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract embedding vectors from DataFrame

        Parameters:
        -----------
        df : pd.DataFrame
            Original DataFrame
        exclude_columns : List[str], optional
            Column names to exclude
        embedding_columns : List[str], optional
            Column names to use as embedding vectors

        Returns:
        --------
        np.ndarray
            Embedding vector matrix
        """
        # If embedding columns specified, use these directly
        if embedding_columns:
            missing_cols = [col for col in embedding_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Specified embedding columns do not exist: {missing_cols}")
            embeddings_df = df[embedding_columns]
        else:
            # Otherwise exclude non-numerical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            # If exclude columns specified, remove from numerical columns
            if exclude_columns:
                exclude_set = set(exclude_columns)
                numeric_cols = [col for col in numeric_cols if col not in exclude_set]

            embeddings_df = df[numeric_cols]

        # Convert to numpy array
        embeddings = embeddings_df.astype(float).to_numpy()

        print(f"Extracted embeddings: {embeddings.shape[0]} samples, {embeddings.shape[1]} dimensions")

        # Check for NaN values
        nan_count = np.isnan(embeddings).sum()
        if nan_count > 0:
            print(f"Warning: Embeddings contain {nan_count} NaN values")
            # Fill NaN with column means
            col_means = np.nanmean(embeddings, axis=0)
            nan_indices = np.where(np.isnan(embeddings))
            embeddings[nan_indices] = np.take(col_means, nan_indices[1])

        return embeddings

    def save_results(self, results: Dict, timestamp: str = None) -> str:
        """
        Save clustering results to files

        Parameters:
        -----------
        results : Dict
            Clustering results
        timestamp : str, optional
            Timestamp

        Returns:
        --------
        str
            Saved file directory path
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create timestamp directory
        timestamp_dir = os.path.join(self.output_dir, timestamp)
        os.makedirs(timestamp_dir, exist_ok=True)

        # Use closure to store DataFrame counter
        df_counter = [0]  # Use list for mutability

        def convert_to_serializable(obj):
            """Recursively convert object to JSON serializable format"""
            if isinstance(obj, np.ndarray):
                # For arrays, convert to list
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                # For DataFrame, save as CSV and return file path
                df_file = os.path.join(timestamp_dir, f"dataframe_{df_counter[0]}.csv")
                obj.to_csv(df_file, index=False)
                df_counter[0] += 1
                return df_file
            elif isinstance(obj, pd.Series):
                # For Series, convert to dictionary
                return obj.to_dict()
            elif isinstance(obj, dict):
                # For dict, recursively process each value
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                # For list or tuple, recursively process each element
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                # For other objects, try to convert to dict
                return convert_to_serializable(obj.__dict__)
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                # Basic types directly return
                return obj
            else:
                # Other types convert to string
                return str(obj)

        try:
            # Convert all results to serializable format
            json_serializable_results = convert_to_serializable(results)

            # Save results dictionary
            results_file = os.path.join(timestamp_dir, "clustering_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(json_serializable_results, f, indent=2, ensure_ascii=False, default=str)

            print(f"Results saved to: {timestamp_dir}")
            return timestamp_dir
        except Exception as e:
            print(f"Error saving results: {e}")
            # If conversion fails, try simpler method
            import traceback
            traceback.print_exc()

            results_file = os.path.join(timestamp_dir, "clustering_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                # Only save basic information
                simple_results = {
                    'file_path': results.get('file_path'),
                    'original_shape': results.get('original_shape'),
                    'reduced_shape': results.get('reduced_shape'),
                    'timestamp': results.get('timestamp'),
                    'parameters': results.get('parameters'),
                    'saved_dir': timestamp_dir
                }
                json.dump(simple_results, f, indent=2, ensure_ascii=False)

            # Separately save arrays
            if 'best_labels' in results and isinstance(results['best_labels'], np.ndarray):
                labels_file = os.path.join(timestamp_dir, "best_labels.npy")
                np.save(labels_file, results['best_labels'])
                print(f"Cluster labels saved to: {labels_file}")

            if 'reduced_embeddings' in results and isinstance(results['reduced_embeddings'], np.ndarray):
                embeddings_file = os.path.join(timestamp_dir, "reduced_embeddings.npy")
                np.save(embeddings_file, results['reduced_embeddings'])
                print(f"Dimensionality reduced embeddings saved to: {embeddings_file}")

            if 'df' in results and isinstance(results['df'], pd.DataFrame):
                df_file = os.path.join(timestamp_dir, "data.csv")
                results['df'].to_csv(df_file, index=False)
                print(f"Data saved to: {df_file}")

            return timestamp_dir

    def save_clustered_data(self,
                            df: pd.DataFrame,
                            labels: np.ndarray,
                            top_results: List[Tuple[str, Dict]],
                            output_dir: str,
                            identifier_columns: Optional[List[str]] = None) -> str:
        """
        Save simplified clustering results, only including identifier columns and cluster labels

        Parameters:
        -----------
        df : pd.DataFrame
            Original DataFrame
        labels : np.ndarray
            Best algorithm cluster labels
        top_results : List[Tuple[str, Dict]]
            Top N best clustering results
        output_dir : str
            Output directory
        identifier_columns : List[str], optional
            Specified identifier column names, if None auto-detect

        Returns:
        --------
        str
            Saved file path
        """
        # Create copy to avoid modifying original data
        result_df = df.copy()

        # Add best cluster labels
        result_df['best_cluster_label'] = labels

        # Only add top 5 algorithm cluster labels, avoid duplicates
        columns_to_add = {}

        for i, (name, result) in enumerate(top_results[:5], 1):
            # Generate unique column name
            algorithm_type = result['algorithm']
            silhouette = result['silhouette']
            n_clusters = result['n_clusters']

            # Create unique column name
            column_name = f'cluster_{algorithm_type}_rank{i}_sil{silhouette:.4f}_k{n_clusters}'

            # Add to result DataFrame
            result_df[column_name] = result['labels']

            # Save column information
            columns_to_add[column_name] = {
                'algorithm': algorithm_type,
                'silhouette': silhouette,
                'n_clusters': n_clusters,
                'rank': i,
                'original_name': name
            }

        # Determine identifier columns
        if identifier_columns is None:
            identifier_columns = []
            # Prioritize common identifier column names
            common_id_names = ['phID', 'bio_id', 'id', 'ID', 'sample_id', 'sample_name',
                               'patient_id', 'subject_id', 'case_id', 'record_id']

            for col_name in common_id_names:
                if col_name in result_df.columns:
                    identifier_columns.append(col_name)

            # If no common identifier columns found, use first two non-numerical columns
            if not identifier_columns:
                non_numeric_cols = result_df.select_dtypes(exclude=[np.number]).columns.tolist()
                identifier_columns = non_numeric_cols[:2] if len(non_numeric_cols) >= 2 else non_numeric_cols

        # Collect all cluster-related columns
        cluster_columns = ['best_cluster_label'] + list(columns_to_add.keys())

        # Filter out potentially duplicate columns
        columns_to_keep = identifier_columns + cluster_columns
        columns_to_keep = list(dict.fromkeys(columns_to_keep))  # Deduplicate while preserving order

        # Create simplified result DataFrame
        simplified_df = result_df[columns_to_keep].copy()

        # Save to file
        output_path = os.path.join(output_dir, "clustered_data_simplified.csv")
        simplified_df.to_csv(output_path, index=False)

        # Create algorithm information file
        algorithms_info = []
        for column_name, info in columns_to_add.items():
            algorithms_info.append({
                'rank': info['rank'],
                'algorithm_name': info['original_name'],
                'algorithm_type': info['algorithm'],
                'silhouette_score': info['silhouette'],
                'n_clusters': info['n_clusters'],
                'column_name': column_name
            })

        # Sort by rank
        algorithms_info.sort(key=lambda x: x['rank'])

        info_path = os.path.join(output_dir, "clustering_algorithms_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(algorithms_info, f, indent=2, ensure_ascii=False)

        # Print summary information
        print(f"\nSimplified clustering results saved:")
        print(f"  File: {output_path}")
        print(f"  Contains {len(simplified_df)} rows, {len(simplified_df.columns)} columns")
        print(f"  Identifier columns: {', '.join(identifier_columns)}")
        print(f"  Clustering algorithm information: {info_path}")

        # Display column information
        print("\nGenerated cluster columns:")
        for info in algorithms_info:
            print(
                f"  {info['column_name']} - {info['algorithm_type']} (Rank {info['rank']}, Silhouette={info['silhouette_score']:.4f})")

        return output_path

    def visualize_clustering_results(self, output_dir: str, save_as_pdf: bool = True):
        """
        Plot and save clustering visualization results

        Parameters:
        -----------
        output_dir : str
            Output directory
        save_as_pdf : bool, default=True
            Whether to save as PDF
        """
        if self.evaluator is None or 'reduced_embeddings' not in self.results:
            print("Error: Please run clustering pipeline first")
            return

        reduced_embeddings = self.results['reduced_embeddings']
        labels = self.results['best_labels']

        if labels is not None:
            # Plot best algorithm results
            algorithm_name = self.results['best_result']['algorithm'] if self.results['best_result'] else "Best"

            if save_as_pdf:
                save_path = os.path.join(output_dir, "best_clustering_results.pdf")
            else:
                save_path = os.path.join(output_dir, "best_clustering_results.png")

            fig = self.evaluator.plot_clustering_results(
                reduced_embeddings=reduced_embeddings,
                labels=labels,
                save_path=save_path,
                algorithm_name=f"{algorithm_name} (Best)",
                show_plots=False
            )

            print(f"Best clustering results plot saved to: {save_path}")

            # Plot top 5 algorithm results
            top_results = self.evaluator.get_best_results(metric='silhouette', top_k=5)
            for i, (key, result) in enumerate(top_results, 1):
                if i > 1:  # Skip first (best result)
                    algo_labels = result['labels']
                    algorithm_name = result['algorithm']

                    # Create independent plot for each algorithm
                    if save_as_pdf:
                        algo_save_path = os.path.join(output_dir, f"clustering_{algorithm_name}_rank{i}.pdf")
                    else:
                        algo_save_path = os.path.join(output_dir, f"clustering_{algorithm_name}_rank{i}.png")

                    fig = self.evaluator.plot_clustering_results(
                        reduced_embeddings=reduced_embeddings,
                        labels=algo_labels,
                        save_path=algo_save_path,
                        algorithm_name=f"{algorithm_name} (Rank {i})",
                        show_plots=False
                    )

                    print(f"{algorithm_name} (Rank {i}) results plot saved to: {algo_save_path}")

            # Create algorithm comparison summary chart
            self.evaluator.create_algorithm_comparison_chart(output_dir, save_as_pdf=save_as_pdf)

            # Create comprehensive PDF report
            if save_as_pdf:
                pdf_report_path = self.evaluator.save_all_plots_as_pdf(
                    output_dir,
                    filename="clustering_analysis_report.pdf"
                )
                print(f"Comprehensive PDF report saved to: {pdf_report_path}")

    def run_pipeline(self,
                     file_path: str,
                     target_dim: int = 200,
                     reduction_method: str = 'two_stage_pca',
                     pca_intermediate_dim: int = 200,
                     exclude_columns: Optional[List[str]] = None,
                     embedding_columns: Optional[List[str]] = None,
                     sample_size: Optional[int] = None,
                     save_results: bool = True,
                     save_as_pdf: bool = True,
                     **kwargs) -> Dict:
        """
        Run complete clustering analysis pipeline

        Parameters:
        -----------
        file_path : str
            Data file path
        target_dim : int
            Target dimensionality reduction dimension
        reduction_method : str
            Dimensionality reduction method ('pca', 'svd', 'two_stage_pca',
                                           'pca_then_tsne', 'pca_then_umap', 'tsne_direct')
        pca_intermediate_dim : int
            PCA intermediate dimension (for pca_then_tsne, etc.)
        exclude_columns : List[str], optional
            Column names to exclude
        embedding_columns : List[str], optional
            Column names to use as embedding vectors
        sample_size : int, optional
            Sample size (for large datasets)
        save_results : bool
            Whether to save results
        save_as_pdf : bool
            Whether to save plots as PDF (True) or PNG (False)
        **kwargs : dict
            Additional parameters for RobustClusteringEvaluator

        Returns:
        --------
        Dict
            Dictionary containing all results
        """
        # 1. Load data
        df = self.load_data(file_path)

        # 2. Sampling (if needed)
        if sample_size and sample_size < len(df):
            print(f"Sampling {sample_size} from {len(df)} samples")
            df = df.sample(n=sample_size, random_state=self.random_state)

        # 3. Extract embeddings
        embeddings = self.extract_embeddings(df, exclude_columns, embedding_columns)

        # 4. Create evaluator and run clustering
        self.evaluator = RobustClusteringEvaluator(random_state=self.random_state, verbose=True)

        # Store original dimensions for reporting
        self.evaluator.original_dim = embeddings.shape[1]

        labels, best_result, reduced_embeddings = self.evaluator.robust_high_dim_clustering(
            embeddings=embeddings,
            target_dim=target_dim,
            reduction_method=reduction_method,
            pca_intermediate_dim=pca_intermediate_dim,
            **kwargs
        )

        # 5. Get top results
        top_results = self.evaluator.get_best_results(metric='silhouette', top_k=5)

        print("\nTop 5 clustering results:")
        for i, (name, result) in enumerate(top_results, 1):
            print(f"{i}. {name}: "
                  f"Silhouette={result['silhouette']:.4f}, "
                  f"Clusters={result['n_clusters']}, "
                  f"Algorithm={result['algorithm']}")

        # 6. Prepare results dictionary
        self.results = {
            'file_path': file_path,
            'original_shape': embeddings.shape,
            'reduced_shape': reduced_embeddings.shape,
            'best_labels': labels,
            'best_result': best_result,
            'top_results': dict(top_results),
            'reduced_embeddings': reduced_embeddings,
            'df': df,  # Contains original data
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'target_dim': target_dim,
                'reduction_method': reduction_method,
                'pca_intermediate_dim': pca_intermediate_dim,
                'exclude_columns': exclude_columns,
                'embedding_columns': embedding_columns,
                'sample_size': sample_size
            }
        }

        # 7. Add cluster labels back to original DataFrame
        if labels is not None:
            df['cluster_label'] = labels
            cluster_stats = df.groupby('cluster_label').size().reset_index(name='count')
            print("\nCluster distribution:")
            print(cluster_stats)

            self.results['cluster_distribution'] = cluster_stats

        # 8. Save results
        if save_results and labels is not None:
            saved_dir = self.save_results(self.results)
            self.results['saved_dir'] = saved_dir

            # Save simplified clustering results
            _ = self.save_clustered_data(
                df=df,
                labels=labels,
                top_results=top_results,
                output_dir=saved_dir,
                identifier_columns=['phID', 'bio_id']  # Explicitly specify identifier columns
            )

            # 9. Plot and save clustering results
            self.visualize_clustering_results(saved_dir, save_as_pdf=save_as_pdf)

            # 生成轮廓评分折线图
            self.evaluator.plot_silhouette_rank_lineplot(output_dir=saved_dir, save_as_pdf=True)
            self.evaluator.plot_silhouette_progression_lineplot(output_dir=saved_dir, save_as_pdf=True)

            # 或者生成完整的分析报告
            self.evaluator.create_silhouette_rank_analysis(output_dir=saved_dir, save_as_pdf=True)


        return self.results

    def visualize_best_clustering(self, save_fig: bool = True, save_as_pdf: bool = True):
        """
        Visualize best clustering results

        Parameters:
        -----------
        save_fig : bool
            Whether to save the figure
        save_as_pdf : bool
            Whether to save as PDF (True) or PNG (False)
        """
        if self.evaluator is None or 'reduced_embeddings' not in self.results:
            print("Error: Please run clustering pipeline first")
            return

        try:
            from visualization_utils import visualize_clustering_results
        except ImportError:
            # If visualization_utils doesn't exist, create simple visualization
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA

            print("Visualization tools not available, using simple PCA visualization")

            reduced_embeddings = self.results['reduced_embeddings']
            labels = self.results['best_labels']

            if labels is not None:
                # Use PCA to reduce to 2D for visualization
                pca = PCA(n_components=2, random_state=self.random_state)
                embeddings_2d = pca.fit_transform(reduced_embeddings)

                fig, ax = plt.subplots(figsize=(10, 8))

                # Plot scatter plot
                scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                     c=labels, cmap='tab20', s=30, alpha=0.7)

                ax.set_title(f"Cluster Visualization (PCA)", fontsize=14, fontname='Arial')
                ax.set_xlabel("First Principal Component", fontsize=12, fontname='Arial')
                ax.set_ylabel("Second Principal Component", fontsize=12, fontname='Arial')

                # Add color bar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Cluster Label', fontname='Arial')

                if save_fig and 'saved_dir' in self.results:
                    if save_as_pdf:
                        fig_path = os.path.join(self.results['saved_dir'], "clustering_visualization.pdf")
                        plt.savefig(fig_path, dpi=300, bbox_inches='tight',
                                    facecolor='white', edgecolor='none',
                                    format='pdf', transparent=False)
                    else:
                        fig_path = os.path.join(self.results['saved_dir'], "clustering_visualization.png")
                        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    print(f"Visualization image saved to: {fig_path}")

                plt.show()
                return fig
        else:
            # Use visualization_utils
            reduced_embeddings = self.results['reduced_embeddings']
            labels = self.results['best_labels']

            if labels is not None:
                title = f"Best Clustering Results (Algorithm: {self.results['best_result']['algorithm']})"

                # Visualize
                fig = visualize_clustering_results(
                    reduced_embeddings,
                    labels,
                    title=title
                )

                # Save image
                if save_fig and 'saved_dir' in self.results:
                    import matplotlib.pyplot as plt
                    if save_as_pdf:
                        fig_path = os.path.join(self.results['saved_dir'], "clustering_visualization.pdf")
                        plt.savefig(fig_path, dpi=300, bbox_inches='tight',
                                    facecolor='white', edgecolor='none',
                                    format='pdf', transparent=False)
                    else:
                        fig_path = os.path.join(self.results['saved_dir'], "clustering_visualization.png")
                        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    print(f"Visualization image saved to: {fig_path}")
                    plt.close()


def main(file_path: str,
         target_dim: int = 200,
         reduction_method: str = 'two_stage_pca',
         pca_intermediate_dim: int = 200,
         exclude_columns: Optional[List[str]] = None,
         embedding_columns: Optional[List[str]] = None,
         sample_size: Optional[int] = None,
         output_dir: str = "./clustering_results",
         random_state: int = 42,
         save_as_pdf: bool = True,
         **kwargs):
    """
    Main function - run complete clustering analysis pipeline

    Parameters:
    -----------
    file_path : str
        Data file path
    target_dim : int, default=200
        Target dimensionality reduction dimension
    reduction_method : str, default='two_stage_pca'
        Dimensionality reduction method ('pca', 'svd', 'two_stage_pca',
                                       'pca_then_tsne', 'pca_then_umap', 'tsne_direct')
    pca_intermediate_dim : int, default=200
        PCA intermediate dimension (for pca_then_tsne, etc.)
    exclude_columns : List[str], optional
        Column names to exclude
    embedding_columns : List[str], optional
        Column names to use as embedding vectors
    sample_size : int, optional
        Sample size (for large datasets)
    output_dir : str, default='./clustering_results'
        Results save directory
    random_state : int, default=42
        Random seed
    save_as_pdf : bool, default=True
        Whether to save plots as PDF (True) or PNG (False)
    **kwargs : dict
        Additional parameters for RobustClusteringEvaluator

    Returns:
    --------
    Dict
        Clustering results
    """
    # Create clustering pipeline
    pipeline = ClusteringPipeline(random_state=random_state, output_dir=output_dir)

    # Run pipeline
    results = pipeline.run_pipeline(
        file_path=file_path,
        target_dim=target_dim,
        reduction_method=reduction_method,
        pca_intermediate_dim=pca_intermediate_dim,
        exclude_columns=exclude_columns,
        embedding_columns=embedding_columns,
        sample_size=sample_size,
        save_results=True,
        save_as_pdf=save_as_pdf,
        **kwargs
    )

    return results

def get_base_dir() -> str:
    """获取当前脚本所在目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    base_path = get_base_dir()
    # Test code
    results = main(
        file_path=os.path.join(base_path, 'demo_data', 'pathology_ollama_embed_deepseek.csv'),
        target_dim=3, #由于demo数据，改为3
        reduction_method="two_stage_pca", #"two_stage_pca",
        pca_intermediate_dim = 5, #由于demo数据，改为5
        exclude_columns=[
            "phID", "bio_id", "source_file", "total_embedding_dim", "timestamp",
            "glomerular_lesions_text_preview", "tubulointerstitial_lesions_text_preview",
            "vascular_lesions_text_preview", "immunofluorescence_text_preview"
        ],
        # sample_size=1000,  # 采样1000个样本进行快速测试
        output_dir=os.path.join(base_path,"deepseek_clustering_results"),
        random_state=42
    )

    # 访问结果
    print(f"最佳聚类算法: {results['best_result']['algorithm']}")
    print(f"轮廓系数: {results['best_result']['silhouette']:.4f}")
    print(f"找到 {results['best_result']['n_clusters']} 个簇")

    # 结果保存在 ./clustering_results/时间戳目录/
    print(f"结果保存目录: {results.get('saved_dir', '未保存')}")

    # 如果数据有聚类标签，可以查看聚类分布
    if 'cluster_distribution' in results:
        print("\n聚类分布:")
        print(results['cluster_distribution'])