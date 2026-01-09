"""
MDOD - Multi-Dimensional Outlier Detection (v3.0.6.1 - High-Performance & Large-Scale)

Features:
- Fully backward compatible with published API
- Supports >500k samples without OOM (batch + float32 + incremental)
- Preserves high AUC by only down-sampling when absolutely necessary
- MemoryError auto-fallback (rarely triggered)

A Python implementation of outlier detection in multi-dimensional data using vector cosine similarity with an added virtual dimension.

Core algorithm from:
"Outlier Detection Using Vector Cosine Similarity by Adding a Dimension" 
DOI: 10.48550/arXiv.2601.00883

Author: Z Shen
Optimized with assistance from Grok (xAI)

Copyright (c) 2024-2026 Z Shen
Licensed under the BSD 3-Clause License - see LICENSE.txt for details.

Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations.
"""

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.neighbors import NearestNeighbors


def _compute_scores_incremental(
    X,
    norm_distance,
    top_n,
    sampling_rate,
    random_state=None,
    dtype=np.float32,
    batch_size=8192,  #  batch 
):
    """
    Compute MDOD scores incrementally with batch processing.
    Never stores full (n_samples, n_sampled) similarity matrix.
    """
    n_samples, n_features = X.shape
    rng = np.random.RandomState(random_state) if random_state is not None else np.random.RandomState()

    scores = np.zeros(n_samples, dtype=dtype)

    # KNN mode (exact top_n nearest neighbors)
    if sampling_rate >= 1.0:
        k = min(top_n + 1, n_samples)  # +1 to exclude self
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto", n_jobs=-1)
        nn.fit(X)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_X = X[start:end]
            dists = nn.kneighbors(batch_X)[0][:, 1:]  # exclude self
            sim = norm_distance / np.sqrt(dists**2 + norm_distance**2 + 1e-10)
            scores[start:end] = np.sum(sim, axis=1)

    # Sampling mode
    else:
        eff_sample_size = max(1, int(n_samples * sampling_rate))
        sampled_idx = rng.choice(n_samples, size=eff_sample_size, replace=False)
        sampled_points = X[sampled_idx]

        eff_k = min(top_n, eff_sample_size)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_X = X[start:end]

            # Shape: (batch, eff_sample_size, features)
            diff = batch_X[:, np.newaxis, :] - sampled_points[np.newaxis, :, :]
            norm_oi_xj = np.sqrt(np.sum(diff**2, axis=2) + norm_distance**2)
            sim = norm_distance / (norm_oi_xj + 1e-10)

            # Select top-k similarities
            if eff_k < sim.shape[1]:
                top_sim = np.partition(sim, -eff_k, axis=1)[:, -eff_k:]
            else:
                top_sim = sim

            scores[start:end] = np.sum(top_sim, axis=1)

    return scores


class MDOD:
    """
    MDOD - High-performance version fully compatible with original API.
    All parameters are controlled externally via constructor arguments.
    Only performs emergency down-sampling if MemoryError occurs.
    """

    def __init__(
        self,
        norm_distance=1.0,
        top_n=5,
        contamination=0.1,
        sampling_rate=1.0,
        random_state=None,
    ):
        self.norm_distance = float(norm_distance)
        self.top_n = int(top_n)
        self.contamination = float(contamination)
        self.sampling_rate = float(sampling_rate)
        self.random_state = random_state

        # Runtime attributes
        self.scores_ = None
        self.decision_scores_ = None
        self.threshold_ = None
        self.labels_ = None
        self._X = None

    def fit(self, X, y=None):
        X = check_array(X, ensure_2d=True, dtype=np.float32)
        n_samples = X.shape[0]

        self._X = X.copy()

        # Primary attempt with user parameters
        current_rate = self.sampling_rate
        max_retries = 5

        for attempt in range(max_retries):
            try:
                scores = _compute_scores_incremental(
                    self._X,
                    self.norm_distance,
                    self.top_n,
                    current_rate,
                    self.random_state,
                    dtype=np.float32,
                    batch_size=8192,
                )

                self.scores_ = scores
                self.decision_scores_ = -scores
                self.threshold_ = np.percentile(
                    self.decision_scores_, 100 * (1 - self.contamination)
                )
                self.labels_ = (self.decision_scores_ >= self.threshold_).astype(int)
                return self

            except MemoryError:
                current_rate *= 0.5
                current_rate = max(current_rate, 0.001)  # 
                print(
                    f"  MDOD MemoryError (attempt {attempt+1}/{max_retries}): "
                    f"emergency down-sampling to {current_rate:.4f}"
                )

        raise MemoryError("MDOD failed after multiple emergency retries.")

    def decision_function(self, X):
        check_is_fitted(self, ["_X"])
        X = check_array(X, ensure_2d=True, dtype=np.float32)
        n_test = X.shape[0]
        if n_test == 0:
            return np.array([])

        train_X = self._X
        scores = np.zeros(n_test, dtype=np.float32)

        if self.sampling_rate >= 1.0:
            k = min(self.top_n, train_X.shape[0])
            nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
            nn.fit(train_X)
            batch_size = 8192
            for i in range(0, n_test, batch_size):
                batch = X[i : i + batch_size]
                dists = nn.kneighbors(batch)[0]
                sim = self.norm_distance / np.sqrt(dists**2 + self.norm_distance**2 + 1e-10)
                scores[i : i + batch_size] = np.sum(sim, axis=1)
        else:
            eff_sample_size = max(1, int(train_X.shape[0] * self.sampling_rate))
            rng = np.random.RandomState(self.random_state)
            sampled = train_X[rng.choice(train_X.shape[0], eff_sample_size, replace=False)]
            eff_k = min(self.top_n, eff_sample_size)
            batch_size = 8192
            for i in range(0, n_test, batch_size):
                batch = X[i : i + batch_size]
                diff = batch[:, np.newaxis, :] - sampled[np.newaxis, :, :]
                norm = np.sqrt(np.sum(diff**2, axis=2) + self.norm_distance**2)
                sim = self.norm_distance / (norm + 1e-10)
                top_sim = np.partition(sim, -eff_k, axis=1)[:, -eff_k:]
                scores[i : i + batch_size] = np.sum(top_sim, axis=1)

        return -scores

    def predict(self, X=None):
        if X is None:
            return self.labels_
        scores = self.decision_function(X)
        return (scores >= self.threshold_).astype(int)