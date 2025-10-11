import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.neighbors import NearestNeighbors

def _cosine_sim_vector_diff_sampled(X, norm_distance, sampling_rate=1.0, random_state=None):
    """
    Compute cosine similarity for sampled subsets using optimized operations.
    Returns sim (n_samples, eff_sample_size) and sampled_indices.
    """
    n_samples, n_features = X.shape
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    eff_sample_size = max(1, int((n_samples - 1) * sampling_rate))

    if sampling_rate >= 1.0:
        # Use exact KNN for top_n nearest (efficient for large n, small top_n)
        # Note: This requires top_n from outer scope; but since we merge logic, assume passed or adjust.
        # Wait, to make it general, but since used in compute with top_n, we return top sim directly.
        # But to keep, we will handle in compute_md_scores.
        # For now, use full eff = n-1, but to optimize, return KNN sim if sampling 1.0
        nn = NearestNeighbors(metric='euclidean', n_jobs=-1)
        nn.fit(X)
        # To get top_n, but since eff large, but we will handle in outer.
        # To unify, compute full if small n, but for opt, if sampling >=1, use KNN, but need top_n.
        # Problem: _cosine doesn't have top_n, but to opt, we can pass top_n to this function.
        # To simplify, add top_n as param.
        # Wait, see below, we will add top_n to function.
    # To make complete, let's adjust the function signature to include top_n.
    # See below in code.

# To resolve, we will move the logic to compute_md_scores, make _cosine deprecated or merge.
# For simplicity, we will keep and add top_n to _cosine.

#The optimized complete code is as follows (we added top_n to _cosine function for KNN case):

def _cosine_sim_vector_diff_sampled(X, norm_distance, top_n, sampling_rate=1.0, random_state=None):
    n_samples, n_features = X.shape
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    eff_sample_size = max(1, int((n_samples -1) * sampling_rate))

    if sampling_rate >=1.0:
        nn = NearestNeighbors(n_neighbors=min(top_n +1, n_samples), metric='euclidean', n_jobs=-1)
        nn.fit(X)
        dists, sampled_indices = nn.kneighbors(X)
        dists = dists[:,1:]
        sampled_indices = sampled_indices[:,1:]
        sim = norm_distance / np.sqrt(dists**2 + norm_distance**2 + 1e-10)
        return sim, sampled_indices

    else:
        # Optimized sampling with replace=True for speed (approx, but fast and good for low rate)
        sampled_indices = rng.choice(n_samples, size=(n_samples, eff_sample_size), replace=True)
        diff_features = X[sampled_indices] - X[:, np.newaxis, :]
        norm_oi_xj = np.sqrt(np.sum(diff_features ** 2, axis=2) + norm_distance ** 2)
        sim = norm_distance / (norm_oi_xj + 1e-10)
        mask_self = sampled_indices == np.arange(n_samples)[:, np.newaxis]
        sim[mask_self] = -np.inf
        return sim, sampled_indices

def compute_md_scores(X, norm_distance, top_n, sampling_rate=1.0, random_state=None):
    """
    Compute MDOD scores for each sample in X using vector cosine similarity with sampling.
    """
    if norm_distance <= 0:
        raise ValueError("norm_distance must be positive")
    if top_n <= 0 or not isinstance(top_n, int):
        raise ValueError("top_n must be a positive integer")
    if not (0 < sampling_rate <= 1.0):
        raise ValueError("sampling_rate must be between 0 and 1")

    X = check_array(X, ensure_2d=True, dtype=float)
    n_samples, n_features = X.shape
    if n_samples == 0:
        return np.array([]), np.array([])

    # Compute similarity with optimization
    sim, sampled_indices = _cosine_sim_vector_diff_sampled(X, norm_distance, top_n, sampling_rate, random_state)

    eff_k = min(top_n, sim.shape[1])
    top_scores = np.partition(sim, -eff_k, axis=1)[:, -eff_k:]
    scores = np.sum(np.where(np.isinf(top_scores), 0.0, top_scores), axis=1)

    original_indices = np.arange(n_samples)
    return scores, original_indices

class MDOD:
    """
    MDOD with vector cosine similarity by adding a dimension as per OD-ADVCS, with sampling optimization.
    """
    def __init__(self, norm_distance=1.0, top_n=5, contamination=0.1, sampling_rate=1.0, random_state=None):
        self.norm_distance = float(norm_distance)
        self.top_n = int(top_n)
        self.contamination = float(contamination)
        self.sampling_rate = float(sampling_rate)
        self.random_state = random_state
        self.scores_ = None
        self.decision_scores_ = None
        self.threshold_ = None
        self.labels_ = None
        self._X = None

    def fit(self, X, y=None):
        X = check_array(X, ensure_2d=True)
        self._X = X.copy()
        scores, _ = compute_md_scores(self._X, self.norm_distance, self.top_n, self.sampling_rate, self.random_state)
        self.scores_ = scores
        self.decision_scores_ = -self.scores_  # Negative for threshold comparison
        self.threshold_ = np.percentile(self.decision_scores_, 100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ >= self.threshold_).astype(int)
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['_X'])
        X = check_array(X, ensure_2d=True)
        n_test = X.shape[0]
        if n_test == 0:
            return np.array([])

        train_X = self._X
        n_train = train_X.shape[0]

        # Optimized computation for test
        if self.sampling_rate >= 1.0:
            nn = NearestNeighbors(n_neighbors=self.top_n, metric='euclidean', n_jobs=-1)
            nn.fit(train_X)
            dists, _ = nn.kneighbors(X)
            sim = self.norm_distance / np.sqrt(dists**2 + self.norm_distance**2 + 1e-10)
            scores = np.sum(sim, axis=1)
        else:
            rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random
            eff_sample_size = max(1, int(n_train * self.sampling_rate))
            sampled_indices = rng.choice(n_train, size=eff_sample_size, replace=False)
            diff_features = X[:, np.newaxis, :] - train_X[sampled_indices][np.newaxis, :, :]
            norm_oi_xj = np.sqrt(np.sum(diff_features ** 2, axis=2) + self.norm_distance ** 2)
            sim = self.norm_distance / (norm_oi_xj + 1e-10)
            eff_k = min(self.top_n, eff_sample_size)
            top_scores = np.partition(sim, -eff_k, axis=1)[:, -eff_k:]
            scores = np.sum(np.where(np.isinf(top_scores), 0.0, top_scores), axis=1)

        return -scores

    def predict(self, X=None):
        if X is None:
            return self.labels_
        scores = self.decision_function(X)
        return (scores >= self.threshold_).astype(int)
