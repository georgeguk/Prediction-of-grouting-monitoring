import numpy as np
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
SEED = 42
START_ROUND = 2
END_ROUND = 8

def fit(energy_matrix):
    energy = np.asarray(energy_matrix, dtype=float)
    if not np.isfinite(energy).all() or np.any(energy[:, :-1] == 0):
        raise ValueError('Invalid energy ratio observations')
    rates = np.abs(np.diff(energy, axis=1) / energy[:, :-1])
    values = rates[:, START_ROUND - 1:END_ROUND].reshape(-1, 1)
    with threadpool_limits(limits=1):
        model = GaussianMixture(n_components=2, covariance_type='full', init_params='kmeans', n_init=1, random_state=SEED, max_iter=100, tol=0.001, reg_covar=1e-06).fit(values)
    grid = np.linspace(values.min(), values.max(), 2000)[:, None]
    p = model.predict_proba(grid)
    threshold = float(grid[np.argmin(abs(p[:, 0] - p[:, 1])), 0])
    return (model, threshold, rates)
