import numpy as np


def simulate_portfolio(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    horizon_days: int = 10,
    n_sims: int = 10000,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    chol = np.linalg.cholesky(cov)

    dt = 1 / 252
    drift = (mu - 0.5 * np.diag(cov)) * dt

    sims = np.empty((n_sims, horizon_days, len(weights)))
    for d in range(horizon_days):
        z = rng.standard_normal((n_sims, len(weights)))
        eps = z @ chol.T
        daily_ret = np.exp(drift + eps * np.sqrt(dt))
        sims[:, d, :] = daily_ret

    cuml_factors = sims.prod(axis=1)

    pnl_array = (cuml_factors @ weights) - 1
    return pnl_array


def var_es(pnl_array: np.ndarray, conf: float) -> tuple[float, float]:
    var = -np.percentile(pnl_array, (1 - conf) * 100)
    es = -pnl_array[pnl_array <= -var].mean()

    return var, es


def simulate_price_paths(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    horizon_days: int = 10,
    n_sims: int = 5,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(cov)

    dt = 1.0 / 252.0
    drift = (mu - 0.5 * np.diag(cov)) * dt

    daily = np.empty((n_sims, horizon_days, len(weights)))

    for d in range(horizon_days):
        z = rng.standard_normal((n_sims, len(weights)))  # (n_sims, n_assets)
        eps = z @ chol.T
        daily[:, d, :] = np.exp(drift + eps * np.sqrt(dt))

    price_paths = np.concatenate(
        [
            np.ones((n_sims, 1, len(weights))),  # t=0
            np.cumprod(daily, axis=1),
        ],
        axis=1,
    )

    return price_paths


def portfolio_trajectory(price_paths: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return price_paths @ weights
