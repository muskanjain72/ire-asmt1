"""
Q4 (part 4): Bootstrap 95% confidence intervals.

Works on any array of per-impression metric values (e.g. the "values"
array returned by metrics.evaluate_impressions for AUC/MRR/nDCG) —
resample with replacement, recompute the mean, repeat many times,
take the 2.5th/97.5th percentiles of the resulting distribution.

Distribution-free: makes no assumption about normality, unlike a
t-test-based CI, which matters here since ranking metrics like nDCG
are bounded in [0,1] and often skewed, not Gaussian.
"""

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Returns {mean, lower, upper, n} for the given per-impression metric
    values. `n` is included since a CI is meaningless without knowing
    how many impressions it's based on.
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return {"mean": None, "lower": None, "upper": None, "n": 0}

    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_iterations)

    for i in range(n_iterations):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return {
        "mean": float(values.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "n": n,
    }


def bootstrap_all_metrics(metric_results: dict, n_iterations: int = 1000) -> dict:
    """
    Convenience wrapper: takes the dict returned by
    metrics.evaluate_impressions() and returns bootstrap CIs for every
    metric in one call.
    """
    return {
        name: bootstrap_ci(result["values"], n_iterations=n_iterations)
        for name, result in metric_results.items()
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # a metric with true mean ~0.6, some noise, n=300 "impressions"
    fake_ndcg_values = np.clip(rng.normal(0.6, 0.15, size=300), 0, 1)

    result = bootstrap_ci(fake_ndcg_values)
    print("Bootstrap CI for a synthetic nDCG-like metric:")
    print(f"  mean={result['mean']:.4f}, 95% CI=[{result['lower']:.4f}, {result['upper']:.4f}], n={result['n']}")

    # smaller sample -> wider CI, sanity check
    small_sample = fake_ndcg_values[:20]
    result_small = bootstrap_ci(small_sample)
    print(f"\nSame metric, only 20 samples (expect wider CI):")
    print(f"  mean={result_small['mean']:.4f}, 95% CI=[{result_small['lower']:.4f}, {result_small['upper']:.4f}], n={result_small['n']}")