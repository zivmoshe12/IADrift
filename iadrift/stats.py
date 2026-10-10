"""Paired statistics for method comparisons (docs/PROTOCOL.md, Section 6).

For methods a, b and one metric, take per-image differences d_i = m_i(a) - m_i(b) over n images and report the mean
d_bar with the 95% interval d_bar +- 1.96 s_d / sqrt(n) (s_d with divisor n - 1), the Wilcoxon signed-rank p-value,
and the number of images where a is better. p-values of the primary tests are Holm-corrected.
"""

import numpy as np
from scipy import stats

LOWER_IS_BETTER = ("lpips", "blockiness", "lr_loss")


def paired_stats(a, b, key):
    """a, b: per-image values of one metric for two methods, same image order. key sets which direction is better."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    better = d < 0 if key in LOWER_IS_BETTER else d > 0
    return {"mean": float(d.mean()), "lo": float(d.mean() - half), "hi": float(d.mean() + half),
            "p": float(stats.wilcoxon(d).pvalue), "better": float(np.sum(better)), "n": float(len(d))}


def holm(pvalues):
    """Holm-adjusted p-values: with p_(1) <= ... <= p_(m), p~_(k) = max_{j<=k} min(1, (m - j + 1) p_(j))."""
    pvalues = np.asarray(pvalues, np.float64)
    order, adjusted, running = np.argsort(pvalues), np.empty(len(pvalues)), 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[k]))
        adjusted[k] = running
    return adjusted


def mean_ci(values):
    """Mean and the half-width of its 95% t-interval, t_{0.975, n-1} s / sqrt(n)."""
    values = np.asarray(values, np.float64)
    return float(values.mean()), float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
