"""Paired statistics for method comparisons (docs/PROTOCOL.md, Section 6).

For methods a, b and a metric, per-image differences d_i = m_i(a) - m_i(b), i = 1..n (n = number of images): mean
d_bar with a 95% interval d_bar +- 1.96 s_d / sqrt(n), where s_d is the sample standard deviation of the d_i (divisor
n - 1), so s_d / sqrt(n) is the standard error of d_bar; the Wilcoxon signed-rank p-value; and the number of images
where a is better.
Primary tests are corrected with Holm's step-down method.
"""

import numpy as np
from scipy import stats

LOWER_IS_BETTER = ("lpips", "blockiness", "lr_loss")


def paired_stats(a, b, key):
    """a, b: per-image metric values of two methods (same image order). key decides the 'better' direction."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    better = d < 0 if key in LOWER_IS_BETTER else d > 0
    return {"mean": float(d.mean()), "lo": float(d.mean() - half), "hi": float(d.mean() + half),
            "p": float(stats.wilcoxon(d).pvalue), "better": float(np.sum(better)), "n": float(len(d))}


def holm(pvalues):
    """Holm-adjusted p-values: sort p_(1) <= ... <= p_(m); p~_(k) = max_{j<=k} min(1, (m - j + 1) p_(j))."""
    pvalues = np.asarray(pvalues, np.float64)
    order, adjusted, running = np.argsort(pvalues), np.empty(len(pvalues)), 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[k]))
        adjusted[k] = running
    return adjusted


def mean_ci(values):
    """Mean and t-based 95% half-width t_{0.975, n-1} * s / sqrt(n) (s = sample standard deviation, divisor n - 1;
    n = number of values)."""
    values = np.asarray(values, np.float64)
    return float(values.mean()), float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
