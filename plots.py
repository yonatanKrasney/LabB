"""
plots.py -- the matplotlib figure that goes into the report.

Kept separate from app.py so it can be produced without the interface:

    from plots import report_figure
    fig = report_figure(result, x, dx, y, dy, mask, "w [rad/s]", "e0 [V]", "Fit")
    fig.savefig("figure.pdf")
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INCLUDED_COLOR = "#1f4e9c"
EXCLUDED_COLOR = "#9aa0a6"
FIT_COLOR = "#c0392b"


def report_figure(
    result,
    x: np.ndarray,
    dx: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    mask: np.ndarray,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "",
    annotate_statistics: bool = True,
) -> plt.Figure:
    """Fit panel and residual panel, side by side, ready for the report.

    `mask` is True for excluded points. They are drawn as hollow grey circles
    rather than removed, so the figure shows what was left out.
    """
    plt.close("all")
    mask = np.asarray(mask, dtype=bool)
    keep = ~mask

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="white")

    # ---- left panel: the data and the fit ----------------------------
    ax = axes[0]
    ax.errorbar(x[keep], y[keep], xerr=dx[keep], yerr=dy[keep],
                fmt="o", ms=4, color=INCLUDED_COLOR, ecolor="gray",
                elinewidth=1, capsize=2, label="Data", zorder=3)
    if mask.any():
        ax.errorbar(x[mask], y[mask], xerr=dx[mask], yerr=dy[mask],
                    fmt="o", ms=5, mfc="none", color=EXCLUDED_COLOR,
                    ecolor="lightgray", elinewidth=1, capsize=2,
                    label="Excluded from fit", zorder=2)
    if result is not None:
        curve_x, curve_y = result.curve()
        ax.plot(curve_x, curve_y, color=FIT_COLOR, lw=1.6, label="Fit", zorder=4)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=9, loc="best")
    ax.set_facecolor("white")

    if annotate_statistics and result is not None and np.isfinite(result.chi2_reduced):
        ax.text(
            0.03, 0.97,
            f"$\\chi^2_{{red}}$ = {result.chi2_reduced:.2f}\n"
            f"p = {result.p_value:.3f}\n"
            f"dof = {result.dof}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
        )

    # ---- right panel: residuals --------------------------------------
    ax = axes[1]
    if result is not None:
        ax.errorbar(result.x, result.residuals,
                    xerr=result.dx, yerr=result.residual_errors,
                    fmt="o", ms=4, color=INCLUDED_COLOR, ecolor="gray",
                    elinewidth=1, capsize=2, zorder=3)
        ax.axhline(0, color=FIT_COLOR, ls="--", lw=1.2, zorder=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(f"residual: {y_label} - f(x)")
    ax.set_title("Residuals")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("white")

    fig.tight_layout()
    return fig
