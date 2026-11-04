"""Shared look of the GIWAXS sector-cut panels (SI 2 and SI 12).

SI2.ipynb writes the cuts to ``SI2_cuts.csv`` and draws them with
:func:`draw_cut_panel`; FIG4.ipynb (SI 12) reads the same file and draws the
same panel, so the two figures carry identical curves, axes and labels.
"""

from __future__ import annotations

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

C_OOP = "red"
C_IP = "blue"
C_GRID = "#4040ff"
LW_CUT = 2.2
FS_LABEL = 9           # axis labels
FS_PEAK = 8.5          # reflection labels
FS_TICK = 8
FS_MINOR = 6.5
Q_LIM = (0.1, 2.5)

RC = {
    "axes.grid": False,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "axes.labelsize": FS_LABEL,
    "legend.fontsize": FS_TICK,
}

MAP_TEXT = dict(color="white", fontsize=FS_PEAK, fontweight="bold", ha="center", va="center",
                path_effects=[pe.Stroke(linewidth=2.0, foreground="black", alpha=0.75), pe.Normal()])


def _minor_label(v, _):
    """Leading digit of a minor log tick (2 to 9), blank elsewhere."""
    m = v / 10 ** np.floor(np.log10(v) + 1e-9)
    return f"{int(round(m))}" if 1.5 < m < 9.5 else ""


def draw_cut_panel(ax, cuts, labels, *, q_lim=Q_LIM, legend=True):
    """Out-of-plane and in-plane cuts on log-log axes, in the SI 2 style.

    Parameters
    ----------
    ax
        Target axes.
    cuts
        Table with ``q``, ``oop_smooth`` and ``ip_smooth`` (``SI2_cuts.csv``).
    labels
        ``{text: (column, q)}`` or ``{text: (column, q, x_factor)}``; each label is set
        above the curve at ``q``, shifted horizontally by ``x_factor`` on the log axis.
    q_lim
        Horizontal limits in 1/A.
    legend
        Draw the boxed legend.
    """
    q = cuts["q"].to_numpy(float)
    ymin, ymax = np.inf, 0.0
    for col, c, lab in (("oop_smooth", C_OOP, "Out of plane"), ("ip_smooth", C_IP, "In plane")):
        y = cuts[col].to_numpy(float)
        ok = np.isfinite(y) & (y > 0) & (q >= q_lim[0]) & (q <= q_lim[1])
        ax.plot(q[ok], y[ok], color=c, lw=LW_CUT, label=lab, solid_capstyle="round", zorder=3)
        hi = ok & (q >= 0.3)
        ymin, ymax = min(ymin, y[ok].min()), max(ymax, y[hi].max())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*q_lim)
    ax.set_ylim(0.75 * ymin, 5.0 * ymax)   # the low-q specular tail leaves the frame, as in the NIKA figure
    ax.set_xlabel(r"Q vector (Å$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: _minor_label(v, _) if v < q_lim[1] else ""))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 4, 6, 8)))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: _minor_label(v, _) if round(v / 10 ** np.floor(np.log10(v) + 1e-9)) in (2, 4, 6) else ""))
    ax.tick_params(which="major", length=5, width=1.0)
    ax.tick_params(which="minor", length=2.8, width=0.8, labelsize=FS_MINOR)
    ax.grid(True, which="both", axis="x", color=C_GRID, ls=(0, (1.5, 1.5)), lw=0.5, alpha=0.7)
    ax.grid(True, which="major", axis="y", color=C_GRID, ls=(0, (1.5, 1.5)), lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    if legend:
        leg = ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black",
                        framealpha=1.0, handlelength=2.0, borderpad=0.4)
        leg.get_frame().set_linewidth(1.0)
    for txt, spec in labels.items():
        col, q0, fx = (*spec, 1.0) if len(spec) == 2 else spec
        y = cuts[col].to_numpy(float)
        ok = np.isfinite(y) & (y > 0)
        ax.text(q0 * fx, np.interp(np.log(q0), np.log(q[ok]), y[ok]) * 1.25, txt, ha="center", va="bottom",
                fontsize=FS_PEAK, fontweight="bold", zorder=5)
    return ax
