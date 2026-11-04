"""Full-box verification view of a reconstructed ZnPc film MD structure.

This module owns the check that the computed structure behind FIG4 panel (d)
is what the profiles claim: every molecule of the last dump frame is loaded,
its plane-normal tilt is measured, and the film is drawn in full depth
alongside the per-molecule tilt, the averaged ``gamma_eff(z)`` and ``rho(z)``
profiles, and the build targets stored in ``{tag}_meta.json``.

It reuses the dump and profile readers of ``fig4_md_strip`` and deliberately
excludes the oblique single-slab cutout styling used in the manuscript figure.

Invariants: depth follows the FIG4 convention (vacuum at zero, substrate at
``total_thick``), and bonds are inferred from distances on heavy atoms only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import fig4_md_strip as strip
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

DEFAULT_TOTAL_A = 188.75
HEAVY_CUT_A = 1.75
ZN_CUT_A = 2.2
SILICA_BAND_A = 8.0
SLAB_PAD_A = 6.0
TILT_CMAP = "viridis"


@dataclass(frozen=True)
class Molecule:
    """One ZnPc molecule with heavy-atom geometry relative to its centre.

    Attributes
    ----------
    rel
        Heavy-atom offsets from the molecular centre of mass, shape ``(n, 3)``,
        angstrom, ordered ``(x, y, depth)`` with depth increasing toward the
        substrate.
    bonds
        Index pairs into ``rel``, shape ``(m, 2)``.
    com
        Centre of mass ``(x, y, depth)`` with ``x`` and ``y`` wrapped into the
        box.
    gamma
        Angle in degrees between the ring-plane normal and the film normal.
    kinds
        Bond class per row of ``bonds``: 0 carbon-only, 1 touching nitrogen,
        2 touching zinc.
    normal
        Unit ring-plane normal ``(x, y, up)`` with a non-negative vertical
        component.
    """

    rel: np.ndarray
    bonds: np.ndarray
    com: np.ndarray
    gamma: float
    kinds: np.ndarray
    normal: np.ndarray


@dataclass(frozen=True)
class Frame:
    """Last-frame molecule list plus box edge lengths in angstrom."""

    molecules: list[Molecule]
    lx: float
    ly: float
    silica_depth: float


def load_frame(
    tag: str = strip.DEFAULT_TAG, *, total_thick: float = DEFAULT_TOTAL_A
) -> Frame:
    """Read every ZnPc molecule of the final dump frame of ``tag``.

    Parameters
    ----------
    tag
        Dump stem under ``md/znpc_film/bulk``.
    total_thick
        Graded film thickness in angstrom; sets the vacuum-at-zero depth.

    Returns
    -------
    Frame
        All molecules with wrapped centres, heavy-atom bonds, and tilt, and
        the depth of the silica top surface in the same convention.

    Raises
    ------
    FileNotFoundError
        If the dump file does not exist.
    RuntimeError
        If the dump contains no frames.
    """
    pt = strip._load_profile_tools()
    meta = strip.load_meta(tag)
    zref = float(meta["silica_top"]) + float(meta.get("d0", 2.0))
    box, data = strip.trailing_frames(tag)[-1]
    lx = float(box[0, 1] - box[0, 0])
    ly = float(box[1, 1] - box[1, 0])
    molecules: list[Molecule] = []
    for types, xyz, normal, gamma in pt.molecules(box, data):
        elems = np.array([strip.TYPE_TO_ELEM[int(t)] for t in types])
        heavy = elems != "H"
        pos = xyz[heavy].copy()
        els = elems[heavy]
        com = xyz.mean(axis=0)
        rel = pos - com
        rel_fig = np.column_stack([rel[:, 0], rel[:, 1], -rel[:, 2]])
        dm = np.linalg.norm(rel[:, None, :] - rel[None, :, :], axis=-1)
        zn = (els == "Zn")[:, None] | (els == "Zn")[None, :]
        ii, jj = np.nonzero(np.triu(dm < np.where(zn, ZN_CUT_A, HEAVY_CUT_A), k=1))
        any_n = (els == "N")[ii] | (els == "N")[jj]
        kinds = np.where(zn[ii, jj], 2, np.where(any_n, 1, 0))
        unit = np.asarray(normal, dtype=float)
        unit = unit if unit[2] >= 0.0 else -unit
        molecules.append(
            Molecule(
                rel=rel_fig,
                bonds=np.column_stack([ii, jj]),
                kinds=kinds,
                normal=unit,
                com=np.array([com[0] % lx, com[1] % ly, total_thick - (com[2] - zref)]),
                gamma=float(gamma),
            )
        )
    silica_depth = total_thick - (float(meta["silica_top"]) - zref)
    return Frame(molecules, lx, ly, silica_depth)


def slab_segments(
    frame: Frame,
    *,
    horizontal: int,
    slab_axis: int,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project molecules whose centre lies in a slab onto a depth plane.

    Parameters
    ----------
    frame
        Output of :func:`load_frame`.
    horizontal
        Box axis drawn horizontally: ``0`` for x, ``1`` for y.
    slab_axis
        Box axis the slab is cut along; must differ from ``horizontal``.
    lo, hi
        Slab limits in angstrom along ``slab_axis``; a molecule is kept whole
        when its wrapped centre of mass satisfies ``lo <= c < hi``.

    Returns
    -------
    segments, gamma
        Bond segments ``(m, 2, 2)`` as ``(horizontal, depth)`` pairs, with one
        periodic image on each side of the box, and the per-segment tilt
        ``(m,)`` in degrees.
    """
    length = frame.lx if horizontal == 0 else frame.ly
    segs: list[np.ndarray] = []
    gam: list[np.ndarray] = []
    for mol in frame.molecules:
        if not lo <= mol.com[slab_axis] < hi:
            continue
        for image in (-1, 0, 1):
            h = mol.com[horizontal] + image * length + mol.rel[:, horizontal]
            pts = np.column_stack([h, mol.com[2] + mol.rel[:, 2]])
            segs.append(np.stack([pts[mol.bonds[:, 0]], pts[mol.bonds[:, 1]]], axis=1))
            gam.append(np.full(len(mol.bonds), mol.gamma))
    return np.concatenate(segs), np.concatenate(gam)


def _draw_slab(  # noqa: PLR0913
    ax,
    frame: Frame,
    *,
    horizontal: int,
    slab_axis: int,
    lo: float,
    hi: float,
    norm: Normalize,
    linewidth: float,
) -> int:
    segs, gam = slab_segments(
        frame, horizontal=horizontal, slab_axis=slab_axis, lo=lo, hi=hi
    )
    cmap = colormaps[TILT_CMAP]
    ax.add_collection(
        LineCollection(segs, colors=cmap(norm(gam)), linewidths=linewidth, alpha=0.9)
    )
    length = frame.lx if horizontal == 0 else frame.ly
    ax.axvspan(
        0.0,
        length,
        facecolor="none",
        edgecolor="#444444",
        linewidth=0.8,
        linestyle="--",
    )
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.axhspan(
        frame.silica_depth,
        frame.silica_depth + SILICA_BAND_A,
        facecolor="#b9b9b9",
        edgecolor="#444444",
        linewidth=0.8,
    )
    ax.set_xlim(-SLAB_PAD_A, length + SLAB_PAD_A)
    return len(segs)


def render_full(
    tag: str = strip.DEFAULT_TAG,
    *,
    out: Path,
    total_thick: float = DEFAULT_TOTAL_A,
    slab_width_a: float = 20.0,
    dpi: int = 300,
) -> dict[str, float]:
    """Write the full-box verification figure for ``tag``.

    Panels share one depth axis (vacuum at the top, silica at the bottom):
    per-molecule tilt against the averaged ``gamma_eff`` and the build targets,
    density, an x-depth slab, and a y-depth slab. Bonds are coloured by the
    tilt of their molecule.

    Parameters
    ----------
    tag
        Dump stem under ``md/znpc_film/bulk``.
    out
        Destination PNG path.
    total_thick
        Graded film thickness in angstrom.
    slab_width_a
        Width in angstrom of the two slabs.
    dpi
        Raster resolution of the saved PNG.

    Returns
    -------
    dict
        Counts and extents: ``n_molecules``, ``lx``, ``ly``, ``depth_min``,
        ``depth_max``, ``gamma_mean``.
    """
    frame = load_frame(tag, total_thick=total_thick)
    meta = strip.load_meta(tag)
    z_fig, gamma_eff, rho = strip.md_profiles(tag, total_thick=total_thick)
    depth_mol = np.array([m.com[2] for m in frame.molecules])
    gamma_mol = np.array([m.gamma for m in frame.molecules])
    norm = Normalize(vmin=0.0, vmax=90.0)

    ylim = (frame.silica_depth + SILICA_BAND_A + 2.0, -10.0)
    span = ylim[0] - ylim[1]
    height_in = 9.0
    scale = height_in / span
    widths = [
        1.7,
        1.3,
        (frame.lx + 2 * SLAB_PAD_A) * scale,
        (frame.ly + 2 * SLAB_PAD_A) * scale,
    ]
    gap_in, left_in, bottom_in, right_in = 0.15, 0.8, 0.7, 1.1
    fig_w = left_in + sum(widths) + gap_in * (len(widths) - 1) + right_in
    fig = plt.figure(figsize=(fig_w, height_in + bottom_in + 0.4))
    axes = []
    x_in = left_in
    for w in widths:
        axes.append(
            fig.add_axes(
                (
                    x_in / fig_w,
                    bottom_in / fig.get_figheight(),
                    w / fig_w,
                    height_in / fig.get_figheight(),
                ),
                sharey=axes[0] if axes else None,
            )
        )
        x_in += w + gap_in
    ax_g, ax_r, ax_x, ax_y = axes

    ax_g.scatter(
        gamma_mol,
        depth_mol,
        s=1.2,
        c="#8aa1b8",
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    ax_g.plot(
        gamma_eff,
        z_fig,
        color="#c0392b",
        linewidth=1.6,
        label=r"$\gamma_\mathrm{eff}(z)$ MD",
    )
    targets = [
        (total_thick - float(lam["z_pxr"]), float(lam["gamma"]))
        for lam in meta["lamellae"]
        if "gamma" in lam
    ]
    ax_g.scatter(
        [g for _, g in targets],
        [z for z, _ in targets],
        marker="D",
        s=22,
        facecolor="none",
        edgecolor="black",
        linewidth=0.9,
        label="build target",
        zorder=5,
    )
    ax_g.set_xlim(0.0, 90.0)
    ax_g.set_xlabel(r"$\gamma$ (deg)")
    ax_g.set_ylabel("depth from vacuum (A)")
    ax_g.legend(loc="lower left", fontsize=7, frameon=False)

    ax_r.plot(rho, z_fig, color="#1f6f8b", linewidth=1.6)
    ax_r.set_xlim(0.0, max(2.0, float(np.max(rho)) * 1.05))
    ax_r.set_xlabel(r"$\rho$ (g cm$^{-3}$)")

    n_x = _draw_slab(
        ax_x,
        frame,
        horizontal=0,
        slab_axis=1,
        lo=0.0,
        hi=slab_width_a,
        norm=norm,
        linewidth=0.45,
    )
    ax_x.set_xlabel(f"x (A), slab y < {slab_width_a:.0f} A")
    n_y = _draw_slab(
        ax_y,
        frame,
        horizontal=1,
        slab_axis=0,
        lo=0.0,
        hi=slab_width_a,
        norm=norm,
        linewidth=0.45,
    )
    ax_y.set_xlabel(f"y (A), slab x < {slab_width_a:.0f} A")
    ax_x.text(
        0.5 * frame.lx,
        frame.silica_depth + 0.5 * SILICA_BAND_A,
        "silica",
        ha="center",
        va="center",
        fontsize=8,
    )
    ax_y.text(
        0.5 * frame.ly,
        frame.silica_depth + 0.5 * SILICA_BAND_A,
        "silica",
        ha="center",
        va="center",
        fontsize=8,
    )
    for ax in (ax_r, ax_x, ax_y):
        ax.tick_params(labelleft=False)
    ax_g.set_ylim(*ylim)

    cax = fig.add_axes((1.0 - 0.6 / fig_w, 0.3, 0.1 / fig_w, 0.4))
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=TILT_CMAP),
        cax=cax,
        label=r"molecule $\gamma$ (deg)",
    )
    fig.suptitle(
        f"{tag}: {len(frame.molecules)} molecules, "
        f"box {frame.lx:.1f} x {frame.ly:.1f} A, {n_x} + {n_y} bonds drawn",
        fontsize=9,
    )
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return {
        "n_molecules": float(len(frame.molecules)),
        "lx": frame.lx,
        "ly": frame.ly,
        "depth_min": float(depth_mol.min()),
        "depth_max": float(depth_mol.max()),
        "gamma_mean": float(gamma_mol.mean()),
    }


def all_bonds(frame: Frame, *, total_thick: float) -> tuple[np.ndarray, np.ndarray]:
    """Collect every bond of every molecule in height-up box coordinates.

    Parameters
    ----------
    frame
        Output of :func:`load_frame`.
    total_thick
        Graded film thickness in angstrom; the vacuum plane sits at height
        ``total_thick`` and larger depth maps to lower height.

    Returns
    -------
    segments, gamma
        Bond endpoints ``(m, 2, 3)`` as ``(x, y, height)`` in angstrom with no
        periodic images (exactly the simulated molecules), and the tilt of the
        owning molecule ``(m,)`` in degrees.
    """
    segs: list[np.ndarray] = []
    gam: list[np.ndarray] = []
    for mol in frame.molecules:
        pts = np.column_stack(
            [
                mol.com[0] + mol.rel[:, 0],
                mol.com[1] + mol.rel[:, 1],
                total_thick - (mol.com[2] + mol.rel[:, 2]),
            ]
        )
        segs.append(np.stack([pts[mol.bonds[:, 0]], pts[mol.bonds[:, 1]]], axis=1))
        gam.append(np.full(len(mol.bonds), mol.gamma))
    return np.concatenate(segs), np.concatenate(gam)


def project_orthographic(
    xyz: np.ndarray, *, azim: float, elev: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project height-up points orthographically for a viewing direction.

    Parameters
    ----------
    xyz
        Points ``(..., 3)`` as ``(x, y, height)``.
    azim
        Rotation of the box about the vertical axis in degrees.
    elev
        Elevation of the camera above the horizon in degrees; positive values
        look down on the film.

    Returns
    -------
    screen, distance
        Screen coordinates ``(..., 2)`` with the vertical axis up, and the
        distance along the line of sight ``(...)``; larger is farther away.
    """
    a, e = np.radians(azim), np.radians(elev)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    u = x * np.cos(a) + y * np.sin(a)
    w = -x * np.sin(a) + y * np.cos(a)
    screen = np.stack([u, z * np.cos(e) + w * np.sin(e)], axis=-1)
    return screen, w * np.cos(e) - z * np.sin(e)


def render_full_3d(  # noqa: PLR0913
    tag: str = strip.DEFAULT_TAG,
    *,
    out: Path,
    total_thick: float = DEFAULT_TOTAL_A,
    azim: float = 32.0,
    elev: float = 24.0,
    dpi: int = 400,
    figsize: tuple[float, float] = (10.0, 11.0),
    linewidth: float = 0.3,
) -> dict[str, float]:
    """Draw every simulated molecule of the last frame as a 3D wireframe.

    Bonds are painted far to near with a depth-dependent alpha and coloured
    by the tilt of their molecule. The simulation box is outlined, the silica
    slab is drawn as a grey block, and no periodic images or cut-aways are
    used, so the figure contains exactly the molecules that were simulated.

    Parameters
    ----------
    tag
        Dump stem under ``md/znpc_film/bulk``.
    out
        Destination PNG path.
    total_thick
        Graded film thickness in angstrom.
    azim, elev
        Viewing angles in degrees, see :func:`project_orthographic`.
    dpi
        Raster resolution of the saved PNG.
    figsize
        Figure size in inches.
    linewidth
        Bond line width in points.

    Returns
    -------
    dict
        ``n_molecules``, ``n_bonds`` and ``n_atoms`` (heavy atoms drawn).
    """
    frame = load_frame(tag, total_thick=total_thick)
    segs, gam = all_bonds(frame, total_thick=total_thick)
    screen, dist = project_orthographic(segs, azim=azim, elev=elev)
    mid = dist.mean(axis=1)
    order = np.argsort(-mid)
    t = (mid - mid.min()) / (mid.max() - mid.min())
    norm = Normalize(vmin=0.0, vmax=90.0)
    rgba = colormaps[TILT_CMAP](norm(gam))
    rgba[:, 3] = 0.95 - 0.8 * t**0.7
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((0.0, 0.0, 1.0, 0.96))
    z_si = total_thick - frame.silica_depth
    z_top = total_thick
    lx, ly = frame.lx, frame.ly

    def corner(x: float, y: float, z: float) -> np.ndarray:
        return project_orthographic(np.array([x, y, z]), azim=azim, elev=elev)[0]

    slab_bottom = z_si - SILICA_BAND_A
    top = [corner(*c, z_si) for c in ((0, 0), (lx, 0), (lx, ly), (0, ly))]
    low = [corner(*c, slab_bottom) for c in ((0, 0), (lx, 0), (lx, ly), (0, ly))]
    sides = (
        (0, 1, (0.0, -1.0), "#8f949a"),
        (1, 2, (1.0, 0.0), "#80858b"),
        (2, 3, (0.0, 1.0), "#8f949a"),
        (3, 0, (-1.0, 0.0), "#80858b"),
    )
    a_rad = np.radians(azim)
    for i, j, (nx, ny), colour in sides:
        if -nx * np.sin(a_rad) + ny * np.cos(a_rad) >= 0.0:
            continue
        arr = np.array([low[i], low[j], top[j], top[i]])
        ax.fill(
            arr[:, 0],
            arr[:, 1],
            facecolor=colour,
            edgecolor="#3d4247",
            lw=0.6,
            zorder=0,
        )
    arr = np.array(top)
    ax.fill(
        arr[:, 0],
        arr[:, 1],
        facecolor="#d6d6d6",
        edgecolor="#3d4247",
        lw=0.8,
        zorder=0.5,
    )

    ax.add_collection(
        LineCollection(
            screen[order], colors=rgba[order], linewidths=linewidth, zorder=2
        )
    )

    z_vac = z_top
    ceiling = [corner(*c, z_vac) for c in ((0, 0), (lx, 0), (lx, ly), (0, ly))]
    ring = np.array([*ceiling, ceiling[0]])
    ax.plot(ring[:, 0], ring[:, 1], color="#5f7f93", lw=0.9, ls="--", zorder=3)
    for k in range(4):
        pair = np.array([top[k], ceiling[k]])
        ax.plot(pair[:, 0], pair[:, 1], color="#5f7f93", lw=0.6, ls=":", zorder=3)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.03)
    ax.set_axis_off()
    cax = fig.add_axes((0.90, 0.40, 0.015, 0.2))
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=TILT_CMAP),
        cax=cax,
        label=r"molecule $\gamma$ (deg)",
    )
    n_atoms = int(sum(len(m.rel) for m in frame.molecules))
    fig.suptitle(
        f"{tag}: all {len(frame.molecules)} molecules, {n_atoms} heavy atoms, "
        f"{len(segs)} bonds; box {lx:.1f} x {ly:.1f} A, film {total_thick:.1f} A",
        fontsize=9,
        y=0.995,
    )
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return {
        "n_molecules": float(len(frame.molecules)),
        "n_bonds": float(len(segs)),
        "n_atoms": float(n_atoms),
    }


def write_interactive_html(
    tag: str = strip.DEFAULT_TAG,
    *,
    out: Path,
    total_thick: float = DEFAULT_TOTAL_A,
) -> dict[str, float]:
    """Write a rotatable WebGL view of every simulated molecule.

    Parameters
    ----------
    tag
        Dump stem under ``md/znpc_film/bulk``.
    out
        Destination HTML path; the plotly runtime is embedded so the file
        works offline.
    total_thick
        Graded film thickness in angstrom.

    Returns
    -------
    dict
        ``n_molecules`` and ``n_bonds`` drawn.
    """
    frame = load_frame(tag, total_thick=total_thick)
    segs, gam = all_bonds(frame, total_thick=total_thick)
    pts = np.full((len(segs), 3, 3), np.nan, dtype=np.float32)
    pts[:, :2] = segs
    flat = pts.reshape(-1, 3)
    lines = go.Scatter3d(
        x=flat[:, 0],
        y=flat[:, 1],
        z=flat[:, 2],
        mode="lines",
        line={
            "color": np.repeat(gam, 3).astype(np.float32),
            "colorscale": "Viridis",
            "cmin": 0.0,
            "cmax": 90.0,
            "width": 2,
            "colorbar": {"title": "molecule gamma (deg)", "len": 0.5},
        },
        hoverinfo="skip",
        name=tag,
    )
    z_si = total_thick - frame.silica_depth
    z_lo = z_si - SILICA_BAND_A
    corners = np.array(
        [
            (x, y, z)
            for z in (z_lo, z_si)
            for x in (0.0, frame.lx)
            for y in (0.0, frame.ly)
        ]
    )
    tri = np.array(
        [
            *[(0, 1, 3), (0, 3, 2), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4)],
            *[(2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 3, 7), (1, 7, 5)],
        ]
    )
    silica = go.Mesh3d(
        x=corners[:, 0],
        y=corners[:, 1],
        z=corners[:, 2],
        i=tri[:, 0],
        j=tri[:, 1],
        k=tri[:, 2],
        color="#9aa0a6",
        opacity=1.0,
        name="silica",
        hoverinfo="skip",
    )
    vacuum = go.Mesh3d(
        x=[0.0, frame.lx, frame.lx, 0.0],
        y=[0.0, 0.0, frame.ly, frame.ly],
        z=[total_thick] * 4,
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color="#9fb8c8",
        opacity=0.15,
        name="vacuum surface",
        hoverinfo="skip",
    )
    fig = go.Figure([lines, silica, vacuum])
    fig.update_layout(
        title=f"{tag}: all {len(frame.molecules)} molecules, {len(segs)} bonds",
        scene={
            "aspectmode": "data",
            "xaxis_title": "x (A)",
            "yaxis_title": "y (A)",
            "zaxis_title": "height above silica reference (A)",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        showlegend=False,
    )
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    return {"n_molecules": float(len(frame.molecules)), "n_bonds": float(len(segs))}


def main() -> None:
    """Render the verification and full 3D figures for each tag on the CLI.

    Usage: ``python fig4_md_full.py [TAG ...]``; defaults to ``REC_L3c``, the
    column-thinned run that replaces ``REC_L3`` once relaxed.
    """
    here = Path(__file__).parent
    for tag in sys.argv[1:] or ["REC_L3c"]:
        out = here / f"FIG4_md_full_check_{tag}.png"
        print(tag, render_full(tag, out=out))  # noqa: T201
        out3 = here / f"FIG4_md_full_3d_{tag}.png"
        print(tag, render_full_3d(tag, out=out3))  # noqa: T201
        html = here / f"FIG4_md_full_3d_{tag}.html"
        print(tag, write_interactive_html(tag, out=html))  # noqa: T201


if __name__ == "__main__":
    main()
