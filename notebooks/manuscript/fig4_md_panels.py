"""Structure painters for FIG4 panel (d), drawn on a shared depth axis.

This module owns how the last frame of a ZnPc reconstruction is drawn into an
axes whose vertical coordinate is film depth in nanometres (vacuum at the top,
the substrate below), so that the panel can sit flush against the ``gamma(z)``
and ``rho(z)`` panels and share their ``y`` limits and ticks exactly.

Two painters are provided:

``draw_wire``
    Every simulated molecule as a thin ball-less wireframe in an oblique
    orthographic view. Depth is never rescaled, so a horizontal line at depth
    ``z`` in this panel is the same depth as in the neighbouring panels for
    the centre plane of the box; nearer and farther atoms shift by
    ``tan(elev)`` times their distance from that plane.

``draw_cell_ticks``
    The relaxed alpha cell itself, seen along the column axis: lamellae at the
    ``d200`` period, the cell's slip along ``b``, the two herringbone classes in
    successive columns (front in colour, behind greyed), tilt and occupancy
    from the fitted orientation and density profiles, no MD frame involved.

``draw_ticks``
    One tick per molecule in a slab, drawn at its true ring-plane tilt
    ``gamma`` from the horizontal and leaning to the side its plane normal
    leans to, so the herringbone families read as mirrored ticks and the tilt
    is read directly against depth.

It excludes axis furniture (depth rules, zone tints, substrate bar); the
notebook owns those so all three depth panels share one implementation.

Invariants: coordinates passed to matplotlib are ``(horizontal nm, depth nm)``
with depth increasing downward, and the caller sets the inverted ``y`` limits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import fig4_md_full as full
import fig4_md_strip as strip
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

A2NM = 0.1
ZN_COLOR = "#8e55b8"
TICK_COLOR = "#23323b"
MAX_AZIM_DEG = 40.0
CELL_PATH = strip.BULK / "relaxed_cell_g73L.json"
BEHIND_STYLE = ((0.60, 0.80), (0.72, 0.55), (0.80, 0.40))   # (grey level, alpha) of the columns behind the front one
# Slightly oblique view: each column behind the front one is offset by this (x, depth) in
# angstrom, about a 10 deg azimuth and 6 deg elevation over the c/2 column separation, so the
# columns behind sit between the front ticks instead of hidden directly behind them.
VIEW_SHIFT_A = (2.0, -1.2)
BEHIND_WIDTH = 0.75                                          # line width and Zn size of those columns, relative to the front
BEHIND_ZN_MIX = 0.75
# Global multiplier on every drawn disorder term (libration, interfacial orientational
# spread, positional jitter, vacancy sag); 1 is the published drawing, 0 a perfect lattice.
DISORDER_SCALE = 1.0
# Substrate-side vacancies. In the rows within SUBSTRATE_ZONE_TAU * tau_si of the substrate
# the vacancy fraction 1 - p is multiplied by SUBSTRATE_VACANCY_SCALE (1 keeps the occupancy
# the fitted density implies) and, with VACANCY_ISLANDS, vacancies are laterally correlated
# so the contact rows read as islands of partial coverage rather than random holes.
SUBSTRATE_VACANCY_SCALE = 1.0
# Contact lamella. Every molecule of the row next to the substrate rests on it: its lowest
# point sits CONTACT_GAP_A above the substrate (half the ring thickness), whatever its tilt.
# Along that row molecules are slid sideways until no two ring traces come closer than
# CLASH_A, so a lying ring takes the room it needs instead of being cut through by a neighbour.
CONTACT_GAP_A = 1.7
# Along that row islands are recentred on where they were drawn, so the extra width of lying
# rings spreads into the neighbouring holes, and a ring that still collides is dropped.
# Over each lying ring the SETTLE_ROWS lamellae above sag into the room it leaves: a smooth
# Gaussian dip as wide as the ring, as deep as the room over it (at most MAX_FALL_A), full in
# the first lamella and tapering linearly to zero above, so stacks bend together instead of
# single molecules dropping through their neighbours. A trace that cuts a lying ring by more
# than SETTLE_OVERLAP_A is lifted clear of it. SETTLE_GAP_A is the trace to trace clearance.
SETTLE_ROWS = 2
MAX_FALL_A = 8.0
SETTLE_GAP_A = 1.5
SETTLE_OVERLAP_A = 0.5   # interpenetration of a lying ring tolerated before the molecule above is lifted
FLAT_TZ_FRACTION = 0.35   # a contact ring counts as lying when its depth half extent is below this x half diameter (gamma < ~20 deg)
CLASH_A = 2.4   # below the 3.3 A pi stacking, so only rings that would interpenetrate are resolved
SUBSTRATE_ZONE_TAU = 1.0   # the contact lamella and the one above it; rows above keep uncorrelated vacancies
VACANCY_ISLANDS = True
ISLAND_WAVENUMBERS = np.array([4.0, 6.3, 9.1])
# The data fix only rho(z) and <cos^2 gamma>, not how they are arranged laterally, so the
# contact row shows two readings side by side at the same density. For x < ISLAND_SPLIT_NM the
# molecules form face-on islands (cofacial stacks, see _stack_islands) separated by holes, with
# the vacancies of the lamella above lined up over them; beyond it the vacancies are dispersed
# at random (sparse coverage) and the tilt spread is recentred so the whole row still has the
# fitted <cos^2 gamma>, which makes the dispersed half more upright.
ISLAND_SPLIT_NM = 0.0
ISLAND_TILT_DEG = 8.0   # islands are face-on (lying) rings stacked cofacially
ISLAND_TILT_SPREAD_DEG = 4.0   # tilt scatter within one island
ISLAND_MAX_LEVELS = 3   # pi stacks laid on top of each other within one island
# Columns behind the front one are drawn fainter within BEHIND_FADE_NM of the substrate, so a
# hole in the front contact row reads as a hole instead of being filled in by grey.
BEHIND_FADE_NM = 2.0
BEHIND_FADE_ALPHA = 0.3
LIBRATION_DEG = 1.5
AMORPHOUS_DEG_VAC = 5.0
AMORPHOUS_DEG_SI = 20.0   # Si-side tilt spread; the row mean <cos^2> stays at the fit (see _spread_center)
JITTER_A_BASE = 0.25
JITTER_A_VAC = 0.55
JITTER_A_SI = 0.85
VACANCY_SAG_A = 2.0
# Where the fitted density exceeds the crystal (vacuum side, density_vac > rho_crystal) the
# lattice is either contracted isotropically (True) or the occupancy is capped at 1 (False).
# GIWAXS shows one lamellar spacing and no compressed component, so the cap is the default.
CONTRACT_DENSE = False
# Where sin(gamma) < n_b the ring normal cannot keep its crystal stack-axis component. True
# widens the stack pitch so the pi-plane spacing stays at the crystal value and raises the
# occupancy by the same factor, so the number density still follows the fit; False keeps the
# pitch and lets the pi-plane spacing shrink to b sin(gamma).
CONSERVE_PI = True
# Surface terraces. In rows the roughness mask reaches (reference depth within
# SURFACE_ZONE * surface_roughness of the nominal surface), a molecule is kept only if it
# belongs to a run of at least MIN_TERRACE consecutive molecules along its row, so the free
# surface ends in terraces rather than isolated molecules floating above the film.
SURFACE_ZONE = 2.0
MIN_TERRACE = 3
# The surface itself (which molecules survive the roughness mask) uses only the longest
# wavelengths of the corrugation, so exposed lamellae form wide terraces with the same
# coverage statistics instead of scattered single molecules.
SURFACE_WAVENUMBERS = np.array([1.0, 1.7, 2.6])
CORRUGATION_FRACTION = 0.5
CORRUGATION_FLOOR = 0.15
CORRUGATION_WAVENUMBERS = np.array([1.0, 1.7, 2.6, 3.9, 5.3, 7.1])


@dataclass(frozen=True)
class PanelGeometry:
    """Drawable area of an axes in nanometres.

    Attributes
    ----------
    half_width_nm
        Half of the horizontal extent at equal nm-per-inch in both directions.
    nm_per_pt
        Nanometres of depth per typographic point of axes height.
    """

    half_width_nm: float
    nm_per_pt: float


@lru_cache(maxsize=4)
def cached_frame(tag: str, total_thick: float) -> full.Frame:
    """Load ``tag`` once per process; see :func:`fig4_md_full.load_frame`."""
    return full.load_frame(tag, total_thick=total_thick)


def panel_geometry(ax, depth_lim_nm: tuple[float, float]) -> PanelGeometry:
    """Measure ``ax`` so horizontal and vertical nanometres are equal.

    Parameters
    ----------
    ax
        Axes already placed in its figure (gridspec cell resolved).
    depth_lim_nm
        Depth limits in drawing order; only their span is used.

    Returns
    -------
    PanelGeometry
        Half width and depth scale of the axes.
    """
    fig = ax.get_figure()
    pos = ax.get_position()
    width_in = pos.width * fig.get_figwidth()
    height_in = pos.height * fig.get_figheight()
    span = abs(depth_lim_nm[1] - depth_lim_nm[0])
    return PanelGeometry(
        half_width_nm=0.5 * span * width_in / height_in,
        nm_per_pt=span / (height_in * 72.0),
    )


def fit_azimuth(lx: float, ly: float, width_a: float) -> float:
    """Largest view azimuth whose projected box width fits ``width_a``.

    Parameters
    ----------
    lx, ly
        Box edge lengths in angstrom.
    width_a
        Available horizontal extent in angstrom.

    Returns
    -------
    float
        Azimuth in degrees in ``[0, MAX_AZIM_DEG]``; ``0`` when even the
        face-on view is wider than ``width_a``.
    """
    grid = np.linspace(0.0, MAX_AZIM_DEG, 401)
    rad = np.radians(grid)
    ok = lx * np.cos(rad) + ly * np.sin(rad) <= width_a
    return float(grid[ok].max()) if ok.any() else 0.0


def _bond_arrays(
    frame: full.Frame, total_thick: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(segments, kinds, centres, tilt)`` in ``(x, y, depth)`` angstrom."""
    segs, kinds, centres, tilt = [], [], [], []
    for mol in frame.molecules:
        pts = mol.com + mol.rel
        segs.append(np.stack([pts[mol.bonds[:, 0]], pts[mol.bonds[:, 1]]], axis=1))
        kinds.append(mol.kinds)
        centres.append(mol.com)
        tilt.append(mol.gamma)
    del total_thick
    return (
        np.concatenate(segs),
        np.concatenate(kinds),
        np.array(centres),
        np.array(tilt),
    )


def draw_wire(  # noqa: PLR0913
    ax,
    tag: str,
    *,
    total_thick: float,
    depth_lim_nm: tuple[float, float],
    elev: float = 9.0,
    azim: float | None = None,
    linewidth: float = 0.32,
    near_alpha: float = 0.95,
    far_alpha: float = 0.10,
    margin_nm: float = 0.25,
) -> dict[str, float]:
    """Draw every simulated molecule as a wireframe on a depth-nm axes.

    Parameters
    ----------
    ax
        Target axes; the caller sets limits, ticks, and decorations.
    tag
        Dump stem under ``md/znpc_film/bulk``.
    total_thick
        Graded film thickness in angstrom (vacuum at depth zero).
    depth_lim_nm
        Depth limits in drawing order; used for the equal-scale measurement.
    elev
        Camera elevation in degrees. Far atoms shift upward by
        ``tan(elev)`` per angstrom of distance behind the central plane.
    azim
        View azimuth in degrees; when ``None`` the largest azimuth for which
        the whole box fits the panel width is used.
    linewidth
        Bond line width in points.
    near_alpha, far_alpha
        Bond opacity at the near and far faces of the box.
    margin_nm
        Horizontal clearance kept between the box silhouette and the frame.

    Returns
    -------
    dict
        ``azim``, ``elev``, ``n_bonds``, ``n_molecules`` of the drawing.
    """
    frame = cached_frame(tag, total_thick)
    geo = panel_geometry(ax, depth_lim_nm)
    width_a = 2.0 * (geo.half_width_nm - margin_nm) / A2NM
    az = fit_azimuth(frame.lx, frame.ly, width_a) if azim is None else azim
    segs, kinds, _, _ = _bond_arrays(frame, total_thick)
    a, e = np.radians(az), np.radians(elev)
    xc, yc = 0.5 * frame.lx, 0.5 * frame.ly
    dx, dy = segs[..., 0] - xc, segs[..., 1] - yc
    u = dx * np.cos(a) + dy * np.sin(a)
    w = -dx * np.sin(a) + dy * np.cos(a)
    row = segs[..., 2] - np.tan(e) * w
    xy = np.stack([u * A2NM, row * A2NM], axis=-1)
    wm = w.mean(axis=1)
    order = np.argsort(-wm)
    t = (wm - wm.min()) / (wm.max() - wm.min())
    rgb = np.array([strip.BOND_RGB_3D[int(k)] for k in range(3)])[kinds]
    rgba = np.column_stack([rgb, near_alpha - (near_alpha - far_alpha) * t**0.8])
    ax.add_collection(
        LineCollection(
            xy[order],
            colors=rgba[order],
            linewidths=linewidth,
            zorder=3,
            capstyle="round",
        )
    )
    ax.set_xlim(-geo.half_width_nm, geo.half_width_nm)
    return {
        "azim": az,
        "elev": elev,
        "n_bonds": float(len(segs)),
        "n_molecules": float(len(frame.molecules)),
    }


def densest_slab_start(com_axis: np.ndarray, length: float, width: float) -> float:
    """Start coordinate of the periodic window of ``width`` holding most centres.

    Parameters
    ----------
    com_axis
        Wrapped centre coordinates along the slab axis, shape ``(n,)``.
    length
        Periodic box length along that axis.
    width
        Slab thickness.

    Returns
    -------
    float
        Window start in ``[0, length)`` maximising the enclosed count.
    """
    starts = np.arange(0.0, length, 0.5)
    counts = [int((((com_axis - s) % length) < width).sum()) for s in starts]
    return float(starts[int(np.argmax(counts))])


def draw_ticks(  # noqa: PLR0913
    ax,
    tag: str,
    *,
    total_thick: float,
    depth_lim_nm: tuple[float, float],
    slab_a: float | None = 9.0,
    n_columns: int = 14,
    overlay_alpha: float = 0.3,
    tick_len_a: float = 12.0,
    linewidth: float = 1.1,
    zn_size: float = 2.2,
    margin_nm: float = 0.25,
) -> dict[str, float]:
    """Draw one ring-plane tick per molecule of a y-slab at its true tilt.

    A molecule with plane normal tilted ``gamma`` from vertical has a ring
    plane tilted ``gamma`` from horizontal. The tick lies along the steepest
    in-plane direction, leaning toward ``-sign(n_x)`` so molecules of the two
    herringbone families lean opposite ways. Viewing along ``y`` is the bulk
    mirror axis of the herringbone, so the picture needs no further rotation.

    Parameters
    ----------
    ax
        Target axes; the caller sets limits, ticks, and decorations.
    tag
        Dump stem under ``md/znpc_film/bulk``.
    total_thick
        Graded film thickness in angstrom (vacuum at depth zero).
    depth_lim_nm
        Depth limits in drawing order.
    slab_a
        Window width along ``y`` in angstrom, re-centred on the same
        pi-stack column at every depth (see
        :func:`fig4_md_strip.column_following_mask`); molecules are kept whole
        by centre of mass. ``None`` draws every molecule of the box instead,
        each tick at ``overlay_alpha``, so columns that share a position stack
        to full opacity and a lamella with whole-column vacancies reads as
        fainter rather than as a void.
    n_columns
        Number of pi-stack columns along ``y`` in the periodic box.
    overlay_alpha
        Opacity of one tick when ``slab_a`` is ``None``; ignored otherwise.
    tick_len_a
        Tick length in angstrom (about the ring-plane diameter).
    linewidth
        Tick width in points.
    zn_size
        Zinc marker size in points.
    margin_nm
        Horizontal clearance between the outermost tick and the frame.

    Returns
    -------
    dict
        ``n_ticks``, ``slab_start`` and ``gamma_mean`` of the drawing.
    """
    frame = cached_frame(tag, total_thick)
    geo = panel_geometry(ax, depth_lim_nm)
    com = np.array([m.com for m in frame.molecules])
    gamma = np.radians([m.gamma for m in frame.molecules])
    normal = np.array([m.normal for m in frame.molecules])
    if slab_a is None:
        keep = np.ones(len(com), dtype=bool)
        alpha = overlay_alpha
    else:
        keep = strip.column_following_mask(
            com, frame.ly, n_columns=n_columns, width=slab_a
        )
        alpha = 1.0
    start = float(np.median(com[keep, 1]))
    half_a = (geo.half_width_nm - margin_nm) / A2NM
    n_img = int(np.ceil(half_a / frame.lx)) + 1
    lean = -np.sign(normal[:, 0])
    lean[lean == 0.0] = 1.0
    half = 0.5 * tick_len_a
    x_tick = lean * np.cos(gamma) * half
    z_tick = np.sin(gamma) * half
    x0 = com[:, 0] - 0.5 * frame.lx
    segs = []
    zn_xy = []
    for k in range(-n_img, n_img + 1):
        xs = x0[keep] + k * frame.lx
        inside = np.abs(xs) <= half_a
        xs = xs[inside]
        zs = com[keep, 2][inside]
        tx, tz = x_tick[keep][inside], z_tick[keep][inside]
        segs.append(
            np.stack(
                [
                    np.column_stack([xs - tx, zs + tz]),
                    np.column_stack([xs + tx, zs - tz]),
                ],
                axis=1,
            )
        )
        zn_xy.append(np.column_stack([xs, zs]))
    segs_a = np.concatenate(segs) * A2NM
    zn = np.concatenate(zn_xy) * A2NM
    ax.add_collection(
        LineCollection(
            segs_a,
            colors=to_rgba(TICK_COLOR, alpha),
            linewidths=linewidth,
            zorder=3,
            capstyle="round",
        )
    )
    ax.scatter(
        zn[:, 0],
        zn[:, 1],
        s=zn_size**2,
        c=ZN_COLOR,
        alpha=alpha,
        linewidths=0,
        zorder=4,
    )
    ax.set_xlim(-geo.half_width_nm, geo.half_width_nm)
    return {
        "n_ticks": float(len(segs_a)),
        "slab_start": start,
        "gamma_mean": float(np.degrees(gamma[keep]).mean()),
    }


@dataclass(frozen=True)
class CellSite:
    """One ZnPc site of the relaxed alpha cell as seen along the in-plane column axis.

    Attributes
    ----------
    parity
        0 for the lamella at fractional ``a = 0`` and 1 for ``a = 1/2``.
    column
        0 for the column at fractional ``c = 0`` (in front) and 1 for ``c = 1/2``.
    b_frac
        Fractional position along the stack axis ``b``.
    lean
        Lean sense of the ring-plane trace, ``-sign(n_a)``, so the two
        herringbone classes lean opposite ways.
    n_b
        Component of the ring normal along the stack axis (unit normal).
    """

    parity: int
    column: int
    b_frac: float
    lean: float
    n_b: float


def load_cell(path: Path = CELL_PATH) -> tuple[dict, dict[tuple[int, int], CellSite]]:
    """Read the MD-relaxed alpha cell and index its four sites by lamella and column.

    Parameters
    ----------
    path
        ``relaxed_cell_g73L.json`` written by the NPT cell relaxation; supplies
        the edges ``a, b, c``, ``d200``, ``pi_spacing``, ``rho`` and the four
        site-averaged molecules with fractional position and ring normal.

    Returns
    -------
    tuple
        The parsed cell dictionary and a map ``(parity, column) -> CellSite``.

    Raises
    ------
    ValueError
        If the four sites do not cover both lamellae and both columns once.
    """
    cell = json.loads(Path(path).read_text())
    sites: dict[tuple[int, int], CellSite] = {}
    for site in cell["sites"]:
        a_frac, b_frac, c_frac = site["frac"]
        n_a, n_b, _ = site["normal"]
        key = (round(2.0 * (a_frac % 1.0)) % 2, round(2.0 * (c_frac % 1.0)) % 2)
        sites[key] = CellSite(
            parity=key[0],
            column=key[1],
            b_frac=float(b_frac % 1.0),
            lean=-float(np.sign(n_a)),
            n_b=float(abs(n_b)),
        )
    if sorted(sites) != [(0, 0), (0, 1), (1, 0), (1, 1)]:
        raise ValueError(
            f"cell sites do not tile two lamellae and two columns: {sorted(sites)}"
        )
    return cell, sites


def _tick_half_extent(
    gamma_rad: np.ndarray, n_b_crystal: float, half: float
) -> tuple[np.ndarray, np.ndarray]:
    """Lateral and depth half extents of the ring-plane trace at polar tilt ``gamma``.

    The ring normal keeps its crystal component along the stack axis (so the
    pi-plane spacing ``b * n_b`` is conserved) unless the tilt is too flat to
    allow it, where the whole in-plane remainder goes to ``n_b``.
    """
    n_a = np.cos(gamma_rad)
    n_b = np.minimum(n_b_crystal, np.sin(gamma_rad))
    norm = np.hypot(n_a, n_b)
    return half * np.abs(n_a) / norm, half * n_b / norm


def _spread_center(gamma_deg: float, sigma_deg: float) -> float:
    """Centre of a Gaussian tilt spread that keeps the row's <cos^2 gamma> at cos^2 gamma_fit.

    P-RSoXR fixes only <cos^2 gamma> at each depth. A spread of width sigma about the fitted
    angle would bias that average, so the centre mu is shifted to satisfy
    <cos^2(mu + e)> = (1 + cos(2 mu) exp(-2 sigma^2)) / 2 = cos^2 gamma_fit, with e ~ N(0, sigma).
    """
    s = np.radians(sigma_deg)
    c = (2.0 * np.cos(np.radians(gamma_deg)) ** 2 - 1.0) / np.exp(-2.0 * s * s)
    return float(np.degrees(0.5 * np.arccos(np.clip(c, -1.0, 1.0))))


def _segment_distance(p0, p1, q0, q1) -> float:
    """Shortest distance between the 2D segments p0-p1 and q0-q1 (0 if they cross)."""
    def point_seg(a, b, c):
        ab = b - a
        t = np.clip(np.dot(c - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
        return float(np.hypot(*(a + t * ab - c)))

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    if ccw(p0, q0, q1) != ccw(p1, q0, q1) and ccw(p0, p1, q0) != ccw(p0, p1, q1):
        return 0.0
    return min(point_seg(p0, p1, q0), point_seg(p0, p1, q1),
               point_seg(q0, q1, p0), point_seg(q0, q1, p1))


def _pack_contact(x, tx, tz, *, floor: float, clear: float, x_max: float, step: float = 0.2):
    """Seat one row on a flat floor and resolve ring clashes along it.

    Every molecule rests with its lowest point at ``floor`` (depth increases downward). Runs of
    molecules separated by vacancies are packed in order: a molecule is slid toward +x, in
    steps of ``step``, until its trace is at least ``clear`` from the traces already placed, so
    a flat ring takes the room it needs instead of being cut through. Each run is then shifted back so its mean position is unchanged, and
    where two runs now touch the later molecule of the clashing pair is dropped. Returns the
    new x, the depth of each centre, and a mask of molecules kept inside ``x_max``.
    """
    x0 = np.asarray(x, dtype=float)
    o = np.argsort(x0, kind="stable")
    x_new = x0.copy()
    z = floor - np.asarray(tz, dtype=float)

    def seg(k):
        return (np.array([x_new[k] - tx[k], z[k] + tz[k]]), np.array([x_new[k] + tx[k], z[k] - tz[k]]))

    xs = x0[o]
    dx = np.diff(xs)
    brk = np.flatnonzero(dx > 1.8 * np.median(dx)) + 1 if dx.size else np.array([], int)
    for run in np.split(o, brk):
        placed: list[int] = []
        for i in run:
            c = clear
            while x_new[i] <= x_max + 50.0:
                p0, p1 = seg(i)
                hit = False
                for j in placed[-4:]:
                    if abs(x_new[i] - x_new[j]) > abs(tx[i]) + abs(tx[j]) + c:
                        continue
                    if _segment_distance(p0, p1, *seg(j)) < c:
                        hit = True
                        break
                if not hit:
                    break
                x_new[i] += step
            placed.append(int(i))
        x_new[run] -= (x_new[run] - x0[run]).mean()
    keep = np.ones(x0.size, dtype=bool)
    last = None
    for i in np.argsort(x_new, kind="stable"):
        if last is not None and _segment_distance(*seg(i), *seg(last)) < clear:
            keep[i] = False
            continue
        last = i
    return x_new, z, keep & (np.abs(x_new) <= x_max)


def _stack_islands(x, *, floor, n_b, half, lean, pi_spacing, height, rng):
    """Condense the contact molecules at lattice sites ``x`` into face-on, stacked islands.

    Runs of ``x`` separated by vacancies become islands of lying rings (tilt
    ISLAND_TILT_DEG with ISLAND_TILT_SPREAD_DEG scatter). Rings are stacked cofacially,
    ``pi_spacing`` apart along their normal, in columns as tall as ``height`` allows (at most
    ISLAND_MAX_LEVELS); columns stand side by side, CLASH_A apart. Each island keeps the
    molecule count and mean position of its run, so the lateral room the stacking saves
    opens as holes. Returns x, z, tx, tz (angstrom) and the islands' mean cos^2 gamma.
    """
    o = np.sort(np.asarray(x, dtype=float))
    gaps = np.diff(o)
    cut = np.flatnonzero(gaps > 1.8 * np.median(gaps)) + 1 if gaps.size else np.array([], int)
    out, cos2 = [], []
    for run in np.split(o, cut):
        g = np.radians(np.clip(np.abs(ISLAND_TILT_DEG + rng.normal(0.0, ISLAND_TILT_SPREAD_DEG, run.size)), 1.0, 89.0))
        tx, tz = _tick_half_extent(g, n_b, half)
        tx = lean * tx
        cos2.append(np.cos(g) ** 2)
        mtx, mtz = float(np.mean(tx)), float(np.mean(tz))
        nx, nz = mtz, mtx
        nrm = np.hypot(nx, nz)
        nx, nz = nx / nrm, nz / nrm
        if nz > 0:                                    # normal pointing away from the substrate
            nx, nz = -nx, -nz
        levels = int(np.clip((height - 2.0 * tz.max()) // (pi_spacing * abs(nz)) + 1, 1, ISLAND_MAX_LEVELS))
        step = 2.0 * float(np.abs(tx).max()) + CLASH_A + (levels - 1) * pi_spacing * abs(nx)
        idx = np.arange(run.size)
        col, lev = idx // levels, idx % levels
        xr = col * step + lev * pi_spacing * nx
        xr += run.mean() - xr.mean()
        zr = floor - tz + lev * pi_spacing * nz
        out.append((xr, zr, tx, tz))
    xs, zs, txs, tzs = (np.concatenate(c) for c in zip(*out))
    return xs, zs, txs, tzs, float(np.concatenate(cos2).mean())


def _no_clash(x, z, tx, tz, *, clear, group):
    """Keep mask that drops, pair by pair, the later of two traces closer than ``clear``.

    Pairs inside the same nonzero ``group`` (one stacked island) were placed together and are
    not tested against each other.
    """
    keep = np.ones(x.size, dtype=bool)
    order = np.argsort(x, kind="stable")
    for a_, i in enumerate(order):
        p0 = np.array([x[i] - tx[i], z[i] + tz[i]]); p1 = np.array([x[i] + tx[i], z[i] - tz[i]])
        for j in order[max(0, a_ - 8):a_]:
            if not keep[j] or (group[i] and group[j]):
                continue
            if abs(x[i] - x[j]) > abs(tx[i]) + abs(tx[j]) + clear:
                continue
            q0 = np.array([x[j] - tx[j], z[j] + tz[j]]); q1 = np.array([x[j] + tx[j], z[j] - tz[j]])
            if _segment_distance(p0, p1, q0, q1) < clear:
                keep[i] = False
                break
    return keep


def _room(x, z, tx, tz, below, flat, *, gap: float, width: float, n_pts: int = 21):
    """Vertical room (angstrom, depth down) from each trace to ``gap`` above the row ``below``.

    The support under a point at x is the shallowest point of the row below within ``width``,
    so a molecule rests on its neighbours below rather than slipping between their traces.
    Negative room is interpenetration. Also returns whether the limiting trace is ``flat``.
    """
    room = np.full(x.size, np.inf)
    on_flat = np.zeros(x.size, dtype=bool)
    if len(below) == 0:
        return room, on_flat
    t = np.linspace(0.0, 1.0, 15)
    bx = (below[:, 0, 0][:, None] + (below[:, 1, 0] - below[:, 0, 0])[:, None] * t).ravel()
    bz = (below[:, 0, 1][:, None] + (below[:, 1, 1] - below[:, 0, 1])[:, None] * t).ravel()
    bflat = np.repeat(np.asarray(flat, dtype=bool), t.size)
    u = np.linspace(-1.0, 1.0, n_pts)
    for i in range(x.size):
        px = x[i] + u * tx[i]
        pz = z[i] - u * tz[i]
        near = np.abs(px[:, None] - bx[None, :]) <= width
        if not near.any():
            continue
        g = np.where(near, bz[None, :] - gap - pz[:, None], np.inf)
        j = np.unravel_index(np.argmin(g), g.shape)
        room[i], on_flat[i] = g[j], bflat[j[1]]
    return room, on_flat


def _drop_short_runs(mask: np.ndarray, n_min: int) -> np.ndarray:
    """``mask`` with every run of consecutive True shorter than ``n_min`` set False."""
    out = mask.copy()
    start = None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start < n_min:
                out[start:i] = False
            start = None
    return out


def _contraction(z: float, z_a, rho_sharp, rho_crystal: float) -> float:
    """Isotropic lattice contraction where the fitted density exceeds the crystal (1 unless CONTRACT_DENSE)."""
    if not CONTRACT_DENSE:
        return 1.0
    return float(max(1.0, np.interp(z, z_a, rho_sharp) / rho_crystal) ** (-1.0 / 3.0))


def _pi_stretch(gamma_deg, n_b: float):
    """Factor on the stack pitch that keeps the pi-plane spacing ``b n_b`` (1 unless CONSERVE_PI)."""
    s = np.sin(np.radians(np.asarray(gamma_deg, dtype=float)))
    return n_b / np.minimum(n_b, s) if CONSERVE_PI else np.ones_like(s)


def _row_depths(  # noqa: PLR0913
    *,
    total_thick: float,
    z_a: np.ndarray,
    gamma_deg: np.ndarray,
    rho_sharp: np.ndarray,
    d_row: float,
    rho_crystal: float,
    n_b_max: float,
    half: float,
    top_nm: float,
) -> list[float]:
    """Reference depths of the lamellar rows, laid from the silicon outward.

    Shared by :func:`draw_cell_ticks` and :func:`tick_spacings`, so the spacings
    reported for the GIWAXS comparison are those of the drawn rows.
    """
    g_si = np.radians(np.interp(total_thick, z_a, gamma_deg))
    z_row = total_thick - float(_tick_half_extent(g_si, n_b_max, half)[1])
    rows: list[float] = []
    while z_row * A2NM > top_nm:
        rows.append(z_row)
        z_row -= d_row * _contraction(z_row, z_a, rho_sharp, rho_crystal)
    return rows


def tick_spacings(  # noqa: PLR0913
    *,
    total_thick: float,
    depth_lim_nm: tuple[float, float],
    z_a: np.ndarray,
    gamma_deg: np.ndarray,
    rho_sharp: np.ndarray,
    tau_si: float,
    tau_vac: float,
    tick_len_a: float = 13.0,
    cell_path: Path = CELL_PATH,
) -> dict[str, np.ndarray | float]:
    """Per-row spacings of the :func:`draw_cell_ticks` drawing, for the GIWAXS check.

    Takes the same profile arguments as :func:`draw_cell_ticks` and evaluates
    the same rows without the random draw. For row ``k`` at depth ``z_k`` the
    lamellar period is ``d200 * c(z_k)`` and the pi-plane spacing is
    ``b * c(z_k) * k(z_k) * min(n_b, sin gamma(z_k))``, with ``c`` the density
    contraction (see CONTRACT_DENSE), ``k`` the pitch stretch (see CONSERVE_PI)
    and ``n_b`` the crystal stack-axis component of the ring normal (site mean),
    which is the spacing the drawn ticks imply.

    Returns
    -------
    dict
        Arrays ``z`` (A), ``gamma`` (deg), ``d_lam`` and ``d_pi`` (A), ``occupancy``
        (``rho_sharp / rho_crystal`` clipped to [0, 1]), ``sigma_pos`` (A, the
        positional jitter of the drawing), ``molecules_per_a`` (occupancy over
        the row pitch, the scattering weight before a Debye-Waller factor), and
        the cell scalars ``d200``, ``pi_spacing``, ``n_b``.
    """
    cell, sites = load_cell(cell_path)
    pitch, d_row, rho_crystal = float(cell["b"]), float(cell["d200"]), float(cell["rho"])
    n_b_max = max(s.n_b for s in sites.values())
    n_b_mean = float(np.mean([s.n_b for s in sites.values()]))
    rows = np.asarray(
        _row_depths(
            total_thick=total_thick,
            z_a=z_a,
            gamma_deg=gamma_deg,
            rho_sharp=rho_sharp,
            d_row=d_row,
            rho_crystal=rho_crystal,
            n_b_max=n_b_max,
            half=0.5 * tick_len_a,
            top_nm=min(depth_lim_nm),
        )
    )
    z_clip = np.clip(rows, 0.0, total_thick)
    c = np.array([_contraction(z, z_a, rho_sharp, rho_crystal) for z in z_clip])
    gamma = np.interp(z_clip, z_a, gamma_deg)
    k = _pi_stretch(gamma, n_b_mean)
    occ = np.clip(np.interp(rows, z_a, rho_sharp) / rho_crystal * k, 0.0, 1.0)
    sub = total_thick - z_clip < SUBSTRATE_ZONE_TAU * tau_si
    occ = np.where(sub, np.clip(1.0 - SUBSTRATE_VACANCY_SCALE * (1.0 - occ), 0.0, 1.0), occ)
    amorph_si = np.exp(-(total_thick - z_clip) / tau_si)
    amorph_vac = np.exp(-z_clip / tau_vac)
    return {
        "z": rows,
        "gamma": gamma,
        "d_lam": d_row * c,
        "d_pi": pitch * c * k * np.minimum(n_b_mean, np.sin(np.radians(gamma))),
        "occupancy": occ,
        "sigma_pos": DISORDER_SCALE * (JITTER_A_BASE + JITTER_A_VAC * amorph_vac + JITTER_A_SI * amorph_si),
        "molecules_per_a": occ / (pitch * c * k),
        "d200": d_row,
        "pi_spacing": float(cell["pi_spacing"]),
        "n_b": n_b_mean,
    }


def draw_cell_ticks(  # noqa: PLR0913, PLR0915
    ax,
    *,
    total_thick: float,
    depth_lim_nm: tuple[float, float],
    z_a: np.ndarray,
    gamma_deg: np.ndarray,
    rho_sharp: np.ndarray,
    tau_si: float,
    tau_vac: float,
    surface_roughness: float,
    n_layers: int = 2,
    tick_len_a: float = 13.0,
    linewidth: float = 1.1,
    zn_size: float = 2.2,
    margin_nm: float = 0.25,
    seed: int = 7,
    cell_path: Path = CELL_PATH,
) -> dict[str, float]:
    """Draw the film as the relaxed alpha cell seen along the column axis, with depth.

    Lamellae are laid from the silicon outward at the true ``d200`` period.
    Along the stack axis ``b`` molecules sit at the pitch and slip of the cell
    (the two lamellae of the cell differ by about half a pitch), and each
    molecule is a ring-plane tick at its polar tilt ``gamma(z)`` from the
    fitted orientation profile. Columns alternate the two herringbone classes
    along the viewing direction: the front column is drawn in full colour and
    the columns behind it are greyed progressively, in the same positions the
    cell puts them, so the mirrored class is visible behind the front one.

    The analytic fitted density (without the vacuum roughness smear) sets
    occupancy, ``p(z) = k(z) rho_sharp(z) / rho_crystal`` clipped to [0, 1], so a
    molecule is missing with probability ``1 - p``. Where ``sin gamma < n_b`` and
    CONSERVE_PI is set, the stack pitch is widened by ``k = n_b / sin gamma`` so the
    pi-plane spacing stays at the crystal ``b n_b`` (flat-lying rings cannot stack
    closer than that), and the occupancy is raised by the same ``k`` so the number
    density still follows the fit. Where the fit is denser than the crystal
    (``rho_sharp > rho_crystal``) the occupancy saturates at 1, or, with
    CONTRACT_DENSE, the lattice is contracted by ``(rho_crystal / rho_sharp)^(1/3)``.
    The return value ``min_pi_spacing`` reports the smallest pi-plane spacing drawn.
    The vacuum interface is a laterally correlated
    height field ``h(x)`` with zero mean and RMS ``surface_roughness``; a
    molecule exists only where it lies deeper than ``h(x)``, so lamellar rows
    terminate in terraces and the laterally averaged density reproduces the
    Nevot-Croce error-function step of panel (c) instead of a hard cut.
    Disorder follows the film model: libration and positional jitter grow into
    both amorphous interface zones of widths ``tau_vac`` and ``tau_si``, the
    lamellae undulate with the same height field, and molecules next to
    vacancies sag vertically.

    Parameters
    ----------
    ax
        Target axes; the caller sets limits, ticks and decorations.
    total_thick
        Film thickness in angstrom (vacuum at depth zero).
    depth_lim_nm
        Depth limits in drawing order, inverted so depth increases downward.
    z_a, gamma_deg, rho_sharp
        Depth grid in angstrom (increasing), polar tilt in degrees and analytic
        density in g cm^-3 before the vacuum roughness smear, all of one
        length, from the film model.
    tau_si, tau_vac
        Decay lengths of the amorphous zones in angstrom.
    surface_roughness
        RMS roughness of the vacuum interface in angstrom.
    n_layers
        Number of columns along the viewing direction, front first.
    tick_len_a
        Ring-plane diameter drawn as the tick length, in angstrom.
    linewidth
        Tick width in points.
    zn_size
        Zinc marker size in points.
    margin_nm
        Horizontal clearance between the outermost tick and the frame.
    seed
        Seed of the disorder draw.
    cell_path
        Relaxed cell JSON, see :func:`load_cell`.

    Returns
    -------
    dict
        ``n_ticks`` (front column), ``n_rows``, the cell numbers used and
        ``min_pi_spacing`` (angstrom) after contraction.
    """
    cell, sites = load_cell(cell_path)
    geo = panel_geometry(ax, depth_lim_nm)
    rng = np.random.default_rng(seed)
    pitch, d_row, rho_crystal = (
        float(cell["b"]),
        float(cell["d200"]),
        float(cell["rho"]),
    )
    half_a = (geo.half_width_nm - margin_nm) / A2NM
    half = 0.5 * tick_len_a
    top_nm = min(depth_lim_nm)
    n_b_max = max(s.n_b for s in sites.values())
    phase = rng.uniform(0.0, 2.0 * np.pi, CORRUGATION_WAVENUMBERS.size)

    def raw_corrugation(x: np.ndarray) -> np.ndarray:
        arg = (
            2.0 * np.pi * np.outer(x + half_a, CORRUGATION_WAVENUMBERS) / (2.0 * half_a)
        )
        return np.sin(arg + phase).sum(axis=1)

    grid = raw_corrugation(np.linspace(-half_a, half_a, 1024, endpoint=False))
    grid_mean, grid_std = float(grid.mean()), float(grid.std())

    def corrugation(x: np.ndarray) -> np.ndarray:
        return (raw_corrugation(x) - grid_mean) / grid_std

    # separate generator, so the rest of the drawing keeps its random stream
    phase_s = np.random.default_rng(seed + 1).uniform(0.0, 2.0 * np.pi, SURFACE_WAVENUMBERS.size)

    def raw_surface(x: np.ndarray) -> np.ndarray:
        arg = 2.0 * np.pi * np.outer(x + half_a, SURFACE_WAVENUMBERS) / (2.0 * half_a)
        return np.sin(arg + phase_s).sum(axis=1)

    s_grid = raw_surface(np.linspace(-half_a, half_a, 1024, endpoint=False))
    s_mean, s_std = float(s_grid.mean()), float(s_grid.std())

    def surface_height(x: np.ndarray) -> np.ndarray:
        """Unit-variance height of the free surface, long wavelengths only."""
        return (raw_surface(x) - s_mean) / s_std

    rng_i = np.random.default_rng(seed + 2)
    rng_t = np.random.default_rng(seed + 3)   # island tilts, kept off the main stream
    phase_i = rng_i.uniform(0.0, 2.0 * np.pi, (256, ISLAND_WAVENUMBERS.size))   # one phase set per row index
    i_grid = np.linspace(-half_a, half_a, 1024, endpoint=False)

    def island_draw(x: np.ndarray, row: int) -> np.ndarray:
        """Laterally correlated stand-in for ``rng.random``, uniform on [0, 1] across the row."""
        def raw(xx):
            arg = 2.0 * np.pi * np.outer(xx + half_a, ISLAND_WAVENUMBERS) / (2.0 * half_a)
            return np.sin(arg + phase_i[row]).sum(axis=1)
        ref = np.sort(raw(i_grid))
        return np.searchsorted(ref, raw(x)) / ref.size   # empirical CDF, so the kept fraction is p

    def contraction(z: float) -> float:
        return _contraction(z, z_a, rho_sharp, rho_crystal)

    rows = _row_depths(
        total_thick=total_thick,
        z_a=z_a,
        gamma_deg=gamma_deg,
        rho_sharp=rho_sharp,
        d_row=d_row,
        rho_crystal=rho_crystal,
        n_b_max=n_b_max,
        half=half,
        top_nm=top_nm,
    )

    segs: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    n_contact = [0] * n_layers
    flat_below = [np.zeros(0, dtype=bool) for _ in range(n_layers)]
    sag_src = [(np.zeros(0), np.zeros(0), None) for _ in range(n_layers)]
    n_sag = [0] * n_layers
    cos2_check: list = []
    n_lift_drop = [0] * n_layers
    n_contact_drop = [0] * n_layers
    zns: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    n_b_mean = float(np.mean([s.n_b for s in sites.values()]))
    min_pi = np.inf
    for k, z_ref in enumerate(rows):
        z_clip = float(np.clip(z_ref, 0.0, total_thick))
        gamma_row = float(np.interp(z_clip, z_a, gamma_deg))
        stretch = float(_pi_stretch(gamma_row, n_b_mean))
        pitch_row = pitch * contraction(z_clip) * stretch
        min_pi = min(min_pi, pitch_row * min(n_b_mean, float(np.sin(np.radians(gamma_row)))))
        n_j = int(np.ceil(half_a / pitch_row)) + 1
        amorph_si = float(np.exp(-(total_thick - z_clip) / tau_si))
        amorph_vac = float(np.exp(-z_clip / tau_vac))
        sigma_gamma = DISORDER_SCALE * (
            LIBRATION_DEG
            + AMORPHOUS_DEG_VAC * amorph_vac
            + AMORPHOUS_DEG_SI * amorph_si
        )
        sigma_pos = DISORDER_SCALE * (JITTER_A_BASE + JITTER_A_VAC * amorph_vac + JITTER_A_SI * amorph_si)
        substrate_row = total_thick - z_clip < SUBSTRATE_ZONE_TAU * tau_si
        corr_amp = (
            CORRUGATION_FRACTION
            * surface_roughness
            * (CORRUGATION_FLOOR + (1.0 - CORRUGATION_FLOOR) * amorph_vac)
        )
        for layer in range(n_layers):
            site = sites[(k % 2, layer % 2)]
            x = (np.arange(-n_j, n_j + 1) + site.b_frac) * pitch_row + layer * VIEW_SHIFT_A[0]
            x = x + rng.normal(0.0, sigma_pos, x.size)
            height = corrugation(x)
            z = z_ref + layer * VIEW_SHIFT_A[1] + corr_amp * height + rng.normal(0.0, sigma_pos, x.size)
            p_keep = np.clip(np.interp(z, z_a, rho_sharp) / rho_crystal * stretch, 0.0, 1.0)
            if substrate_row:
                p_keep = np.clip(1.0 - SUBSTRATE_VACANCY_SCALE * (1.0 - p_keep), 0.0, 1.0)
            z = z + DISORDER_SCALE * rng.normal(0.0, VACANCY_SAG_A, x.size) * (1.0 - p_keep)
            # |.| folds tilts through the film normal, so flat-lying (face-on) rings stay possible
            mu = _spread_center(gamma_row, sigma_gamma)
            gamma = np.radians(
                np.clip(np.abs(mu + rng.normal(0.0, sigma_gamma, x.size)), 1.0, 89.0)
            )
            tx, tz = _tick_half_extent(gamma, site.n_b, half)
            z = np.minimum(z, total_thick - tz)
            draw = rng.random(x.size)   # always drawn, so the random stream is the same either way
            if substrate_row and VACANCY_ISLANDS:
                draw = np.where(x * A2NM < ISLAND_SPLIT_NM, island_draw(x, 0), draw)   # holes line up through the zone
            keep = (
                (draw < p_keep)
                & (np.abs(x) <= half_a)
                & (z * A2NM >= top_nm)
                & (z >= surface_roughness * surface_height(x))
            )
            if z_ref < SURFACE_ZONE * surface_roughness:
                keep = _drop_short_runs(keep, MIN_TERRACE)
            x, z, tx, tz = x[keep], z[keep], site.lean * tx[keep], tz[keep]
            lifted = np.zeros(x.size, dtype=bool)
            if k == 0:   # contact lamella: rest on the substrate, no ring cutting through another
                floor = total_thick - CONTACT_GAP_A + layer * VIEW_SHIFT_A[1]
                isl = VACANCY_ISLANDS & (x * A2NM < ISLAND_SPLIT_NM)
                if isl.sum() > 1 and (~isl).sum() > 1:
                    xi, zi, txi, tzi, c_isl = _stack_islands(
                        x[isl], floor=floor, n_b=site.n_b, half=half, lean=site.lean,
                        pi_spacing=float(cell["pi_spacing"]), height=d_row - CONTACT_GAP_A, rng=rng_t)
                    # the dispersed half carries the rest of the row's <cos^2 gamma>
                    f = isl.mean()
                    c_sp = (np.cos(np.radians(gamma_row)) ** 2 - f * c_isl) / (1.0 - f)
                    g_sp = float(np.degrees(np.arccos(np.sqrt(np.clip(c_sp, 0.0, 1.0)))))
                    mu_sp = _spread_center(g_sp, sigma_gamma)
                    cos2_check.append((layer, gamma_row, c_isl, float(c_sp), g_sp, f))
                    g = np.radians(np.clip(np.abs(mu_sp + rng_t.normal(0.0, sigma_gamma, (~isl).sum())), 1.0, 89.0))
                    txs, tzs = _tick_half_extent(g, site.n_b, half)
                    xs, zs, oks = _pack_contact(x[~isl], site.lean * txs, tzs, floor=floor, clear=CLASH_A, x_max=half_a)
                    x = np.concatenate([xi, xs]); z = np.concatenate([zi, zs])
                    tx = np.concatenate([txi, site.lean * txs]); tz = np.concatenate([tzi, tzs])
                    group = np.r_[np.ones(xi.size), np.zeros(xs.size)]
                    ok = np.concatenate([np.abs(xi) <= half_a, oks]) & _no_clash(x, z, tx, tz, clear=CLASH_A, group=group)
                else:
                    x, z, ok = _pack_contact(x, tx, tz, floor=floor, clear=CLASH_A, x_max=half_a)
                n_contact_drop[layer] += int((~ok).sum())
                n_contact[layer] += int(x.size)
                x, z, tx, tz = x[ok], z[ok], tx[ok], tz[ok]
            if k == 0:   # lying rings leave room above them; record where and how wide
                flat = tz < FLAT_TZ_FRACTION * half
                sag_src[layer] = (x[flat], np.abs(tx[flat]) + 0.5 * pitch_row, None)
            elif k <= SETTLE_ROWS and segs[layer]:
                room, on_flat = _room(x, z, tx, tz, segs[layer][-1] / A2NM, flat_below[layer],
                                      gap=SETTLE_GAP_A, width=pitch_row)
                xs, ws, amp = sag_src[layer]
                near = np.abs(x[:, None] - xs[None, :]) <= ws[None, :]
                if amp is None:   # first lamella above: the room left over each lying ring
                    amp = np.clip(np.where(near, room[:, None], np.inf).min(axis=0), 0.0, MAX_FALL_A)
                    amp[~np.isfinite(amp)] = 0.0
                    sag_src[layer] = (xs, ws, amp)
                f_k = 1.0 - (k - 1) / SETTLE_ROWS
                sag = np.zeros(x.size) if xs.size == 0 else f_k * (
                    amp[None, :] * np.exp(-0.5 * ((x[:, None] - xs[None, :]) / ws[None, :]) ** 2)).max(axis=1)
                dz = np.minimum(sag, np.maximum(room, 0.0))
                lift = on_flat & (room < -SETTLE_OVERLAP_A)    # nothing cuts through a contact ring
                dz[lift] = room[lift]
                z = z + dz
                n_sag[layer] += int((dz > 1.0).sum())
                # a molecule moved into its own lamella's neighbours leaves a vacancy instead
                bad = np.zeros(x.size, dtype=bool)
                for i in np.flatnonzero(np.abs(dz) > 0.05):
                    p0 = np.array([x[i] - tx[i], z[i] + tz[i]]); p1 = np.array([x[i] + tx[i], z[i] - tz[i]])
                    for j in np.flatnonzero((np.abs(x - x[i]) < np.abs(tx) + abs(tx[i]) + 1.0) & ~bad):
                        if j != i and _segment_distance(p0, p1, np.array([x[j] - tx[j], z[j] + tz[j]]),
                                                        np.array([x[j] + tx[j], z[j] - tz[j]])) < 1.0:
                            bad[i] = True
                            break
                n_lift_drop[layer] += int(bad.sum())
                x, z, tx, tz, lift = x[~bad], z[~bad], tx[~bad], tz[~bad], lift[~bad]
                lifted = lift
            # nothing in the lamella above cuts a contact ring, or a molecule lifted out of one
            flat_below[layer] = np.full(x.size, True) if k == 0 else lifted
            segs[layer].append(
                np.stack(
                    [
                        np.column_stack([x - tx, z + tz]),
                        np.column_stack([x + tx, z - tz]),
                    ],
                    axis=1,
                )
                * A2NM
            )
            zns[layer].append(np.column_stack([x, z]) * A2NM)

    zn_front = np.array(to_rgba(ZN_COLOR))
    n_front = 0
    for layer in reversed(range(n_layers)):
        seg = np.concatenate(segs[layer])
        zn = np.concatenate(zns[layer])
        if layer == 0:
            tick_rgba, zn_rgba, z_order = to_rgba(TICK_COLOR), zn_front, 3.0
            n_front = len(seg)
        else:
            grey, alpha = BEHIND_STYLE[min(layer, len(BEHIND_STYLE)) - 1]
            fade = np.where(total_thick * A2NM - seg[:, :, 1].mean(axis=1) < BEHIND_FADE_NM,
                            BEHIND_FADE_ALPHA, 1.0)
            tick_rgba = np.column_stack([np.full((fade.size, 3), grey), alpha * fade])
            zfade = np.where(total_thick * A2NM - zn[:, 1] < BEHIND_FADE_NM, BEHIND_FADE_ALPHA, 1.0)
            zn_rgba = np.column_stack([
                np.tile((1.0 - BEHIND_ZN_MIX) * zn_front[:3] + BEHIND_ZN_MIX * grey, (zfade.size, 1)),
                alpha * zfade,
            ])
            z_order = 1.0 + 0.1 * (n_layers - layer)
        ax.add_collection(
            LineCollection(
                seg,
                colors=[tick_rgba] if layer == 0 else tick_rgba,
                linewidths=linewidth * (1.0 if layer == 0 else BEHIND_WIDTH),
                zorder=z_order,
                capstyle="round",
            )
        )
        ax.scatter(
            zn[:, 0],
            zn[:, 1],
            s=(zn_size * (1.0 if layer == 0 else BEHIND_WIDTH)) ** 2,
            color=[zn_rgba] if layer == 0 else zn_rgba,
            linewidths=0,
            zorder=z_order + 0.05,
        )
    ax.set_xlim(-geo.half_width_nm, geo.half_width_nm)
    return {
        "n_ticks": float(n_front),
        "n_rows": float(len(rows)),
        "b": pitch,
        "d200": d_row,
        "pi_spacing": float(cell["pi_spacing"]),
        "min_pi_spacing": float(min_pi),
        "contact_clash_removed": float(n_contact_drop[0]),
        "contact_molecules": float(n_contact[0]),
        "sagged_molecules": float(n_sag[0]),
        "island_cos2": cos2_check,
        "lift_removed": float(n_lift_drop[0]),
    }
