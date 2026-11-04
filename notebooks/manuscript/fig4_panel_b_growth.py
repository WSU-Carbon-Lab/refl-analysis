"""FIG4 panel-(b) molecule packing (lamellar herringbone strip).

Source of truth for the structure strip. Lamellae lie parallel to the
substrate and are laid down from the buried interface outward; inside each
lamella molecules tile laterally along a slipped-stack column axis that lies
in the growth plane, so the only tilted object is the macrocycle itself. The
herringbone alternates between successive lamellae rather than within one.
Vacuum is left; Si is right.

Standalone::

    uv run python fig4_panel_b_growth.py

From ``fig_5_graded.ipynb``, call ``configure_packing`` then ``draw_structure``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
from numpy.typing import NDArray
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

SHOW_H = False
D_LAM = 14.2
PI_STACK = 3.4
L_MOL = 13.6
LAT_FRAC = 0.90
LIBRATION_DEG = 1.5
DISORDER_VAC_DEG = 5.0
DISORDER_SI_DEG = 12.0
POS_DISORDER_VAC = 0.55
POS_DISORDER_SI = 2.5  # depth scatter at the Si, fading over tau_Si into the bulk
STRAIN_MIN = 0.90
STRAIN_MAX = 1.35
STRAIN_ANISO_SI = 0.08
HERRINGBONE_MODE = "reflect"
HERRINGBONE_SPLAY_DEG = 28.9
GLIDE_FRAC = 0.5
TILT_FLOOR_DEG = 5.0
SIN_GAMMA_MIN = 0.35

# Packing. "crystal" (default) keeps the strip of the main branch (lamellae laid from
# Si outward, gamma(z) per lamella, density split between the two periods) with two
# changes. Every lamella carries the same herringbone class, as one column of the alpha
# cell does viewed along c (successive lamellae are related by the GLIDE_FRAC offset
# only, not by a mirror), and the cell it scales is the crystal one. The cell
# is PI_SPACING / n_b wide, so neighbouring ring planes sit PI_SPACING apart along their
# normal (n_b is the crystal stack-axis component N_B_CRYSTAL of the normal, or sin gamma
# where the tilt is too flat for it), and k_lam sin gamma deep, equal to the bulk period
# at the median tilt. Where rho(z) is below the density of that cell both periods dilate
# together, as on main; where it is above, only the lamellar period compresses, so the
# pi-plane spacing never closes below the crystal value. COLUMN_SLAB (c / 2 of the alpha
# cell) is the thickness one drawn column stands for, which makes the bulk lamellar
# period about 12.5 A. "main" reproduces the main-branch strip, whose bulk cell put the
# ring planes about 3.1 A apart.
PACKING = "crystal"  # "crystal" | "main"
PI_SPACING = 3.3215  # pi-plane spacing of the alpha cell (angstrom)
N_B_CRYSTAL = 0.848  # stack-axis component of the ring normal in the alpha cell
COLUMN_SLAB = 11.99  # c / 2 of the alpha cell (angstrom)
MAIN_CONTACT_ROWS = (
    1  # lamellae from the Si generated exactly as on main (the partial covering)
)
# The main-branch contact cell deepens with sin(gamma). A larger substrate tilt
# opens a gap under the crystal lamella above that partial row. Close this
# fraction of the free gap and leave CONTACT_GAP_CLEAR_A between the envelopes.
CONTACT_GAP_FILL = 1.0
CONTACT_GAP_CLEAR_A = 0.6
# After that register shift, the first crystal lamellae drop individually into
# whatever lateral room the contact rings leave. Atom centres stay this far
# apart, and only this fraction of the open drop is taken.
NESTLE_ROWS = 2
NESTLE_CLEAR_A = 1.8
# Gravity at the interface: each molecule of the first lamella above the contact patches
# falls until it meets the oxide (with its roughness allowance) or a molecule already
# placed, so over bare oxide it drops all the way down and stands between the lying
# patches; the second lamella falls at most 6 A into the holes of the first, and only
# the CARRY_QUANTILE of its drops carries the bulk down, so the holes fill without
# packing the zone above rho(z).
NESTLE_MAX_A = (30.0, 6.0)  # the second value also caps every settled lamella
NESTLE_FILL = (1.0, 1.0)
# The lamellae above the nestled ones, up to SETTLE_ZONE_TAU * tau_Si from the Si, then
# settle onto the lamella below each, by up to SETTLE_MAX_A and SETTLE_FILL of the open
# drop, so a dip in the lamella below is followed instead of leaving a void above it. The
# bulk above keeps its rho(z) lamellar register, carried down by the median settle of the
# last settled lamella, and the cap on the settle decays as exp(-dz / tau_Si) beyond the
# zone, so local dips are still followed but the bulk register is left alone.
SETTLE_FILL = 1.0
# quantile of a lamella's drops that carries the lamellae above it down: below 0.5 the
# molecules that fall into holes fill them without pulling the whole bulk after them
CARRY_QUANTILE = 0.1
SETTLE_MAX_A = 4.0
SETTLE_ZONE_TAU = 10.0
# Closed interfacial film and density check; see draw_structure. With FILM_DENSE the
# settled lamellae keep the crystal pitch, so the film has no voids, and DENSITY_CORRECT
# removes the resulting excess over the interfacial zone from the partial contact
# covering (keep fraction at least CONTACT_KEEP_MIN), the one place vacancies may sit.
FILM_DENSE = True
# The partial contact covering is drawn as closed patches of lying molecules: pi stacked
# as a slipped cofacial stack at CONTACT_TILT_DEG (ring planes PI_SPACING apart, tilt
# scatter CONTACT_SPREAD_DEG), with the vacancies grouped by a field with lateral periods
# CONTACT_PATCH_PERIODS_A into bare oxide between patches. False restores the
# main-branch contact row with random vacancies.
CONTACT_PATCHES = True
CONTACT_SPREAD_DEG = 4.0
CONTACT_JITTER_A = 0.3
CONTACT_PATCH_PERIODS_A = (22.0, 33.0, 51.0)  # short enough for several islands across the panel
PATCH_SEED = 2  # seed of the patch layout alone (None: SEED + 13); the rest of the strip is unchanged
# Lying contact patches are marked by a soft red halo (a PowerPoint-style glow) under
# the front column, with a thin white rim between halo and molecule.
PATCH_GLOW = True
PATCH_GLOW_COLOR = "red"  # the gradient-trace red
PATCH_GLOW_WIDTH_A = 6.5  # outer reach of the glow (line width, A)
PATCH_GLOW_STEPS = 20  # translucent layers, widths spaced for a Gaussian falloff
PATCH_GLOW_ALPHA = 0.05  # opacity per layer; the core gets about STEPS x ALPHA
PATCH_RIM_A = 1.5  # thin rim right around the molecule
PATCH_RIM_COLOR = "white"  # None draws the glow alone
# "glow" marks the islands with the halo above; "color" instead recolours their atoms and
# bonds with PATCH_PALETTES[PATCH_PALETTE] (element contrast kept, no halo).
PATCH_STYLE = "color"
PATCH_PALETTE = "red"
PATCH_PALETTES = {  # element colours and bond colour of the island molecules
    "green": {"C": "#238b45", "N": "#74c476", "Zn": "#00441b", "bond": "#1a6b35"},
    "red": {"C": "#cb181d", "N": "#fc9272", "Zn": "#67000d", "bond": "#a50f15"},
}
# CONTACT_TILT_DEG is the patch tilt (more lying than standing). LAST_DRAW
# ["contact_band_gamma"] reports the drawn and fitted cos^2-mean tilt over the first
# CONTACT_BAND_A from the Si, patches and the lamella nestled into them together.
CONTACT_TILT_DEG = 28.0
PATCH_SINK_A = 3.0  # how far the lying patches reach into the rough oxide (rms roughness 2.7 A)
CONTACT_BAND_A = 12.0
DENSITY_CORRECT = True
DENSITY_PASSES = 5
CONTACT_KEEP_MIN = 0.3
CONTACT_KEEP: dict[int, float] = {}
# The first standing lamella sandwiches onto the islands, so the first FILL_BAND_A from
# the Si would hold more molecules than rho(z) allows. The density passes leave out the
# filling-layer molecules right over the island cores (patch field below FILL_CUT)
# until that band follows rho(z); the next lamella then falls into the space.
FILL_BAND_A = 0.0  # 0: off; the cascade below fills the holes instead, at a low density
FILL_CUT: dict[int, float] = {}
# A lamella that settles ends up deeper than its register row, where rho(z) is lower and
# gamma(z) differs. The density passes record each lamella's median drop, and the next
# pass takes its lateral pitch and tilt from rho and gamma at the settled depth, so the
# lamellae can fall into the gaps without packing the interface above rho(z).
ROW_DROP: dict[int, float] = {}
# Cascade above the filling layer: a molecule with nothing of the lamella below within
# HOLE_WIDTH_A of it laterally sits over a hole and falls up to HOLE_MAX_A into it; one
# that is supported may fall CASCADE_DECAY of the median drop of the molecules under it
# plus an allowance that decays away from the Si, so the disorder seeded at the Si
# cascades up and dies out before the bulk.
HOLE_WIDTH_A = 4.0
HOLE_MAX_A = 20.0
CASCADE_DECAY = 0.7
CASCADE_BASE_A = 10.0  # extra fall allowed at the Si, decaying over CASCADE_TAU * tau_Si
CASCADE_TAU = 2.0
CRYSTAL_SENSE = -1  # herringbone sense of every other lamella of the front column
# Contact and clash handling (crystal packing). No drawn atom crosses the Si: a molecule
# whose lowest atom would sit closer than SUB_REST_A to the surface is lifted until it
# rests there, so the partial contact covering lies on the substrate rather than in it.
# No two molecules of one column come closer than CLASH_A, atom centre to atom centre
# (drawn C discs are 1.3 A wide, so 1.8 A leaves a visible margin; tighter values push apart lamellae that touch end to end in the crystal itself): a molecule that
# would is lifted away from the Si by up to MAX_LIFT_A, then nudged sideways by up to
# MAX_SLIDE_A, and otherwise left out as a vacancy. The two columns lie in different
# planes along the view, so they are not tested against each other.
SUB_REST_A = 0.6
# The native oxide is rough (rms 2.7 A in the graded fit, oxide "rough"), so a molecule
# may reach below the mean Si surface: each draws an allowance |N(0, SUB_ROUGH_A)|, capped
# at 2 SUB_ROUGH_A, measured from the surface itself. The contact covering then keeps the
# tilt it anchors with instead of being lifted wholesale. 0 restores the hard SUB_REST_A wall.
SUB_ROUGH_A = 2.7
# Free-surface terrace (topmost lamella) of the front column: True leaves out a random
# SURFACE_VACANCY_FRAC of its molecules, giving a broken surface; False fills it.
SURFACE_VACANCIES = True
SURFACE_VACANCY_FRAC = 0.15
SURFACE_WINDOW_A = 60.0  # depth over which the top lamellae are drawn together (0: off)
SURFACE_SQUEEZE_MIN = 0.85
CLASH_A = 1.8
MAX_LIFT_A = 2.5
MAX_SLIDE_A = 0.5
SLIDE_COST = 4.0  # a sideways move costs this many times a lift of the same size
# In-lamella disorder: the tilt deviation is TILT_CORR correlated along the lamella (smooth
# over 3 to 9 nm) and the lateral jitter is LATERAL_JITTER_FRAC of the depth jitter, so
# neighbours do not cut into each other or open gaps; the spread in depth is unchanged.
TILT_CORR = 0.9
LATERAL_JITTER_FRAC = 0.3
# Inside the film (every lamella above the partial contact covering) a molecule is never
# left out, so no void opens there: the search widens to FILM_LIFT_A and FILM_SLIDE_A, and
# as a last resort accepts FILM_CLASH_MIN_A, where the drawn discs just meet. Vacancies are
# therefore confined to the partial covering on the Si.
FILM_LIFT_A = 8.0
FILM_SLIDE_A = 1.0
FILM_CLASH_MIN_A = 1.35
LAST_DRAW: dict = {}   # molecules of the last draw_structure call, for diagnostics
DRAW_BEHIND = True  # the column behind (c = 1/2): same generation, opposite tilt
BEHIND_ALPHA = (
    0.25  # chosen from a sweep: legible behind the front column without crowding it
)
BEHIND_GREY = "0.70"
BEHIND_SHIFT_A = (
    -1.2,
    2.0,
)  # (depth, lateral) offset of the column behind, an oblique view

# Substrate exclusion, added for the interface comparison. "off" reproduces the
# published strip byte for byte and is the default. "clip" rejects any molecule
# whose drawn footprint crosses into the substrate; the test sits after the
# random draws for that molecule, so every other molecule keeps the random
# numbers it had and the two renders differ only by the molecules removed.
# "shift" additionally seeds the lamellar register where a tilted molecule can
# sit flush against the substrate, so the first row is thinned rather than
# deleted wholesale. That moves every row, so the whole strip redraws.
SUBSTRATE_MODE = "off"  # "off" | "clip" | "shift" | "tilt"
TILT_SEARCH_N = 60  # grid points for the substrate tilt clamp
SUB_CLEAR = 0.0  # extra clearance, angstrom, e.g. a drawn atomic radius
SUB_LIFT = 0.0  # extra clearance when "tilt" has to translate a molecule
SLAB_THICK = 14.0
ATOMIC_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "Zn": 65.38}
N_AVOGADRO = 6.02214076e23
CORR_FRAC = 0.5
CORR_FLOOR = 0.35
SEED = 17
ALIGN_EPS = 1e-8
N_BULK_Z_MIN = 0.25

ELEM_COLORS = {"C": "#6e6e6e", "N": "#3b55d0", "Zn": "#9a63c0"}
ELEM_RADIUS = {"C": 0.60, "N": 0.66, "Zn": 1.00}
BOND_COLOR = "#585858"
ATOM_DRAW_SCALE = 1.07
BOND_WIDTH_A = 0.38

XYZ = HERE / "tpznpc_0m.xyz"
OUT_PNG = HERE / "FIG4_panel_b_replay.png"
OUT_PDF = HERE / "FIG4_panel_b_replay.pdf"
COMPARE_PNG = HERE / "FIG4_panel_b_vs_41.png"
REF_FIG41 = HERE / "FIG4.1.png"

Z_LEFT = -6.0
PANEL_B_HEIGHT_IN = 1.035
TAU_COLOR = "#0b7f8a"
TAU_LW = 0.85
TAU_LS = (0, (4, 3))

Z_G: NDArray[np.float64] | None = None
RHO_MAT: NDArray[np.float64] | None = None
ALPHA_G: NDArray[np.float64] | None = None
TOTAL = 0.0
TAU_SI = 0.0
TAU_VAC = 0.0
SURF_ROUGH = 0.0
RHO_BULK_REF = 1.0
MOLAR_MASS = 0.0
DEPTH_XLIM: tuple[float, float] = (Z_LEFT, 0.0)

ELS: NDArray[np.str_] | None = None
MOL: NDArray[np.float64] | None = None
BONDS: list[tuple[int, int]] = []
KEEP: NDArray[np.bool_] | None = None

_HighlightFn = Callable[..., None]
_highlight_fn: _HighlightFn | None = None


def nevot_croce_density(film: Any, *, n_points: int = 2000, n_sigma: int = 5):
    """Convolve book-ended density with the surface Névot-Croce Gaussian.

    Parameters
    ----------
    film
        Book-ended film component from the graded fit.
    n_points
        Samples across ``[0, total_thick]``.
    n_sigma
        Kernel half-width in units of surface roughness.

    Returns
    -------
    depth, rho_sharp, rho_rough
        Depth (angstrom) and analytic / smeared densities (g/cm^3).
    """
    total = float(film.total_thick.value)
    sigma = float(film.surface_roughness.value)
    depth = np.linspace(0.0, total, n_points)
    rho_sharp = np.asarray(film.local_density(depth), dtype=float)
    if sigma <= 0.0:
        return depth, rho_sharp, rho_sharp.copy()

    dz = depth[1] - depth[0]
    pad = int(np.ceil(n_sigma * sigma / dz))
    z_ext = np.concatenate(
        [
            depth[0] + dz * np.arange(-pad, 0),
            depth,
            depth[-1] + dz * np.arange(1, pad + 1),
        ]
    )
    rho_ext = np.empty_like(z_ext)
    below, above = z_ext < 0.0, z_ext > total
    inside = ~(below | above)
    rho_ext[below] = 0.0
    rho_ext[inside] = np.asarray(film.local_density(z_ext[inside]), dtype=float)
    rho_ext[above] = float(film.local_density(total))

    kernel = np.exp(-0.5 * (dz * np.arange(-pad, pad + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    rho_rough = np.convolve(rho_ext, kernel, mode="same")[pad : pad + n_points]
    return depth, rho_sharp, rho_rough


def _load_znpc() -> tuple[NDArray[np.str_], NDArray[np.float64]]:
    """Load ZnPc XYZ with the ring plane in xy and ring normal along +z."""
    lines = XYZ.read_text().splitlines()
    n = int(lines[0].split()[0])
    els = np.array([ln.split()[0] for ln in lines[2 : 2 + n]])
    xyz = np.array([[float(v) for v in ln.split()[1:4]] for ln in lines[2 : 2 + n]])
    c = xyz - xyz.mean(0)
    nrm = np.linalg.svd(c)[2][2]
    a = nrm / np.linalg.norm(nrm)
    b = np.array([0.0, 0.0, 1.0])
    v, s = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < ALIGN_EPS:
        rot = np.eye(3) if s > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)
        rot = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + s))
    return els, c @ rot.T


def configure_packing(  # noqa: PLR0913
    film: Any,
    *,
    z_left: float = -6.0,
    z_g: NDArray[np.floating] | None = None,
    rho_sharp: NDArray[np.floating] | None = None,
    alpha_g: NDArray[np.floating] | None = None,
    slab_thick: float | None = None,
    highlight_fn: _HighlightFn | None = None,
    xyz_path: Path | None = None,
) -> dict[str, float]:
    """Bind graded-film profiles and load the ZnPc geometry for ``draw_structure``.

    Parameters
    ----------
    film
        Book-ended film from ``extract_graded_fit_context(...).film``.
    z_left
        Left depth-axis pad (angstrom), matching the manuscript ``Z_LEFT``.
    z_g, rho_sharp, alpha_g
        Optional precomputed profiles; when omitted they are built from ``film``.
    slab_thick
        Thickness (angstrom) of the slab the strip projects through. Sets the
        drawn areal density through ``cell_area`` and belongs in the caption.
        Defaults to the module ``SLAB_THICK``. Values much above one molecular
        width compress the in-plane lattice past van der Waals contact rather
        than adding out-of-plane neighbours, so raise it only alongside a
        multi-sublayer draw.
    highlight_fn
        Optional ``(ax, *, halo=True)`` marker drawer. Defaults to the module
        tau markers.
    xyz_path
        Override path to ``tpznpc_0m.xyz``.

    Returns
    -------
    dict
        ``total``, ``tau_vac``, ``tau_si``, ``gamma_median``, ``slab_thick``
        and the bulk ``cell_area`` for callers and captions.
    """
    global XYZ, Z_LEFT, Z_G, RHO_MAT, ALPHA_G, TOTAL, TAU_SI, TAU_VAC
    global SURF_ROUGH, RHO_BULK_REF, DEPTH_XLIM, ELS, MOL, BONDS, KEEP, MOLAR_MASS
    global SLAB_THICK
    global _highlight_fn

    if xyz_path is not None:
        XYZ = Path(xyz_path)
    if slab_thick is not None:
        SLAB_THICK = float(slab_thick)
    Z_LEFT = float(z_left)
    _highlight_fn = highlight_fn

    if z_g is None or rho_sharp is None:
        z_g, rho_sharp, _rho_rough = nevot_croce_density(film)
    if alpha_g is None:
        alpha_g = np.degrees(np.asarray(film.orientation(z_g), dtype=float))

    Z_G = np.asarray(z_g, dtype=float)
    RHO_MAT = np.asarray(rho_sharp, dtype=float)
    ALPHA_G = np.asarray(alpha_g, dtype=float)
    TOTAL = float(film.total_thick.value)
    TAU_SI = float(film.tau_si.value)
    TAU_VAC = float(film.tau_vac.value)
    SURF_ROUGH = float(film.surface_roughness.value)
    RHO_BULK_REF = float(film.density_bulk.value)
    DEPTH_XLIM = (Z_LEFT, TOTAL)

    ELS, MOL = _load_znpc()
    MOLAR_MASS = float(sum(ATOMIC_MASS[e] for e in ELS))
    dm = np.linalg.norm(MOL[:, None, :] - MOL[None, :, :], axis=-1)
    BONDS = [
        (i, j)
        for i in range(len(MOL))
        for j in range(i + 1, len(MOL))
        if dm[i, j] < (2.2 if "Zn" in (ELS[i], ELS[j]) else 1.75)
        and (SHOW_H or ("H" not in (ELS[i], ELS[j])))
    ]
    KEEP = np.array([SHOW_H or e != "H" for e in ELS])

    return {
        "total": TOTAL,
        "tau_vac": TAU_VAC,
        "tau_si": TAU_SI,
        "gamma_median": float(np.median(ALPHA_G)),
        "slab_thick": SLAB_THICK,
        "cell_area": MOLAR_MASS / (RHO_BULK_REF * N_AVOGADRO * 1e-24 * SLAB_THICK),
    }


def _gamma(z: float) -> float:
    if Z_G is None:
        raise RuntimeError("configure_packing required")
    if ALPHA_G is None:
        raise RuntimeError("configure_packing required")
    return float(np.interp(z, Z_G, ALPHA_G))


def highlight_interface(ax, *, halo: bool = True) -> None:
    """Draw vacuum and buried tau markers on the structure strip."""
    if _highlight_fn is not None:
        _highlight_fn(ax, halo=halo)
        return
    for z_tau in (TAU_VAC, TOTAL - TAU_SI):
        if halo:
            ax.axvline(
                z_tau,
                color="white",
                lw=TAU_LW + 1.1,
                solid_capstyle="butt",
                zorder=4,
            )
        ax.axvline(
            z_tau,
            color=TAU_COLOR,
            lw=TAU_LW,
            ls=TAU_LS,
            solid_capstyle="butt",
            zorder=5,
        )


def shade_substrate(ax) -> None:
    """Fade a grey wedge into the Si side of the strip."""
    z0 = TOTAL - 2.0 * TAU_SI
    z1 = TOTAL
    n = 80
    edges = np.linspace(z0, z1, n + 1)
    alphas = 0.22 * np.linspace(0.0, 1.0, n) ** 2.0
    for x0, x1, a in zip(edges[:-1], edges[1:], alphas, strict=True):
        ax.axvspan(x0, x1, color="0.40", alpha=float(a), lw=0, zorder=0)


def _molecule_2d(
    gamma_deg: float, centre: NDArray[np.float64], sense: int = 1
) -> NDArray[np.float64]:
    """Project one macrocycle into (depth, lateral) for polar tilt ``gamma``.

    In splay mode ``sense`` only selects the partner after the caller has
    already offset gamma by ``+/- HERRINGBONE_SPLAY_DEG/2``. In reflect mode
    the molecular frame is mirrored across the depth axis.
    """
    assert MOL is not None
    g = np.radians(gamma_deg)
    s = 1.0 if (sense >= 0 or HERRINGBONE_MODE != "reflect") else -1.0
    n_hat = np.array([np.cos(g), s * np.sin(g)])
    t_hat = np.array([-s * np.sin(g), np.cos(g)])
    xy = np.outer(MOL[:, 0], t_hat) + np.outer(MOL[:, 2], n_hat)
    return xy + centre


def _n_b(gamma_deg: float) -> float:
    """Stack-axis component of the ring normal at tilt ``gamma`` (crystal packing)."""
    return float(
        np.clip(
            min(N_B_CRYSTAL, abs(np.sin(np.radians(gamma_deg)))), SIN_GAMMA_MIN, 1.0
        )
    )


def _molecule_2d_crystal(
    gamma_deg: float, centre: NDArray[np.float64], sense: int = 1
) -> NDArray[np.float64]:
    """Project one macrocycle for ``PACKING = "crystal"``.

    Same as ``_molecule_2d`` in reflect mode, except that the in-plane normal is
    (cos gamma, n_b) with the crystal stack-axis component, so the drawn trace is the
    ring seen edge on along the column axis.
    """
    assert MOL is not None
    g = np.radians(gamma_deg)
    n_a, n_b = np.cos(g), min(N_B_CRYSTAL, abs(np.sin(g)))
    norm = float(np.hypot(n_a, n_b))
    s = 1.0 if (sense >= 0 or HERRINGBONE_MODE != "reflect") else -1.0
    n_hat = np.array([n_a, s * n_b]) / norm
    t_hat = np.array([-n_hat[1], n_hat[0]])
    xy = np.outer(MOL[:, 0], t_hat) + np.outer(MOL[:, 2], n_hat)
    return xy + centre


def _lattice_scale(z: float) -> float:
    """Areal lattice scale ``sqrt(rho_bulk / rho)`` clamped to the strain window."""
    if Z_G is None:
        raise RuntimeError("configure_packing required")
    if RHO_MAT is None:
        raise RuntimeError("configure_packing required")
    r = max(float(np.interp(z, Z_G, RHO_MAT)) / RHO_BULK_REF, 1e-3)
    s_ideal = float(np.sqrt(1.0 / r))
    return float(np.clip(s_ideal, STRAIN_MIN, STRAIN_MAX))


def cell_area(z: float) -> float:
    """Area per molecule (angstrom^2) implied by ``rho(z)`` and ``SLAB_THICK``.

    The strip is a projection through a slab ``SLAB_THICK`` thick, so one
    molecule must occupy ``M / (rho N_A t)``. Drawn number density is then a
    consequence of panel (d) rather than of the drawing constants, which set
    only the shape of the cell. The clamp is the same strain window as
    ``_lattice_scale`` and bites only where the fitted density collapses.
    """
    if Z_G is None or RHO_MAT is None:
        raise RuntimeError("configure_packing required")
    rho = max(float(np.interp(z, Z_G, RHO_MAT)), 1e-3)
    a_ideal = MOLAR_MASS / (rho * N_AVOGADRO * 1e-24 * SLAB_THICK)
    a_bulk = MOLAR_MASS / (RHO_BULK_REF * N_AVOGADRO * 1e-24 * SLAB_THICK)
    return float(np.clip(a_ideal, a_bulk * STRAIN_MIN**2, a_bulk * STRAIN_MAX**2))


def _points_per_angstrom(ax) -> float:
    """Typographic points per angstrom of depth for the axes as laid out.

    ``ELEM_RADIUS`` and ``BOND_WIDTH_A`` are physical sizes, but scatter and
    line widths are given in points, so the conversion has to follow the axes
    box. Requires the figure to have been laid out, which the callers do.
    """
    fig = ax.get_figure()
    w_in = ax.get_position().width * fig.get_size_inches()[0]
    span = DEPTH_XLIM[1] - DEPTH_XLIM[0]
    return float(w_in * 72.0 / span)


def _max_depth_offset(gamma_deg: float, sense: int) -> float:
    """Largest depth offset of any drawn atom for a molecule at this tilt.

    Exact rather than ``0.5 * L_MOL * sin(gamma)``, since it uses the real
    atomic positions and the same projection ``_molecule_2d`` applies.
    """
    assert MOL is not None and KEEP is not None
    g = np.radians(gamma_deg)
    s = 1.0 if (sense >= 0 or HERRINGBONE_MODE != "reflect") else -1.0
    a, b = MOL[KEEP, 0], MOL[KEEP, 2]
    return float(np.max(-s * a * np.sin(g) + b * np.cos(g)))


def _tilt_that_fits(gamma_deg: float, z_centre: float, sense: int) -> float:
    """Largest tilt at or below ``gamma_deg`` whose footprint clears the substrate.

    A molecule close to the substrate cannot stand at the bulk tilt, so it lies
    flatter instead of disappearing. Keeping the molecule is what holds the
    drawn number density tied to rho(z). Flat is nearly always admissible,
    since the macrocycle is planar to a fraction of an angstrom, so this fails
    only when the centre is already inside the substrate.
    """
    head = TOTAL - SUB_CLEAR - z_centre
    if _max_depth_offset(gamma_deg, sense) <= head:
        return gamma_deg
    grid = np.linspace(0.0, abs(gamma_deg), TILT_SEARCH_N)
    fits = [g for g in grid if _max_depth_offset(g, sense) <= head]
    if not fits:
        return float("nan")
    return float(max(fits)) * (1.0 if gamma_deg >= 0 else -1.0)


def _fit_to_substrate(gamma_deg, centre, sense):
    """Tilt, then translate, until a molecule's footprint clears the substrate.

    Rotation is spent first. A molecule pinned against the substrate is laid
    flatter, down to flat if it must be, which is what a molecule against a
    hard wall does and costs nothing in position. Only a molecule whose centre
    is so close to the interface that even a flat macrocycle still crosses is
    then translated, by exactly the residual plus ``SUB_LIFT``. The translation
    is uncapped, so a molecule is never deleted however deep its jittered
    centre landed.

    Returns the tilt to draw and the possibly shifted centre.
    """
    z = float(centre[0])
    head = TOTAL - SUB_CLEAR - z
    if _max_depth_offset(gamma_deg, sense) <= head:
        return gamma_deg, centre
    g_fit = _tilt_that_fits(gamma_deg, z, sense)
    if np.isfinite(g_fit):
        return g_fit, centre  # flattening alone was enough
    shift = _max_depth_offset(0.0, sense) - head + SUB_LIFT
    return 0.0, np.array([z - shift, float(centre[1])])


def _depth_half_extents(
    gamma_deg: float, sense: int, *, crystal: bool
) -> tuple[float, float]:
    """Return how far drawn atoms reach from the molecule centre.

    Parameters
    ----------
    gamma_deg
        Polar tilt in degrees.
    sense
        Herringbone sign passed to the projector.
    crystal
        Use the crystal column projection when true, and the main-branch
        projection when false.

    Returns
    -------
    tuple of float
        ``(toward_vacuum, toward_substrate)`` in angstrom, both nonnegative.
        Vacuum is the direction of decreasing depth.
    """
    if MOL is None or KEEP is None:
        raise RuntimeError("call configure_packing(...) before drawing")
    centre = np.zeros(2)
    project = _molecule_2d_crystal if crystal else _molecule_2d
    depth = project(gamma_deg, centre, sense=sense)[KEEP, 0]
    return float(-depth.min()), float(depth.max())


def _settled_contact_step(z_contact: float, step: float) -> float:
    """Shorten the lamellar step from a partial contact row to the crystal film.

    Parameters
    ----------
    z_contact
        Depth of the contact-row centres, in angstrom.
    step
        Unsettled lamellar period of that row, in angstrom.

    Returns
    -------
    float
        Step to the next row, in angstrom, at most ``step``. Closes
        ``CONTACT_GAP_FILL`` of the free gap between the contact envelope and
        the crystal envelope and leaves ``CONTACT_GAP_CLEAR_A``.
    """
    toward_vacuum, _toward_si = _depth_half_extents(_gamma(z_contact), 1, crystal=False)
    z_next = float(np.clip(z_contact - step, 0.0, TOTAL))
    _vac_next, toward_si = _depth_half_extents(
        _gamma(z_next), CRYSTAL_SENSE, crystal=True
    )
    excess = step - toward_vacuum - toward_si - CONTACT_GAP_CLEAR_A
    if excess <= 0.0:
        return step
    return step - CONTACT_GAP_FILL * excess


def _nestle_drop(
    offsets: NDArray[np.float64],
    z: float,
    y: float,
    obstacles: NDArray[np.float64],
    *,
    fill: float,
    max_drop: float | None = None,
    floor: float = -CONTACT_GAP_CLEAR_A,
) -> float:
    """Return how far a molecule centre may move toward the substrate.

    Parameters
    ----------
    offsets
        Drawn-atom positions relative to the centre, shape ``(n, 2)``,
        depth first.
    z, y
        Centre depth and lateral position, angstrom.
    obstacles
        Atom positions already placed underneath, shape ``(m, 2)``.
    fill
        Fraction of the open drop to take, in ``[0, 1]``.

    Returns
    -------
    float
        Depth shift in angstrom, nonnegative. Stops ``NESTLE_CLEAR_A`` short
        of an obstacle atom and short of the substrate.
    """
    toward_si = float(np.max(offsets[:, 0]))
    near = np.abs(obstacles[:, 1] - y) < float(np.ptp(offsets[:, 1])) + 2.0 * NESTLE_CLEAR_A
    obstacles = obstacles[near]
    span = (TOTAL + floor) - (z + toward_si)  # floor > 0 reaches into the rough oxide
    if max_drop is not None:
        span = min(span, max_drop)
    if span <= 0.2 or fill <= 0.0:
        return 0.0
    if obstacles.size == 0:  # nothing underneath within reach: fall the whole span
        return float(fill * span)
    grid = np.arange(0.0, span + 1e-9, 0.25)
    pts = offsets[None, :, :] + np.array([z, y])
    pts = pts + grid[:, None, None] * np.array([1.0, 0.0])
    delta = pts[:, :, None, :] - obstacles[None, None, :, :]
    dist = np.hypot(delta[..., 0], delta[..., 1]).min(axis=(1, 2))
    open_drop = grid[dist >= NESTLE_CLEAR_A]
    if open_drop.size == 0:
        return 0.0
    return float(fill * open_drop[-1])


def _resolve_clash(
    xy: NDArray[np.float64],
    placed: list[NDArray[np.float64]],
    tree,
    *,
    in_film: bool = False,
    poke: float = 0.0,
) -> NDArray[np.float64] | None:
    """Move one molecule clear of the Si and of the molecules already placed.

    Parameters
    ----------
    xy
        Drawn-atom positions, shape ``(n, 2)``, depth first (angstrom).
    placed
        Atom arrays already drawn in the same column.
    tree
        KD-tree over ``np.vstack(placed)``, or ``None`` when nothing is placed.

    Returns
    -------
    ndarray or None
        The moved positions, or ``None`` when no move within ``MAX_LIFT_A`` and
        ``MAX_SLIDE_A`` clears ``CLASH_A``; the molecule is then left out.
    """
    over = float(xy[:, 0].max()) - (TOTAL - SUB_REST_A + poke)
    if over > 0.0:
        xy = xy - np.array([over, 0.0])  # rest on the substrate, not in it
    if tree is None:
        return xy
    def search(max_lift: float, max_slide: float, clear: float):
        lifts = np.arange(0.0, max_lift + 1e-9, 0.25)
        slides = np.arange(-max_slide, max_slide + 1e-9, 0.25)
        moves = sorted(((dz, dy) for dz in lifts for dy in slides),
                       key=lambda m: np.hypot(m[0], SLIDE_COST * m[1]))
        for dz, dy in moves:  # smallest cost first; sliding is penalised, since it opens a gap in the row
                trial = xy + np.array([-dz, dy])
                if np.all(np.isinf(tree.query(trial, k=1, distance_upper_bound=clear)[0])):
                    return trial
        return None

    out = search(MAX_LIFT_A, MAX_SLIDE_A, CLASH_A)
    if out is None and in_film:
        # inside the film a molecule is never left out: search further, then accept
        # the touching clearance FILM_CLASH_MIN_A (drawn discs just meet, no overlap)
        out = search(FILM_LIFT_A, FILM_SLIDE_A, CLASH_A)
        if out is None:
            out = search(FILM_LIFT_A, FILM_SLIDE_A, FILM_CLASH_MIN_A)
        if out is None:  # last resort: lift further; later lamellae settle around it
            out = search(2.0 * FILM_LIFT_A, FILM_SLIDE_A, FILM_CLASH_MIN_A)
    return out


def draw_structure(ax, y_half: float) -> None:
    """Draw the strip; with the crystal packing, first dry passes for the density check.

    The lamellae settled onto the contact covering are closed (crystal pitch), so the
    film has no voids, but near the Si they then hold more molecules than rho(z) asks
    for. Dry passes count that excess over the interfacial zone and remove it from the
    partial contact covering as vacancies, so the drawn number density integrated over
    the zone follows rho(z). ``LAST_DRAW["density_zone"]`` reports drawn and expected
    counts per column.
    """
    CONTACT_KEEP.clear()
    FILL_CUT.clear()
    ROW_DROP.clear()
    if PACKING == "crystal" and DENSITY_CORRECT:
        for _ in range(DENSITY_PASSES):
            _draw_structure(ax, y_half, dry=True)
            CONTACT_KEEP.clear()
            CONTACT_KEEP.update(LAST_DRAW["contact_keep"])
            FILL_CUT.update(LAST_DRAW["fill_cut"])
            ROW_DROP.update(LAST_DRAW["row_drop"])
    _draw_structure(ax, y_half)


def _draw_structure(ax, y_half: float, *, dry: bool = False) -> None:  # noqa: PLR0912, PLR0915
    """Draw the FIG4.1 herringbone strip: stack along bulk orientation.

    The lamellar axis is the substrate normal, not the ring normal, so rows of
    molecules are iso-depth and grain boundaries are coplanar with the Si
    interface. Within a lamella every molecule shares ``gamma(z)`` and one
    tilt sense, since neighbours along the column axis are related by pure
    translation; the herringbone partner is the next lamella, reached by the
    mirror in ``_molecule_2d`` plus a ``GLIDE_FRAC`` lateral offset. Both
    sublattices therefore carry the same polar tilt and the structural repeat
    is two lamellae. The cell is ``(L_MOL * LAT_FRAC * sin gamma)`` deep by
    ``(PI_STACK / sin gamma)`` wide, so its area is set by the local density
    alone and the orientation profile only reshapes it.

    With ``PACKING = "crystal"`` (the default) the cell scaled by rho(z) is the
    crystal one, ``PI_SPACING / n_b`` wide and ``k_lam sin gamma`` deep, so the
    bulk pi-plane spacing is the crystal value; see ``PACKING``.
    """
    if MOL is None or ELS is None or KEEP is None or Z_G is None or ALPHA_G is None:
        raise RuntimeError("call configure_packing(...) before draw_structure")

    rng = np.random.default_rng(SEED)
    segs: list[list[NDArray[np.float64]]] = []
    pts: list[NDArray[np.float64]] = []
    elem: list[NDArray[np.str_]] = []
    segs_b: list[list[NDArray[np.float64]]] = []  # column behind (crystal packing)
    pts_b: list[NDArray[np.float64]] = []
    elem_b: list[NDArray[np.str_]] = []
    rng_b = np.random.default_rng(
        SEED + 1
    )  # own stream, so the front column is unchanged
    r_keep = 0.28 * L_MOL
    placed: list[list[NDArray[np.float64]]] = [[], []]   # drawn atoms and bond midpoints per column
    BOND_I = np.array([a for a, _b in BONDS], dtype=int)
    BOND_J = np.array([b for _a, b in BONDS], dtype=int)
    trees: list = [None, None]
    LAST_DRAW.clear()
    LAST_DRAW.update({"molecules": [], "dropped": 0, "lifted": 0, "fill_removed": [0, 0]})
    glow_segs: list[list[NDArray[np.float64]]] = []  # lying contact patches, front column
    isl_pts: list[NDArray[np.float64]] = []
    isl_elem: list[NDArray[np.str_]] = []
    row_drop: dict[tuple[int, int], list[float]] = {}
    rng_poke = np.random.default_rng(SEED + 7)  # own stream for the oxide roughness
    rng_patch = np.random.default_rng(SEED + 19)  # patch sinking into the oxide
    rng_thin = np.random.default_rng(SEED + 9)  # own stream for contact vacancies
    _tk = 2.0 * np.pi / np.array([31.0, 53.0, 89.0])
    _tph = np.random.default_rng(SEED + 17).uniform(0, 2 * np.pi, (64, 2, _tk.size))

    def tilt_field(y: float, row: int, col: int) -> float:
        """Unit-variance tilt deviation that varies smoothly along a lamella."""
        return float(np.sin(_tk * y + _tph[row % 64, col]).sum() / np.sqrt(_tk.size / 2.0))
    _pk = 2.0 * np.pi / np.asarray(CONTACT_PATCH_PERIODS_A)
    _pseed = SEED + 13 if PATCH_SEED is None else PATCH_SEED
    _pph = np.random.default_rng(_pseed).uniform(0, 2 * np.pi, (2, _pk.size))
    _pgrid = np.linspace(-y_half, y_half, 2048)

    def _praw(y, col):
        return np.sin(np.outer(np.atleast_1d(y), _pk) + _pph[col]).sum(axis=1)

    _pref = [np.sort(_praw(_pgrid, c)) for c in (0, 1)]

    def patch_field(y: float, col: int) -> float:
        """Laterally correlated stand-in for a uniform draw: its quantile along the strip."""
        return float(np.searchsorted(_pref[col], _praw(y, col)[0]) / _pref[col].size)
    c_rows: dict[int, float] = {}

    _kk = np.array([1.0, 1.7, 2.6, 3.9, 5.3, 7.1])
    _ph = rng.uniform(0, 2 * np.pi, _kk.size)

    def corrugation(y: float) -> float:
        """Unit-amplitude growth-front undulation, shared by every lamella."""
        s_ = np.sin(2 * np.pi * _kk * (y + y_half) / (2 * y_half) + _ph).sum()
        return float(s_ / np.sqrt(_kk.size / 2.0))

    def z_surface(y: float) -> float:
        return 0.35 * SURF_ROUGH * corrugation(y)

    g_med = float(np.median(ALPHA_G))
    a_bulk = MOLAR_MASS / (RHO_BULK_REF * N_AVOGADRO * 1e-24 * COLUMN_SLAB)
    k_lam = (
        a_bulk
        * _n_b(g_med)
        / (
            PI_SPACING
            * float(np.clip(abs(np.sin(np.radians(g_med))), SIN_GAMMA_MIN, 1.0))
        )
    )

    def crystal_cell(z: float) -> tuple[float, float, float]:
        """Lamellar period, lateral pitch and gamma at ``z`` for the crystal packing.

        The crystal cell is ``PI_SPACING / n_b`` wide (the pi-plane spacing) and
        ``k_lam sin gamma`` deep (the molecular height). Where rho(z) is below the
        density of that cell, both periods are dilated by the same factor, as the
        main-branch strip does, so the sparse interfacial rows keep their shape;
        where rho(z) is above it (the densified free surface), both are compressed
        by the same factor, so the pi spacing closes below the crystal value. The area per
        molecule is clamped to the same strain window as ``cell_area``.
        """
        g = _gamma(z)
        c0 = PI_SPACING / _n_b(g)
        d0 = k_lam * float(np.clip(abs(np.sin(np.radians(g))), SIN_GAMMA_MIN, 1.0))
        rho = max(float(np.interp(z, Z_G, RHO_MAT)), 1e-3)
        area = MOLAR_MASS / (rho * N_AVOGADRO * 1e-24 * COLUMN_SLAB)
        area = float(np.clip(area, a_bulk * STRAIN_MIN**2, a_bulk * STRAIN_MAX**2))
        dil = area / (c0 * d0)
        lam = float(np.sqrt(dil))
        return d0 * lam, c0 * lam, g

    def cell(z: float, *, main: bool = False) -> tuple[float, float, float]:
        """Lamellar period (depth), column period (lateral), and gamma at ``z``.

        The cell area comes from ``rho(z)``; ``LAT_FRAC`` and ``PI_STACK`` set
        only its aspect ratio, and the near-Si uniaxial strain is applied as an
        area-preserving shear so the drawn density stays tied to panel (d).
        """
        g = _gamma(z)
        if PACKING == "crystal" and not main:
            return crystal_cell(z)
        s_g = float(np.clip(abs(np.sin(np.radians(g))), SIN_GAMMA_MIN, 1.0))
        area = cell_area(z)
        ratio = L_MOL * LAT_FRAC * s_g**2 / PI_STACK
        aniso = 1.0 + STRAIN_ANISO_SI * float(np.exp(-(TOTAL - z) / TAU_SI))
        return np.sqrt(area * ratio) * aniso, np.sqrt(area / ratio) / aniso, g

    # Growth runs from the buried interface outward, so the lamellar register is
    # anchored at Si and the incommensurate layer is truncated at the free
    # surface. Stacking columns along the ring normal instead would rotate the
    # entire lattice by gamma and produce a diagonal grain that the GIWAXS
    # excludes.
    z_rows: list[float] = []
    z_cur = TOTAL
    if SUBSTRATE_MODE == "shift":
        # Seed the register where a molecule at the local tilt sits flush
        # against the substrate instead of straddling it.
        z_cur = TOTAL - 0.5 * L_MOL * abs(np.sin(np.radians(_gamma(TOTAL))))
    while z_cur > -0.5 * L_MOL:
        z_rows.append(z_cur)
        _main = PACKING == "crystal" and len(z_rows) <= MAIN_CONTACT_ROWS
        step = cell(float(np.clip(z_cur, 0.0, TOTAL)), main=_main)[0]
        if PACKING == "crystal" and len(z_rows) == MAIN_CONTACT_ROWS:
            step = _settled_contact_step(z_cur, step)
        z_cur -= step
    if PACKING == "crystal" and SURFACE_WINDOW_A:
        # Surface densification: the lamellae within SURFACE_WINDOW_A of the free surface
        # are drawn together so the topmost one, otherwise cut by the surface and left as
        # stray molecules, sits half a period inside it as a whole terrace. Falls back to
        # dropping that row when the squeeze would pass SURFACE_SQUEEZE_MIN.
        z_top = 0.5 * cell(0.0)[0]
        z_w = SURFACE_WINDOW_A
        if z_rows[-1] < z_top:
            f = (z_w - z_top) / (z_w - z_rows[-1])
            if f < SURFACE_SQUEEZE_MIN:
                z_rows, f = z_rows[:-1], (z_w - z_top) / (z_w - z_rows[-2])
            z_rows = [z if z >= z_w else z_w - (z_w - z) * f for z in z_rows]
            LAST_DRAW["surface_squeeze"] = round(f, 3)

    below_atoms: list[list[NDArray[np.float64]]] = [[], []]
    # Lamellae above the nestled ones are carried down by the median drop of the
    # last nestled lamella, so no gap opens between the two.
    nestle_dz: list[list[float]] = [[], []]
    carry = [0.0, 0.0]
    row_atoms: list[list[NDArray[np.float64]]] = [[], []]
    prev_fall: list[list[tuple[float, float]]] = [[], []]  # (y, drop) of the lamella below
    cur_fall: list[list[tuple[float, float]]] = [[], []]
    # lamellae (counted above the contact row) that settle: the nestled ones, then those
    # still within SETTLE_ZONE_TAU * tau_Si of the Si; the bulk keeps its rho(z) register
    settle_rows = max(
        NESTLE_ROWS,
        sum(1 for z in z_rows[MAIN_CONTACT_ROWS:] if TOTAL - z < SETTLE_ZONE_TAU * TAU_SI),
    )
    d_top = cell(0.0)[0]  # rows centred within one period of the surface form the top terrace
    for i, z_row in enumerate(z_rows):
        z_ref = float(np.clip(z_row, 0.0, TOTAL))
        # the first, partial covering of the Si is generated exactly as on main
        as_main = PACKING == "crystal" and i < MAIN_CONTACT_ROWS
        dense_row = False
        _d_lam, c_stack, g_row = cell(z_ref, main=as_main)
        if PACKING == "crystal" and not as_main and ROW_DROP.get(i, 0.0) > 0.0:
            _d, c_stack, g_row = cell(float(np.clip(z_ref + ROW_DROP[i], 0.0, TOTAL)))
        if PACKING == "crystal" and FILM_DENSE and MAIN_CONTACT_ROWS <= i < MAIN_CONTACT_ROWS + settle_rows:
            # closed film: the settled lamellae keep the crystal pitch instead of dilating
            # with rho(z); the density deficit is taken up by the contact covering below
            c_stack = PI_SPACING / _n_b(g_row)
            dense_row = True
        if PACKING == "crystal" and as_main and CONTACT_PATCHES:
            # partial covering of lying molecules: a slipped cofacial stack on the oxide,
            # ring planes PI_SPACING apart, so each patch is closed and the bare oxide
            # lies between patches
            g_row = CONTACT_TILT_DEG
            c_stack = PI_SPACING / float(np.sin(np.radians(g_row)))
        c_rows[i] = c_stack
        amorph_si = float(np.exp(-(TOTAL - z_ref) / TAU_SI))
        amorph_vac = float(np.exp(-z_ref / TAU_VAC))
        spread = (
            LIBRATION_DEG + DISORDER_VAC_DEG * amorph_vac + DISORDER_SI_DEG * amorph_si
        )
        jit = POS_DISORDER_VAC * amorph_vac + POS_DISORDER_SI * amorph_si
        if PACKING == "crystal" and dense_row:
            # room for the flattest molecules the correlated tilt field reaches (2 sigma),
            # so neighbours tilting together still clear each other
            spread_row = LIBRATION_DEG + DISORDER_VAC_DEG * amorph_vac + DISORDER_SI_DEG * amorph_si
            c_stack = max(c_stack, PI_SPACING / _n_b(g_row - 2.0 * spread_row))
        if PACKING == "crystal" and as_main and CONTACT_PATCHES:
            spread, jit = CONTACT_SPREAD_DEG, CONTACT_JITTER_A
        corr_amp = (
            CORR_FRAC * SURF_ROUGH * (CORR_FLOOR + (1.0 - CORR_FLOOR) * amorph_vac)
        )
        n_j = int(np.ceil((y_half + L_MOL) / c_stack)) + 2
        columns = (0, 1) if (PACKING == "crystal" and DRAW_BEHIND) else (0,)
        for col in columns:
            if PACKING != "crystal":
                sense = 1 if i % 2 == 0 else -1
            else:
                # one herringbone class per column, so no flip between lamellae;
                # the partial contact covering takes the same sense as the bulk
                sense = CRYSTAL_SENSE
            if col == 1:
                sense = -sense
            phase = (
                GLIDE_FRAC * c_stack * (i % 2) + 0.5 * c_stack * col
            )  # cell: columns half a stack period apart
            rng_c = rng if col == 0 else rng_b
            nestle_i = i - MAIN_CONTACT_ROWS
            if PACKING == "crystal" and nestle_i == NESTLE_ROWS and nestle_dz[col]:
                carry[col] = float(np.quantile(nestle_dz[col], CARRY_QUANTILE))
            if PACKING == "crystal" and nestle_i == settle_rows and nestle_dz[col]:
                carry[col] = carry[col] + float(np.quantile(nestle_dz[col], CARRY_QUANTILE))
            if PACKING == "crystal" and 0 <= nestle_i < settle_rows:
                nestle_dz[col] = []
            z_shift = carry[col] if (PACKING == "crystal" and nestle_i >= NESTLE_ROWS) else 0.0
            if row_atoms[col]:  # the lamella just drawn becomes the floor of this one
                below_atoms[col].append(np.vstack(row_atoms[col]))
            row_atoms[col] = []
            prev_fall[col], cur_fall[col] = cur_fall[col], []
            for j in range(-n_j, n_j + 1):
                y = j * c_stack + phase
                z = z_row + z_shift + corr_amp * corrugation(y)
                if PACKING == "crystal" and as_main and CONTACT_PATCHES:
                    # patches lie on the oxide: centre half the tilted footprint above it
                    # lying molecules settle into the rough oxide: lowest atom PATCH_SINK_A
                    # below the mean surface (small scatter), so the lamella above can come
                    # down onto them and sandwich in
                    z = TOTAL - 0.5 * L_MOL * float(np.sin(np.radians(g_row)))
                    z += float(np.clip(rng_patch.normal(PATCH_SINK_A, 0.5), 0.0, 2.0 * SUB_ROUGH_A))
                if not (Z_LEFT + r_keep <= z <= TOTAL and abs(y) <= y_half - r_keep):
                    continue
                if z < max(z_surface(y), 0.5):
                    continue
                centre = np.array([z, y]) + rng_c.normal(0.0, jit, 2)
                g = g_row + rng_c.normal(0.0, spread)
                if PACKING == "crystal" and not as_main:
                    # Disorder within a lamella is correlated along it, so neighbours tilt
                    # and shift together: no ring cuts its neighbour and no gap opens
                    # between them, while the depth statistics are unchanged.
                    dev_c = centre - np.array([z, y])
                    centre = np.array([z, y]) + np.array([dev_c[0], LATERAL_JITTER_FRAC * dev_c[1]])
                    g = g_row + spread * (
                        TILT_CORR * tilt_field(float(y), i, col)
                        + np.sqrt(1.0 - TILT_CORR**2) * (g - g_row) / max(spread, 1e-9)
                    )
                if HERRINGBONE_MODE == "splay" and PACKING != "crystal":
                    g += 0.5 * HERRINGBONE_SPLAY_DEG * sense
                    g = max(g, TILT_FLOOR_DEG)
                if SUBSTRATE_MODE == "tilt":
                    g, centre = _fit_to_substrate(g, centre, sense)
                if col == 1:
                    centre = centre + np.asarray(BEHIND_SHIFT_A)
                if PACKING == "crystal" and not as_main:
                    xy = _molecule_2d_crystal(g, centre, sense=sense)
                else:
                    xy = _molecule_2d(g, centre, sense=sense)
                # oxide roughness allowance of this molecule, drawn once and used both for
                # its fall and for its rest on the substrate
                poke = min(abs(rng_poke.normal(0.0, SUB_ROUGH_A)), 2.0 * SUB_ROUGH_A)
                settle = PACKING == "crystal" and nestle_i >= 0 and below_atoms[col]
                cap = SETTLE_MAX_A
                if nestle_i >= settle_rows:  # above the zone the settle fades out
                    cap *= float(np.exp(-((TOTAL - z_row) - SETTLE_ZONE_TAU * TAU_SI) / TAU_SI))
                    settle = settle and cap > 0.25
                if settle:
                    nestled = nestle_i < max(NESTLE_ROWS, settle_rows)  # the whole interfacial zone falls
                    max_drop = NESTLE_MAX_A[min(nestle_i, len(NESTLE_MAX_A) - 1)] if nestled else cap
                    if nestled and nestle_i >= 1:
                        under = [d_ for y_, d_ in prev_fall[col] if abs(y_ - centre[1]) < HOLE_WIDTH_A]
                        max_drop = (HOLE_MAX_A if not under
                                    else CASCADE_DECAY * float(np.median(under))
                                    + CASCADE_BASE_A * float(np.exp(-(TOTAL - z_row) / (CASCADE_TAU * TAU_SI))))
                    dz = _nestle_drop(
                        xy[KEEP] - centre,
                        float(centre[0]),
                        float(centre[1]),
                        # what lies below, and the neighbours of this lamella already
                        # placed, so a falling molecule stops on whatever it meets first
                        # (same-lamella neighbours only for the filling layer, so a molecule
                        # above a hole is not held up by a neighbour that is supported)
                        np.vstack(below_atoms[col][-3:] + (row_atoms[col][-3:] if nestle_i == 0 else [])),
                        fill=NESTLE_FILL[min(nestle_i, len(NESTLE_FILL) - 1)] if nestled else SETTLE_FILL,
                        max_drop=max_drop,
                        floor=poke if SUB_ROUGH_A > 0 else -CONTACT_GAP_CLEAR_A,
                    )
                    if dz > 0.05:
                        xy = xy + np.array([dz, 0.0])
                    nestle_dz[col].append(max(dz, 0.0))
                    cur_fall[col].append((float(centre[1]), max(dz, 0.0)))
                    row_drop.setdefault((col, i), []).append(max(dz, 0.0) + z_shift)
                if (
                    SUBSTRATE_MODE in ("clip", "shift")
                    and xy[KEEP, 0].max() + SUB_CLEAR > TOTAL
                ):
                    continue  # drawn footprint would cross into the substrate
                if PACKING == "crystal" and as_main:
                    # vacancy in the partial covering, which carries the density deficit;
                    # with CONTACT_PATCHES the vacancies are laterally correlated, so they
                    # group into bare oxide between closed patches
                    u = patch_field(float(centre[1]), col) if CONTACT_PATCHES else rng_thin.random()
                    if u > CONTACT_KEEP.get(col, 1.0):
                        continue
                if (PACKING == "crystal" and CONTACT_PATCHES and i == MAIN_CONTACT_ROWS
                        and patch_field(float(centre[1]), col) < FILL_CUT.get(col, 0.0)):
                    LAST_DRAW["fill_removed"][col] += 1  # over an island core: left out
                    continue
                if (SURFACE_VACANCIES and col == 0 and z_row < d_top
                        and rng_thin.random() < SURFACE_VACANCY_FRAC):
                    continue
                if PACKING == "crystal":
                    # atoms plus bond midpoints, so two rings cannot cross between atoms
                    probe = np.vstack([xy[KEEP], 0.5 * (xy[BOND_I] + xy[BOND_J])])
                    moved = _resolve_clash(
                        probe,
                        placed[col],
                        trees[col],
                        # the free-surface terrace of the front column may hold vacancies, so
                        # the surface reads as a broken terrace with the column behind showing
                        # through; everywhere else in the film no molecule is left out
                        in_film=i >= MAIN_CONTACT_ROWS,
                        poke=poke + SUB_REST_A if SUB_ROUGH_A > 0 else 0.0,
                    )
                    if moved is None:
                        LAST_DRAW["dropped"] += 1
                        continue
                    shift = moved[0] - xy[KEEP][0]
                    if np.any(np.abs(shift) > 1e-9):
                        LAST_DRAW["lifted"] += 1
                        xy = xy + shift
                    placed[col].append(probe + shift)
                    trees[col] = cKDTree(np.vstack(placed[col]))
                    LAST_DRAW["molecules"].append((col, i, xy[KEEP].copy()))
                    LAST_DRAW.setdefault("gamma", []).append(float(g))
                if PACKING == "crystal":
                    row_atoms[col].append(xy[KEEP])
                if col == 0 and as_main and PATCH_GLOW:  # island molecules, drawn on top
                    glow_segs.extend([xy[a], xy[b]] for a, b in BONDS)
                    isl_pts.append(xy[KEEP])
                    isl_elem.append(ELS[KEEP])
                elif col == 0:
                    segs += [[xy[a], xy[b]] for a, b in BONDS]
                    pts.append(xy[KEEP])
                    elem.append(ELS[KEEP])
                else:
                    segs_b += [[xy[a], xy[b]] for a, b in BONDS]
                    pts_b.append(xy[KEEP])
                    elem_b.append(ELS[KEEP])

    # Number density check. The settled lamellae are closed (crystal pitch, no voids),
    # so near the Si they hold more molecules than rho(z) asks for. The excess over the
    # interfacial zone, counted per column, is removed from the partial contact covering,
    # where vacancies are allowed, as a keep fraction for the next pass.
    n_of_z = RHO_MAT * N_AVOGADRO * 1e-24 * COLUMN_SLAB / MOLAR_MASS   # per angstrom^2
    zone_top = z_rows[min(MAIN_CONTACT_ROWS + settle_rows, len(z_rows) - 1)]
    zz = np.linspace(zone_top, TOTAL, 256)
    keep = {}
    for c_ in (0, 1):
        mols = [a_ for cc, i_, a_ in LAST_DRAW["molecules"] if cc == c_]
        if not mols:
            continue
        yc = np.array([a_[:, 1].mean() for a_ in mols])
        zc_ = np.array([a_[:, 0].mean() for a_ in mols])
        width = float(np.ptp(yc)) + PI_SPACING
        expected = width * float(np.trapezoid(np.interp(zz, Z_G, n_of_z), zz))
        drawn = int(np.sum(zc_ >= zone_top))
        contact = sum(1 for cc, i_, _a in LAST_DRAW["molecules"] if cc == c_ and i_ < MAIN_CONTACT_ROWS)
        prev = CONTACT_KEEP.get(c_, 1.0)
        n_contact_full = contact / max(prev, 1e-6)
        want = contact - (drawn - expected)
        keep[c_] = float(np.clip(want / max(n_contact_full, 1.0), CONTACT_KEEP_MIN, 1.0))
        LAST_DRAW.setdefault("density_zone", {})[c_] = (drawn, round(expected, 1))
    scale = keep
    # Orientation check over the contact band (CONTACT_BAND_A from the Si): drawn
    # <cos^2 gamma> of the front column against the rho-weighted fit.
    zb = np.linspace(TOTAL - CONTACT_BAND_A, TOTAL, 128)
    wb = np.interp(zb, Z_G, RHO_MAT)
    c2_fit = float(np.sum(wb * np.cos(np.radians(np.interp(zb, Z_G, ALPHA_G))) ** 2) / np.sum(wb))
    gam = LAST_DRAW.get("gamma", [])
    c2 = [np.cos(np.radians(gam[k_])) ** 2 for k_, (cc, i_, a_) in enumerate(LAST_DRAW["molecules"])
          if cc == 0 and k_ < len(gam) and TOTAL - float(a_[:, 0].mean()) <= CONTACT_BAND_A]
    if c2:
        LAST_DRAW["contact_band_gamma"] = (
            round(float(np.degrees(np.arccos(np.sqrt(np.mean(c2))))), 1),
            round(float(np.degrees(np.arccos(np.sqrt(c2_fit)))), 1),
        )
    LAST_DRAW["contact_keep"] = scale
    # Filling layer: the excess over the first FILL_BAND_A comes out over the islands.
    zb_ = np.linspace(TOTAL - FILL_BAND_A, TOTAL, 64)
    fill_cut = {}
    for c_ in (0, 1):
        mols = [(i_, a_) for cc, i_, a_ in LAST_DRAW["molecules"] if cc == c_]
        if not mols:
            continue
        width = float(np.ptp([a_[:, 1].mean() for _i, a_ in mols])) + PI_SPACING
        exp_b = width * float(np.trapezoid(np.interp(zb_, Z_G, n_of_z), zb_))
        got_b = sum(1 for _i, a_ in mols if TOTAL - a_[:, 0].mean() <= FILL_BAND_A)
        n_fill = sum(1 for i_, _a in mols if i_ == MAIN_CONTACT_ROWS) + LAST_DRAW["fill_removed"][c_]
        cut = FILL_CUT.get(c_, 0.0) + (got_b - exp_b) / max(n_fill, 1)
        fill_cut[c_] = float(np.clip(cut, 0.0, CONTACT_KEEP.get(c_, 1.0))) if FILL_BAND_A > 0 else 0.0
        LAST_DRAW.setdefault("fill_band", {})[c_] = (got_b, round(exp_b, 1))
    LAST_DRAW["fill_cut"] = fill_cut
    LAST_DRAW["row_drop"] = {
        i_: float(np.median(row_drop.get((0, i_), []) + row_drop.get((1, i_), [])))
        for i_ in {k_[1] for k_ in row_drop}
    }
    if dry:
        return
    if not pts:
        raise RuntimeError("draw_structure produced no molecules")
    pts_arr = np.vstack(pts)
    elem_arr = np.concatenate(elem)

    # Marker sizes and line widths are typographic units, so a panel that is
    # narrower for the same depth span silently draws fatter molecules and
    # closes the intermolecular gaps. Convert once here and everything below is
    # specified in angstrom, which keeps the strip identical across panel
    # geometries and dpi.
    pts_per_a = _points_per_angstrom(ax)
    if pts_b:  # column behind, the other herringbone class, drawn first in grey
        pts_b_arr, elem_b_arr = np.vstack(pts_b), np.concatenate(elem_b)
        ax.add_collection(
            LineCollection(
                segs_b,
                colors=BEHIND_GREY,
                alpha=BEHIND_ALPHA,
                linewidths=BOND_WIDTH_A * pts_per_a,
                zorder=2,
                capstyle="round",
            )
        )
        for e in ELEM_COLORS:
            m = elem_b_arr == e
            if m.any():
                ax.scatter(
                    pts_b_arr[m, 0],
                    pts_b_arr[m, 1],
                    s=(2 * ELEM_RADIUS[e] * ATOM_DRAW_SCALE * pts_per_a) ** 2,
                    c=BEHIND_GREY,
                    alpha=BEHIND_ALPHA,
                    edgecolors="none",
                    zorder=2.5,
                )
    LAST_DRAW["n_patch_molecules"] = len(glow_segs) // max(len(BONDS), 1)
    # Island heights: front-column contact molecules grouped where neighbours are less
    # than 1.5 contact pitches apart; height is the depth span of their atom centres
    # (add one drawn atom diameter, about 1.3 A, for the van der Waals extent).
    contact = sorted((a for c_, i_, a in LAST_DRAW["molecules"] if c_ == 0 and i_ < MAIN_CONTACT_ROWS),
                     key=lambda a: a[:, 1].mean())
    islands, pitch = [], c_rows.get(0, PI_SPACING)
    for a in contact:
        if islands and a[:, 1].mean() - islands[-1][-1][:, 1].mean() < 1.5 * pitch:
            islands[-1].append(a)
        else:
            islands.append([a])
    LAST_DRAW["islands"] = [
        {
            "n": len(isl),
            "height_A": round(float(np.ptp(np.vstack(isl)[:, 0])), 2),
            "top_above_Si_A": round(float(TOTAL - np.vstack(isl)[:, 0].min()), 2),
            "width_A": round(float(np.ptp(np.vstack(isl)[:, 1])), 1),
        }
        for isl in islands
    ]
    ax.add_collection(
        LineCollection(
            segs,
            colors=BOND_COLOR,
            linewidths=BOND_WIDTH_A * pts_per_a,
            zorder=3,
            capstyle="round",
        )
    )
    for e, col in ELEM_COLORS.items():
        m = elem_arr == e
        if m.any():
            ax.scatter(
                pts_arr[m, 0],
                pts_arr[m, 1],
                s=(2 * ELEM_RADIUS[e] * ATOM_DRAW_SCALE * pts_per_a) ** 2,
                c=col,
                edgecolors="none",
                zorder=4,
            )
    if glow_segs:  # soft halo marking the lying, sub-monolayer contact patches
        # All bonds as one NaN-separated polyline: a single path per layer, so overlapping
        # bonds do not stack alpha within a layer, and still a LineCollection, so the
        # notebook's rotation into the panel applies to it. Stacked translucent layers of
        # shrinking width give a soft, blurred halo.
        seg = np.asarray(glow_segs, dtype=float)
        poly = np.concatenate([seg, np.full((len(seg), 1, 2), np.nan)], axis=1).reshape(-1, 2)
        # Widths chosen so the number of layers covering radius r falls as a Gaussian,
        # n exp(-(r/sigma)^2), giving a feathered edge instead of a hard-edged cone.
        k = np.arange(1, PATCH_GLOW_STEPS + 1)
        reach = PATCH_GLOW_WIDTH_A - PATCH_RIM_A
        r_k = np.sqrt(np.log(PATCH_GLOW_STEPS / (k - 0.5)) / np.log(2 * PATCH_GLOW_STEPS))
        layers = [(PATCH_RIM_A + reach * r, PATCH_GLOW_COLOR, PATCH_GLOW_ALPHA) for r in r_k]
        pal = {**ELEM_COLORS, "bond": BOND_COLOR}
        if PATCH_STYLE == "color":
            layers, pal = [], PATCH_PALETTES[PATCH_PALETTE]
        elif PATCH_RIM_COLOR is not None:
            layers.append((PATCH_RIM_A, PATCH_RIM_COLOR, 1.0))
        for width, colour, alpha in layers:
            ax.add_collection(LineCollection([poly], colors=colour, alpha=alpha,
                                             linewidths=width * pts_per_a, capstyle="round",
                                             joinstyle="round", zorder=4.5))
        # the island molecules themselves, above the glow and above the rest of the film
        ax.add_collection(LineCollection(glow_segs, colors=pal["bond"],
                                         linewidths=BOND_WIDTH_A * pts_per_a, zorder=4.6,
                                         capstyle="round"))
        ip, ie = np.vstack(isl_pts), np.concatenate(isl_elem)
        for e in ELEM_COLORS:
            m = ie == e
            if m.any():
                ax.scatter(ip[m, 0], ip[m, 1], c=pal[e], edgecolors="none", zorder=4.7,
                           s=(2 * ELEM_RADIUS[e] * ATOM_DRAW_SCALE * pts_per_a) ** 2)
    highlight_interface(ax, halo=True)
    lab_bbox = {"fc": "white", "ec": "none", "alpha": 0.78, "pad": 0.6}
    ax.annotate(
        "vacuum",
        xy=(0.010, 0.97),
        xycoords="axes fraction",
        fontsize=6.5,
        color="0.3",
        ha="left",
        va="top",
        bbox=lab_bbox,
        zorder=10,
    )
    ax.annotate(
        "Si",
        xy=(0.990, 0.97),
        xycoords="axes fraction",
        fontsize=6.5,
        color="0.3",
        ha="right",
        va="top",
        bbox=lab_bbox,
        zorder=10,
    )
    ax.set_yticks([])
    ax.tick_params(
        which="both",
        left=False,
        right=False,
        bottom=False,
        top=False,
        labelleft=False,
        labelright=False,
        labelbottom=False,
        labeltop=False,
        length=0,
    )
    for tick in list(ax.xaxis.get_major_ticks()) + list(ax.xaxis.get_minor_ticks()):
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    ax.grid(False)
    for sp in ("left", "right", "top", "bottom"):
        ax.spines[sp].set_visible(False)


def render_panel_b() -> plt.Figure:
    """Build a single-column structure strip matching the FIG4 panel-(b) aspect.

    ``fig_h`` sets how many molecules fall across the lateral direction, since
    the axes are equal-aspect and the depth span is fixed. It must track the
    panel-(b) height ratio used in ``fig_5_graded.ipynb`` or the standalone
    replay will not match the figure.
    """
    fig_w = 3.35
    fig_h = PANEL_B_HEIGHT_IN
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)

    fig.canvas.draw()
    pos = ax.get_position()
    w_in = pos.width * fig_w
    h_in = pos.height * fig_h
    xspan = DEPTH_XLIM[1] - DEPTH_XLIM[0]
    y_half = 0.5 * xspan * (h_in / w_in)

    shade_substrate(ax)
    draw_structure(ax, y_half)
    ax.set_xlim(*DEPTH_XLIM)
    ax.set_ylim(-y_half, y_half)
    ax.set_aspect("equal", adjustable="box")
    return fig


def write_comparison(replay_path: Path) -> None:
    """Save a side-by-side crop of FIG4.1 panel (b) vs the replay strip."""
    if not REF_FIG41.exists():
        return
    ref = Image.open(REF_FIG41).convert("RGB")
    rw, rh = ref.size
    box = (int(0.12 * rw), int(0.42 * rh), int(0.95 * rw), int(0.58 * rh))
    ref_crop = ref.crop(box)
    replay = Image.open(replay_path).convert("RGB")
    hgt = 220
    ref_r = ref_crop.resize((int(ref_crop.width * hgt / ref_crop.height), hgt))
    rep_r = replay.resize((int(replay.width * hgt / replay.height), hgt))
    width = ref_r.width + rep_r.width + 16
    canvas = Image.new("RGB", (width, hgt + 28), (255, 255, 255))
    canvas.paste(ref_r, (0, 24))
    canvas.paste(rep_r, (ref_r.width + 16, 24))
    canvas.save(COMPARE_PNG)


def main() -> None:
    """Load the graded fit, render the FIG4.1 strip, and write PNG/PDF."""
    from utils import objectives_retarget_energies, read_fit, read_xrr
    from utils.graded_objective import extract_graded_fit_context
    from utils.models import configure_refloxide_fitting

    configure_refloxide_fitting()
    energy = 283.7
    data = read_xrr("reflectivity_data", material="znpc", source="hub")
    graded_fit = read_fit("graded/graded_fit.pkl", material="znpc", source="local")
    graded_obj = objectives_retarget_energies(graded_fit, [energy], data=data)[energy]
    film = extract_graded_fit_context(graded_obj).film
    configure_packing(film, z_left=Z_LEFT, xyz_path=HERE / "tpznpc_0m.xyz")
    fig = render_panel_b()
    fig.savefig(OUT_PNG, dpi=600, transparent=True)
    fig.savefig(OUT_PDF, dpi=600, transparent=True)
    plt.close(fig)
    write_comparison(OUT_PNG)


if __name__ == "__main__":
    main()
