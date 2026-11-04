"""REC_L3 atomistic cutout and depth profiles for FIG4 panel (b).

Loads the reconstructed film dump under ``md/znpc_film/bulk``, averages the last
frames into ``gamma_eff(z)`` and ``rho(z)`` (lamellar boxcar), and draws a
y-slab ball-and-stick cutout of ZnPc for the structure panel (2D projection or
matplotlib 3D). Depth uses the FIG4 convention (vacuum at z = 0, substrate at
``total_thick``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from itertools import pairwise
from pathlib import Path

import matplotlib.colors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

REPO = Path(__file__).resolve().parents[2]
BULK = REPO / "md" / "znpc_film" / "bulk"
DEFAULT_TAG = "REC_L3"
INITIAL_SUFFIX = ":initial"
ATOM_COLUMNS = 7
# Cutout depth into the film (y): ~one herringbone cell so registry is continuous.
SLAB_Y = 20.0
TYPE_TO_ELEM = {1: "C", 2: "H", 3: "N", 4: "N", 5: "Zn"}
ELEM_COLORS = {"C": "#6e6e6e", "N": "#3b55d0", "Zn": "#9a63c0"}
# Skeleton radii (well below covalent), so macrocycle rings stay open.
ELEM_RADIUS = {"C": 0.22, "N": 0.26, "Zn": 0.40}
BOND_COLOR = "#6a6a6a"
ATOM_DRAW_SCALE = 0.85
BOND_WIDTH_A = 0.14
LATERAL_HALF_NM = 1.5
# Lateral half-width (A) for the 3D panel; one herringbone pair.
LATERAL_HALF_A_3D = 16.0
COLUMN_SEED_DEPTH_A = 20.0
# Molecules within this depth of each other compete for the front column.
FRONT_DEPTH_WINDOW_A = 8.0
# A molecule is behind the front stack once it sits this far (A) behind it.
FRONT_THICKNESS_A = 6.0
FRONT_EDGE_WIDTH_A = 2.0
BEHIND_GREY = 0.62
BEHIND_MIX = 0.88
BEHIND_ALPHA = 0.55
FRONT_ALPHA = 0.97
# Thin wireframe: atoms are marks only (Zn), bonds carry the structure.
ATOM_SIZE_3D = {"C": 0.0, "N": 0.0, "Zn": 3.5}
BOND_LW_3D = 0.28
TAU_COLOR_3D = "#0b7f8a"


def _load_profile_tools():
    """Import ``profile_tools`` from the bulk MD tree without package install."""
    path = BULK / "profile_tools.py"
    spec = importlib.util.spec_from_file_location("znpc_profile_tools", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load profile_tools from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _offsets(path: Path) -> list[int]:
    off: list[int] = []
    with path.open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.startswith(b"ITEM: TIMESTEP"):
                off.append(pos)
                f.readline()
                f.readline()
                n = int(f.readline())
                for _ in range(5 + n):
                    f.readline()
    return off


def _frame_at(path: Path, pos: int):
    with path.open() as f:
        f.seek(pos)
        f.readline()
        step = int(f.readline())
        f.readline()
        n = int(f.readline())
        f.readline()
        box = np.array([list(map(float, f.readline().split()[:2])) for _ in range(3)])
        cols = f.readline().split()[2:]
        arr = np.loadtxt([f.readline() for _ in range(n)])
    return step, box, {c: arr[:, i] for i, c in enumerate(cols)}


def split_tag(tag: str) -> tuple[str, bool]:
    """Split ``tag`` into the run stem and whether it names the initial structure.

    A tag ending in :data:`INITIAL_SUFFIX` (for example ``"REC_L4c:initial"``)
    selects the built, unrelaxed structure in ``{stem}.data`` instead of the
    dump of the relaxed run.
    """
    stem, sep, _ = tag.partition(INITIAL_SUFFIX)
    return stem, bool(sep)


def _read_data_state(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Parse the ``Atoms`` section of a LAMMPS ``full`` data file.

    Returns the box as ``[[xlo, xhi], [ylo, yhi], [zlo, zhi]]`` and the same
    per-atom column dictionary that :func:`_frame_at` produces.
    """
    lines = path.read_text().split("\n")
    box = np.zeros((3, 2))
    for line in lines[:30]:
        for k, key in enumerate(("xlo xhi", "ylo yhi", "zlo zhi")):
            if line.endswith(key):
                box[k] = [float(v) for v in line.split()[:2]]
    start = next(i for i, line in enumerate(lines) if line.startswith("Atoms")) + 2
    rows = [r.split() for r in lines[start:] if len(r.split()) >= ATOM_COLUMNS]
    arr = np.array([[float(v) for v in r[:ATOM_COLUMNS]] for r in rows])
    cols = ("id", "mol", "type", "q", "x", "y", "z")
    return box, {c: arr[:, i] for i, c in enumerate(cols)}


def trailing_frames(
    tag: str, nlast: int = 1
) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Return ``(box, per-atom data)`` for the last ``nlast`` dump frames of ``tag``.

    Parameters
    ----------
    tag
        Run stem under ``md/znpc_film/bulk``, optionally with
        :data:`INITIAL_SUFFIX`, which returns the single built structure from
        ``{stem}.data``.
    nlast
        Number of trailing dump frames; ignored for the initial structure.

    Raises
    ------
    FileNotFoundError
        If the dump or data file does not exist.
    RuntimeError
        If the dump contains no frames.
    """
    stem, initial = split_tag(tag)
    if initial:
        path = BULK / f"{stem}.data"
        if not path.is_file():
            raise FileNotFoundError(path)
        return [_read_data_state(path)]
    path = BULK / f"{stem}.dump"
    if not path.is_file():
        raise FileNotFoundError(path)
    off = _offsets(path)
    if not off:
        raise RuntimeError(f"no frames in {path}")
    return [_frame_at(path, pos)[1:] for pos in off[-nlast:]]


def load_meta(tag: str = DEFAULT_TAG) -> dict:
    """Read ``{tag}_meta.json`` from the bulk directory (initial suffix ignored)."""
    return json.loads((BULK / f"{split_tag(tag)[0]}_meta.json").read_text())


def md_profiles(
    tag: str = DEFAULT_TAG,
    *,
    total_thick: float,
    nlast: int = 10,
    dz: float = 1.0,
    d200: float = 12.33,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average last frames to FIG4-depth ``z``, ``gamma_eff``, ``rho``.

    Parameters
    ----------
    tag
        Reconstruction dump stem under ``md/znpc_film/bulk``.
    total_thick
        Graded-film thickness in angstrom (vacuum at 0, substrate at this value).
    nlast
        Trailing dump frames to average.
    dz, d200
        Histogram bin and lamellar boxcar width in angstrom.

    Returns
    -------
    z_fig, gamma_eff, rho
        Depth from the vacuum surface (A), effective tilt (deg), density (g/cm3).
    """
    pt = _load_profile_tools()
    meta = load_meta(tag)
    zref = float(meta["silica_top"]) + float(meta.get("d0", 2.0))
    edges = np.arange(-10.0, 220.0, dz)
    zc = 0.5 * (edges[1:] + edges[:-1])
    mass_hists, c2_hists = [], []
    box = None
    for box, data in trailing_frames(tag, nlast):
        mols = pt.molecules(box, data)
        z = np.concatenate([x[:, 2] for _, x, _, _ in mols]) - zref
        m = np.concatenate([np.vectorize(pt.MASS.get)(tt) for tt, _, _, _ in mols])
        c2 = np.concatenate(
            [np.full(len(tt), np.cos(np.radians(g)) ** 2) for tt, _, _, g in mols]
        )
        mass_hists.append(np.histogram(z, edges, weights=m)[0])
        c2_hists.append(np.histogram(z, edges, weights=m * c2)[0])
    area = (box[0, 1] - box[0, 0]) * (box[1, 1] - box[1, 0])
    rho, gam = pt.reduce(zc, np.array(mass_hists), np.array(c2_hists), area, dz, d200)
    z_fig = total_thick - zc
    order = np.argsort(z_fig)
    return z_fig[order], gam[order], rho[order]


def _bonds_for_molecule(
    els: np.ndarray, xyz: np.ndarray, *, show_h: bool
) -> list[tuple[int, int]]:
    dm = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    bonds: list[tuple[int, int]] = []
    for i in range(len(els)):
        for j in range(i + 1, len(els)):
            if not show_h and ("H" in (els[i], els[j])):
                continue
            cut = 2.2 if "Zn" in (els[i], els[j]) else 1.75
            if dm[i, j] < cut:
                bonds.append((i, j))
    return bonds


def _points_per_angstrom(ax, *, depth_span_a: float, horiz: bool) -> float:
    """Typographic points per angstrom along the depth axis of ``ax``."""
    fig = ax.get_figure()
    pos = ax.get_position()
    size = fig.get_size_inches()
    axis_in = pos.width * size[0] if horiz else pos.height * size[1]
    return float(axis_in * 72.0 / max(depth_span_a, 1.0))


def load_md_cutout_3d(  # noqa: PLR0915
    tag: str = DEFAULT_TAG,
    *,
    total_thick: float,
    slab_y: float = SLAB_Y,
    show_h: bool = False,
    lateral_half_a: float | None = LATERAL_HALF_A_3D,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Load last-frame ZnPc atoms in a y-slab as 3D FIG4-depth coordinates.

    Parameters
    ----------
    tag
        Dump stem under ``md/znpc_film/bulk``.
    total_thick
        Graded thickness in angstrom; flips MD height into vacuum-at-zero depth.
    slab_y
        Front-slab thickness in y (angstrom).
    show_h
        When false, drop hydrogen atoms and H-involving bonds.
    lateral_half_a
        Keep whole molecules whose COM lies within this lateral half-width
        (angstrom) of a depth-continuous column; ``None`` keeps the full slab.

    Returns
    -------
    pts, elems, segs
        Atom positions ``(n, 3)`` in angstrom as ``(x, y, z_fig)`` with x
        centered, element labels, and bond segments as ``(2, 3)`` arrays.
    """
    pt = _load_profile_tools()
    meta = load_meta(tag)
    zref = float(meta["silica_top"]) + float(meta.get("d0", 2.0))
    box, data = trailing_frames(tag)[-1]
    mols = pt.molecules(box, data)
    lx = float(box[0, 1] - box[0, 0])
    ly = float(box[1, 1] - box[1, 0])
    cand: list[tuple[float, float, float, np.ndarray, np.ndarray]] = []
    for tt, xyz, _n, _g in mols:
        com = xyz.mean(0)
        y_com = float(com[1] % ly)
        if y_com >= slab_y:
            continue
        els = np.array([TYPE_TO_ELEM[int(t)] for t in tt])
        keep = np.array([show_h or e != "H" for e in els])
        if not keep.any():
            continue
        z_com = float(total_thick - (com[2] - zref))
        z_fig = total_thick - (xyz[:, 2] - zref)
        dx = xyz[:, 0] - com[0]
        dx -= lx * np.round(dx / lx)
        dy = xyz[:, 1] - com[1]
        dy -= ly * np.round(dy / ly)
        xyz_fig = np.column_stack([com[0] + dx, y_com + dy, z_fig])
        cand.append((float(com[0]), y_com, z_com, xyz_fig[keep], els[keep]))
    if not cand:
        raise RuntimeError(f"no ZnPc atoms in front slab of {tag}")
    z_coms = np.array([c[2] for c in cand])
    mid = 0.5 * (float(z_coms.min()) + float(z_coms.max()))
    mid_band = [c for c in cand if abs(c[2] - mid) < COLUMN_SEED_DEPTH_A] or cand
    x0 = float(np.median([c[0] for c in mid_band]))
    half = None if lateral_half_a is None else float(lateral_half_a)
    kept: list[tuple[float, np.ndarray, np.ndarray, list[np.ndarray]]] = []
    for x_com, y_com, _z_com, xyz_fig, els in cand:
        dx_com = x_com - x0
        dx_com -= lx * np.round(dx_com / lx)
        if half is not None and abs(dx_com) > half:
            continue
        mol_xyz = xyz_fig.copy()
        mol_xyz[:, 0] = xyz_fig[:, 0] - x_com + dx_com
        bond_idx = _bonds_for_molecule(els, mol_xyz, show_h=True)
        mol_segs = [np.vstack([mol_xyz[i], mol_xyz[j]]) for i, j in bond_idx]
        kept.append((y_com, mol_xyz, els, mol_segs))
    if not kept:
        raise RuntimeError(f"no ZnPc molecules in lateral column of {tag}")
    kept.sort(key=lambda t: t[0])
    pts = np.vstack([k[1] for k in kept])
    elems = np.concatenate([k[2] for k in kept])
    segs = [s for k in kept for s in k[3]]
    y0 = float(np.min(pts[:, 1]))
    pts = pts.copy()
    pts[:, 1] -= y0
    segs = [s - np.array([0.0, y0, 0.0]) for s in segs]
    return pts, elems, segs


MIN_BIN_MOLECULES = 3
N_COLUMNS = 14


def column_following_mask(  # noqa: PLR0913
    com: np.ndarray,
    ly: float,
    *,
    n_columns: int = N_COLUMNS,
    width: float,
    bin_a: float = 6.0,
    centre_shift: float = 0.0,
) -> np.ndarray:
    """Select a y window of pi-stack columns that follows the lattice with depth.

    Columns are the rows of pi-stacks along ``x`` that repeat every
    ``ly / n_columns`` along ``y``. Lamellae shear laterally against each
    other by up to a few angstrom, so a fixed ``y`` window drops a column from
    some lamellae and shows an artificial void. This mask re-centres the window
    on the circular-mean column phase of every depth bin, unwrapped so the same
    physical column is tracked from one bin to the next. A bin whose window
    holds no molecule (a column removed as a whole-column vacancy) is moved,
    in steps of two columns so the herringbone class is kept, to the nearest
    occupied column, so vacancies do not read as voids.

    Parameters
    ----------
    com
        Centres of mass ``(n, 3)`` with columns ``x``, ``y`` and depth.
    ly
        Periodic box length along ``y``.
    n_columns
        Number of columns in the box.
    width
        Window width along ``y`` in angstrom.
    bin_a
        Depth bin height used to estimate the local column phase.
    centre_shift
        Offset of the window centre from a column centre in angstrom; half a
        column pitch centres the window between two neighbouring columns.

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape ``(n,)``.
    """
    pitch = ly / n_columns
    edges = np.arange(com[:, 2].min(), com[:, 2].max() + bin_a, bin_a)
    mid = 0.5 * (edges[:-1] + edges[1:])
    phase = np.full(len(mid), np.nan)
    bins = list(pairwise(edges))
    for i, (lo, hi) in enumerate(bins):
        m = (com[:, 2] >= lo) & (com[:, 2] < hi)
        if m.sum() > MIN_BIN_MOLECULES:
            phase[i] = np.angle(np.exp(2j * np.pi * com[m, 1] / pitch).mean())
    ok = ~np.isnan(phase)
    unwrapped = np.full(len(mid), np.nan)
    unwrapped[ok] = np.unwrap(phase[ok]) * pitch / (2 * np.pi)
    ref = int(np.nanargmin(np.abs(mid - np.nanmedian(com[:, 2]))))
    anchor = ly / 2 + ((unwrapped[ref] - ly / 2 + 0.5 * pitch) % pitch) - 0.5 * pitch
    centre = anchor + centre_shift + unwrapped - unwrapped[ref]
    mask = np.zeros(len(com), dtype=bool)
    for i, (lo, hi) in enumerate(bins):
        if np.isnan(centre[i]):
            continue
        m = (com[:, 2] >= lo) & (com[:, 2] < hi)
        d_class = ((com[m, 1] - centre[i] + pitch) % (2 * pitch)) - pitch
        if m.any() and np.abs(d_class).min() > 0.5 * width:
            centre[i] += d_class[np.argmin(np.abs(d_class))]
        d = ((com[:, 1] - centre[i] + 0.5 * ly) % ly) - 0.5 * ly
        mask |= m & (np.abs(d) < 0.5 * width)
    return mask


def load_md_cutout_wide(  # noqa: PLR0915
    tag: str = DEFAULT_TAG,
    *,
    total_thick: float,
    slab_y: float = SLAB_Y,
    lateral_half_a: float,
    show_h: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load last-frame ZnPc molecules in a y-slab, tiled periodically along x.

    Parameters
    ----------
    tag
        Dump stem under ``md/znpc_film/bulk``.
    total_thick
        Graded thickness in angstrom; flips MD height into vacuum-at-zero depth.
    slab_y
        Window width in y (angstrom), centred between two neighbouring
        pi-stack columns and following their lateral drift with depth (see
        :func:`column_following_mask`); a molecule is kept whole by its centre
        of mass.
    lateral_half_a
        Keep molecules whose centre of mass lies within this x half-width
        (angstrom) of the box centre. Wider than the box half-width pulls in
        periodic images, so the cutout fills a panel wider than ``Lx``.
    show_h
        Keep hydrogen when true.

    Returns
    -------
    pts, elems, bonds, kind, behind_atom, behind_bond
        Atom positions ``(n, 3)`` in angstrom as ``(x, y, z_fig)`` with x centred
        and y shifted to start at zero, element labels ``(n,)``, bond segment
        endpoints ``(m, 2, 3)``, a bond class ``(m,)`` (0 for carbon-only,
        1 for bonds touching nitrogen, 2 for bonds touching zinc), and for every
        atom and bond the distance in angstrom of its molecule behind the front
        pi-stack column at the same depth (zero for the front column, taken as
        the smallest column ``y`` among molecules within
        :data:`FRONT_DEPTH_WINDOW_A` in depth).
    """
    pt = _load_profile_tools()
    meta = load_meta(tag)
    zref = float(meta["silica_top"]) + float(meta.get("d0", 2.0))
    box, data = trailing_frames(tag)[-1]
    lx = float(box[0, 1] - box[0, 0])
    ly = float(box[1, 1] - box[1, 0])
    mols: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    raw = list(pt.molecules(box, data))
    coms = np.array([xyz.mean(0) for _, xyz, _, _ in raw])
    coms_depth = np.column_stack(
        [coms[:, 0] % lx, coms[:, 1] % ly, total_thick - (coms[:, 2] - zref)]
    )
    in_slab = column_following_mask(
        coms_depth, ly, width=slab_y, centre_shift=0.5 * ly / N_COLUMNS
    )
    for (tt, xyz, _n, _g), com, keep_mol in zip(raw, coms, in_slab, strict=True):
        if not keep_mol:
            continue
        y_com = float(com[1] % ly)
        els = np.array([TYPE_TO_ELEM[int(t)] for t in tt])
        keep = show_h | (els != "H")
        if not keep.any():
            continue
        d = xyz - com
        d[:, 0] -= lx * np.round(d[:, 0] / lx)
        d[:, 1] -= ly * np.round(d[:, 1] / ly)
        rel = d[keep]
        rel_els = els[keep]
        rel[:, 2] = total_thick - (com[2] + d[keep][:, 2] - zref)
        dm = np.linalg.norm(rel[:, None, :] - rel[None, :, :], axis=-1)
        zn = (rel_els == "Zn")[:, None] | (rel_els == "Zn")[None, :]
        ii, jj = np.nonzero(np.triu(dm < np.where(zn, 2.2, 1.75), k=1))
        n_any = (rel_els == "N")[ii] | (rel_els == "N")[jj]
        kind = np.where(zn[ii, jj], 2, np.where(n_any, 1, 0))
        idx = np.column_stack([ii, jj])
        mols.append((float(com[0]), y_com, rel, rel_els, idx, kind))
    if not mols:
        raise RuntimeError(f"no ZnPc atoms in front slab of {tag}")
    x0 = float(np.median([m[0] for m in mols]))
    pts_l, el_l, bond_l, kind_l, behind_atom_l, behind_bond_l = [], [], [], [], [], []
    n_img = int(np.ceil(lateral_half_a / lx)) + 1
    y_coms = np.array([m[1] for m in mols])
    z_coms = np.array([m[2][:, 2].mean() for m in mols])
    behind_a = np.array(
        [
            y_coms[i] - y_coms[np.abs(z_coms - z_coms[i]) < FRONT_DEPTH_WINDOW_A].min()
            for i in range(len(mols))
        ]
    )
    for (x_com, y_com, rel, els, idx, kind), behind in zip(mols, behind_a, strict=True):
        base = x_com - x0
        base -= lx * np.round(base / lx)
        for k in range(-n_img, n_img + 1):
            dx = base + k * lx
            if abs(dx) > lateral_half_a:
                continue
            xyz = rel + np.array([dx, y_com, 0.0])
            pts_l.append(xyz)
            el_l.append(els)
            bond_l.append(np.stack([xyz[idx[:, 0]], xyz[idx[:, 1]]], axis=1))
            kind_l.append(kind)
            behind_atom_l.append(np.full(len(els), behind))
            behind_bond_l.append(np.full(len(idx), behind))
    if not pts_l:
        raise RuntimeError(f"no ZnPc molecules in lateral window of {tag}")
    pts = np.vstack(pts_l)
    y_min = float(pts[:, 1].min())
    pts[:, 1] -= y_min
    bonds = np.concatenate(bond_l)
    bonds[:, :, 1] -= y_min
    return (
        pts,
        np.concatenate(el_l),
        bonds,
        np.concatenate(kind_l),
        np.concatenate(behind_atom_l),
        np.concatenate(behind_bond_l),
    )


def load_md_slab(
    tag: str = DEFAULT_TAG,
    *,
    total_thick: float,
    slab_y: float = SLAB_Y,
    show_h: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
    """Load last-frame ZnPc atoms as a 2D ``(x, z_fig)`` projection.

    Parameters
    ----------
    tag
        Dump stem under ``md/znpc_film/bulk``.
    total_thick
        Graded thickness in angstrom; flips MD height into vacuum-at-zero depth.
    slab_y
        Front-slab thickness in y (angstrom).
    show_h
        When false, drop hydrogen atoms and H-involving bonds.

    Returns
    -------
    pts, elems, segs_xyz
        Atom positions in angstrom as ``(n, 2)`` of ``(lateral_x, z_fig)``,
        element labels, and bond segments as pairs of 2D points (still in A).
    """
    pts3, elems, segs3 = load_md_cutout_3d(
        tag,
        total_thick=total_thick,
        slab_y=slab_y,
        show_h=show_h,
        lateral_half_a=None,
    )
    pts = pts3[:, [0, 2]]
    segs = [[s[0, [0, 2]], s[1, [0, 2]]] for s in segs3]
    return pts, elems, segs


def _lateral_half_for_axes(
    ax, *, depth_span_nm: float, horiz: bool, override: float | None
) -> float:
    """Half-width (nm) so one angstrom is square in the laid-out axes box."""
    if override is not None:
        return float(override)
    fig = ax.get_figure()
    pos = ax.get_position()
    size = fig.get_size_inches()
    w_in = pos.width * size[0]
    h_in = pos.height * size[1]
    if horiz:
        return 0.5 * depth_span_nm * (h_in / max(w_in, 1e-9))
    return 0.5 * depth_span_nm * (w_in / max(h_in, 1e-9))


def draw_md_structure(  # noqa: PLR0913
    ax,
    tag: str = DEFAULT_TAG,
    *,
    total_thick: float,
    slab_y: float = SLAB_Y,
    show_h: bool = False,
    a2nm: float = 0.1,
    horiz: bool = False,
    lateral_half_nm: float | None = LATERAL_HALF_NM,
    depth_lim_nm: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Draw an orthographic ZnPc ball-and-stick cutout onto ``ax``.

    Projects a thin y-slab of the last dump frame into the (lateral, depth)
    plane, matching the visual language of ``fig4_panel_b_growth`` (CPK atoms,
    distance bonds, no hydrogen by default). Depth shares the FIG4 vacuum-at-
    zero convention. Lateral limits are chosen so length units match on both
    axes without fighting the GridSpec box (no ``set_aspect``).

    Parameters
    ----------
    ax
        Target matplotlib axes (structure column of FIG4).
    tag
        Dump stem.
    total_thick
        Graded thickness in angstrom.
    slab_y
        Front-slab thickness in y (angstrom).
    show_h
        Include hydrogen when true.
    a2nm
        Angstrom to nanometre scale factor.
    horiz
        When true, depth runs on x (historical layout); otherwise depth is y.
    lateral_half_nm
        Half-width of the lateral window in nm. ``None`` sizes it from the
        axes aspect so one angstrom is square without shrinking the box.
    depth_lim_nm
        Optional depth-axis limits in nm; when omitted, span the film from the
        projected atom extents padded slightly.

    Returns
    -------
    dict
        ``n_atoms``, ``n_bonds``, ``lateral_half_nm``, and depth span used.
    """
    pts_a, elems, segs_a = load_md_slab(
        tag, total_thick=total_thick, slab_y=slab_y, show_h=show_h
    )
    pts = pts_a * a2nm
    segs = [[p * a2nm, q * a2nm] for p, q in segs_a]

    if depth_lim_nm is None:
        zmin = float(pts[:, 1].min()) - 0.3
        zmax = float(pts[:, 1].max()) + 0.3
        depth_lim = (zmin, zmax) if horiz else (zmax, zmin)
    else:
        depth_lim = depth_lim_nm

    depth_span_nm = abs(depth_lim[1] - depth_lim[0])
    depth_span_a = depth_span_nm / a2nm
    ax.figure.canvas.draw()
    half = _lateral_half_for_axes(
        ax, depth_span_nm=depth_span_nm, horiz=horiz, override=lateral_half_nm
    )
    pad = 0.4
    keep = np.abs(pts[:, 0]) <= half + pad
    pts = pts[keep]
    elems = elems[keep]
    segs = [
        [p, q] for p, q in segs if abs(p[0]) <= half + pad and abs(q[0]) <= half + pad
    ]

    if horiz:
        bond_segs = [[(p[1], p[0]), (q[1], q[0])] for p, q in segs]
        scatter_x, scatter_y = pts[:, 1], pts[:, 0]
        ax.set_xlim(*depth_lim)
        ax.set_ylim(-half, half)
    else:
        bond_segs = [[(p[0], p[1]), (q[0], q[1])] for p, q in segs]
        scatter_x, scatter_y = pts[:, 0], pts[:, 1]
        ax.set_xlim(-half, half)
        ax.set_ylim(*depth_lim)

    # Equal data aspect; shrink the axes box inside the GridSpec cell so a
    # narrow cutout stays geometrically correct without imshow squash. Marker
    # sizes must be computed after the box settles.
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.figure.canvas.draw()
    ppa = _points_per_angstrom(ax, depth_span_a=depth_span_a, horiz=horiz)

    ax.add_collection(
        LineCollection(
            bond_segs,
            colors=BOND_COLOR,
            linewidths=BOND_WIDTH_A * ppa,
            zorder=3,
            capstyle="round",
        )
    )
    for e, col in ELEM_COLORS.items():
        m = elems == e
        if m.any():
            ax.scatter(
                scatter_x[m],
                scatter_y[m],
                s=(2 * ELEM_RADIUS[e] * ATOM_DRAW_SCALE * ppa) ** 2,
                c=col,
                edgecolors="none",
                zorder=4,
            )
    return {
        "n_atoms": float(len(pts)),
        "n_bonds": float(len(segs)),
        "lateral_half_nm": float(half),
        "depth_span_nm": float(depth_span_nm),
    }


BOND_RGB_3D = {0: (0.18, 0.18, 0.18), 1: (0.16, 0.30, 0.78), 2: (0.52, 0.28, 0.72)}
_SUB_TOP_3D = "#d2d2d2"
_SUB_FRONT_3D = ("#a6a6a6", "#4f5459")
_SUB_SIDE_3D = ("#8c8c8c", "#3d4247")


def _mix(c0: str, c1: str, t: float) -> np.ndarray:
    """Linear RGB blend of two colour specs at fraction ``t`` in ``[0, 1]``."""
    a = np.array(matplotlib.colors.to_rgb(c0))
    b = np.array(matplotlib.colors.to_rgb(c1))
    return (1.0 - t) * a + t * b


def _behind_rgba(rgb: np.ndarray, behind_a: np.ndarray) -> np.ndarray:
    """Colour front-column items in full and grey out those behind the front stack.

    Parameters
    ----------
    rgb
        Element or bond colours ``(n, 3)`` in ``[0, 1]``.
    behind_a
        Distance ``(n,)`` in angstrom behind the front pi-stack column.

    Returns
    -------
    numpy.ndarray
        ``(n, 4)`` colours. Items whose ``behind_a`` is below
        :data:`FRONT_THICKNESS_A` keep their colour at :data:`FRONT_ALPHA`; items
        beyond it blend :data:`BEHIND_MIX` of the way to a neutral grey and drop to
        :data:`BEHIND_ALPHA`, over a smooth :data:`FRONT_EDGE_WIDTH_A` transition.
    """
    w = np.clip((behind_a - FRONT_THICKNESS_A) / FRONT_EDGE_WIDTH_A + 0.5, 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    grey = np.full_like(rgb, BEHIND_GREY)
    colour = (1.0 - BEHIND_MIX * w)[:, None] * rgb + (BEHIND_MIX * w)[:, None] * grey
    alpha = FRONT_ALPHA + (BEHIND_ALPHA - FRONT_ALPHA) * w
    return np.column_stack([colour, alpha])


def _view_geometry(azim: float, elev: float) -> tuple[float, float, float]:
    """Return ``(cos a, sin a, shear)`` for an oblique orthographic view.

    Depth ``z`` stays on the screen's vertical axis unscaled, so the panel shares
    its depth axis with the profile panels; looking from above only shifts a
    point by ``shear`` times its distance behind the cutout centre.
    """
    a = np.radians(azim)
    return float(np.cos(a)), float(np.sin(a)), float(np.tan(np.radians(elev)))


def draw_md_structure_3d(  # noqa: PLR0913, PLR0915
    ax,
    tag: str = DEFAULT_TAG,
    *,
    total_thick: float,
    depth_lim_nm: tuple[float, float],
    slab_y: float = SLAB_Y,
    show_h: bool = False,
    a2nm: float = 0.1,
    azim: float = 18.0,
    elev: float = 9.0,
    tau_vac_nm: float | None = None,
    tau_si_nm: float | None = None,
    sub_nm: float = 1.0,
    label_tau: bool = True,
    margin_nm: float = 0.35,
) -> dict[str, float]:
    """Draw a thin-wire ZnPc film cutout as an oblique 3D scene on a 2D axes.

    The vertical screen axis is film depth in nm, drawn on ``ax`` with the same
    scale and limits as the profile panels (``ax`` must share its y axis with
    them), so the tau hyperplanes, the vacuum surface and the silicon surface
    land on the same rows as the dashed lines in the neighbouring panels. The
    lateral scale equals the depth scale; the cutout is tiled periodically along
    x to fill the panel width, and every molecule is drawn whole. Silicon is a
    solid block behind the molecules, so nothing is clipped at the interface.

    Parameters
    ----------
    ax
        Target axes sharing the depth axis (inverted, vacuum at the top).
    tag
        Dump stem.
    total_thick
        Graded film thickness in angstrom.
    depth_lim_nm
        ``(bottom, top)`` depth limits in nm, identical to the profile panels.
    slab_y
        Slab thickness along the viewing depth in angstrom.
    show_h
        Include hydrogen when true.
    a2nm
        Angstrom to nanometre factor.
    azim, elev
        Azimuth about the depth axis and apparent elevation in degrees.
    tau_vac_nm, tau_si_nm
        Decay lengths in nm; hyperplanes are drawn at ``tau_vac`` and at
        ``total - tau_si`` when supplied.
    sub_nm
        Silicon block thickness below the film in nm.
    label_tau
        Annotate the hyperplanes when true.
    margin_nm
        Free space kept between the scene and the axes edge in nm.

    Returns
    -------
    dict
        Atom and bond counts and the half-widths used.
    """
    cos_a, sin_a, shear = _view_geometry(azim, elev)
    total_nm = float(total_thick) * a2nm
    fig = ax.get_figure()
    pos = ax.get_position()
    w_in = pos.width * fig.get_size_inches()[0]
    h_in = pos.height * fig.get_size_inches()[1]
    span_nm = abs(depth_lim_nm[0] - depth_lim_nm[1])
    half_x_nm = 0.5 * w_in * span_nm / h_in
    y_half_nm = 0.5 * slab_y * a2nm + 0.6
    x_box = (half_x_nm - margin_nm - y_half_nm * sin_a) / cos_a

    pts_a, elems, bonds_a, kind, behind_atom, behind_bond = load_md_cutout_wide(
        tag,
        total_thick=total_thick,
        slab_y=slab_y,
        lateral_half_a=x_box / a2nm - 3.0,
        show_h=show_h,
    )
    y_mid = 0.5 * (pts_a[:, 1].min() + pts_a[:, 1].max())

    def project(p: np.ndarray, *, scale: float = a2nm, y_c: float = y_mid * a2nm):
        """Screen ``(sx, depth_row, distance)`` of ``(.., 3)`` points in nm or A."""
        x, y, z = p[..., 0] * scale, p[..., 1] * scale - y_c, p[..., 2] * scale
        dist = y * cos_a - x * sin_a
        return x * cos_a + y * sin_a, z - shear * dist, dist

    y0, y1 = -y_half_nm, y_half_nm

    def corner(x: float, y: float, z: float) -> tuple[float, float]:
        sx, row, _ = project(np.array([x, y + y_mid * a2nm, z]), scale=1.0)
        return float(sx), float(row)

    def face(pts: list[tuple[float, float, float]], color, *, z_order: float, **kw):
        xy = [corner(*p) for p in pts]
        ax.fill(
            [p[0] for p in xy], [p[1] for p in xy], color=color, zorder=z_order, **kw
        )

    def gradient_face(  # noqa: PLR0913
        c_top, c_bot, edge, x_a, y_a, x_b, y_b, z_a, z_b, z_order
    ):
        n_strip = 28
        zs = np.linspace(z_a, z_b, n_strip + 1)
        for i in range(n_strip):
            face(
                [
                    (x_a, y_a, zs[i]),
                    (x_b, y_b, zs[i]),
                    (x_b, y_b, zs[i + 1]),
                    (x_a, y_a, zs[i + 1]),
                ],
                _mix(c_top, c_bot, (i + 0.5) / n_strip),
                z_order=z_order,
                lw=0.0,
                antialiased=False,
            )
        xs = [
            corner(x, y, z)
            for x, y, z in [
                (x_a, y_a, z_a),
                (x_b, y_b, z_a),
                (x_b, y_b, z_b),
                (x_a, y_a, z_b),
            ]
        ]
        ax.plot(
            [p[0] for p in xs] + [xs[0][0]],
            [p[1] for p in xs] + [xs[0][1]],
            color=edge,
            lw=0.5,
            zorder=z_order + 0.2,
            solid_joinstyle="miter",
        )

    z_si = total_nm
    _, rows_front, _ = project(
        np.array([[sx, y0 + y_mid * a2nm, 0.0] for sx in (-x_box, x_box)]), scale=1.0
    )
    z_bot = min(
        total_nm + float(sub_nm),
        max(depth_lim_nm) - float(np.max(rows_front)) - 0.01,
    )
    face(
        [(-x_box, y0, z_si), (x_box, y0, z_si), (x_box, y1, z_si), (-x_box, y1, z_si)],
        _SUB_TOP_3D,
        z_order=1.0,
        lw=0.0,
    )
    for t in np.linspace(-1.0, 1.0, 9)[1:-1]:
        xa, ya = corner(t * x_box, y0, z_si), corner(t * x_box, y1, z_si)
        ax.plot([xa[0], ya[0]], [xa[1], ya[1]], color="0.62", lw=0.3, zorder=1.1)
    for t in np.linspace(-1.0, 1.0, 5)[1:-1]:
        xa, ya = corner(-x_box, t * y_half_nm, z_si), corner(x_box, t * y_half_nm, z_si)
        ax.plot([xa[0], ya[0]], [xa[1], ya[1]], color="0.62", lw=0.3, zorder=1.1)
    gradient_face(*_SUB_SIDE_3D, "0.3", x_box, y0, x_box, y1, z_si, z_bot, 1.2)
    gradient_face(*_SUB_FRONT_3D, "0.3", -x_box, y0, x_box, y0, z_si, z_bot, 1.2)
    top_edge = [
        corner(-x_box, y0, z_si),
        corner(x_box, y0, z_si),
        corner(x_box, y1, z_si),
        corner(-x_box, y1, z_si),
        corner(-x_box, y0, z_si),
    ]
    ax.plot(
        [p[0] for p in top_edge],
        [p[1] for p in top_edge],
        color="0.3",
        lw=0.6,
        zorder=1.4,
    )
    sx_c, row_c = corner(0.0, y0, 0.5 * (z_si + z_bot))
    ax.text(
        sx_c,
        row_c,
        "Si",
        color="white",
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=1.5,
    )

    for sx_e, y_e in ((-x_box, y1), (x_box, y1), (-x_box, y0), (x_box, y0)):
        a_, b_ = corner(sx_e, y_e, 0.0), corner(sx_e, y_e, z_si)
        ax.plot(
            [a_[0], b_[0]],
            [a_[1], b_[1]],
            color="0.6",
            lw=0.35,
            ls=(0, (2, 2)),
            zorder=0.5,
        )
    top = [
        corner(-x_box, y0, 0.0),
        corner(x_box, y0, 0.0),
        corner(x_box, y1, 0.0),
        corner(-x_box, y1, 0.0),
        corner(-x_box, y0, 0.0),
    ]
    face(
        [(-x_box, y0, 0.0), (x_box, y0, 0.0), (x_box, y1, 0.0), (-x_box, y1, 0.0)],
        "#9fb8c8",
        z_order=4.2,
        alpha=0.16,
        lw=0.0,
    )
    ax.plot(
        [p[0] for p in top], [p[1] for p in top], color="#5f7f93", lw=0.7, zorder=4.3
    )
    for t in np.linspace(-1.0, 1.0, 9)[1:-1]:
        xa, ya = corner(t * x_box, y0, 0.0), corner(t * x_box, y1, 0.0)
        ax.plot(
            [xa[0], ya[0]],
            [xa[1], ya[1]],
            color="#5f7f93",
            lw=0.25,
            alpha=0.6,
            zorder=4.3,
        )
    for t in np.linspace(-1.0, 1.0, 5)[1:-1]:
        xa = corner(-x_box, t * y_half_nm, 0.0)
        ya = corner(x_box, t * y_half_nm, 0.0)
        ax.plot(
            [xa[0], ya[0]],
            [xa[1], ya[1]],
            color="#5f7f93",
            lw=0.25,
            alpha=0.6,
            zorder=4.3,
        )

    sx0, row0, d0 = project(bonds_a[:, 0, :])
    sx1, row1, d1 = project(bonds_a[:, 1, :])
    dist = 0.5 * (d0 + d1)
    order = np.argsort(-dist)
    rgb = np.array([BOND_RGB_3D[int(k)] for k in kind])
    rgba = _behind_rgba(rgb, behind_bond)
    segs = np.stack(
        [np.column_stack([sx0, row0]), np.column_stack([sx1, row1])], axis=1
    )
    ax.add_collection(
        LineCollection(segs[order], colors=rgba[order], linewidths=BOND_LW_3D, zorder=3)
    )
    zn = elems == "Zn"
    if zn.any():
        szn, rzn, _ = project(pts_a[zn])
        zn_rgba = _behind_rgba(
            np.tile(matplotlib.colors.to_rgb(ELEM_COLORS["Zn"]), (int(zn.sum()), 1)),
            behind_atom[zn],
        )
        zn_front = zn_rgba[:, 3] > 0.5 * (FRONT_ALPHA + BEHIND_ALPHA)
        for sel, z_order in ((~zn_front, 2.9), (zn_front, 3.5)):
            ax.scatter(
                szn[sel],
                rzn[sel],
                s=ATOM_SIZE_3D["Zn"],
                c=zn_rgba[sel],
                edgecolors="none",
                zorder=z_order,
            )

    planes: list[tuple[float, str, str]] = []
    if tau_vac_nm is not None:
        planes.append((float(tau_vac_nm), r"$\tau_\mathrm{vac}$", "bottom"))
    if tau_si_nm is not None:
        planes.append((total_nm - float(tau_si_nm), r"$\tau_\mathrm{Si}$", "top"))
    for z_tau, text, va in planes:
        face(
            [
                (-x_box, y0, z_tau),
                (x_box, y0, z_tau),
                (x_box, y1, z_tau),
                (-x_box, y1, z_tau),
            ],
            TAU_COLOR_3D,
            z_order=4,
            alpha=0.20,
            lw=0.0,
        )
        ring = [
            corner(-x_box, y0, z_tau),
            corner(x_box, y0, z_tau),
            corner(x_box, y1, z_tau),
            corner(-x_box, y1, z_tau),
            corner(-x_box, y0, z_tau),
        ]
        ax.plot(
            [p[0] for p in ring],
            [p[1] for p in ring],
            color=TAU_COLOR_3D,
            lw=0.8,
            ls=(0, (4, 3)),
            zorder=4.5,
        )
        if label_tau:
            sx_l, row_l = corner(-x_box, y0, z_tau)
            ax.text(
                sx_l,
                row_l + (-0.1 if va == "bottom" else 0.1),
                text,
                color=TAU_COLOR_3D,
                fontsize=8,
                ha="left",
                va=va,
                zorder=6,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )

    ax.set_xlim(-half_x_nm, half_x_nm)
    ax.set_ylim(*depth_lim_nm)
    ax.set_axis_off()
    return {
        "n_atoms": float(len(pts_a)),
        "n_bonds": float(len(bonds_a)),
        "half_x_nm": float(half_x_nm),
        "x_box_nm": float(x_box),
    }


def render_molecule_strip(  # noqa: PLR0913
    tag: str = DEFAULT_TAG,
    *,
    total_thick: float,
    out: Path | None = None,
    slab_y: float = SLAB_Y,
    dpi: int = 220,
    a2nm: float = 0.1,
) -> Path:
    """Write a standalone preview PNG of the 2D atomistic cutout.

    Parameters
    ----------
    tag
        Dump stem.
    total_thick
        Graded thickness in angstrom.
    out
        Output PNG path; defaults to ``FIG4_md_strip.png`` next to this module.
    slab_y
        Front-slab thickness in y (angstrom).
    dpi
        Raster resolution.
    a2nm
        Angstrom to nanometre scale factor.

    Returns
    -------
    Path
        Written PNG path.
    """
    out = out or (Path(__file__).resolve().parent / "FIG4_md_strip.png")
    depth_nm = total_thick * a2nm
    fig_h = max(4.0, depth_nm * 0.35)
    fig = plt.figure(figsize=(1.6, fig_h), dpi=dpi)
    ax = fig.add_axes([0.05, 0.02, 0.9, 0.96])
    draw_md_structure(
        ax,
        tag,
        total_thick=total_thick,
        slab_y=slab_y,
        a2nm=a2nm,
        horiz=False,
        depth_lim_nm=(depth_nm + 1.0, -0.5),
    )
    ax.set_axis_off()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.savefig(out, dpi=dpi, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return out


def render_molecule_strip_3d(  # noqa: PLR0913
    tag: str = DEFAULT_TAG,
    *,
    total_thick: float,
    out: Path | None = None,
    slab_y: float = SLAB_Y,
    dpi: int = 220,
    a2nm: float = 0.1,
    tau_vac_nm: float | None = None,
    tau_si_nm: float | None = None,
    sub_nm: float = 1.0,
    figsize: tuple[float, float] = (2.9, 3.9),
    azim: float = 28.0,
    elev: float = 12.0,
) -> Path:
    """Write a standalone preview of the structure panel as PNG and PDF.

    Parameters
    ----------
    tag
        Dump stem.
    total_thick
        Graded thickness in angstrom.
    out
        Output PNG path; defaults to ``FIG4_md_strip_3d_preview.png``.
    slab_y
        Slab thickness along the viewing depth in angstrom.
    dpi
        Raster resolution.
    a2nm
        Angstrom to nanometre scale factor.
    tau_vac_nm, tau_si_nm
        Decay lengths in nm for the hyperplanes (optional).
    sub_nm
        Silicon block thickness below the film in nm.
    figsize
        Figure size in inches; its aspect sets how wide the cutout is tiled.
    azim, elev
        View angles in degrees.

    Returns
    -------
    Path
        Written PNG path.
    """
    out = out or (Path(__file__).resolve().parent / "FIG4_md_strip_3d_preview.png")
    depth_nm = total_thick * a2nm
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    draw_md_structure_3d(
        ax,
        tag,
        total_thick=total_thick,
        slab_y=slab_y,
        a2nm=a2nm,
        depth_lim_nm=(depth_nm + sub_nm, -0.1),
        tau_vac_nm=tau_vac_nm,
        tau_si_nm=tau_si_nm,
        sub_nm=sub_nm,
        azim=azim,
        elev=elev,
    )
    fig.savefig(out, dpi=dpi, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    """CLI: write strip preview PNG and profile cache for FIG4."""
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--total-thick", type=float, default=188.7)
    ap.add_argument("--nlast", type=int, default=10)
    ap.add_argument("--slab-y", type=float, default=SLAB_Y)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--mode",
        choices=("2d", "3d"),
        default="3d",
        help="Preview projection: thin-wire 3D (default) or orthographic 2D.",
    )
    a = ap.parse_args()
    z, g, r = md_profiles(a.tag, total_thick=a.total_thick, nlast=a.nlast)
    out = a.out or Path(__file__).resolve().parent / (
        "FIG4_md_strip_3d_preview.png" if a.mode == "3d" else "FIG4_md_strip.png"
    )
    if a.mode == "3d":
        png = render_molecule_strip_3d(
            a.tag, total_thick=a.total_thick, out=out, slab_y=a.slab_y
        )
    else:
        png = render_molecule_strip(
            a.tag, total_thick=a.total_thick, out=out, slab_y=a.slab_y
        )
    cache = out.with_name("FIG4_md_profiles.npz")
    np.savez_compressed(
        cache, z=z, gamma=g, rho=r, total_thick=a.total_thick, tag=a.tag
    )
    print(f"wrote {png}")  # noqa: T201
    print(  # noqa: T201
        f"wrote {cache}  ({len(z)} bins, gamma[{np.nanmin(g):.1f},{np.nanmax(g):.1f}])"
    )


if __name__ == "__main__":
    main()
