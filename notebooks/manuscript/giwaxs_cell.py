# ---------------------------------------------------------------------------
# GIWAXS consistency check for the panel (c) packing.
#
# Self-contained. Run any time after the packing cell. Imports
# fig4_panel_b_growth directly rather than borrowing the notebook's _fig41
# alias, so it depends on no notebook names at all. Replaces the whole of the
# previous GIWAXS cell.
#
# The strip's projected lattice is orthogonal: a lamellar period along depth
# and a column period in-plane. Their PRODUCT is the cell area, fixed by rho(z)
# and SLAB_THICK; their RATIO is fixed by L_MOL * LAT_FRAC / PI_STACK and
# sin^2(gamma). So the two measured in-plane / out-of-plane peaks pin the two
# knobs independently.
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

import fig4_panel_b_growth as growth

# ---- inputs ----------------------------------------------------------------
# Measured peak centres (1/A). Edit to your fitted values.
Q_OOP = 0.51    # out-of-plane lamellar, indexed (200)
Q_002 = 0.52    # in-plane along the viewing direction, indexed (002)
Q_IP = 1.90     # in-plane column stacking, pi-pi

# Packing knobs. None keeps whatever the packing cell set. A number overrides
# the module in place, which is enough for the numbers below, but the DRAWING
# only follows once you put the same value in the packing cell and re-run it.
SLAB_THICK_OVR = None    # try 14.37
L_MOL_OVR = None         # try 14.95, the H-inclusive in-plane diameter
LAT_FRAC_OVR = None      # try 0.955 alongside L_MOL = 14.95

Z_CELL = 4               # molecules per unit cell, alpha-ZnPc convention
BETA_LIT = 121.6         # alpha-ZnPc, monoclinic C2/c
ABC_LIT = (25.92, 3.79, 23.92)

BG = 1.0                 # flat pedestal so peaks sit on a log axis
SHOW_LABELS = False      # Miller labels are an assignment, not a model output
AMP = {"200": 100.0, "400": 2.5, "002": 12.0, "pipi": 4.5}

_applied = []
for _name, _val in (("SLAB_THICK", SLAB_THICK_OVR), ("L_MOL", L_MOL_OVR),
                    ("LAT_FRAC", LAT_FRAC_OVR)):
    if _val is not None:
        setattr(growth, _name, float(_val))
        _applied.append(f"{_name}={_val}")

# ---- 1. the lattice the simulation predicts --------------------------------
# This is the only place L_MOL, LAT_FRAC and PI_STACK do any work. They set the
# cell's aspect ratio; rho(z) and SLAB_THICK set its area.
def _lattice(z):
    """Lamellar period (depth) and column period (lateral) at depth z, in A."""
    s_g = float(np.clip(abs(np.sin(np.radians(growth._gamma(z)))),
                        growth.SIN_GAMMA_MIN, 1.0))
    area = growth.cell_area(z)
    ratio = growth.L_MOL * growth.LAT_FRAC * s_g**2 / growth.PI_STACK
    aniso = 1.0 + growth.STRAIN_ANISO_SI * float(
        np.exp(-(growth.TOTAL - z) / growth.TAU_SI))
    return np.sqrt(area * ratio) * aniso, np.sqrt(area / ratio) / aniso


_z = growth.Z_G[(growth.Z_G >= 0.0) & (growth.Z_G <= growth.TOTAL)]
_dl, _cs = np.array([_lattice(z) for z in _z]).T

# GIWAXS samples the film by scattering power, not by volume. Weight each depth
# by the local density times a Debye-Waller factor built from the same
# positional disorder the strip draws, so the buried interface contributes
# material but little coherent Bragg intensity.
_u = growth.POS_DISORDER_VAC + (growth.POS_DISORDER_SI - growth.POS_DISORDER_VAC) * (
    _z / growth.TOTAL)
_rho = np.clip(np.interp(_z, growth.Z_G, growth.RHO_MAT), 0.0, None)


def _stat(d):
    """Weighted mean q and its depth dispersion for a spacing profile d(z)."""
    q = 2 * np.pi / d
    w = _rho * np.exp(-(q**2) * _u**2)
    w = w / w.sum()
    qm = float(w @ q)
    return qm, float(np.sqrt(w @ (q - qm) ** 2)), float(w @ d)


q_oop, s_oop, d_oop = _stat(_dl)
q_ip, s_ip, d_ip = _stat(_cs)

_vol_mol = growth.MOLAR_MASS / (growth.RHO_BULK_REF * growth.N_AVOGADRO * 1e-24)
_gam_med = float(np.median(np.interp(_z, growth.Z_G, growth.ALPHA_G)))
_sin2 = np.sin(np.radians(_gam_med)) ** 2

print("=" * 64)
if _applied:
    print("OVERRIDES APPLIED TO THE MODULE: " + ", ".join(_applied))
    print("  (re-run the packing cell with the same values to move the drawing)")
print("FROM THE SIMULATION")
print(f"  fitted plateau density         {growth.RHO_BULK_REF:8.3f} g/cm3")
print(f"  molar mass (from the xyz)      {growth.MOLAR_MASS:8.2f} g/mol")
print(f"  volume per molecule            {_vol_mol:8.1f} A^3")
print(f"  SLAB_THICK                     {growth.SLAB_THICK:8.2f} A   (area)")
print(f"  L_MOL x LAT_FRAC               "
      f"{growth.L_MOL * growth.LAT_FRAC:8.2f} A   (aspect ratio)")
print(f"  PI_STACK                       {growth.PI_STACK:8.2f} A")
print(f"  median gamma                   {_gam_med:8.1f} deg")
print(f"  bulk cell area = V/SLAB_THICK  {_vol_mol / growth.SLAB_THICK:8.2f} A^2")
print("\n  depth-weighted lattice (rho x Debye-Waller over the fitted profile)")
print(f"    lamellar (out-of-plane)   d = {d_oop:6.2f} A   "
      f"q = {q_oop:.3f} +/- {s_oop:.3f}")
print(f"    column   (in-plane)       d = {d_ip:6.2f} A   "
      f"q = {q_ip:.3f} +/- {s_ip:.3f}")

# ---- 2. measured, and the knobs it implies ---------------------------------
_d200, _d002, _d010 = 2 * np.pi / Q_OOP, 2 * np.pi / Q_002, 2 * np.pi / Q_IP
print("\nMEASURED vs MODEL")
for _lab, _qm, _dm, _qs, _ds in (
        ("(200) out-of-plane", Q_OOP, _d200, q_oop, d_oop),
        ("(pi-pi) in-plane  ", Q_IP, _d010, q_ip, d_ip)):
    print(f"  {_lab}  q {_qm:6.3f} / {_qs:6.3f}   "
          f"d {_dm:6.2f} / {_ds:6.2f}   {100 * (_ds / _dm - 1):+5.1f}%")
print("\nIMPLIED KNOBS (from the out-of-plane / pi-pi pair)")
print(f"  SLAB_THICK       = {_vol_mol / (_d200 * _d010):7.2f} A   "
      f"(now {growth.SLAB_THICK:.2f})")
print(f"  L_MOL x LAT_FRAC = {(_d200 / _d010) * growth.PI_STACK / _sin2:7.2f} A   "
      f"(now {growth.L_MOL * growth.LAT_FRAC:.2f})")
print("    L_MOL and LAT_FRAC are exactly degenerate, only the product matters:")
for _lm in (13.18, 13.60, 14.95):
    print(f"      L_MOL = {_lm:5.2f}  ->  LAT_FRAC = "
          f"{(_d200 / _d010) * growth.PI_STACK / _sin2 / _lm:.3f}")

# ---- 3. beta and the implied unit cell -------------------------------------
# beta is the monoclinic angle between a and c. It is NOT a simulation output:
# the packing model is a 2D projection with an orthogonal projected lattice and
# a slab thickness measured perpendicular to the page, so it holds no inclined
# axis. beta only appears when that perpendicular thickness is set against a
# spacing measured along an inclined crystallographic axis.
#
# For monoclinic b-unique with Z molecules per cell,
#   d_200 = (a/2) sin(beta),  d_002 = (c/2) sin(beta),  d_010 = b,
#   V_cell = a b c sin(beta) = Z * V_molecule
# eliminating a, b and c gives
#   sin(beta) = 4 d_200 d_002 d_010 / (Z V_molecule)
# so beta comes from three MEASURED spacings; the simulation contributes only
# the volume per molecule.
_sinb = 4 * _d200 * _d002 * _d010 / (Z_CELL * _vol_mol)
_beta = 180.0 - np.degrees(np.arcsin(np.clip(_sinb, 0.0, 1.0)))
_a_f, _b_f, _c_f = 2 * _d200 / _sinb, _d010, 2 * _d002 / _sinb
_rho_lit = growth.MOLAR_MASS / (
    (ABC_LIT[0] * ABC_LIT[1] * ABC_LIT[2] * np.sin(np.radians(BETA_LIT))
     / Z_CELL) * growth.N_AVOGADRO * 1e-24)

print("\nUNIT CELL (all three measured peaks + the fitted density)")
print(f"  {'':16s}{'this film':>10s}{'alpha-ZnPc':>12s}{'diff':>8s}")
for _n, _v, _l in (("a (A)", _a_f, ABC_LIT[0]), ("b (A)", _b_f, ABC_LIT[1]),
                   ("c (A)", _c_f, ABC_LIT[2]), ("beta (deg)", _beta, BETA_LIT),
                   ("density", growth.RHO_BULK_REF, _rho_lit)):
    print(f"  {_n:16s}{_v:10.2f}{_l:12.2f}{100 * (_v / _l - 1):+7.0f}%")
print("\n  CAUTION. beta lands within ~2 deg of the literature value, but a, b")
print("  and c are off by 13-20%, so the beta agreement alone is not evidence")
print("  the assignment is right. Treat it as a consistency figure only.")
print("=" * 64)

# ---- 4. figure --------------------------------------------------------------
# The third direction has no depth resolution in a 2D projection, so (002) gets
# the disorder floor for a width rather than a dispersion.
q_002 = 2 * np.pi / (growth.SLAB_THICK * np.sin(np.radians(BETA_LIT)))
s_002 = 0.02

fig, ax = plt.subplots(figsize=(3.6, 2.7))
_qq = np.logspace(np.log10(0.10), np.log10(2.4), 1200)


def _peak(q0, sig, amp):
    return amp * np.exp(-0.5 * ((_qq - q0) / max(sig, 0.02)) ** 2)


_oop = BG + _peak(q_oop, s_oop, AMP["200"]) + _peak(2 * q_oop, 2 * s_oop,
                                                    AMP["400"])
_ip = BG + _peak(q_002, s_002, AMP["002"]) + _peak(q_ip, s_ip, AMP["pipi"])

ax.plot(_qq, _oop, color="#e8291c", lw=2.0, label="Out-of-Plane", zorder=3)
ax.plot(_qq, _ip, color="#1f37c4", lw=2.0, label="In-Plane", zorder=2)
for _qm, _col in ((Q_OOP, "#e8291c"), (2 * Q_OOP, "#e8291c"),
                  (Q_002, "#1f37c4"), (Q_IP, "#1f37c4")):
    ax.axvline(_qm, color=_col, ls=(0, (3, 2)), lw=0.8, alpha=0.55, zorder=1)

if SHOW_LABELS:
    for _lab, _qc, _yy in ((r"(200)", q_oop, AMP["200"] * 1.5),
                           (r"(002)", q_002, AMP["002"] * 1.6),
                           (r"(400)", 2 * q_oop, AMP["400"] * 1.9),
                           (r"$(\pi-\pi)$", q_ip, AMP["pipi"] * 1.9)):
        ax.annotate(_lab, xy=(_qc, _yy), ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.10, 2.4)
ax.set_ylim(0.7, 600)
ax.xaxis.set_major_locator(LogLocator(base=10.0))
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10))))
ax.xaxis.set_major_formatter(FuncFormatter(
    lambda v, _: f"{v:g}" if v >= 1 else f"{v:.1f}"))
ax.xaxis.set_minor_formatter(FuncFormatter(
    lambda v, _: f"{v / 10 ** np.floor(np.log10(v)):.0f}"))
ax.tick_params(axis="x", which="minor", labelsize=6.5, pad=1.5)
ax.tick_params(axis="x", which="major", labelsize=9)
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(which="both", color="#6a6ad0", ls=(0, (2, 3)), lw=0.45, alpha=0.45)
ax.set_xlabel(r"$Q$ vector ($\mathrm{\AA}^{-1}$)")
ax.set_ylabel("Intensity (a.u.)")
ax.legend(fontsize=7, frameon=True, edgecolor="k", handlelength=1.6,
          loc="upper right")
fig.tight_layout()
