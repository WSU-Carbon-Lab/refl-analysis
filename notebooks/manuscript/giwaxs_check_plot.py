# ---------------------------------------------------------------------------
# Replaces everything in giwaxs_check_cell.py from "# Linecuts" onward, and
# supersedes the old beta block at the bottom of that cell. Keep the
# _cell / _stat / q_oop / q_ip definitions above it unchanged.
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

Z_CELL = 4          # molecules per unit cell, alpha-ZnPc convention
BETA_LIT = 121.6    # alpha-ZnPc, monoclinic C2/c
ABC_LIT = (25.92, 3.79, 23.92)

BG = 1.0            # flat pedestal so the peaks sit on a log axis
SHOW_LABELS = False # Miller labels are an assignment, not a model output
AMP = {"200": 100.0, "400": 2.5, "002": 12.0, "pipi": 4.5}

# ---- 1. what the simulation itself gives ------------------------------------
_vol_mol = _P.MOLAR_MASS / (_P.RHO_BULK_REF * _P.N_AVOGADRO * 1e-24)
_w = _rho * np.exp(-((2 * np.pi / _dl) ** 2) * _u**2)
_w = _w / _w.sum()
_area_w = float(_w @ (_dl * _cs))
_d_lam_w, _d_col_w = float(_w @ _dl), float(_w @ _cs)

print("=" * 62)
print("FROM THE SIMULATION")
print(f"  fitted plateau density        {_P.RHO_BULK_REF:8.3f} g/cm3")
print(f"  molar mass (from the xyz)     {_P.MOLAR_MASS:8.2f} g/mol")
print(f"  volume per molecule           {_vol_mol:8.1f} A^3")
print(f"  SLAB_THICK (input)            {_P.SLAB_THICK:8.2f} A")
print(f"  LAT_FRAC   (input)            {_P.LAT_FRAC:8.3f}")
print(f"  bulk cell area = V/SLAB       {_vol_mol / _P.SLAB_THICK:8.2f} A^2")
print(f"  median gamma                  {_PACK['gamma_median']:8.1f} deg")
print("\n  depth-weighted lattice (rho x Debye-Waller over the fitted profile)")
print(f"    lamellar (out-of-plane)  d = {_d_lam_w:6.2f} A    "
      f"q = {q_oop:.3f} +/- {s_oop:.3f}")
print(f"    column   (in-plane)      d = {_d_col_w:6.2f} A    "
      f"q = {q_ip:.3f} +/- {s_ip:.3f}")
print(f"    cell area from the pair     {_area_w:6.2f} A^2  "
      f"(bulk {_vol_mol / _P.SLAB_THICK:.2f})")

# ---- 2. measured, and what it implies for the two knobs ---------------------
_d200, _d002, _d010 = 2 * np.pi / Q_OOP, 2 * np.pi / Q_002, 2 * np.pi / Q_IP
_sin2 = np.sin(np.radians(_PACK["gamma_median"])) ** 2
print("\nMEASURED")
for _lab, _q, _d in (("(200) out-of-plane", Q_OOP, _d200),
                     ("(002) in-plane    ", Q_002, _d002),
                     ("(pi-pi) in-plane  ", Q_IP, _d010)):
    print(f"  {_lab}   q = {_q:6.3f}   d = {_d:6.2f} A")
print("\nIMPLIED KNOBS (from the out-of-plane / pi-pi pair)")
print(f"  SLAB_THICK = {_vol_mol / (_d200 * _d010):6.2f} A    "
      f"(now {_P.SLAB_THICK:.2f})")
print(f"  LAT_FRAC   = {(_d200 / _d010) * _P.PI_STACK / (_P.L_MOL * _sin2):6.3f}"
      f"    (now {_P.LAT_FRAC:.2f})")

# ---- 3. beta ----------------------------------------------------------------
# beta is the monoclinic angle between a and c. It is NOT a simulation output.
# The packing model is a 2D projection with an orthogonal projected lattice and
# a slab thickness measured perpendicular to the drawing plane, so no inclined
# axis exists in it. beta only appears when the perpendicular SLAB_THICK is set
# against a crystallographic spacing measured along an inclined axis.
#
# For monoclinic b-unique with Z molecules per cell,
#   d_200 = (a/2) sin(beta),  d_002 = (c/2) sin(beta),  d_010 = b,
#   V_cell = a b c sin(beta) = Z * V_molecule
# eliminating a, b, c gives
#   sin(beta) = 4 d_200 d_002 d_010 / (Z V_molecule).
# So beta follows from three MEASURED spacings plus the simulation's volume per
# molecule. One measured input, the density, is all the simulation contributes.
_sinb = 4 * _d200 * _d002 * _d010 / (Z_CELL * _vol_mol)
_beta = 180.0 - np.degrees(np.arcsin(np.clip(_sinb, 0.0, 1.0)))
_a, _b, _c = 2 * _d200 / _sinb, _d010, 2 * _d002 / _sinb
_rho_lit = _P.MOLAR_MASS / ((ABC_LIT[0] * ABC_LIT[1] * ABC_LIT[2]
                             * np.sin(np.radians(BETA_LIT)) / Z_CELL)
                            * _P.N_AVOGADRO * 1e-24)

print("\nUNIT CELL (needs all three measured peaks + the fitted density)")
print(f"  sin(beta) = {_sinb:.4f}   ->   beta = {_beta:6.1f} deg   "
      f"(lit {BETA_LIT})")
print(f"  {'':14s}{'this film':>10s}{'alpha-ZnPc':>12s}{'diff':>8s}")
for _n, _v, _l in (("a (A)", _a, ABC_LIT[0]), ("b (A)", _b, ABC_LIT[1]),
                   ("c (A)", _c, ABC_LIT[2]), ("beta (deg)", _beta, BETA_LIT)):
    print(f"  {_n:14s}{_v:10.2f}{_l:12.2f}{100 * (_v / _l - 1):+7.0f}%")
print(f"  density (g/cm3){_P.RHO_BULK_REF:10.3f}{_rho_lit:12.3f}"
      f"{100 * (_P.RHO_BULK_REF / _rho_lit - 1):+7.0f}%")
print("\n  CAUTION. beta lands within ~2 deg of the literature value, but a, b")
print("  and c are off by 13-20%, so the agreement in beta alone is not")
print("  evidence the assignment is right. Treat it as a consistency figure,")
print("  not as a confirmed polymorph.")
print("=" * 62)

# ---- 4. figure --------------------------------------------------------------
q_002 = 2 * np.pi / (_P.SLAB_THICK * np.sin(np.radians(BETA_LIT)))
s_002 = 0.02        # SLAB_THICK is rigid in the model, so no depth dispersion

fig, ax = plt.subplots(figsize=(3.6, 2.7))
_q = np.logspace(np.log10(0.10), np.log10(2.4), 1200)


def _peak(q0, sig, amp):
    return amp * np.exp(-0.5 * ((_q - q0) / max(sig, 0.02)) ** 2)


_oop = BG + _peak(q_oop, s_oop, AMP["200"]) + _peak(2 * q_oop, 2 * s_oop, AMP["400"])
_ip = BG + _peak(q_002, s_002, AMP["002"]) + _peak(q_ip, s_ip, AMP["pipi"])

ax.plot(_q, _oop, color="#e8291c", lw=2.0, label="Out-of-Plane", zorder=3)
ax.plot(_q, _ip, color="#1f37c4", lw=2.0, label="In-Plane", zorder=2)
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
