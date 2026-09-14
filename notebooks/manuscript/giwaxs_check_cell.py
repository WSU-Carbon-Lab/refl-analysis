# ---------------------------------------------------------------------------
# GIWAXS check. Run after the packing cell. Reads the drawn lattice straight
# out of fig4_panel_b_growth and compares it to the measured peak positions.
#
# The strip's lattice is orthogonal, lamellar period along depth and column
# period lateral, so the two measured peaks pin both knobs independently:
#   d_lam * c_stack = cell area  -> SLAB_THICK  (set by rho, the film density)
#   d_lam / c_stack = cell ratio -> LAT_FRAC    (set by the aspect ratio)
# Nothing here changes the figure. It only reports what the packing predicts.
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# Measured peak centres (1/A). Edit to your fitted values.
Q_OOP = 0.51   # (200), out-of-plane lamellar
Q_IP = 1.90    # (pi-pi), in-plane column stacking
Q_002 = 0.52   # (002), in-plane along the viewing direction

_P = _fig41


def _cell(z):
    """Lamellar period (depth), column period (lateral) at depth z, angstrom."""
    g = _P._gamma(z)
    s_g = float(np.clip(abs(np.sin(np.radians(g))), _P.SIN_GAMMA_MIN, 1.0))
    area = _P.cell_area(z)
    ratio = _P.L_MOL * _P.LAT_FRAC * s_g**2 / _P.PI_STACK
    aniso = 1.0 + _P.STRAIN_ANISO_SI * float(np.exp(-(TOTAL - z) / TAU_SI))
    return np.sqrt(area * ratio) * aniso, np.sqrt(area / ratio) / aniso


# Depth-resolved periods, weighted the way GIWAXS samples the film: local
# density times a Debye-Waller factor built from the same positional disorder
# the strip draws. Density alone would overweight the buried interface, which
# holds material but contributes no coherent Bragg intensity.
_z = Z_G[(Z_G >= 0.0) & (Z_G <= TOTAL)]
_dl, _cs = np.array([_cell(z) for z in _z]).T
_u = _P.POS_DISORDER_VAC + (_P.POS_DISORDER_SI - _P.POS_DISORDER_VAC) * (_z / TOTAL)
_rho = np.clip(np.interp(_z, Z_G, _P.RHO_MAT), 0.0, None)


def _stat(d):
    q = 2 * np.pi / d
    w = _rho * np.exp(-(q**2) * _u**2)
    w = w / w.sum()
    qm = float(w @ q)
    return qm, float(np.sqrt(w @ (q - qm) ** 2))


q_oop, s_oop = _stat(_dl)
q_ip, s_ip = _stat(_cs)

# What the two measured peaks imply for the two knobs.
_d1, _d2 = 2 * np.pi / Q_OOP, 2 * np.pi / Q_IP
_vol = _P.MOLAR_MASS / (_P.RHO_BULK_REF * _P.N_AVOGADRO * 1e-24)
_sin2 = np.sin(np.radians(_PACK["gamma_median"])) ** 2
print(f"          {'model q':>12s} {'meas q':>9s} {'d model':>9s} {'d meas':>8s}")
print(f"out-of-pl {q_oop:7.3f}+/-{s_oop:.3f} {Q_OOP:9.3f} "
      f"{2 * np.pi / q_oop:9.2f} {_d1:8.2f}")
print(f"in-plane  {q_ip:7.3f}+/-{s_ip:.3f} {Q_IP:9.3f} "
      f"{2 * np.pi / q_ip:9.2f} {_d2:8.2f}")
print(f"\nimplied by the measured pair:  SLAB_THICK = {_vol / (_d1 * _d2):.2f} A"
      f"   (now {_P.SLAB_THICK:.2f})")
print(f"                               LAT_FRAC   = "
      f"{(_d1 / _d2) * _P.PI_STACK / (_P.L_MOL * _sin2):.3f}"
      f"   (now {_P.LAT_FRAC:.2f})")

# Linecuts, same layout as the measured panel (b).
fig, ax = plt.subplots(figsize=(3.4, 2.4))
_q = np.logspace(np.log10(0.3), np.log10(2.4), 800)
for qc, sc, col, lab, order in ((q_oop, s_oop, "tab:red", "out-of-plane", (1, 2)),
                                (q_ip, s_ip, "tab:blue", "in-plane", (1,))):
    prof = sum(np.exp(-0.5 * ((_q - n * qc) / (np.hypot(sc, 0.02) * n)) ** 2) / n**3
               for n in order)
    ax.semilogx(_q, prof / prof.max(), color=col, lw=1.4, label=f"model {lab}")
for qm, col in ((Q_OOP, "tab:red"), (2 * Q_OOP, "tab:red"), (Q_IP, "tab:blue")):
    ax.axvline(qm, color=col, ls=(0, (3, 2)), lw=0.8, alpha=0.75)
ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
ax.set_ylabel("intensity (norm.)")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=6, handlelength=1.2, frameon=False)
ax.set_title("dashed = measured, solid = drawn packing", fontsize=6.5)
fig.tight_layout()


# ---------------------------------------------------------------------------
# Third direction: the (002) peak and the monoclinic angle.
#
# The drawing plane holds only two lattice directions, lamellar along depth and
# column along the lateral axis. The (002) peak is the third, along the viewing
# direction, and its repeat is SLAB_THICK. That is why it never appears in the
# linecuts above, not because the model lacks it.
#
# SLAB_THICK is the PERPENDICULAR slab thickness, since cell_area = V /
# SLAB_THICK is what ties the drawing to rho(z). The measured d_002 is an
# interplanar spacing along an inclined axis, so the two differ by sin(beta):
#
#     d_002 = SLAB_THICK * sin(beta),   V = d_200 * d_pipi * d_002 / sin(beta)
#
# Treating the three directions as mutually orthogonal with one molecule per
# block forces sin(beta) = 1 and breaks the volume bookkeeping by ~19%.
# ---------------------------------------------------------------------------
_vol_mol = _P.MOLAR_MASS / (_P.RHO_BULK_REF * _P.N_AVOGADRO * 1e-24)
_d200, _dpp, _d002 = 2 * np.pi / Q_OOP, 2 * np.pi / Q_IP, 2 * np.pi / Q_002
_sinb = _d200 * _dpp * _d002 / _vol_mol
_beta = 180.0 - np.degrees(np.arcsin(np.clip(_sinb, 0.0, 1.0)))

print(f"\nvolume/molecule from rho   {_vol_mol:7.1f} A^3")
print(f"product of the three d     {_d200 * _dpp * _d002:7.1f} A^3")
print(f"implied sin(beta)          {_sinb:7.3f}   ->  beta = {_beta:.1f} deg")
print(f"model (002) at current SLAB_THICK  q = "
      f"{2 * np.pi / (_P.SLAB_THICK * _sinb):.3f}   meas {Q_002:.3f}")
