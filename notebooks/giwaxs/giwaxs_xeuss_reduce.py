"""Reduce Xeuss GIWAXS EDF frames of ZnPc to (q_xy, q_z), chi, and sector cuts.

The EDF header beam centre (Center_1, Center_2) is not the direct-beam position
for these frames. The direct beam was located from the transmitted spot below the
sample horizon, (585.0, 983.3) for ZnPc Xuess GIWAXS.edf and (584.0, 982.0) for
ZnPC_Xuess GIWAXS 1.edf. With these centres the ZnPc_20 lamellar peak falls at
q = 0.497 1/A and its second order at 0.987 1/A, which reproduces the earlier
reduction in notebooks/manuscript/XuessGiwaxs.csv (0.500, 0.989). Incidence angle
is taken from the header 'om' (0.2001 deg). SDD and pixel size from the header.
"""
import sys
import fabio
import gemmi
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FRAME = sys.argv[1] if len(sys.argv) > 1 else "ZnPC_Xuess GIWAXS 1.edf"
CENTRE = {"ZnPC_Xuess GIWAXS 1.edf": (584.0, 982.0), "ZnPc Xuess GIWAXS.edf": (585.0, 983.3)}
CIF = sys.argv[2] if len(sys.argv) > 2 else "tpZnPc_0m.cif"


def qmap(path, cx, cy):
    im = fabio.open(path)
    h = im.header
    d = im.data.astype(float)
    d[d < 0] = np.nan
    L, p = float(h["SampleDistance"]), float(h["PSize_1"])
    lam = float(h["WaveLength"]) * 1e10
    ai = np.radians(float(h["om"]))
    ny, nx = d.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    dx, dz = (X - cx) * p, (cy - Y) * p
    k = 2 * np.pi / lam
    tth = np.arctan2(dx, L)
    af = np.arctan2(dz, np.hypot(L, dx)) - ai
    qx = k * (np.cos(af) * np.cos(tth) - np.cos(ai))
    qy = k * np.cos(af) * np.sin(tth)
    qz = k * (np.sin(af) + np.sin(ai))
    d[af < 0] = np.nan
    return d, np.sign(qy) * np.hypot(qx, qy), qz


def beta_reflections(cif, qmax=2.1, rel=0.1):
    st = gemmi.read_small_structure(cif)
    st.setup_cell_images()
    ops = gemmi.SpaceGroup(st.spacegroup_hm).operations()
    calc = gemmi.StructureFactorCalculatorX(st.cell)
    out = {}
    for h in range(-8, 9):
        for k in range(0, 5):
            for l in range(-9, 10):
                if (h, k, l) == (0, 0, 0) or ops.is_systematically_absent([h, k, l]):
                    continue
                q = 2 * np.pi / st.cell.calculate_d([h, k, l])
                if q > qmax:
                    continue
                key = round(q, 3)
                out[key] = out.get(key, 0) + abs(calc.calculate_sf_from_small_structure(st, [h, k, l])) ** 2
    imax = max(out.values())
    return sorted((q, i / imax) for q, i in out.items() if i / imax > rel)


d, qxy, qz = qmap(FRAME, *CENTRE[FRAME])
m = np.isfinite(d) & (d > 0)
q = np.hypot(qxy, qz)
chi = np.degrees(np.arctan2(qxy, qz))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
# detector space, so the grazing-incidence forbidden wedge never appears
cx, cy = CENTRE[FRAME]
raw = fabio.open(FRAME).data.astype(float)[548:984, 0:780]
okp = np.isfinite(raw) & (raw > 0)
ax[0].imshow(np.log10(np.where(okp, raw, 1e-3)), cmap="jet", vmin=np.log10(1.4), vmax=np.log10(490),
             extent=(0, 780, 984, 548), interpolation="nearest")
ax[0].axis("off")
ax[0].set_title(FRAME)

b = np.linspace(-90, 90, 91)
xc = 0.5 * (b[1:] + b[:-1])
def chiprof(a, c):
    sel = m & (q > a) & (q < c)
    s, _ = np.histogram(chi[sel], b, weights=d[sel])
    n, _ = np.histogram(chi[sel], b)
    return s / np.maximum(n, 1)
net = chiprof(0.46, 0.56) - 0.5 * (chiprof(0.40, 0.44) + chiprof(0.60, 0.64))
ax[1].plot(xc, net / np.nanmax(net), "k")
ax[1].set_xlabel(r"$\chi$ from $q_z$ (deg)")
ax[1].set_ylabel("lamellar ring, background subtracted")

qb = np.linspace(0.25, 2.4, 431)
for name, (a, c) in {"out of plane, |chi| < 10": (0, 10), "in plane, |chi| 75 to 90": (75, 90)}.items():
    sel = m & (np.abs(chi) >= a) & (np.abs(chi) < c)
    s, _ = np.histogram(q[sel], qb, weights=d[sel])
    n, _ = np.histogram(q[sel], qb)
    ax[2].semilogy(0.5 * (qb[1:] + qb[:-1]), s / np.maximum(n, 1), label=name)
for qq, rr in beta_reflections(CIF):
    ax[2].axvline(qq, color="tab:red", alpha=0.25 + 0.6 * rr, lw=0.8)
ax[2].set_xlabel(r"$q$ (1/Å)")
ax[2].legend(fontsize=8, title="red, beta-ZnPc tpZnPc_0m.cif, I > 10%", title_fontsize=7)
plt.tight_layout()
plt.savefig(FRAME.replace(".edf", "_reduced.png"), dpi=150)
