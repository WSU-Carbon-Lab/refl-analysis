"""Pole figure of the q ~ 0.50 to 0.53 ring, ZnPC_Xuess GIWAXS 1.edf.

Per chi bin (chi = polar angle of q from the surface normal, both q_xy sides folded),
the ring is integrated over 0.45 < q < 0.59 1/A after subtracting a linear background
fit to the flanks 0.38 to 0.43 and 0.63 to 0.68. The ring centroid q(chi) is reported
alongside, because (200) at 0.504 and (002) at 0.532 overlap within the ring width and
a two-peak decomposition per bin is not well conditioned. Bins inside the forbidden
wedge are absent by construction.
"""
import fabio, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

FRAME, CX, CY = "ZnPC_Xuess GIWAXS 1.edf", 584.0, 982.0
im = fabio.open(FRAME); h = im.header
d = im.data.astype(float); d[d < 0] = np.nan
L, p = float(h["SampleDistance"]), float(h["PSize_1"])
k = 2 * np.pi / (float(h["WaveLength"]) * 1e10); ai = np.radians(float(h["om"]))
ny, nx = d.shape
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
dx, dz = (X - CX) * p, (CY - Y) * p
tth = np.arctan2(dx, L); af = np.arctan2(dz, np.hypot(L, dx)) - ai
qx = k * (np.cos(af) * np.cos(tth) - np.cos(ai)); qy = k * np.cos(af) * np.sin(tth)
qz = k * (np.sin(af) + np.sin(ai)); qxy = np.hypot(qx, qy)
q = np.sqrt(qxy**2 + qz**2); chi = np.degrees(np.arctan2(qxy, qz))
v = np.isfinite(d) & (d > 0) & (af >= 0)

cb = np.arange(0, 92, 3.0); cc = 0.5 * (cb[1:] + cb[:-1])
qb = np.arange(0.38, 0.681, 0.005); qc = 0.5 * (qb[1:] + qb[:-1])
ring = (qc > 0.45) & (qc < 0.59); flank = ((qc > 0.38) & (qc < 0.43)) | ((qc > 0.63) & (qc < 0.68))
A, E, Q0, cake = [], [], [], []
for lo, hi in zip(cb[:-1], cb[1:]):
    sel = v & (chi >= lo) & (chi < hi) & (q > 0.37) & (q < 0.69)
    s, _ = np.histogram(q[sel], qb, weights=d[sel]); n, _ = np.histogram(q[sel], qb)
    y = np.where(n > 0, s / np.maximum(n, 1), np.nan); cake.append(y)
    okr, okf = ring & np.isfinite(y), flank & np.isfinite(y)
    if okr.sum() < 0.8 * ring.sum() or okf.sum() < 6:
        A.append(np.nan); E.append(np.nan); Q0.append(np.nan); continue
    bc = np.polyfit(qc[okf], y[okf], 1); net = y[okr] - np.polyval(bc, qc[okr])
    dq = qb[1] - qb[0]
    A.append(np.sum(net) * dq)
    var = np.where(n[okr] > 0, np.maximum(y[okr], 0.5) / np.maximum(n[okr], 1), 0)
    resid = y[okf] - np.polyval(bc, qc[okf])
    E.append(np.sqrt(np.sum(var) + okr.sum() * np.var(resid)) * dq)
    Q0.append(np.sum(qc[okr] * np.clip(net, 0, None)) / max(np.sum(np.clip(net, 0, None)), 1e-12))
A, E, Q0, cake = map(np.array, (A, E, Q0, cake))
ok = np.isfinite(A)
w = np.sin(np.radians(cc))
pole = A * w
print(f"first measured chi bin centre {cc[ok][0]:.1f} deg")
I = A[ok]; ch = cc[ok]
half = ch[np.where(I >= I.max() / 2)[0].max()]
print(f"ring intensity HWHM about {half:.1f} deg (bin width 3 deg)")
floor = np.nanmedian(A[(cc > 40) & (cc < 70)])
print(f"floor (median 40-70 deg) / peak = {floor / np.nanmax(A):.3f}")
tot = np.nansum(pole)
for c0 in (15, 30):
    print(f"share of sin-weighted pole density at chi < {c0}: {np.nansum(pole[cc < c0]) / tot:.2f}")
print(f"share at chi > 70: {np.nansum(pole[cc > 70]) / tot:.2f}")
print("centroid q by chi:", [(int(c), round(x, 3)) for c, x in zip(cc[ok], Q0[ok])][::2])
# core-plus-floor model of the chi profile, for an orientation number free of the in-plane (002)
g = lambda x, a, s, b: a * np.exp(-0.5 * (x / s) ** 2) + b
m = ok & (cc < 60)
P, C = curve_fit(g, cc[m], A[m], p0=[np.nanmax(A), 10, floor], sigma=np.maximum(E[m], 1e-6))
print(f"Gaussian core sigma = {P[1]:.1f} +/- {np.sqrt(C[1,1]):.1f} deg (FWHM {2.355*P[1]:.1f}); floor/core amplitude = {P[2]/P[0]:.3f}")
x = np.linspace(0, 90, 901); core = P[0] * np.exp(-0.5 * (x / P[1]) ** 2)
c2 = np.sum(core * np.sin(np.radians(x)) * np.cos(np.radians(x)) ** 2) / np.sum(core * np.sin(np.radians(x)))
print(f"core only <cos^2 chi> = {c2:.3f}, Hermans f = {0.5*(3*c2-1):.3f}")
np.savez("pole_figure.npz", chi=cc, A=A, E=E, q_centroid=Q0, cake=cake, q=qc)

plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 12,
                     "axes.linewidth": 1.4, "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))
cm = matplotlib.colormaps["jet"].copy(); cm.set_bad("white")
ax[0].pcolormesh(qb, cb, np.log10(np.clip(cake, 1e-3, None)), cmap=cm, vmin=np.log10(1.4), vmax=np.log10(np.nanmax(cake) * 1.2))
ax[0].plot(Q0[ok], cc[ok], "w.-", lw=1, ms=4, label="ring centroid")
for c0 in (0.504, 0.532):
    ax[0].axvline(c0, color="w", ls=(0, (3, 2)), lw=0.9)
ax[0].set_xlabel(r"$q$ (Å$^{-1}$)"); ax[0].set_ylabel(r"$\chi$ (deg)"); ax[0].set_title("cake, lines at 0.504 and 0.532")
ax[1].errorbar(cc[ok], A[ok] / np.nanmax(A), E[ok] / np.nanmax(A), color="red", marker="o", ms=4, lw=1.6, label="data")
ax[1].plot(x, (core + P[2]) / np.nanmax(A), "k--", lw=1.2, label=f"Gaussian core, FWHM {2.355*P[1]:.0f}°, + floor")
ax[1].set_yscale("log"); ax[1].set_ylim(5e-3, 1.6); ax[1].set_title("pole figure, q 0.45 to 0.59 ring")
ax[1].set_xlabel(r"$\chi$ (deg)"); ax[1].set_ylabel("integrated intensity, norm."); ax[1].legend(fontsize=9)
ax[2].plot(cc[ok], pole[ok] / np.nanmax(pole), color="red", marker="o", ms=4, lw=1.6)
ax[2].set_xlabel(r"$\chi$ (deg)"); ax[2].set_ylabel(r"$I(\chi)\,\sin\chi$, norm."); ax[2].set_title(r"pole density, $\sin\chi$ weighted")
for a_ in ax[1:]:
    a_.set_xlim(0, 90); a_.axvspan(0, cc[ok][0] - 1.5, color="0.85", lw=0)
plt.tight_layout(); plt.savefig("ZnPC_Xuess GIWAXS 1_pole_figure.png", dpi=170)
