"""Figure, ZnPc GIWAXS (ZnPC_Xuess GIWAXS 1.edf). (a) detector image, log jet, with the
out-of-plane and in-plane integration wedges. (b) wedge cuts, log-log.

Direct-beam centre located from the transmitted spot, (584.0, 982.0); the EDF header
centre is not the direct beam for this frame. Wedges follow the reference figure,
out of plane within 25 deg of the surface normal, in plane within 17 deg of the horizon.
"""
import sys
import fabio
import matplotlib
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
from scipy.signal import savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

FRAME = sys.argv[1] if len(sys.argv) > 1 else "ZnPC_Xuess GIWAXS 1.edf"
CX, CY = 584.0, 982.0
OOP_HALF = 25.0      # deg from the surface normal
IP_ELEV = 17.0       # deg above the horizon
CROP = dict(r0=548, r1=984, c0=0, c1=780)

im = fabio.open(FRAME)
h = im.header
d = im.data.astype(float)
d[d < 0] = np.nan
L, p = float(h["SampleDistance"]), float(h["PSize_1"])
lam = float(h["WaveLength"]) * 1e10
ai = np.radians(float(h["om"]))
ny, nx = d.shape
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
dx, dz = (X - CX) * p, (CY - Y) * p
k = 2 * np.pi / lam
tth = np.arctan2(dx, L)
af = np.arctan2(dz, np.hypot(L, dx)) - ai
qx = k * (np.cos(af) * np.cos(tth) - np.cos(ai))
qy = k * np.cos(af) * np.sin(tth)
qz = k * (np.sin(af) + np.sin(ai))
q = np.sqrt(qx**2 + qy**2 + qz**2)
elev = np.degrees(np.arctan2(dz, np.abs(dx)))          # detector-plane angle above horizon
valid = np.isfinite(d) & (d > 0) & (af >= 0)

def wedge(sel, qb=np.logspace(np.log10(0.1), np.log10(2.45), 180)):
    s, _ = np.histogram(q[sel], qb, weights=d[sel])
    n, _ = np.histogram(q[sel], qb)
    x = np.sqrt(qb[1:] * qb[:-1])
    y = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    ok = np.isfinite(y)
    return x[ok], y[ok]

x_oop, y_oop = wedge(valid & (elev >= 90 - OOP_HALF))
x_ip, y_ip = wedge(valid & (elev <= IP_ELEV) & (dx < 0))
smooth = lambda y: 10 ** savgol_filter(np.log10(y), 9, 2)

plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.bf": "Liberation Sans:bold", "mathtext.it": "Liberation Sans:italic", "font.size": 13,
                     "axes.linewidth": 1.6, "xtick.direction": "in", "ytick.direction": "in"})
fig = plt.figure(figsize=(6.6, 8.1))
axa = fig.add_axes([0.09, 0.585, 0.84, 0.39])
axb = fig.add_axes([0.13, 0.11, 0.79, 0.45])

# (a) detector image
c = CROP
img = d[c["r0"]:c["r1"], c["c0"]:c["c1"]]
from scipy.ndimage import gaussian_filter
fill = np.where(np.isfinite(img) & (img > 0), img, 0.0)
w = gaussian_filter(np.isfinite(img).astype(float) * (img > 0), 0.8)
img_s = gaussian_filter(fill, 0.8) / np.maximum(w, 1e-6)
pk = np.nanpercentile(img_s[(np.abs(np.arange(c["c0"], c["c1"])[None, :] - CX) < 40) & (np.abs(np.arange(c["r0"], c["r1"])[:, None] - 866) < 12)], 90)
vmin, vmax = 1.4, pk * 3.6   # background near jet 0.06, (200) near 0.78
cm = matplotlib.colormaps["jet"].copy(); cm.set_bad(cm(0.0)); cm.set_under(cm(0.0)); cm.set_over(cm(0.88))
axa.imshow(np.log10(np.clip(img_s, 1e-3, None)) , cmap=cm, vmin=np.log10(vmin), vmax=np.log10(vmax),
           extent=(c["c0"], c["c1"], c["r1"], c["r0"]), interpolation="nearest", aspect="equal")
axa.set_xlim(c["c0"], c["c1"]); axa.set_ylim(c["r1"], c["r0"]); axa.axis("off")
glow = [pe.Stroke(linewidth=4.5, foreground="white", alpha=0.35), pe.Normal()]
R = 1400
for sgn in (-1, 1):
    a = np.radians(90 - OOP_HALF)
    axa.plot([CX, CX + sgn * R * np.cos(a)], [CY, CY - R * np.sin(a)], "--", color="red", lw=2.6, dashes=(4, 2.5), path_effects=glow)
for e in (0.0, IP_ELEV):
    a = np.radians(e)
    axa.plot([CX, CX - R * np.cos(a)], [CY - 1.5, CY - R * np.sin(a) - 1.5], "--", color="#2f6bff", lw=2.6, dashes=(4, 2.5), path_effects=glow)

def at(qv, elev_deg, side=1):
    """Detector pixel for |q| = qv at a detector-plane elevation (small ai approximation)."""
    t2 = 2 * np.arcsin(qv / (2 * k))
    r = L * np.tan(t2) / p
    e = np.radians(elev_deg)
    return CX + side * r * np.cos(e), CY - r * np.sin(e)

lab = dict(color="white", fontsize=16, fontweight="bold", ha="center", va="center")
for txt, qv, e, s in [("(200)", 0.50, 90, 1), ("(400)", 1.12, 90, 1), ("(20*)", 1.24, 37, -1),
                      ("(002)", 0.62, 6.0, -1), (r"$\mathbf{(\pi-\pi)}$", 1.90, 5.0, -1)]:
    xx, yy = at(qv, e, s)
    if txt == "(200)": yy -= 26
    axa.text(xx, yy, txt, **lab)
fig.text(0.02, 0.965, "(a)", fontsize=20, va="top")

# (b) wedge cuts
axb.loglog(x_oop, smooth(y_oop), color="red", lw=4.5, label="Out-of-Plane", solid_capstyle="round")
axb.loglog(x_ip, smooth(y_ip), color="blue", lw=4.5, label="In-Plane", solid_capstyle="round")
axb.set_xlim(0.095, 2.5)
axb.set_ylim(1.3, 480)
axb.set_xlabel(r"Q vector (Å$^{-1}$)", fontsize=20)
axb.set_ylabel("Intensity (a.u)", fontsize=20)
axb.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
axb.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
axb.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
axb.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
axb.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{int(round(v / 10 ** np.floor(np.log10(v))))}" if v < 2.45 else ""))
axb.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 4, 6, 8)))
axb.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{int(round(v / 10 ** np.floor(np.log10(v))))}"))
axb.tick_params(which="both", top=True, right=True, width=1.6)
axb.tick_params(which="major", length=9)
axb.tick_params(which="minor", length=5, labelsize=11)
axb.grid(True, which="both", axis="x", color="#4040ff", ls=(0, (2, 2)), lw=0.7, alpha=0.7)
axb.grid(True, which="major", axis="y", color="#4040ff", ls=(0, (2, 2)), lw=0.7, alpha=0.7)
leg = axb.legend(loc="upper right", fontsize=15, frameon=True, fancybox=False, edgecolor="black", handlelength=2.6)
leg.get_frame().set_linewidth(2.0)

def peak(x, y, lo, hi):
    m = (x > lo) & (x < hi)
    i = np.argmax(smooth(y)[m])
    return x[m][i], smooth(y)[m][i]
for txt, (x, y, lo, hi, dxf, dyf) in {"(200)": (x_oop, y_oop, 0.4, 0.62, 0.88, 1.35),
                                      "(002)": (x_ip, y_ip, 0.42, 0.65, 0.98, 1.45),
                                      "(400)": (x_oop, y_oop, 0.85, 1.3, 0.98, 1.45),
                                      r"$\mathbf{(\pi-\pi)}$": (x_ip, y_ip, 1.6, 2.2, 0.98, 1.42)}.items():
    xp, yp = peak(x, y, lo, hi)
    axb.text(xp * dxf, yp * dyf, txt, fontsize=17, fontweight="bold", ha="center")
    print(f"{txt:12s} q = {xp:.3f}  I = {yp:.1f}")
fig.text(0.02, 0.555, "(b)", fontsize=20, va="top")

out = FRAME.replace(".edf", "_figure")
fig.savefig(out + ".png", dpi=200)
fig.savefig(out + ".pdf")
