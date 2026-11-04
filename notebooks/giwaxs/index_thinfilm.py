"""Fiber-textured thin-film indexing of a CIF against the ZnPc GIWAXS frame.

For every allowed (hkl) the reciprocal vector G is split relative to the contact-plane
normal G_c, q_z = |G| cos(psi), q_xy = |G| sin(psi), which is the spot position for a
film with G_c along the surface normal and random azimuth. Intensities are |F|^2 summed
over symmetry-equivalent spots at the same (q_xy, q_z). Optional metal substitution
(for example Cu to Zn in an alpha-CuPc CIF) is applied before the structure factors.

Usage
    python index_thinfilm.py CIF H K L [--swap Cu Zn] [--edf FRAME]
"""
import argparse
import fabio
import gemmi
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("cif")
ap.add_argument("hkl", nargs=3, type=int)
ap.add_argument("--swap", nargs=2, default=None)
ap.add_argument("--edf", default="ZnPC_Xuess GIWAXS 1.edf")
ap.add_argument("--centre", nargs=2, type=float, default=(584.0, 982.0))
ap.add_argument("--qmax", type=float, default=2.1)
ap.add_argument("--rel", type=float, default=0.02)
a = ap.parse_args()

st = gemmi.read_small_structure(a.cif)
if a.swap:
    for s in st.sites:
        if s.element.name == a.swap[0]:
            s.element = gemmi.Element(a.swap[1])
            s.type_symbol = a.swap[1]
st.setup_cell_images()
ops = gemmi.SpaceGroup(st.spacegroup_hm).operations()
calc = gemmi.StructureFactorCalculatorX(st.cell)
B = np.array(st.cell.frac.mat.tolist()).T * 2 * np.pi        # G = B @ hkl, 1/A
gc = B @ np.array(a.hkl, float)
gc /= np.linalg.norm(gc)
spots = {}
n = 10
for h in range(-n, n + 1):
    for k in range(-n, n + 1):
        for l in range(-n, n + 1):
            if (h, k, l) == (0, 0, 0) or ops.is_systematically_absent([h, k, l]):
                continue
            g = B @ np.array([h, k, l], float)
            q = np.linalg.norm(g)
            if q > a.qmax:
                continue
            qz = abs(g @ gc)
            qxy = np.sqrt(max(q * q - qz * qz, 0))
            key = (round(qxy, 2), round(qz, 2))
            I = abs(calc.calculate_sf_from_small_structure(st, [h, k, l])) ** 2
            e = spots.setdefault(key, [0.0, (h, k, l)])
            e[0] += I
imax = max(v[0] for v in spots.values())
rows = sorted(((k[0], k[1], v[0] / imax, v[1]) for k, v in spots.items() if v[0] / imax > a.rel), key=lambda r: np.hypot(r[0], r[1]))
print(f"cell {st.cell}  {st.spacegroup_hm}  contact {tuple(a.hkl)}  swap {a.swap}")
for qxy, qz, r, hkl in rows:
    print(f"  q={np.hypot(qxy, qz):.3f}  q_xy={qxy:.3f}  q_z={qz:.3f}  I={100 * r:5.1f}  {hkl}")

# overlay on the measured frame, in detector space so the forbidden wedge never appears
from scipy.ndimage import gaussian_filter
im = fabio.open(a.edf)
h = im.header
d = im.data.astype(float)
L, p = float(h["SampleDistance"]), float(h["PSize_1"])
k0 = 2 * np.pi / (float(h["WaveLength"]) * 1e10)
ai = np.radians(float(h["om"]))
cx, cy = a.centre
r0, r1, c0, c1 = 548, 984, 0, 780
img = d[r0:r1, c0:c1]
ok = np.isfinite(img) & (img > 0)
img = gaussian_filter(np.where(ok, img, 0.0), 0.8) / np.maximum(gaussian_filter(ok.astype(float), 0.8), 1e-6)
cm = matplotlib.colormaps["jet"].copy()
cm.set_under(cm(0.0)); cm.set_over(cm(0.88))
pk = np.percentile(img[(np.abs(np.arange(c0, c1)[None, :] - cx) < 40) & (np.abs(np.arange(r0, r1)[:, None] - 866) < 12)], 90)


def to_pixel(qxy, qz, side=-1):
    """Inverse grazing-incidence map. Returns None inside the forbidden wedge."""
    saf = qz / k0 - np.sin(ai)
    if abs(saf) > 1:
        return None
    af = np.arcsin(saf)
    c2t = (np.cos(af) ** 2 + np.cos(ai) ** 2 - (qxy / k0) ** 2) / (2 * np.cos(af) * np.cos(ai))
    if c2t > 1:
        return None
    t2 = np.arccos(np.clip(c2t, -1, 1))
    dx = side * L * np.tan(t2)
    dz = np.tan(af + ai) * np.hypot(L, dx)
    return cx + dx / p, cy - dz / p


fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.imshow(np.log10(np.clip(img, 1e-3, None)), cmap=cm, vmin=np.log10(1.4), vmax=np.log10(pk * 3.6),
          extent=(c0, c1, r1, r0), interpolation="nearest")
for x, z, r, hkl in rows:
    pix = to_pixel(x, z)
    style = dict(s=30 + 400 * r, facecolors="none", lw=1.2)
    if pix is None:                       # specular-type spot, drawn on the meridian
        t2 = 2 * np.arcsin(np.hypot(x, z) / (2 * k0))
        pix = (cx, cy - L * np.tan(t2) / p)
        style.update(edgecolors="white", linestyle="--")
    else:
        style.update(edgecolors="white")
    if not (c0 <= pix[0] <= c1 and r0 <= pix[1] <= r1):
        continue
    ax.scatter(*pix, **style)
    if r > 0.08:
        ax.text(pix[0] + 8, pix[1] - 8, "".join(str(i) for i in hkl), color="white", fontsize=7)
ax.set_xlim(c0, c1); ax.set_ylim(r1, r0); ax.axis("off")
ax.set_title(f"{a.cif.split('/')[-1]}, contact {tuple(a.hkl)}" + (f", {a.swap[0]} to {a.swap[1]}" if a.swap else "")
             + "\ndashed, specular-type spots off the Ewald sphere, drawn on the meridian", fontsize=9)
plt.tight_layout()
plt.savefig(f"index_{a.cif.split('/')[-1].replace('.cif', '')}_{''.join(map(str, a.hkl)).replace('-', 'm')}.png", dpi=150)
