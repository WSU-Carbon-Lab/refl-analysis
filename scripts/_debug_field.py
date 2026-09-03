import numpy as np
from periodictable.xsf import index_of_refraction
from refloxide.integrations.pyref import uniaxial_reflectivity

from utils.field_profile import HC_EV_ANGSTROM, uniaxial_field_profile

ENERGY = 250.0
FILM = 200.0
n_vac = complex(1.0, 0.0)
n_film = index_of_refraction("C8H8", density=1.0, energy=ENERGY * 1e-3)
n_si = index_of_refraction("Si", density=2.33, energy=ENERGY * 1e-3)


def tensor_diag(n):
    d = 1.0 - n  # delta + i beta packing s.t. eps = conj(1-2*tensor) = n**2-ish
    return np.diag([d, d, d])


tensor = np.array([tensor_diag(n) for n in (n_vac, n_film, n_si)], dtype=np.complex128)
layers = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [FILM, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

k0 = 2 * np.pi * ENERGY / HC_EV_ANGSTROM


def airy(q, pol):
    kx = k0 * np.sqrt(1 - (q / (2 * k0)) ** 2 + 0j)
    eps = [np.conj(1 - 2 * (1 - n)) for n in (n_vac, n_film, n_si)]
    kz = [np.sqrt(e * k0**2 - kx**2 + 0j) for e in eps]
    if pol == "s":
        w = kz
    else:
        w = [k / e for k, e in zip(kz, eps, strict=True)]
    r01 = (w[0] - w[1]) / (w[0] + w[1])
    r12 = (w[1] - w[2]) / (w[1] + w[2])
    ph = np.exp(1j * kz[1] * FILM)
    r = (r01 + r12 * ph**2) / (1 + r01 * r12 * ph**2)
    return abs(r) ** 2


print("Isotropic vac/C8H8/Si at 250 eV -- reflectance comparison")
for q in (0.02, 0.05, 0.1, 0.15):
    refl, _ = uniaxial_reflectivity(np.array([q]), layers, tensor, ENERGY, use_rust=True)
    fs = uniaxial_field_profile(q, layers, tensor, ENERGY, "s")
    fp = uniaxial_field_profile(q, layers, tensor, ENERGY, "p")
    print(
        f" q={q:.3f} | s recon={fs.reflectance:.6e} kernel[0,0]={refl[0,0,0]:.6e} "
        f"airy_s={airy(q,'s'):.6e} | p recon={fp.reflectance:.6e} "
        f"kernel[1,1]={refl[0,1,1]:.6e} airy_p={airy(q,'p'):.6e}"
    )
