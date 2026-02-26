from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

plate = 275
mjd = 51910
fiber = 141

fn = "results/SDSS/spectra/2d/spPlate-0275-51910.fits"
hdul = fits.open(fn)
flux2d = hdul[0].data  # HDU0 = FLUX, shape = (NFIBER, NPIX)

row = fiber - 1  # 行番号（0始まり）
cut = flux2d[row:row+1, :]  # 1行だけ

plt.figure(figsize=(10, 2))
plt.imshow(cut, aspect='auto', cmap='gray', origin='lower')
plt.yticks([0], [f'fiber {fiber}'])
plt.xlim()
plt.xlabel('Wavelength pixel')
plt.title(f'spPlate {plate}-{mjd}  fiber {fiber}: 2D row view')
plt.colorbar(label='Flux (1e-17 erg/s/cm^2/Å)')
plt.tight_layout()
plt.savefig("results/SDSS/figure/spPlate-0275-51910_141_2d.png")
plt.show()
# 7216.057 7329.931