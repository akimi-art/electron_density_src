"""Median high-z SFR and bootstrap uncertainty in a fixed stellar-mass bin.

Bootstrap errors describe uncertainty of the sample median; they do not propagate
individual SFR measurement errors or mass errors across the selection boundaries.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table

INPUT = 'results/JADES/sample/jades_all_with_flags.fits'
MASS_MIN, MASS_MAX = 9.3, 10.0
N_BOOT = 20000
SEED = 42


def f64(col):
    return np.ma.filled(np.ma.asarray(col).astype(float), np.nan)


def main():
    t = Table.read(INPUT)
    mass, sfr = f64(t['N26_logM']), f64(t['N26_SFR_hb'])
    mask = (np.asarray(np.ma.filled(t['SAMPLE_SSFR'], False), bool)
            & np.isfinite(mass) & (mass >= MASS_MIN) & (mass < MASS_MAX)
            & np.isfinite(sfr) & (sfr > 0))
    values = sfr[mask]
    if not len(values):
        raise ValueError('No galaxies in mass bin')
    rng = np.random.default_rng(SEED)
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True)
    rows = []
    for name, x, boot in [('SFR', values, np.median(draws, axis=1)),
                          ('logSFR', np.log10(values), np.median(np.log10(draws), axis=1))]:
        median = np.median(x)
        lo, hi = np.percentile(boot, [16, 84])
        p16, p84 = np.percentile(x, [16, 84])
        rows.append(dict(quantity=name, N=len(x), median=median,
                         bootstrap_p16=lo, bootstrap_p84=hi,
                         error_minus=median-lo, error_plus=hi-median,
                         population_p16=p16, population_p84=p84))
    out = Path('results/comparison')
    out.mkdir(parents=True, exist_ok=True)
    tag = f'highz_sfr_logM{MASS_MIN}-{MASS_MAX}'
    t[mask].write(out / f'{tag}.fits', overwrite=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f'{tag}.csv', index=False)
    print(df.to_string(index=False))
    print('Mass median:', np.median(mass[mask]))
    print('Saved:', out / f'{tag}.csv')


if __name__ == '__main__':
    main()
