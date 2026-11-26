#!/usr/bin/env python3
"""Download only the DR7 spSpec spectra marked SELECTED_V2 in the master."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from shutil import disk_usage
from threading import local

import requests
from astropy.io import fits
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "results/fits/mpajhu_dr7_v5_2_master_v2_Ka03_Reu.fits"
SPECTRA = ROOT / "data/data_SDSS/DR7/spectra"
DESTINATION = SPECTRA / "master_v2_Ka03_Reu"
ARCHIVE = "https://das.sdss.org/spectro/1d_26"
THREAD = local()


def selected_ids():
    with fits.open(MASTER) as hdus:
        table = hdus[1].data
        ids = zip(table["PLATEID"], table["MJD"], table["FIBERID"], table["SELECTED_V2"])
        return sorted({(int(p), int(m), int(f)) for p, m, f, selected in ids if selected})


def spectrum_name(key):
    plate, mjd, fiber = key
    return f"spSpec-{mjd:05d}-{plate:04d}-{fiber:03d}.fit"


def validate(contents, key):
    if not contents.startswith(b"SIMPLE  =") or len(contents) % 2880:
        raise ValueError("Incomplete or invalid FITS")
    with fits.open(BytesIO(contents)) as hdus:
        header = hdus[0].header
        actual = tuple(header[name] for name in ("PLATEID", "MJD", "FIBERID"))
        if actual != key or header["NAXIS"] != 2 or header["NAXIS2"] < 3:
            raise ValueError(f"Unexpected spectrum metadata: {actual}")
        hdus.verify("exception")
        end = hdus.fileinfo(len(hdus) - 1)
        if end["datLoc"] + end["datSpan"] > len(contents):
            raise ValueError("Truncated FITS extension")


def session():
    if not hasattr(THREAD, "session"):
        THREAD.session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        THREAD.session.mount("https://", HTTPAdapter(max_retries=retry))
    return THREAD.session


def download(key):
    name = spectrum_name(key)
    target = DESTINATION / name
    cached = SPECTRA / "fit" / name
    for path in (target, cached):
        if path.is_file():
            contents = path.read_bytes()
            try:
                validate(contents, key)
            except (OSError, KeyError, ValueError):
                continue
            if path == target:
                return "existing"
            source = "local"
            break
    else:
        url = f"{ARCHIVE}/{key[0]:04d}/1d/{name}"
        response = session().get(url, timeout=(10, 45))
        response.raise_for_status()
        contents = response.content
        validate(contents, key)
        source = "downloaded"
    partial = target.with_suffix(".fit.part")
    partial.write_bytes(contents)
    partial.replace(target)
    return source


def attempt(key):
    try:
        return download(key), None
    except Exception as error:
        return "failed", f"{spectrum_name(key)}\t{error}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, help="Process only the first N selected targets.")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Workers and limit must be positive.")
    ids = selected_ids()[:args.limit]
    DESTINATION.mkdir(parents=True, exist_ok=True)
    error_log = DESTINATION.parent / f"{DESTINATION.name}_download_errors.tsv"
    counts = dict(existing=0, local=0, downloaded=0, failed=0)
    print(f"Target: {len(ids):,} spectra -> {DESTINATION}", flush=True)
    with error_log.open("a", encoding="utf-8") as errors, ThreadPoolExecutor(args.workers) as pool:
        for start in range(0, len(ids), 256):
            if disk_usage(DESTINATION).free < 5 * 1024**3:
                raise SystemExit("Stopped with less than 5 GiB free. Rerun to resume.")
            batch = ids[start:start + 256]
            failed_before = counts["failed"]
            for status, error in pool.map(attempt, batch):
                counts[status] += 1
                if error:
                    errors.write(error + "\n")
            errors.flush()
            print(f"{sum(counts.values()):,}/{len(ids):,}  {counts}", flush=True)
            if counts["failed"] - failed_before == len(batch):
                raise SystemExit(f"All targets in the latest batch failed. See {error_log}")
    if counts["failed"]:
        raise SystemExit(f"Some targets failed. See {error_log}. Rerun to retry.")


if __name__ == "__main__":
    main()
