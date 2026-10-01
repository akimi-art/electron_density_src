# replace_mass_v52b.py
#   merged の質量列（sm_*）を v5_2b の値に差し替える
#   行は消さない。旧版の値は *_v5_2 という列名で残す
# replace_mass_v52b.py
#   元の merged の質量列（sm_*）を v5_2b の値に差し替え、ROW_ID を付ける
#   行は消さない。旧版の値は *_v5_2 という列名で残す
import os
import numpy as np
from astropy.table import Table

current_dir = os.getcwd()
mpa_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged.fits")
new_path = os.path.join(current_dir, "data/data_SDSS/DR7/fits_files/totlgm_dr7_v5_2b.fit")
out_path = os.path.join(current_dir, "results/fits/mpajhu_dr7_v5_2_merged_lgm2b.fits")
N_EXPECTED = 927552

def f64(a):
    return np.ma.filled(np.ma.asarray(a).astype(float), np.nan)

# 1. 読み込みと行数の確認
t   = Table.read(mpa_path)
new = Table.read(new_path, hdu=1)
N = len(t)
assert N == N_EXPECTED and len(new) == N, f"行数が一致しない: merged {N}, v5_2b {len(new)}"

# 2. ROW_ID（元の行番号）を付ける。以後すべてのファイルでこの番号を引き継ぐ
if "ROW_ID" not in t.colnames:
    t["ROW_ID"] = np.arange(N, dtype=np.int64)
assert np.array_equal(np.asarray(t["ROW_ID"]), np.arange(N))

# 3. 差し替える列の対応（v5_2b の列 X → merged の sm_X）
pairs = [(c, f"sm_{c}") for c in new.colnames if f"sm_{c}" in t.colnames]
print("差し替える列:", [p[1] for p in pairs])
assert ("MEDIAN", "sm_MEDIAN") in pairs

# 4. 行順の確認：旧版で値がある行は新旧が一致するはず
old = f64(t["sm_MEDIAN"])
nw  = f64(new["MEDIAN"])
both = (old != -1) & (nw != -1) & np.isfinite(old) & np.isfinite(nw)
frac = np.mean(np.abs(nw[both] - old[both]) < 0.01)
print(f"旧版で値がある行のうち新旧が一致（|差| < 0.01）: {frac:.4f}")
assert frac > 0.999, "新旧が一致しない → 行順がずれている可能性。差し替え中止"
assert np.sum((old != -1) & (nw == -1)) == 0, "旧版にあった値が新版で欠けている行がある"

# 5. 差し替え（旧版は *_v5_2 で残す）
rescued = (old == -1) & (nw != -1)
for c_new, c_old in pairs:
    t.rename_column(c_old, f"{c_old}_v5_2")
    t[c_old] = f64(new[c_new])
t["MASS_RESCUED_v52b"] = rescued

# 6. 確認して保存
assert len(t) == N
print(f"sm_MEDIAN = -1：旧 {np.sum(old == -1):,} → 新 {np.sum(f64(t['sm_MEDIAN']) == -1):,}")
print(f"v5_2b で値が入った行: {rescued.sum():,}")
t.write(out_path, overwrite=True)
print(f"[DONE] {out_path}（全 {N:,} 行）")