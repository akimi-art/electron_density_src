import os
import re
import numpy as np

# === ファイル設定 ===
current_dir = os.getcwd()

def parse_med_pm(s):
    m = re.match(r'([\d\.]+)\+([\d\.]+)-([\d\.]+)', s)
    if m is None:
        return None, None, None
    return map(float, m.groups())

def format_med_pm(med, ep, em):
    return f"{med:.2f}+{ep:.2f}-{em:.2f}"

# ===== 設定 =====
logM_col   = 6   # logM* の列番号（0始まり）
logSFR_col = 7   # logSFR の列番号
insert_col = 8   # log(sSFR) を挿入したい列番号

infile  = os.path.join(current_dir, "results/Harikane25/Harikane25in.txt")
outfile = os.path.join(current_dir, "results/Harikane25/Harikane25in_re.txt")

with open(infile) as f, open(outfile, "w") as g:
    for line in f:
        cols = line.split()

        M, Mp, Mm = parse_med_pm(cols[logM_col])
        S, Sp, Sm = parse_med_pm(cols[logSFR_col])

        if M is None or S is None:
            ssfr_str = "0+0-0"
        else:
            ssfr_med = S - M
            ssfr_p = np.sqrt(Sp**2 + Mm**2)
            ssfr_m = np.sqrt(Sm**2 + Mp**2)
            ssfr_str = format_med_pm(ssfr_med, ssfr_p, ssfr_m)

        # ★ ここが肝
        cols.insert(insert_col, ssfr_str)

        g.write(" ".join(cols) + "\n")