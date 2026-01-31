#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スクリプトの概要:
このスクリプトはneやstellar mass, sfrなどの
情報が記載されたinputファイル（txt）をpyファイルに変換し、
描画しやすい形にします。
2026-01-07に12+log(O/H)（ガス相金属量）の情報も追加しました。

使用方法:
    python make_ne_vs_sm_sfr_input_to_output.py [オプション]

著者: A. M.
作成日: 2026-01-07

参考文献:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 257 (Docstring規約): https://peps.python.org/pep-0257/
    - Python公式ドキュメント: https://docs.python.org/ja/3/
"""


# === 必要なパッケージのインストール === #
import re
import os


# === 全角マイナス (U+2212) を ASCII ハイフンに置換する関数 === #
def normalize_minus(s):
    return s.replace("−", "-")  


# === 数値と誤差を分離する関数（変更予定あり） === #
def parse_value_with_error(s):
    s = normalize_minus(s)  
    match = re.match(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\-([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$", s)
    if match:
        try:
            val = float(match.group(1))
            err_plus = float(match.group(2))
            err_minus = float(match.group(3))
            return {"value": val, "err_plus": err_plus, "err_minus": err_minus}
        except ValueError:
            return {"value": None, "err_plus": None, "err_minus": None}
    else:
        return {"value": None, "err_plus": None, "err_minus": None}


# === inputファイルのデータを読み取ってoutputファイルに変換する関数 === 
def load_galaxy_data(filepath):
    galaxy_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split()  
            name  = parts[0]
            AGN   = int(parts[1]) 
            z     = float(parts[2])
            SM    = parse_value_with_error(parts[7]) 
            SFR   = parse_value_with_error(parts[8]) 
            metal = parse_value_with_error(parts[9]) 
            ne_values = {
                "low":          parse_value_with_error(parts[3]),
                "intermediate": parse_value_with_error(parts[4]),
                "high":         parse_value_with_error(parts[5]),
                "very_high":    parse_value_with_error(parts[6])
            }
            galaxy_dict[name] = {
                "AGN": AGN,
                "z": z,
                "SM": { # 追加
                    "value":     SM["value"],
                    "err_plus":  SM["err_plus"],
                    "err_minus": SM["err_minus"]
                },
                "SFR": { # 追加
                    "value":     SFR["value"],
                    "err_plus":  SFR["err_plus"],
                    "err_minus": SFR["err_minus"]
                },
                "metal": { # 追加
                    "value":     metal["value"],
                    "err_plus":  metal["err_plus"],
                    "err_minus": metal["err_minus"]
                },
                "ne_values": ne_values
            }
    return galaxy_dict

def write_galaxy_dict_as_python(galaxy_dict, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("all_data = {\n")
        for name, info in galaxy_dict.items():
            f.write(f'    "{name}": {{\n')
            f.write(f'        "AGN":   {info["AGN"]}, "z": {info["z"]}, \n') # 追加
            f.write(f'        "SM":    {{"value": {info["SM"]["value"]}, "err_plus": {info["SM"]["err_plus"]}, "err_minus": {info["SM"]["err_minus"]}}},\n')
            f.write(f'        "SFR":   {{"value": {info["SFR"]["value"]}, "err_plus": {info["SFR"]["err_plus"]}, "err_minus": {info["SFR"]["err_minus"]}}},\n')
            f.write(f'        "metal": {{"value": {info["metal"]["value"]}, "err_plus": {info["metal"]["err_plus"]}, "err_minus": {info["metal"]["err_minus"]}}},\n')
            f.write(f'        "ne_values": {{\n')
            for ne_type, ne_info in info["ne_values"].items():
                val = ne_info["value"]
                plus = ne_info["err_plus"]
                minus = ne_info["err_minus"]
                f.write(f'            "{ne_type}": {{"value": {val}, "err_plus": {plus}, "err_minus": {minus}}},\n')
            f.write(f'        }}\n')
            f.write(f'    }},\n')
        f.write("}\n")

# 実行例
current_dir = os.getcwd()
input_file  = os.path.join(current_dir, "results/Rigby21/Rigby21in.txt")   # 適宜変更
output_file = os.path.join(current_dir, "results/Rigby21/Rigby21out.py") # 適宜変更
 
galaxies = load_galaxy_data(input_file)
write_galaxy_dict_as_python(galaxies, output_file)
