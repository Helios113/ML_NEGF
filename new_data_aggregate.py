from os import listdir
import numpy as np
from os.path import isfile, join
import csv
from collections import defaultdict

import matplotlib.pyplot as plt

name = "3x12_16_damp00"

res_dir = "data"
img_dir = res_dir + "/ml_test"
cond_dir = res_dir + "/" + name + "/dat_std"

target_prefix = "NEGFXY"
table = "/info_dat_std_{nm}.csv".format(nm=target_prefix)
info_file = open(res_dir + "/" + name + table, "w+")
writer = csv.writer(info_file)


inp_suffix = "_1.txt"
cmp_suffix = "_4.txt"

tar_suffix = "_{index}.txt"

imgs = listdir(img_dir)

np.seterr(all="raise")
files = [f for f in imgs if isfile(join(img_dir, f)) and f.startswith(target_prefix)]


# Find the same devices everywhere
unique_names = defaultdict(list)
max_index = defaultdict(lambda: 0)
for f in files:
    key = "_".join(f.split("_", 3)[:3])
    val = int(f.rsplit(".", 1)[0].split("_")[-1])
    # key1 = "_".join(f.split("_")[:4])

    unique_names[key].append(f)
    if max_index[key] < val:
        max_index[key] = val

for key, val in unique_names.items():
    val.sort()
    matches = filter(lambda a: a.endswith(inp_suffix), val)
    norm = [[0, 0], [0, 0]]
    for v in val:
        chrg = int("charge" in v)
        if v.endswith(inp_suffix):
            data = (
                np.log10(np.loadtxt(join(img_dir, v)))
                if chrg == 1
                else np.loadtxt(join(img_dir, v))
            )
            norm[chrg][0] = data.mean()
            norm[chrg][1] = data.std()
            np.savetxt(join(cond_dir, v), (data - norm[chrg][0]) / (norm[chrg][1]))
        if v.endswith(cmp_suffix) or v.endswith(
            tar_suffix.format(index=max_index[key])
        ):
            data = (
                np.log10(np.loadtxt(join(img_dir, v)))
                if chrg == 1
                else np.loadtxt(join(img_dir, v))
            )
            np.savetxt(join(cond_dir, v), (data - norm[chrg][0]) / (norm[chrg][1]))
        if v.endswith(inp_suffix) and chrg == 0:
            v1 = v.replace("pot", "charge")
            vd = v.split("_", 4)[1]
            vg = v.split("_", 4)[2]
            loc = v.split("_", 4)[3]
            cmp_p = "_".join(v.split("_", 4)[:4]) + "_pot" + cmp_suffix
            tar_p = (
                "_".join(v.split("_", 4)[:4])
                + "_pot"
                + tar_suffix.format(index=max_index[key])
            )

            cmp_c = "_".join(v.split("_", 4)[:4]) + "_charge" + cmp_suffix
            tar_c = (
                "_".join(v.split("_", 4)[:4])
                + "_charge"
                + tar_suffix.format(index=max_index[key])
            )

            row = [
                v,
                v1,
                cmp_p,
                cmp_c,
                tar_p,
                tar_c,
                norm[0][0],
                norm[0][1],
                norm[1][0],
                norm[1][1],
                vd,
                vg,
                loc,
            ]
            writer.writerow(row)
