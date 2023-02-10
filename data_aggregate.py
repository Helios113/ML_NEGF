from os import listdir
import numpy as np
from os.path import isfile, join
import csv

res_dir = 'data_3x3_10'
img_dir = res_dir+'/ml_res'
cond_dir = res_dir+'/dat_std'

info_file = open(res_dir+'/info.csv', 'w+')
writer = csv.writer(info_file)


inp_suffix = "_inp.txt"
tar_suffix = "_tar.txt"


files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]

files.sort()
# print(files)


# Find the same devices everywhere
unique_names = set()
for f in files:
    unique_names.add("_".join(f.split("_", 3)[:3]))

inp = np.array([])
data = {}
for j in unique_names:
    for f in files:
        if f.startswith(j) and f.endswith(inp_suffix):
            inp = np.concatenate((inp, np.loadtxt(join(img_dir, f)).flatten()))

    data[j] = (inp.mean(), inp.std())
    inp = np.array([])
    

for f in files:
    for i in data.keys():
        if f.startswith(i):
            j = data[i]
            np.savetxt(join(cond_dir, f),(np.loadtxt(join(img_dir, f))-j[0])/j[1])

for f in files:
    if f.endswith(inp_suffix):
        tar = f.replace(inp_suffix, tar_suffix)
        row = [f, tar]
        writer.writerow(row)

info_file.close()