from os import listdir
import numpy as np
from os.path import isfile, join
import csv


import matplotlib.pyplot as plt


res_dir = 'data/3x12_16'
img_dir = res_dir+'/ml_res'
cond_dir = res_dir+'/dat_pot_10_std'

info_file = open(res_dir+'/info_dat_pot_10_std.csv', 'w+')
writer = csv.writer(info_file)


inp_suffix = "_pot_inp.txt"
tar_suffix = "_pot_tar.txt"

np.seterr(all='raise')
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
            dat = np.loadtxt(join(img_dir, f)).flatten()
            if not np.isnan(dat).any():
                inp = np.concatenate((inp,dat))
            else:
                print("HEy")
    
    if np.isnan(inp).any() or np.isinf(inp).any() or not np.all(inp):
        print("Error")
        
    # CONDITIONING
    # ################
    inp = np.power(10, inp)
    
    # #################
    if inp.size != 0:
        data[j] = (inp.max(), inp.min())
    # plt.hist(inp)
    # # plt.hist(np.log(inp))
    # plt.show()
    # input()
    
    inp = np.array([])
    

for f in files:
    for i in data.keys():
        if f.startswith(i) and (f.endswith(inp_suffix) or f.endswith(tar_suffix)):
            j = data[i]
            # np.savetxt(join(cond_dir, f),(np.loadtxt(join(img_dir, f))-j[1])/(j[0]-j[1]))
            try:
                np.savetxt(join(cond_dir, f), (np.power(10, np.loadtxt(join(img_dir, f)))-j[1])/(j[0]-j[1]))
            except Exception as e:
                print(np.loadtxt(join(img_dir, f)))
                print(f)
                print(e)
                input()

for f in files:
    if f.endswith(inp_suffix):
        devId = "_".join(f.split("_", 3)[:3])
        # print(devId)
        tar = f.replace(inp_suffix, tar_suffix)
        row = [f, tar, data[devId][0], data[devId][1]]
        writer.writerow(row)

info_file.close()