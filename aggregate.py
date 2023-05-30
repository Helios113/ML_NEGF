from os import listdir
import numpy as np
from os.path import isfile, join
import csv
import os
import pandas as pd

import matplotlib.pyplot as plt


data = pd.DataFrame(columns=["Cirt", "Widht", "Height", "Plane", "Vg", "Vd", "loc", "type", "iter"])

path = "new_4page"

x = [x[0] for x in os.walk(path) if "ml_test" in x[0]]
cnt = 0
for i in x:
    g = i.split("/")
    w = g[2].split("x")
    for f in listdir(i):
        f = f.split("_")
        f[0] = f[0][-2:]
        f[-1] = f[-1][:-4]
        data.loc[cnt] = [g[1]] + w + f
        cnt+=1
        print(cnt)
        # data = data.append({"Cirt":g[1],
        #                     "Widht":w[0],
        #                     "Height":w[1],
        #                     "Plane":f[0], 
        #                     "Vg":f[1],
        #                     "Vd":f[2],
        #                     "loc":f[3],
        #                     "type":f[4],
        #                     "iter":f[5]}
        #                    )
    print(data)
    exit()


# img_dir = res_dir+'/ml_test'
# cond_dir = res_dir+"/"+name+'/dat_'+target

# info_file = open(res_dir+"/"+name+'/info_dat_{tar}.csv'.format(tar=target), 'w+')
# writer = csv.writer(info_file)


# target_prefix = "NEGFZX"
# inp_suffix = "_{target}_1.txt".format(target=target)
# tar_suffix = "_{target}_{index}.txt".format(target=target, index = "{index}")

# np.seterr(all='raise')
# files = [f for f in listdir(img_dir) if isfile(join(img_dir, f)) and f.startswith(target_prefix)]

# files.sort()
# # print(files)


# # Find the same devices everywhere
# unique_names = set()
# for f in files:
#     unique_names.add("_".join(f.split("_", 3)[:3]))

# # print(unique_names)

# inp = np.array([])
# data = {}
# for j in unique_names:
#     for f in files:
#         if f.startswith(j) and f.endswith(inp_suffix):
#             # print(f)
#             dat = np.loadtxt(join(img_dir, f)).flatten()
#             if not np.isnan(dat).any():
#                 inp = np.concatenate((inp,dat))
#             else:
#                 print("HEy")
    
#     if np.isnan(inp).any() or np.isinf(inp).any() or not np.all(inp):
#         print("Error")
        
#     # CONDITIONING
#     # ################
#     inp = np.log10(inp)
#     # inp = np.power(10, inp)
    
    
#     # #################
#     if inp.size != 0:
#         data[j] = (inp.max(), inp.min())
#     # plt.hist(inp)
#     # # plt.hist(np.log(inp))
#     # plt.show()
#     # input()
    
#     inp = np.array([])
    


# # find all files which have the correct begining and type
# # beggining with and contains, and get lat index, so basically split
# # then take the first, second and last for training


# name_max_index = {}

# for i in data.keys():
#     for f in files:    
#         if f.startswith(i) and target in f:
#             index = int(f.split("_")[-1][:-4])
#             if i+target in name_max_index:
#                 name_max_index[i+target] = name_max_index[i+target] if name_max_index[i+target]>index else index
#             else:
#                 name_max_index[i+target] = index
  
# for i in data.keys():
#     # names = []
#     for f in files:    
#         if f.startswith(i):
#             if f.endswith(inp_suffix) or f.endswith(tar_suffix.format(index = 2)): # or ) :
#                 # j = data[i]
#                 np.savetxt(join(cond_dir, f), (np.log10(np.loadtxt(join(img_dir, f)))))#-j[1])/(j[0]-j[1]))
#                 # np.savetxt(join(cond_dir, f), (np.power(10, np.loadtxt(join(img_dir, f)))-j[1])/(j[0]-j[1]))
#                 # np.savetxt(join(cond_dir, f), (np.loadtxt(join(img_dir, f))))

#                 # j=np.loadtxt(join(img_dir, f))
#                 # j = (j-j.min())/(j.max()-j.min())
#                 # np.savetxt(join(cond_dir, f), j)

                
#                 # names.append(f)
#             elif f.endswith(tar_suffix.format(index = name_max_index[i+target])):
#                 f1 = "_".join(f.split("_", 4)[:4])+tar_suffix.format(index = "tar")
#                 j = data[i]
#                 np.savetxt(join(cond_dir, f1), (np.log10(np.loadtxt(join(img_dir, f)))))#-j[1])/(j[0]-j[1]))
#                 # np.savetxt(join(cond_dir, f1), (np.power(10, np.loadtxt(join(img_dir, f)))-j[1])/(j[0]-j[1]))
#                 # np.savetxt(join(cond_dir, f1), (np.loadtxt(join(img_dir, f))))

#                 # j=np.loadtxt(join(img_dir, f))
#                 # j = (j-j.min())/(j.max()-j.min())
#                 # np.savetxt(join(cond_dir, f1), j)

#                 # names.append(f1)



# for f in files:
#     if f.endswith(inp_suffix):
#         devId = "_".join(f.split("_", 3)[:3])
#         vd = f.split("_", 4)[1]
#         vg = f.split("_", 4)[2]
#         loc = f.split("_", 4)[3]

#         cmp = "_".join(f.split("_", 4)[:4])+tar_suffix.format(index = 2)
#         tar = "_".join(f.split("_", 4)[:4])+tar_suffix.format(index = "tar")
        
#         row = [f, cmp, tar, data[devId][0], data[devId][1], vd, vg, loc]
#         writer.writerow(row)    

# info_file.close()
# #  NEGF_0.51875_0.51875_75_charge_10.txt