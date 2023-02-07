from os import listdir
from os.path import isfile, join
import csv

res_dir = 'data_3x3_10'
img_dir = res_dir+'/ml_res'

info_file = open(res_dir+'/info.csv', 'w+')
writer = csv.writer(info_file)


inp_suffix = "_inp.txt"
tar_suffix = "_tar.txt"


files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]

files.sort()
# print(files)

for f in files:
    if f.endswith(inp_suffix):
        tar = f.replace(inp_suffix, tar_suffix)
        row = [f, tar]
        writer.writerow(row)

info_file.close()