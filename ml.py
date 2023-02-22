import torch
import math
from AE import AE
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
# Download the MNIST Dataset

ml = AE()
ml.load_state_dict(torch.load("3x3_12_charge.mp"))
ml.eval()
img_dir = '/ml_res'
cond_dir = '/ml_data'
vg = float(sys.argv[1])
vd = float(sys.argv[2])

print(vg,vd)

files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]


inp_suffix = "_charge_inp.txt"
# NEGF_0.8_0.05_0_charge_tar
files = [f for f in listdir(img_dir) if f.startswith("NEGF_"+vg+"_"+vd)]

inp = np.array([])

for f in files:
    dat = np.loadtxt(join(img_dir, f)).flatten()
    inp = np.concatenate((inp,dat))
inp = np.log10(inp)
if inp.size != 0:
    mmax = (inp.max(), inp.min())
    
for f in files:
    dat = np.loadtxt(join(img_dir, f))
    rec = ml(dat)
    rec = np.power(10, (rec*(max-min))+min)
    np.savetxt(join(cond_dir, f), (np.power(10, (ml*(mmax[0]-mmax[1])+mmax[1]))))
