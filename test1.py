import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


root_dir = "data/3x12_16_damp00"
data_dir = root_dir+"/dat_std"
labels = pd.read_csv(root_dir+"/info_dat_std_NEGFZX.csv", header=None)



idx = 50


inp_name = os.path.join(data_dir, str(labels.iloc[idx, 1]))
cmp_name = os.path.join(data_dir, str(labels.iloc[idx, 3]))
tar_name = os.path.join(data_dir, str(labels.iloc[idx, 5]))
# print(str(labels.iloc[idx, 5]),labels.iloc[idx, 6], labels.iloc[idx, 0])


inp = np.loadtxt(inp_name, dtype="float32")
cmp = np.loadtxt(cmp_name, dtype="float32")
tar = np.loadtxt(tar_name, dtype="float32")


fig, axs = plt.subplots(nrows = 1, ncols=3)

a = axs[0].imshow(inp)
axs[0].set_title('Input')
v = np.linspace(inp.min(), inp.max(), 5, endpoint=True)
plt.colorbar(a, ticks=v)

a = axs[2].imshow(tar)
axs[2].set_title('Target')

v = np.linspace(tar.min(), tar.max(), 5, endpoint=True)
plt.colorbar(a, ticks=v)


a = axs[1].imshow(tar-inp)
axs[1].set_title('Delta')

v = np.linspace((tar-inp).min(), (tar-inp).max(), 5, endpoint=True)
plt.colorbar(a, ticks=v)


plt.title(str(labels.iloc[idx, 0]))

plt.show()