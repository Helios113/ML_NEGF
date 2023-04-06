import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


class NEFG3x3Set(Dataset):

    def __init__(self, csv_file, root_dir, data_folder, shape = (111,71), device="mps"):

        # self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.shape = shape
        self.data_dir = root_dir+"/"+data_folder
        self.labels = pd.read_csv(root_dir+"/"+csv_file, header=None)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dat = []
        imp = torch.stack((
            torch.from_numpy(np.loadtxt(os.path.join(self.data_dir,str(self.labels.iloc[idx, 0])), dtype="float32")).to(self.device),
            torch.from_numpy(np.loadtxt(os.path.join(self.data_dir,str(self.labels.iloc[idx, 1])), dtype="float32")).to(self.device),
            torch.full(self.shape,self.labels.iloc[idx, 10]).to(self.device),
            torch.full(self.shape,self.labels.iloc[idx, 11]).to(self.device),
            torch.full(self.shape,self.labels.iloc[idx, 12]).to(self.device),
            torch.arange(0,self.shape[0]).reshape(self.shape[0], 1).expand(self.shape[0], self.shape[1]).to(self.device)
            # torch.arange(0,self.shape[0]*self.shape[1]).view(self.shape).to(self.device)
        ), dim = 0)
        dat.append(imp)

        # self.labels.iloc[idx, 2] =  self.labels.iloc[idx, 2][:-5]+"3.txt"
        # self.labels.iloc[idx, 3] =  self.labels.iloc[idx, 2][:-5]+"3.txt"

        # print(self.labels.iloc[idx, 2])
        for i in range(4):
            dat.append(torch.from_numpy(np.loadtxt(os.path.join(self.data_dir,str(self.labels.iloc[idx, i+2])), dtype="float32")).to(self.device))

        return dat
    
# dataset = NEFG3x3Set("info_dat_std_NEGFXY.csv",
#                      "data/3x12_16_damp00", "dat_std")
# dataset[0]
