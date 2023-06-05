import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

class NEFGSet(Dataset):
    # This class should be used to create PyTorch compatible objects from a given dataframe

    def __init__(self, df, use_dimension=True, device="mps"):
        self.df = df.reset_index()
        self.device = device
        self.mode = use_dimension

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        plane = self.df.iloc[idx]["Plane"]
        shape = [0, 0]

        x_val = ((20 + 2) * 5) + 1
        z_val = (((self.df.iloc[idx]["Height"]) + 2) * 5) + 1
        y_val = (((self.df.iloc[idx]["Width"]) + 2) * 5) + 1
        maxLoc = 0

        if plane == "XY":
            shape[0] = x_val
            shape[1] = y_val
            maxLoc = z_val
        elif plane == "YZ":
            shape[0] = y_val
            shape[1] = z_val
            maxLoc = x_val
        elif plane == "ZX":
            shape[0] = z_val
            shape[1] = x_val
            maxLoc = y_val

        tmpList = []
        dat = []
        tmpList.append(
            torch.from_numpy(np.loadtxt(self.df.iloc[idx]["inpPotPath"], dtype=np.float32)).to(
                self.device
            )
        )  # Load POT File
        
        tmpList.append(
            torch.from_numpy(np.loadtxt(self.df.iloc[idx]["inpChargePath"], dtype=np.float32)).to(
                self.device
            )
        )  # Load CHARGE file
        
        tmpList.append(torch.full(shape, self.df.iloc[idx]["VD"]).to(self.device))  # VD
        tmpList.append(torch.full(shape, self.df.iloc[idx]["VG"]).to(self.device))  # VG
        
        tmpList.append(
            torch.full(shape, self.df.iloc[idx]["Location"]/maxLoc).to(self.device)
        )  # Location
        
        print(self.df.iloc[idx]["Location"]/maxLoc)
        print(shape, maxLoc)

        if self.mode:
            tmpList.append(
                torch.full(shape, self.df.iloc[idx]["Height"]).to(self.device)
            )  # Add Height
            tmpList.append(
                torch.full(shape, self.df.iloc[idx]["Width"]).to(self.device)
            )  # Add Width

        tmpList.append(
            (
                torch.arange(0, shape[0])
                .reshape(shape[0], 1)
                .expand(shape[0], shape[1])
                / shape[0]
            ).to(self.device)
        )
        tmpList.append(
            (
                torch.arange(0, shape[1])
                .reshape(1, shape[1])
                .expand(shape[0], shape[1])
                / shape[1]
            ).to(self.device)
        )

        imp = torch.stack(tuple(tmpList), dim=0)

        dat.append(imp)
        
        for i in range(4):
            # Appends cmp and tar filepaths
            dat.append(
                torch.from_numpy(np.loadtxt(self.df.iloc[idx, i + 11], dtype=np.float32)).to(self.device)
            )
            
        for i in range(4):
            # Appends mean and std
            dat.append(
                self.df.iloc[idx, i + 15]
            )
            
        for i in range(3):
            # Appends VG, VD, Location
            dat.append(self.df.iloc[idx, i + 3])

        for i in range(2):
            # Appends height and width
            dat.append(self.df.iloc[idx, i + 7])

        return dat
