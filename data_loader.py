import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


class NEFGSet(Dataset):
    # This class should be used to create PyTorch compatible objects from a given data file

    def __init__(
        self, csv_file, root_dir, data_folder, shape=(111, 71), device="mps"
    ):
        # self.landmarks_frame = pd.read_csv(csv_file)
        # self.root_dir = root_dir
        self.shape = shape


        # self.data_dir = os.path.join(root_dir, data_folder)
        # self.labels = pd.read_csv(os.path.join(root_dir, csv_file), header=True)

        self.df = pd.Dataframe()
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dat = []
        imp = torch.stack(
            (
                torch.from_numpy(
                    np.loadtxt(
                        self.df.iloc[idx]["inpPotPath"]
                    )
                ).to(self.device), # Load POT File

                torch.from_numpy(
                    np.loadtxt(
                        self.df.iloc[idx]["inpChargePath"]
                    )
                ).to(self.device), # Load CHARGE file
                
                torch.full(self.shape, self.df.iloc[idx]["VD"]).to(self.device), # VD

                torch.full(self.shape, self.df.iloc[idx]["VG"]).to(self.device), # VG

                torch.full(self.shape, self.df.iloc[idx]["Location"]).to(self.device), # LOC

                torch.full(self.shape, self.df.iloc[idx]["Height"]).to(self.device), # Add Height

                torch.full(self.shape, self.df.iloc[idx]["Width"]).to(self.device), # Add Width

                (
                    torch.arange(0, self.shape[0])
                    .reshape(self.shape[0], 1)
                    .expand(self.shape[0], self.shape[1])
                    / self.shape[0]
                ).to(self.device),

                (
                    torch.arange(0, self.shape[1])
                    .reshape(1, self.shape[1])
                    .expand(self.shape[0], self.shape[1])
                    / self.shape[1]
                ).to(self.device),

            ),

        )
        dat.append(imp)

        # self.labels.iloc[idx, 2] =  self.labels.iloc[idx, 2][:-5]+"3.txt"ß
        # self.labels.iloc[idx, 3] =  self.labels.iloc[idx, 2][:-5]+"3.txt"

        # print(self.labels.iloc[idx, 2])

        for i in range(8):
            dat.append(
                torch.from_numpy(
                    np.loadtxt(
                        self.df.iloc[idx, i + 9]
                    )
                ).to(self.device)
            )
        for i in range(3):
            dat.append(self.df.iloc[idx, i + 1])

        for i in range(2):
            dat.append(self.df.iloc[idx, i + 5])

        return dat
