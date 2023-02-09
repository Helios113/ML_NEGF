import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


class NEFG3x3Set(Dataset):

    def __init__(self, csv_file, root_dir, data_folder, transform=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.data_dir = root_dir+"/"+data_folder
        self.transform = transform
        self.labels = pd.read_csv(root_dir+"/"+csv_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inp_name = os.path.join(self.data_dir, self.labels.iloc[idx, 0])
        tar_name = os.path.join(self.data_dir, self.labels.iloc[idx, 1])

        inp = np.loadtxt(inp_name)
        tar = np.loadtxt(tar_name)
        if self.transform:
            mean, std = inp.mean(), inp.std()
            transform_norm = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                                ])
            inp = transform_norm(inp)
            tar = transform_norm(tar)
        return inp, tar, (mean, std)
