import math
import torch
import torch.nn.functional as F
from data_loader import NEFGSet
from models.VAE import VAE
import argparse
import os
from datetime import datetime, timedelta
import pandas as pd
from yaml_query import *
import matplotlib.pyplot as plt
torch.manual_seed(42)
import numpy as np
# Instantiate the parser
parser = argparse.ArgumentParser(description='ML_NEGF')


# parser.add_argument('--batch_size', type=int,
#                     help='An optional integer argument')

#Path to yaml files
parser.add_argument('--query_path',type=str,action='store',required=True,help='Path to yaml Query')

#Path to yaml file
parser.add_argument('--cond_data_path',type=str,action='store',required=True,help='Path to conditioned data')

# Optional argument
parser.add_argument('--batch_size', type=int, nargs='?', default=32,
                    help='Batch size')

# Optional argument
parser.add_argument('--epochs', type=int, nargs='?', default=500,
                    help='Epochs')

# Optional argument
parser.add_argument('--lr', type=float, nargs='?', default=1e-3,
                    help='Learningn rate')

# Optional argument
parser.add_argument('--split', type=float, nargs='?', default=0.7,
                    help='Training data portion')

# Switch
parser.add_argument('--notLocXY', action='store_false',
                    help='Generate location map in X and Y')

# Switch
parser.add_argument('--residu', nargs="+", type=int, default=[],
                    help='Residual connections')

parser.add_argument('--layers', nargs="+", type=int, default=[8, 16, 32, 64],
                    help='Layer description')

# Switch
parser.add_argument('--addX', action='store_false',
                    help='Add X to end of the encoder')

parser.add_argument('--gpu', action='store_true',
                    help='Activate GPU support')
parser.add_argument('--name', required=True, action='store',
                    type=str, help='Folder name for result saving')
parser.add_argument('--tar', required=True,choices=['pot', 'charge'], action='store',
                    type=str, help='Folder name for result saving')
parser.add_argument('--res', action='store', default="results",
                    type=str, help='Folder name for parent dir')


args = parser.parse_args()


"""
Parameters
"""
condiditioned_path = args.cond_data_path
query_path = args.query_path
batch_size = args.batch_size
num_epochs = args.epochs
lr = args.lr
locXY = args.notLocXY
residu = {}
for i in args.residu:
    residu[i] = None
train_split_ration = args.split
gpu_support = args.gpu
addX = args.addX
layers = args.layers
target = args.tar
tar = int(target=="pot")+3
"""
Paths
"""
save_path = os.path.join(args.res, args.name)
# model parameters
params_path = os.path.join(save_path, "params.mp")
# text file with the information of the run
info_path = os.path.join(save_path, "info.txt")
# loss vs iter
loss_train_path = os.path.join(save_path, "loss_train.txt")
loss_test_path = os.path.join(save_path, "loss_test.txt")
# statistics
stats_path = os.path.join(save_path, "stats.txt")



"""
Fiels
"""
if not os.path.exists(save_path):
    os.makedirs(save_path)
inf_f = open(info_path, "w+")
lss_train_f = open(loss_train_path, "w+")
lss_test_f = open(loss_test_path, "w+")
sts_f = open(stats_path, "w+")



# Determine hardware availability
if torch.cuda.is_available() and gpu_support:
    device = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available() and gpu_support:
    device = "mps"  # Apple GPU
else:
    device = "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

query = query_data(querypath=query_path,datapath=condiditioned_path)
dataframe1, dataframe2 = query.query_search()

train_dataset_list = list()
test_dataset_list = list()
for df in dataframe1:
    train_dataset_list.append(NEFGSet(df, device=device))

for df in dataframe2:
    test_dataset_list.append(NEFGSet(df, device=device))

imgChannels = 9
# Split data into training and validation sets

# Used in printing no. of samples from each set
train_split,test_split = 0,0
for i in range(len(train_dataset_list)):
    train_split += len(train_dataset_list[i])
for i in range(len(test_dataset_list)):
    test_split += len(test_dataset_list[i])

#length = len(dataset)
#train_split = math.floor(length*train_split_ration)
#test_split = length - train_split
#train_inds, test_inds = torch.utils.data.random_split(
#    dataset, [train_split, test_split], generator=torch.Generator().manual_seed(42))


train_data_list = list()
test_data_list = list()
for train_dataset in train_dataset_list:
    train_data_list.append(torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                        shuffle=True))
for test_dataset in test_dataset_list:
    test_data_list.append(torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                        shuffle=True))

init_time = datetime.now()
net = VAE(imgChannels=imgChannels, layers=layers, residu=residu, addX=addX).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
inf_f.flush()
printed_stat = True
saved_epoch = 0

loss = 0
min_loss = 1000
l1 = 0


len_test_data = 0
for i in range(len(test_data_list)):
    len_test_data += len(test_data_list)


for epoch in range(num_epochs):
    last_time = datetime.now()
    loss1 = 0
    kl_divergence1 = 0
    net.train()
    fig, axs = plt.subplots(3, 3, figsize=(6, 6), layout='constrained')
    for train_data in train_data_list:
        for idx, data in enumerate(train_data, 0):
            print(type(data[0][0].cpu().detach().numpy()))
            im = axs[0,0].imshow(data[0][0][0].cpu().detach().numpy())
            im2 = axs[0,1].imshow(data[0][0][1].cpu().detach().numpy())
            axs[0,2].imshow(data[0][0][2].cpu().detach().numpy())
            axs[1,0].imshow(data[0][0][3].cpu().detach().numpy())
            axs[1,1].imshow(data[0][0][4].cpu().detach().numpy())
            axs[1,2].imshow(data[0][0][5].cpu().detach().numpy())
            axs[2,0].imshow(data[0][0][6].cpu().detach().numpy())
            im8 = axs[2,1].imshow(data[0][0][7].cpu().detach().numpy())
            im9 = axs[2,2].imshow(data[0][0][8].cpu().detach().numpy())
            axs[0,2].text(12, 12,str(np.mean((data[0][0][2].cpu().detach().numpy()))),ha='center', va='center',c="white")
            axs[1,0].text(12, 12,str(np.mean((data[0][0][3].cpu().detach().numpy()))),ha='center', va='center',c="white")
            axs[1,1].text(12, 12,str(np.mean((data[0][0][4].cpu().detach().numpy()))),ha='center', va='center',c="white")
            axs[1,2].text(12, 12,str(np.mean((data[0][0][5].cpu().detach().numpy()))),ha='center', va='center',c="white")
            axs[2,0].text(12, 12,str(np.mean((data[0][0][6].cpu().detach().numpy()))),ha='center', va='center',c="white")
            plt.colorbar(im, ax=axs[0, 0])
            plt.colorbar(im2, ax=axs[0, 1])
            plt.colorbar(im8, ax=axs[2, 1])
            plt.colorbar(im9, ax=axs[2, 2])
            axs[0,0].set_title("Potential")
            axs[0,1].set_title("Charge")
            axs[0,2].set_title("VD")
            axs[1,0].set_title("VG")
            axs[1,1].set_title("Location")
            axs[1,2].set_title("Height")
            axs[2,0].set_title("Width")
            axs[2,1].set_title("X Location Map")
            axs[2,2].set_title("Y Location Map")
        
            fig.savefig("testing_data_loader.png")
            quit()