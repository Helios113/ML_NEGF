import math
import torch
import torch.nn.functional as F
from data_loader import NEFGSet
from models.VAE import VAE
import argparse
import os
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from yaml_query import *

torch.manual_seed(42)
np.random.seed(0)
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



parser.add_argument('--drop', action='store', default=0.5,
                    type=float, help='Dropout value between 0 and 1')

parser.add_argument('--batch', action=argparse.BooleanOptionalAction, default=True)

parser.add_argument('--noise', action=argparse.BooleanOptionalAction, default=False)

parser.add_argument('--locat', action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()


"""
Parameters
"""
condiditioned_path = args.cond_data_path
query_path = args.query_path
batch_size = args.batch_size
batch_use = args.batch
dropout_val = args.drop
location = args.locat
noise = args.noise
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
tar = int(target=="charge")+3 # 4 if pot, 3 if charge
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
Files
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

query = query_data(condiditioned_path)
dataframe1, dataframe2 = query.query_search(query_path)

train_dataset_list = list()
test_dataset_list = list()

for df in dataframe1:
    train_dataset_list.append(NEFGSet(df, use_dimension=True, use_location=location, use_noise=noise, device=device))

for df in dataframe2:
    test_dataset_list.append(NEFGSet(df, use_dimension=True, use_location=location, use_noise=False, device=device))

imgChannels = train_dataset_list[0][0][0].shape[0]
# Split data into training and validation sets

# Used in printing no. of samples from each set
train_split,test_split = 0,0
for i in range(len(train_dataset_list)):
    train_split += len(train_dataset_list[i])
for i in range(len(test_dataset_list)):
    test_split += len(test_dataset_list[i])

# length = len(dataset)
# train_split = math.floor(length*train_split_ration)
# test_split = length - train_split
# train_inds, test_inds = torch.utils.data.random_split(
# dataset, [train_split, test_split], generator=torch.Generator().manual_seed(42))

print(r"{:-^30}".format("PID"), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="pid", val=os.getpid()), file=inf_f)
print(r"{:-^30}".format("Location"), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="folder name", val=save_path), file=inf_f)
print(r"{:-^30}".format("Parameters"), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="layers", val=layers), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="target", val=target), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="batch size", val=batch_size), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="epochs", val=num_epochs), file=inf_f)
print(r"{txt:<20}:{val:.1e}".format(txt="lr", val=lr), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="generate XY", val=locXY), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="Batch Norm", val=batch_use), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="Noise in the input", val=noise), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="Loc in 3rd dim", val=location), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="residu", val=residu), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="device", val=device), file=inf_f)
print(r"{txt:<20}:{val}".format(
    txt="img channels", val=imgChannels), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="addX", val=addX), file=inf_f)
print(r"{:-^30}".format(""), file=inf_f)
print(r"{txt:<20}:{val1:d}/{val2:d}".format(txt="data split:",
    val1=int(train_split_ration*100), val2=int(100-train_split_ration*100)), file=inf_f)
print(r"{txt:<20}:{val}".format(
    txt="train samples", val=train_split), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="test samples", val=test_split), file=inf_f)
print(r"{txt:<20}:{val}".format(txt="total",
    val=train_split+test_split), file=inf_f)
print(r"{:-^30}".format("Model"), file=inf_f)

train_data_list = list()
test_data_list = list()
for train_dataset in train_dataset_list:
    train_data_list.append(torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                        shuffle=True))
for test_dataset in test_dataset_list:
    test_data_list.append(torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                        shuffle=True))

init_time = datetime.now()
net = VAE(imgChannels=imgChannels, layers=layers, residu=residu, addX=addX, use_batch=batch_use, dropout=dropout_val).to(device)

print(net, file=inf_f)
print(r"{:-^30}".format("Optimiser"), file=inf_f)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
print(optimizer, file=inf_f)
inf_f.flush()
printed_stat = True
saved_epoch = 0

loss = 0
min_loss = 1000
l1 = 0
print('total_time,epoch_time,epoch,loss', file=lss_train_f)
print('total_time,epoch_time,epoch,loss', file=lss_test_f)

len_train_data = 0
for train_data in train_data_list:
    len_train_data += len(train_data)
    
len_test_data = 0
for test_data in test_data_list:
    len_test_data += len(test_data)


for epoch in range(num_epochs):
    last_time = datetime.now()
    loss1 = 0
    kl_divergence1 = 0
    net.train()
    for train_data in train_data_list:
        for data in train_data:
            optimizer.zero_grad()
            # print(data)
            # print(data[0])
            # print(len(data)) # 14
            # print(len(data[0])) # 32
            out = net(data[0], tar-3)
            loss = F.mse_loss(out, data[tar].unsqueeze(1))
            loss.backward() 
            optimizer.step()

            if epoch == 1:
                cmp = data[tar-2]
                l1 += F.mse_loss(data[tar], cmp).item()
            loss1 += loss.item()

    loss1 = loss1/len_train_data
    if epoch == 1:
        print("CMP_train:", l1/len_train_data, file=inf_f) # somethings likely wrong with the loss calculation here
        inf_f.flush()
    print('{},{},{},{}'.format(datetime.now()-init_time, datetime.now()-last_time,epoch, loss1), file=lss_train_f)
    l1 = 0
    l2 = 0
    last_time = datetime.now()
    net.eval()
    for test_data in test_data_list:
        for data in test_data:
            out = net(data[0], tar-3)
            if epoch == 1:
                cmp = data[tar-2]
                l1 += F.mse_loss(data[tar], cmp).item()
            l2 += F.mse_loss(out, data[tar].unsqueeze(1)).item()
            
    l2 = l2/len_test_data
    if l2 < min_loss:
        min_loss = l2
        torch.save(net.state_dict(), params_path)
        saved_epoch = epoch
    if epoch == 1:
        l1 = l1/len_test_data
        print("CMP_test:", l1, file=inf_f)
        inf_f.flush()
    print('{},{},{},{}'.format(datetime.now()-init_time, datetime.now()-last_time,epoch, l2), file=lss_test_f)
    lss_test_f.flush()
    lss_train_f.flush()
    if l2 < l1 and printed_stat:
        printed_stat = False
        print(r"test loss < cmp loss @ {}".format(epoch), file=sts_f)


print(r"Cmp loss                :{}".format(l1), file=sts_f)
print(r"Min test loss           :{}".format(min_loss), file=sts_f)
print(r"Min test loss at epoch  :{}".format(saved_epoch), file=sts_f)


inf_f.close()
lss_train_f.close()
lss_test_f.close()
sts_f.close()
