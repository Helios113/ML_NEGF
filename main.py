import math
import torch
import torch.nn.functional as F
from data_loader import NEFGSet
from models.VAE import VAE
import argparse
import os
from datetime import datetime, timedelta
torch.manual_seed(42)
# Instantiate the parser
parser = argparse.ArgumentParser(description='ML_NEGF')


# parser.add_argument('--batch_size', type=int,
#                     help='An optional integer argument')

# Optional argument
parser.add_argument('--batch_size', type=int, nargs='?', default=50,
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


dataset = NEFGSet("info_dat_std_NEGFXY.csv",
                  "data/3x12_16_damp00", "dat_std", device=device, locXY=locXY)

imgChannels = int(dataset[0][0].shape[0])
# Split data into training and validation sets

length = len(dataset)
train_split = math.floor(length*train_split_ration)
test_split = length - train_split
train_inds, test_inds = torch.utils.data.random_split(
    dataset, [train_split, test_split], generator=torch.Generator().manual_seed(42))


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

train_data = torch.utils.data.DataLoader(dataset=train_inds, batch_size=batch_size,
                                         shuffle=True)
test_data = torch.utils.data.DataLoader(dataset=test_inds, batch_size=batch_size,
                                        shuffle=True)

init_time = datetime.now()
net = VAE(imgChannels=imgChannels, layers=layers, residu=residu, addX=addX).to(device)

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

for epoch in range(num_epochs):
    last_time = datetime.now()
    loss1 = 0
    kl_divergence1 = 0
    net.train()
    for idx, data in enumerate(train_data, 0):
        out = net(data[0])
        loss = F.mse_loss(out, data[tar].unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss1 += loss.item()

    loss1 = loss1/len(test_data)
    print('{},{},{},{}'.format(datetime.now()-init_time, datetime.now()-last_time,epoch, loss1), file=lss_train_f)

    l2 = 0
    last_time = datetime.now()
    net.eval()
    for data in test_data:
        out = net(data[0])
        if epoch == 1:
            cmp = data[tar-2]
            # print((data[3].shape))
            l1 += F.mse_loss(data[tar], cmp).item()
        l2 += F.mse_loss(data[tar].unsqueeze(1), out).item()
    if l2 < min_loss:
        min_loss = l2
        torch.save(net.state_dict(), params_path)
        saved_epoch = epoch
    if epoch == 1:
        print("CMP:", l1)
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
