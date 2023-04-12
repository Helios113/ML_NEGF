import numpy as np
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
parser.add_argument('--res', action='store', default="results_gen",
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

if not os.path.exists(save_path):
    os.makedirs(save_path)

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


test_data = torch.utils.data.DataLoader(dataset=dataset, batch_size=1,
                                        shuffle=True)


init_time = datetime.now()
net = VAE(imgChannels=imgChannels, layers=layers, residu=residu, addX=addX).to(device)

net.load_state_dict(torch.load("results/test1/params.mp"))

net.eval()
for data in test_data:
    out = net(data[0]).squeeze().detach().numpy()
    name = f"NEGFXY_{data[-3].item()}_{data[-2].item()}_{data[-1].item()}"
    if target == "charge":
        out = out * data[8].item() + data[7].item()
        out = np.power(10, out)
        name+= "_charge_rec.txt"
    else:
        out = out*data[6].item()+data[5].item()
        name+= "_pot_rec.txt"
    # print(out.shape)
    np.savetxt(os.path.join(save_path, name), out)

    
    