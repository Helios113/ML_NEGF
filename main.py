import torch
import math
import sys
from data_loader import NEFG3x3Set
from AE import AE
from AE import CustomLoss
import matplotlib.pyplot as plt

"""
Determine if any GPUs are available
"""
if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

print(f"Using device: {device}")


device = torch.device(device)


dataset = NEFG3x3Set(info_file, data_folder,
                     test_folder, transform=True)

length = len(dataset)
train_split = math.floor(length*.7)
test_split = length - train_split

train_inds, test_inds = torch.utils.data.random_split(
    dataset, [train_split, test_split], generator=torch.Generator().manual_seed(42))

# Model Initialization
model = AE().to(dev)

loss_function = CustomLoss()

optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-3)

train_data = torch.utils.data.DataLoader(dataset=train_inds, batch_size=32,
                                         shuffle=True)
test_data = torch.utils.data.DataLoader(dataset=test_inds, batch_size=32,
                                        shuffle=True)


epochs = 100
outputs = []
losses = []
local_loss = 0
txt1 = color.BOLD+"Epoch {epoch:0>3d}/"+str(epochs) + color.END
txt2 = "Average training loss:   {loss}"
txt3 = "Time taken for training: {time}"
txt4 = "Average test loss:       {loss}"


start_time = datetime.now()
cur_time = 0

plt.style.use('fivethirtyeight')
fig,ax  = plt.subplots()
plt.tight_layout()

for epoch in range(epochs):
    print(txt1.format(epoch=epoch))
    cur_time = datetime.now()
    for (inp, tar, _, _, _) in train_data:
        model.train()
        inp = inp.reshape(-1, 71*26).float()
        tar = tar.reshape(-1, 71*26).float()
        # Output of Autoencoder
        reconstructed = model(inp)

        # Calculating the loss function
        loss = loss_function(tar, reconstructed)

        local_loss = local_loss+loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(txt2.format(loss=local_loss/len(train_data)))
    print(txt3.format(time=datetime.now()-cur_time))
    
    losses.append(local_loss/len(test_data))
    a = [i.cpu().detach().numpy() for i in losses]
    
    ax.clear()
    ax.plot(a)
    plt.savefig(fig_file)
    
    local_loss = 0

    for (inp, tar, _, _, _) in test_data:
        model.eval()

        inp = inp.reshape(-1, 71*26).float()
        tar = tar.reshape(-1, 71*26).float()

        # Output of Autoencoder
        reconstructed = model(inp)

        # Calculating the loss function
        loss = loss_function(tar, reconstructed)
        local_loss += loss

    print(txt4.format(loss=local_loss/len(test_data)))

    local_loss = 0

print(color.BOLD+"Total time:{time}".format(time = datetime.now()-start_time)+color.END)

torch.save(model.state_dict(), "trained_models/"+save_file)


# Plotting the last 100 values

print(losses[-1])

plt.show(block=True)
