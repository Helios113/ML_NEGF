import torch
import math
from data_loader import NEFG3x3Set
from AE import AE
import matplotlib.pyplot as plt


# Download the MNIST Dataset
dataset = NEFG3x3Set("info.csv", "data_3x3_10", "ml_res",transform=True)

length = len(dataset)
train_split = math.floor(length*.7)
test_split = length - train_split

train_inds, test_inds = torch.utils.data.random_split(dataset, [train_split, test_split], generator=torch.Generator().manual_seed(42))



# Model Initialization
model = AE()

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-2)

train_data = torch.utils.data.DataLoader(dataset=train_inds, batch_size=32,
                                            shuffle=True)
test_data = torch.utils.data.DataLoader(dataset=test_inds, batch_size=32,
                                            shuffle=True)

for i in test_data:
    print(i)

