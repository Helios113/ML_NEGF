import torch
from data_loader import NEFG3x3Set
from AE import AE
import matplotlib.pyplot as plt


# Download the MNIST Dataset
dataset = NEFG3x3Set("info.csv", "data_3x3_10", "ml_res",transform=True)

# DataLoader is used to load the dataset
# for training
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=32,
                                     shuffle=True)

# Model Initialization
model = AE()

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.SGD(model.parameters(),
                             lr=1e-4)


epochs = 20
outputs = []
losses = []
for epoch in range(epochs):
    for (inp, tar, stat) in loader:

        # Reshaping the image to (-1, 676)
        inp = inp.reshape(-1, 26*26).float()
        tar = tar.reshape(-1, 26*26).float()

        # Output of Autoencoder
        reconstructed = model(inp)

        # Calculating the loss function
        loss = loss_function(reconstructed, tar)
        print(reconstructed.shape)
        print(tar.shape)
    

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        print(loss)
        losses.append(loss)
    outputs.append((epochs, tar, reconstructed))

# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
 
# Plotting the last 100 values
a = [i.detach().numpy() for i in losses[-100:]]

plt.plot(a)
plt.show()

for i, item in enumerate(tar):
  # Reshape the array for plotting
  item = item.reshape(-1, 26, 26)
  plt.imshow(item[0].detach().numpy())
 
for i, item in enumerate(reconstructed):
  item = item.reshape(-1, 26, 26)
  plt.imshow(item[0].detach().numpy())
plt.show()


