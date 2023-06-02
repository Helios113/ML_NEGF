
import torch.nn as nn
import torch

"""
A Convolutional Variational Autoencoder
"""
class VAE(nn.Module):
    # ther og net had 128 final channels
    def __init__(self, imgChannels=3, layers = [16,32,64,64], residu = {}, addX = True):
        super(VAE, self).__init__()

        self.enc_modules = nn.ModuleList()
        self.dec_modules = nn.ModuleList()
        self.residu = residu
        self.addX = addX
        inChannels = imgChannels
        outChannels = layers[0]
        for i in layers:
            self.enc_modules.append(
                nn.Sequential(
                nn.Conv2d(in_channels=inChannels, out_channels=i, kernel_size=5),
                nn.BatchNorm2d(i),
                nn.Dropout(0.3),
                nn.LeakyReLU()
                )
            )
            inChannels = i
            
        
        layers.reverse()
        for i in range(len(layers)-1):
            inChannels = layers[i]
            outChannels = layers[i+1]
            self.dec_modules.append(
                nn.Sequential(
                nn.ConvTranspose2d(in_channels=inChannels, out_channels=outChannels, kernel_size=5),
                nn.BatchNorm2d(outChannels),
                nn.Dropout(0.3),
                nn.ReLU()
                )
            )
        self.dec_modules.append(
                nn.Sequential(
                nn.ConvTranspose2d(in_channels=layers[-1], out_channels=1, kernel_size=5),
                )
            )

    def encode(self, x):
        for i, k in enumerate(self.enc_modules):
            x = k(x)
            if i in self.residu:
                self.residu[i] = x
        return x

    def decode(self, x):
        for i, k in enumerate(self.dec_modules):
            x = k(x)
            if len(self.dec_modules)-i-2 in self.residu:
                x=torch.add(x,self.residu[len(self.dec_modules)-i-2 ])
        return x
    
    def forward(self, x, t):
        if self.addX:
            out = torch.add(self.decode(self.encode(x)),x[:,t,...].unsqueeze(1))
        else:
            out = self.decode(self.encode(x))
        return out