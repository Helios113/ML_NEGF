import torch


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1,1,(5,5),1,0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1,1,(6,6),1,0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1,1,(7,7),1,0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1,1,(8,8),1,0),
            
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1,1,(8,8),1,0),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1,1,(7,7),1,0),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1,1,(6,6),1,0),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1,1,(5,5),1,0)
        )
        # Some way of scaling data
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)+x
        return decoded, encoded
    
