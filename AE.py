import torch
import torch.nn.functional as F


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(26 * 71, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 265),
            torch.nn.Dropout(),
            torch.nn.Tanh(),
            torch.nn.Linear(265, 128),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 26 * 71)
        )
        self.short_circuit = torch.nn.Sequential(
            torch.nn.Linear(71*26, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 71*26)


        )
        # Some way of scaling data

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
