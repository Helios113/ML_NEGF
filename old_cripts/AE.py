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
        decoded = self.decoder(encoded)+x
        return decoded


def gram_matrix(input):
    a, b = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a, b)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b)


class StyleLoss(torch.nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, input, target):
        G = gram_matrix(input)
        T = gram_matrix(target)
        return F.l1_loss(G, T)


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.style_loss = StyleLoss()

    def forward(self, input, target):
        return self.style_loss(input, target)+F.l1_loss(input,target)
