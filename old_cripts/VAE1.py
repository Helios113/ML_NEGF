import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class AE(nn.Module):
    def __init__(self, in_channels: int, hidden_dims: list, latent_dim: int):
        super().__init__()


        self.input_channels = in_channels
        modules = []
        self.h_dims = hidden_dims
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                            kernel_size=5, stride=1, padding=0),
                    nn.ReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        
        # This need to give the same size as the whole convnet
        self.res1 = nn.Conv2d(self.input_channels, out_channels=hidden_dims[-1],
                            kernel_size=5*(len(hidden_dims)), stride=1, padding=0)
        
        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=5,
                                       stride=1,
                                       padding=0,
                                       output_padding=0),
                    nn.ReLU())
            )
        modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[-1],
                                       1,
                                       kernel_size=5,
                                       stride=1,
                                       padding=0,
                                       output_padding=0)
                    # nn.Conv2d(hidden_dims[-1], out_channels=hidden_dims[-1],
                    #         kernel_size=5, stride=1, padding=1))
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        self.res2 = nn.ConvTranspose2d(hidden_dims[0], out_channels=self.input_channels,
                            kernel_size=5*(len(hidden_dims)-1), stride=1, padding=0)
        
        self.fc_mu = nn.Conv2d(hidden_dims[0],
                                       hidden_dims[0],
                                       kernel_size=1)
        self.fc_var = nn.Conv2d(hidden_dims[0],
                                       hidden_dims[0],
                                       kernel_size=1)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        # print(self.encoder(input).shape)
        # print(self.res1(input).shape)

        result = self.encoder(input)#+self.res1(input)
      

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)#+self.res2(z)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        
        batches = log_var.shape[0]
        
        log_var=log_var.reshape(batches,-1)
        mu=mu.reshape(batches,-1)
        

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    # def sample(self,
    #            num_samples:int,
    #            current_device: int, **kwargs):
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     # z = torch.randn(num_samples,
    #                     # self.latent_dim)

    #     # z = z.to(current_device)

    #     samples = self.decode(z)
    #     return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]



# t = torch.ones((32,1,71,26))
# model = AE(1,[32,64,128,258,256],32)
# a = model(t)[0]
# print(a[0][0].shape)
# plt.imsave("test.png",a[0][0].detach().numpy())
# plt.show()