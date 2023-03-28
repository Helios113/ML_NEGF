import torch
from torch import nn
from torch.nn import functional as F
from math import floor, pi, log
import matplotlib.pyplot as plt


def conv_out_shape(img_size, k, s, p):
    return [floor((img_size[0] + 2*p - k)/s)+1, floor((img_size[1] + 2*p - k)/s)+1]


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 latent_dim,
                 img_size,
                 kernel,
                 stride,
                 padding):
        super(EncoderBlock, self).__init__()

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())

        out_size = conv_out_shape(img_size, kernel,
                                  stride,
                                  padding)
        self.encoder_mu = nn.Linear(
            out_channels * out_size[0]*out_size[1], latent_dim)
        self.encoder_var = nn.Linear(
            out_channels * out_size[0]*out_size[1], latent_dim)
        # return out_channels, out_size[0], out_size[1]

    def forward(self, input):
        result = self.encoder(input)
        h = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution

        mu = self.encoder_mu(h)
        log_var = self.encoder_var(h)

        return [result, mu, log_var]


class LadderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 latent_dim):
        super(LadderBlock, self).__init__()

        # Build Decoder
        self.decode = nn.Sequential(nn.Linear(in_channels, latent_dim),
                                    nn.BatchNorm1d(latent_dim))
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, z):
        z = self.decode(z)
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)

        return [mu, log_var]


class LVAE(nn.Module):

    def __init__(self,
                 in_channels,
                 latent_dims,
                 hidden_dims,
                 kernel,
                 stride,
                 padding):
        super(LVAE, self).__init__()
        self.out_channels = in_channels
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.num_rungs = len(latent_dims)
        self.after_latent_size = []
        assert len(latent_dims) == len(hidden_dims), "Length of the latent" \
                                                     "and hidden dims must be the same"

        # Build Encoder
        modules = []
        img_size = [71, 26]
        # latent_rec_dims = []
        for i, h_dim in enumerate(hidden_dims):
            latent_rec_dims = modules.append(EncoderBlock(in_channels,
                                        h_dim,
                                        latent_dims[i],
                                        img_size,
                                        kernel,
                                        stride,
                                        padding))

            img_size = conv_out_shape(img_size, kernel,
                                      stride,
                                      padding)
            in_channels = h_dim
        self.after_latent_size = img_size 
        self.encoders = nn.Sequential(*modules)
        # ====================================================================== #
        # Build Decoder
        modules = []

        for i in range(self.num_rungs - 1, 0, -1):
            modules.append(LadderBlock(latent_dims[i],
                                       latent_dims[i-1]))

        self.ladders = nn.Sequential(*modules)

        self.decoder_input = nn.Linear(latent_dims[0], in_channels*img_size[0]*img_size[1])

        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=kernel,
                                       stride=stride,
                                       padding=padding,
                                       output_padding=padding),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               self.out_channels,
                               kernel_size=kernel,
                               stride=stride,
                               padding=padding,
                               output_padding=padding),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU())#,
            # nn.Conv2d(hidden_dims[-1], out_channels=1,
            #           kernel_size=kernel, padding=padding),
            # nn.Tanh())
        hidden_dims.reverse()

    def encode(self, input1):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        h = input1
        # Posterior Parameters
        post_params = []
        for encoder_block in self.encoders:
            h, mu, log_var = encoder_block(h)
            post_params.append((mu, log_var))

        return post_params

    def decode(self, z, post_params):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        kl_div = 0
        post_params.reverse()
        for i, ladder_block in enumerate(self.ladders):
            mu_e, log_var_e = post_params[i]
            mu_t, log_var_t = ladder_block(z)
            mu, log_var = self.merge_gauss(mu_e, mu_t,
                                           log_var_e, log_var_t)
            z = self.reparameterize(mu, log_var)
            kl_div += self.compute_kl_divergence(z,
                                                 (mu, log_var), (mu_e, log_var_e))

        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.after_latent_size[0], self.after_latent_size[1])
        result = self.decoder(result)
        return self.final_layer(result), kl_div

    def merge_gauss(self,
                    mu_1,
                    mu_2,
                    log_var_1,
                    log_var_2):

        p_1 = 1. / (log_var_1.exp())
        p_2 = 1. / (log_var_2.exp())

        mu = (mu_1 * p_1 + mu_2 * p_2)/(p_1 + p_2)
        log_var = torch.log(1./(p_1 + p_2))
        return [mu, log_var]

    def compute_kl_divergence(self, z, q_params, p_params):
        mu_q, log_var_q = q_params
        mu_p, log_var_p = p_params
        #
        # qz = -0.5 * torch.sum(1 + log_var_q + (z - mu_q) ** 2 / (2 * log_var_q.exp() + 1e-8), dim=1)
        # pz = -0.5 * torch.sum(1 + log_var_p + (z - mu_p) ** 2 / (2 * log_var_p.exp() + 1e-8), dim=1)

        kl = (log_var_p - log_var_q) + (log_var_q.exp() +
                                        (mu_q - mu_p)**2)/(2 * log_var_p.exp()) - 0.5
        kl = torch.sum(kl, dim=-1)
        return kl

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

    def forward(self, input):
        post_params = self.encode(input)
        mu, log_var = post_params.pop()
        z = self.reparameterize(mu, log_var)
        recons, kl_div = self.decode(z, post_params)

        #kl_div += -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
        return [recons+input, input, kl_div]

    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        kl_div = args[2]

        # Account for the minibatch samples from the dataset
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(kl_div, dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
               num_samples,
               current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dims[-1])

        # z = z.to(current_device)

        for ladder_block in self.ladders:
            mu, log_var = ladder_block(z)
            z = self.reparameterize(mu, log_var)

        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], 2, 2)
        result = self.decoder(result)
        samples = self.final_layer(result)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


# t = torch.ones(32, 1, 71, 26)
# model = LVAE(1, [16, 16, 16, 16], [8, 16, 32, 64], kernel=3,  padding=0, stride=1)
# a = model(t)
# print(a[0][0].shape)
# plt.imsave("test.png",a[0][0][0].detach().numpy())
# # plt.show()
