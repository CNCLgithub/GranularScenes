import torch
from torch import nn
import torch.nn.functional as F
from . pytypes import *

from torchvision.transforms import Resize

# from : https://gist.githubusercontent.com/hunter-heidenreich/9512636394a23721452046039dd52d90/raw/490eee83bdf5ba7f87c99246ca310c698845f8b8/vae.py
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the hidden layer.
        latent_dim (int): Dimensionality of the latent space.
    """

    def __init__(self, input_dim = 1, hidden_dim = 16, latent_dim = 32):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            # PrintLayer(),
            nn.Conv2d(input_dim, hidden_dim, 5, 2, 2),
            # PrintLayer(),
            nn.SiLU(),  # Swish activation function
            nn.Conv2d(hidden_dim, hidden_dim // 2, 5, 2, 2),
            # PrintLayer(),
            nn.SiLU(),  # Swish activation function
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 5, 2, 2),
            # PrintLayer(),
            nn.SiLU(),  # Swish activation function
            nn.Flatten(),
            # PrintLayer(),
            nn.Linear((hidden_dim // 4) * 256, 2 * latent_dim),# 2 for mean and variance.
            # PrintLayer(),
        )

        # used for logvar
        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            # PrintLayer(),
            nn.Linear(latent_dim, (hidden_dim // 4) * 256),
            # PrintLayer(),
            nn.Unflatten(1,  (hidden_dim // 4, 16, 16)),
            # PrintLayer(),
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 2, 5, 2, 2,
                               output_padding = 1),
            # PrintLayer(),
            nn.SiLU(),  # Swish activation function
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim, 5, 2, 2,
                               output_padding = 1),
            nn.SiLU(),  # Swish activation function
            nn.ConvTranspose2d(hidden_dim, 1, 5, 2, 2,
                               output_padding = 1),
            nn.Sigmoid(),  # Swish activation function
        )

    def encode(self, x):
        """
        Encodes the input data into the latent space.

        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability.

        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + 1e-8
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.

        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.

        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()

    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.

        Args:
            z (torch.Tensor): Data in the latent space.

        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return self.decoder(z) #.repeat(1, self.input_dim, 1, 1)

    def forward(self, x : Tensor):
        """
        Performs a forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.

        Returns:
            VAEOutput: VAE output dataclass.
        """
        if x.shape[1] == 3:
            x = x[:, 0:1, :, :]
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)

        return (dist, z, recon_x)

    def sample(self, n : int, device):
        z = torch.randn(n, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        (_, _, recon) = self.forward(x)
        return recon

class Decoder(nn.Module):

    def __init__(self, input_dim = 1, hidden_dim = 16, latent_dim = 32):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim

        self.decoder = nn.Sequential(
            # PrintLayer(),
            nn.Linear(latent_dim, (hidden_dim // 4) * 256),
            # PrintLayer(),
            nn.Unflatten(1,  (hidden_dim // 4, 16, 16)),
            # PrintLayer(),
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 2, 5, 2, 2,
                               output_padding = 1),
            # PrintLayer(),
            nn.SiLU(),  # Swish activation function
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim, 5, 2, 2,
                               output_padding = 1),
            # PrintLayer(),
            nn.SiLU(),  # Swish activation function
            nn.ConvTranspose2d(hidden_dim, 1, 5, 2, 2,
                               output_padding = 1),
            # PrintLayer(),
            nn.Conv2d(1, 1, 12, 8, 2),
            # PrintLayer(),
            nn.Sigmoid(),
        )


    def forward(self, mu: Tensor, **kwargs) -> Tensor:
        return self.decoder(mu).squeeze(1)


    def generate(self, z: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(z)[0]

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decoder(z)
        return samples

class PrintLayer(nn.Module):
    def forward(self, x:Tensor):
        print(x.shape)
        return x
