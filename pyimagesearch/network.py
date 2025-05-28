import torch
import torch.nn as nn
import torch.optim as optim

from typing import List


# define a convolutional block for the encoder part of the vae
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    

# define a transposed convolutional block for the decoder part of the vae
class ConvTBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvTBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    

class CelebVAE(nn.Module):
    def __init__(self, in_channels: int, latent_dims: int, hidden_dims: List | None = None) -> None:
        super(CelebVAE, self).__init__()

        self.latent_dim = latent_dims # dimensionality of the hidden space
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512] # default hidden dimensions

        # Build the encoder using the convolutional blocks
        blocks = [ConvBlock(in_f, out_f) for in_f, out_f in zip([in_channels] + hidden_dims[:-1], hidden_dims)]
        self.encoder = nn.Sequential(*blocks)

        # fully connected layer for the mean of the latent space
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dims)

        # fully connected layer for the variance of the latent space
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dims)

        self.decoder_input = nn.Linear(latent_dims, hidden_dims[-1]*4)
        hidden_dims.reverse()

        Tblocks = [ConvTBlock(in_f, out_f) for in_f, out_f in zip(hidden_dims[:-1], hidden_dims[1:])]
        self.decoder = nn.Sequential(*Tblocks)
        
        # final layer to reconstruct the original input
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[-1], out_channels=hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            # final convolution to match the output channels
            nn.Conv2d(in_channels=hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # encoding function to map the input to the latent space
    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        result = self.encoder(x)

        # flatten the result for the fully connected layers
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)        

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Expand the latent space
        result = self.decoder_input(z)

        # reshape the result for the transposed convolutions
        '''
        The -1 in the first dimension means that this dimension is inferred from the length of the tensor, ensuring that the total number of elements remains the same.
        The reshaped tensor has dimensions of [batch size, 512 channels, height of 2, width of 2].
        This reshaping is typically done to prepare the data for transposed convolutional layers, which expect multi-dimensional input.
        '''
        result = result.view(-1, 512, 2, 2)

        result = self.decoder(result)

        '''
        This step is often used to ensure the output has the desired shape or characteristics, such as applying a tanh
        activation function to ensure pixel values are between -1 and 1 for image data.
        '''
        result = self.final_layer(result)
        return result
    
        '''
        In summary, the decode method takes points from the latent space and maps them back to the original data space,
        producing a reconstruction of the original input. This is a crucial part of the autoencoder's architecture,
        where the goal is to compress data into a latent space and then reconstruct it with minimal loss of information.
        '''
    
    # reparameterization trick to sample from the latent space
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # compute the standard deviation from the log variance
        '''
        The transformation involves taking the exponential of half the log variance. This is because the variance is the square of the standard deviation,
        and by taking the exponential of half the log variance, we effectively compute the square root of the variance.
        '''
        std = torch.exp(0.5*logvar)

        # sample random noise
        eps = torch.randn_like(std)

        # Compute the sample from the laten space
        '''
        The final step is to compute the sample from the latent space. This is done by multiplying the random noise (eps)
        with the standard deviation (std) and then adding the mean (mu)
        This ensures that the sample is drawn from a distribution with the given mean and variance.
        '''
        return eps * std + mu
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # encode the input to the latent space
        '''
        This method maps the input data to the latent space and returns the mean (mu) and log variance (log_var) of the latent space distribution.
        '''
        mu, log_var = self.encode(x)
        '''
        Using the mean and log variance obtained from the encoding step, the reparameterize method is called to sample a point (z) from the latent space
        This method uses the reparameterization trick to make the sampling process differentiable, which is crucial for training the VAE using gradient descent.
        '''
        z = self.reparameterize(mu, log_var)
        # decode the sample, and return the reconstruction along with the original input, mean, and log variance
        return [self.decode(z), x, mu, log_var]
        '''
        In summary, the forward method of the VAE takes input data, encodes it to a latent space,
        samples a point from this latent space using the reparameterization trick, and then decodes this point to produce a reconstruction of the original input.
        The method returns the reconstructed data along with the original input, mean, and log variance.
        This information is essential for computing the loss during training,
        which typically consists of a reconstruction loss and a regularization term based on the mean and log variance.
        '''