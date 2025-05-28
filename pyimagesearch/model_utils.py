import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image



def VAE_Loss(VAELossParams, kld_weight):
    recons, input, mu, log_var = VAELossParams

    recons_loss = F.mse_loss(recons, input)

    '''
    The KLD loss measures the divergence between the learned latent variable distribution and a standard normal distribution.
    The formula inside the torch.sum computes the KLD for each data point in the batch. The outer torch.mean
    then averages the KLD values over the entire batch.
    '''
    kld_loss = torch.mean(-0.5*torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=1), dim=0)

    loss = recons_loss + kld_loss * kld_weight

    return {
        'loss' : loss,
        'Reconstruction_loss' : recons_loss.detach().item(),
        'KLD' : kld_loss.detach().item()
    }


