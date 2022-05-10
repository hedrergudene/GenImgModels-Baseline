#
# Callbacks
#

# Requirements
from typing import Dict
import numpy as np
import torch
import wandb

# Weights and Biases callback
def wandb_update(history:Dict,
                 model:torch.nn.Module,
                 ):
    """This method logs the gathered loss and metrics in training epoch into Weights and Biases.

    Args:
        history (Dict): Dictionary with the following keys:
            * 'epoch': The current epoch.
            * 'train': Average train loss through batches in that epoch.
            * 'val': Average validation loss through batches in that epoch.
            * 'lr': Learning rate used for the optimiser in that epoch.
            * Optional metrics.
        val_loader (torch.utils.data.DataLoader): DataLoader to extract sample image and evaluate reconstruction
    """
    #
    # Part I: Log history
    #
    
    # Copy input to make changes
    data_log = history.copy()
    
    #
    # Part II: Image sample
    #

    # Latent vectors
    latent_vectors = torch.randn((1, model.generator.latent_dim))
    # Sample image
    with torch.no_grad():
        sample_img = model.generator(latent_vectors)['reconstruction']
        sample_img = (torch.squeeze(sample_img).cpu().numpy()*255.).astype('int')
        sample_img = sample_img.transpose((1,2,0))
    
    # Update wandb
    wandb.log(data_log, step = history['epoch'])
    wandb.log({"Original images": wandb.Image(sample_img, caption=f"Original image on epoch {history['epoch']}")}, step = history['epoch'])
