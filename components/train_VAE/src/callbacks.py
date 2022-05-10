#
# Callbacks
#

# Requirements
from typing import Dict
import numpy as np
import torch
import torchvision
import wandb

# Weights and Biases callback
def wandb_update(history:Dict,
                 val_loader:torch.utils.data.DataLoader,
                 model:torch.nn.Module,
                 device:str='cuda',
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
    # Rename certain keys
    step = data_log['epoch']
    
    #
    # Part II: Image sample
    #

    # Batch
    random_pick = np.random.randint(low=0, high=val_loader.batch_size)
    # Batch (memory efficient)
    batch = next(iter(val_loader))['x'][random_pick:random_pick+1,:,:,:].to(device).detach()
    # Original image
    original_image = batch.cpu()
    num_channels = original_image.shape[0]
    # Sample image
    with torch.no_grad():
        sample_image = torch.sigmoid(model(batch)["recon_batch"])
    sample_image = sample_image.cpu()
    
    # Update wandb
    wandb.log(data_log, step = step)
    wandb.log({"Sample image and reconstruction": wandb.Image(torchvision.utils.make_grid(torch.cat([original_image, sample_image], dim=0),
                                                                   nrow=2,
                                                                   value_range = (0,1),
                                                                  ),
                                             caption=f"Epoch {step}",
                                            ),
              }, step = step)
