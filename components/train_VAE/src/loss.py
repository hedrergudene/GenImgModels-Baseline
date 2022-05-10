#
# Loss
#

# Requirements
import torch
import torchvision
import logging as log


# Auxiliary class for perceptual loss
class VGGPerceptLoss(torch.nn.Module):
    """VGG/Perceptual Loss
    
    Parameters
    ----------
    conv_index : str
        Convolutional layer in VGG model to use as perceptual output

    """
    def __init__(self, device:str, use_amp:bool, conv_index: str = '22'):

        super(VGGPerceptLoss, self).__init__()
        vgg_features = torchvision.models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        
        if conv_index == '22':
            self.vgg = torch.nn.Sequential(*modules[:8]).to(device)
        elif conv_index == '54':
            self.vgg = torch.nn.Sequential(*modules[:35]).to(device)

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229, 0.224, 0.225)
        self.prep = torchvision.transforms.Normalize(vgg_mean, vgg_std)
        self.vgg.requires_grad = False

        self.use_amp = use_amp

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Compute VGG/Perceptual loss between Super-Resolved and High-Resolution

        Parameters
        ----------
        sr : torch.Tensor
            Super-Resolved model output tensor
        hr : torch.Tensor
            High-Resolution image tensor

        Returns
        -------
        loss : torch.Tensor
            Perceptual VGG loss between sr and hr

        """
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            vgg_sr = self.vgg(self.prep(sr))
            vgg_hr = self.vgg(self.prep(hr.detach()))

        loss = torch.mean(torch.nn.functional.mse_loss(vgg_sr, vgg_hr, reduction = 'none'), dim=(-3,-2,-1))

        return loss


# Main class
class KLReconLoss(torch.nn.Module):
    def __init__(self,
                 reconstruction_loss:str="perceptual",
                 reduction:str="mean",
                 beta:float=1.,
                 warmup_epoch:int=1,
                 C:float=0,
                 device:str='cuda',
                 use_amp:bool=False,
                 ):
        super(KLReconLoss, self).__init__()
        # Assert
        assert reconstruction_loss in ["pixelwise", "perceptual"], f"Reconstruction loss should be either 'pixelwise' or 'perceptual' (currently reconstruction_loss={reconstruction_loss})."
        assert reduction in ["mean","sum"], f"Reduction parameter should be either 'mean' or 'sum' (currently reduction={reduction})."
        assert beta>=1, f"Beta value should be greater than or equal to one (currently beta={beta})."
        assert warmup_epoch>0, f"Warmup epoch value should be greater than zero (currently warmup_epoch={warmup_epoch})."
        assert C>=0, f"Temperature value should be greater than or equal to zero (currently C={C})."
        # Print model configuration
        if (beta>1) & (C>0):
            log.info(f"Using Disentangled BetaVAE loss function with warmup_epoch={warmup_epoch}, C={C} and beta={beta}.")
        if (beta==1) & (C>0):
            log.info(f"Using Disentangled VAE loss function with warmup_epoch={warmup_epoch} and C={C}.") 
        if (beta>1) & (C==0):
            log.info(f"Using BetaVAE loss function with beta={beta}.")   
        if (beta==1) & (C==0):
            log.info(f"Using standard VAE loss function.")
        # Layers
        if reconstruction_loss=='perceptual':
            self.vgg_extractor = VGGPerceptLoss(device = device, use_amp = use_amp)
        # Parameters
        self.reconstruction_loss = reconstruction_loss
        self.reduction=reduction
        self.beta = beta
        self.warmup_epoch = warmup_epoch
        self.C = C


    def forward(self,
                output:torch.Tensor,
                batch:torch.Tensor,
                epoch:int=1,
                ):
        # Parameters
        recon_batch = output['recon_batch']
        mu = output['mu']
        log_var = output['log_var']
        # Compute reconstruction loss to measure discrepancies in image reconstruction quality
        if self.reconstruction_loss=='pixelwise':
            recon_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                recon_batch.reshape(recon_batch.shape[0], -1),
                batch.reshape(batch.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
        else:
            recon_loss = self.vgg_extractor(recon_batch, batch)
        # Compute KL divergence to measure discrepancies in latent space distributions
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        C_factor = min(epoch / (self.warmup_epoch + 1), 1)
        KLD_diff = torch.abs(KLD - self.C * C_factor)
        if self.reduction=="mean":
            return {'KLD':KLD.mean(axis=-1), 'recon':recon_loss.mean(axis=-1), 'summary':(recon_loss + self.beta*KLD_diff).mean(axis=-1)}
        elif self.reduction=="sum":
            return {'KLD':KLD.sum(axis=-1), 'recon':recon_loss.sum(axis=-1), 'summary':(recon_loss + self.beta*KLD_diff).sum(axis=-1)}