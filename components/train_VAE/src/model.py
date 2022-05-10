#
# Models
#

# Requirements
import torch

# VAE
class VAE(torch.nn.Module):
    """
    Vanilla Variational Autoencoder model.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
    ):

        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, batch: torch.Tensor):
        # Encode images
        encoder_output = self.encoder(batch)
        # Obtain characteristics of latent space
        mu, log_var = encoder_output['embedding'], encoder_output['log_covariance']
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        # Based on Gauss sampling, reconstruct image
        recon_batch = self.decoder(z)["reconstruction"]
        # Format output
        output = {
            "recon_batch":recon_batch,
            "mu":mu,
            "log_var":log_var,
        }
        return output


    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps