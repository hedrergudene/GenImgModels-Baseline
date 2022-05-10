#
# Models
#

# Requirements
import torch

# GAN
class GAN(torch.nn.Module):
    """
    Vanilla GAN model.
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        device:str,
    ):

        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.device = device


    def forward(self, batch: torch.Tensor):
        #
        # Discriminator training
        #

        # Generate random samples in latent space
        batch_size = batch.shape[0]
        random_latent_vectors = torch.randn((batch_size, self.generator.latent_dim)).to(self.device)
        # Encode images
        generated_images = self.generator(random_latent_vectors)["reconstruction"]
        # Combine them with real images
        combined_images = torch.cat([generated_images, batch], axis=0)
        # Discriminator logits
        discr_preds = self.discriminator(combined_images)

        #
        # Generator training
        #

        # Generate random samples in latent space
        random_latent_vectors = torch.randn((batch_size, self.generator.latent_dim)).to(self.device)
        # Generator logits
        gen_preds = self.discriminator(self.generator(random_latent_vectors)["reconstruction"])

        #
        # Output
        #
        
        output = {
            "generated_images":generated_images,
            "gen_preds":gen_preds, # Shape (batch_size, 1)
            "discr_preds":discr_preds, # Shape (2*batch_size, 1)
        }
        return output
