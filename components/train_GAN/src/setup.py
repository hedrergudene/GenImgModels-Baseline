# Requirements
import torch
import logging as log
import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
# Custom objects
from .model import GAN
from .layers import Generator, Discriminator
from .fitter import GANFitter


# Method to set up any Kaggle dataset previously configured
def setup_kaggle_data(data_flag:str,
                      data_config_path:str='input/info.json',
                      ):
    # Pick up info dict
    with open(data_config_path, 'r') as f:
        info_dct = json.load(f)[data_flag]
    # Kaggle API
    log.info(f"Authenticating via Kaggle API")
    api = KaggleApi()
    api.authenticate()
    log.info(f"Downloading files")
    api.dataset_download_files(f"{os.environ['KAGGLE_USERNAME']}/{data_flag}", path="input/")
    # Unzip file
    log.info(f"Unzip files")
    with zipfile.ZipFile(f"input/{data_flag}.zip", 'r') as zipObj:
        zipObj.extractall("input/")
    os.remove(f"input/{data_flag}.zip")



# Method to set up model
def setup_model(data_flag:str,
                data_config_path:str='input/info.json',
                model_config_path:str='input/model_config.json',
                train_config_path:str='input/training_config.json',
                ):
    # Pick up data info dict
    with open(data_config_path, 'r') as f:
        INFO = json.load(f)
        info_dct = INFO[data_flag]
    # Pick up model info dict
    with open(model_config_path, 'r') as f:
        model_dct = json.load(f)
    # Pick up training info dict
    with open(train_config_path, 'r') as f:
        train_dct = json.load(f)
    # Validate parameters
    assert model_dct['generator_style'] in ['ConvMixer', 'ResNet', 'ResNet18', 'ResNet50'], f"Current supported generator backbones are {', '.join(['ConvMixer', 'ResNet', 'ResNet18', 'ResNet50'])}."
    assert model_dct['discriminator_style'] in ['ConvMixer', 'ResNet', 'HuggingFace'], f"Current supported generator backbones are {', '.join(['ConvMixer', 'ResNet', 'HuggingFace'])}."

    # Encoder
    generator = Generator(
                        style=model_dct['generator_style'],
                        img_size=info_dct['img_size'],
                        n_channels=info_dct['n_channels'],
                        latent_dim=train_dct['latent_dim'],
                        dim=model_dct['dim'],
                        depth=model_dct['generator_depth'],
                        kernel_size=model_dct['kernel_size'],
                        patch_size=model_dct['patch_size'],
                    )
    # Decoder
    discriminator = Discriminator(
                        model_backbone=model_dct['model_backbone'],
                        style=model_dct['discriminator_style'],
                        img_size=info_dct['img_size'],
                        n_channels=info_dct['n_channels'],
                        latent_dim=train_dct['latent_dim'],
                        dim=model_dct['dim'],
                        depth=model_dct['discriminator_depth'],
                        kernel_size=model_dct['kernel_size'],
                        patch_size=model_dct['patch_size'],
                    )
    # Model
    model = GAN(
        generator=generator,
        discriminator=discriminator,
        device=train_dct['device'],
    ).to(train_dct['device'])

    # Loss, optimiser and scheduler
    gen_optimiser = torch.optim.AdamW(model.generator.parameters(), lr=train_dct['learning_rate'], weight_decay = train_dct['weight_decay'])
    discr_optimiser = torch.optim.AdamW(model.discriminator.parameters(), lr=train_dct['learning_rate'], weight_decay = train_dct['weight_decay'])
    gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gen_optimiser,
                                                               mode=train_dct['scheduler_mode'],
                                                               factor=train_dct['scheduler_factor'],
                                                               patience=train_dct['scheduler_patience'],
                                                               min_lr=1e-6,
                                                               )
    discr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(discr_optimiser,
                                                                 mode=train_dct['scheduler_mode'],
                                                                 factor=train_dct['scheduler_factor'],
                                                                 patience=train_dct['scheduler_patience'],
                                                                 min_lr=1e-6,
                                                                 )

    # Training pipeline (put benatools Fitter)
    fitter = GANFitter(
                 model=model,
                 device=train_dct['device'],
                 gen_optimizer=gen_optimiser,
                 gen_scheduler=gen_scheduler,
                 discr_optimizer=discr_optimiser,
                 discr_scheduler=discr_scheduler,
                 validation_scheduler=True,
                 step_scheduler=False,
                 folder=train_dct['output_dir'],
                 verbose=True,
                 save_log=True,
                 use_amp=bool(train_dct['use_amp'])
    )

    return fitter
