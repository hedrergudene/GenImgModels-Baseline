# Requirements
import torch
import logging as log
import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
# Custom objects
from .loss import KLReconLoss
from .model import VAE
from .layers import Encoder, Decoder
from .fitter import VAEFitter


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
    assert model_dct['encoder_style'] in ['HuggingFace', 'ConvMixer', 'ResNet'], f"Current supported encoder backbones are {', '.join(['HuggingFace', 'ConvMixer', 'ResNet'])}."
    assert model_dct['decoder_style'] in ['ConvMixer', 'ResNet', 'ResNet18', 'ResNet50'], f"Current supported decoder backbones are {', '.join(['ConvMixer', 'ResNet', 'ResNet18', 'ResNet50'])}."

    # Encoder
    encoder = Encoder(model_backbone=model_dct['model_backbone'],
                      style=model_dct['encoder_style'],
                      n_channels=info_dct['n_channels'],
                      latent_dim=train_dct['latent_dim'],
                      dim=model_dct['dim'],
                      depth=model_dct['encoder_depth'],
                      kernel_size=model_dct['kernel_size'],
                      patch_size=model_dct['patch_size'],
                      )
    # Decoder
    decoder = Decoder(style=model_dct['decoder_style'],
                      img_size = info_dct['img_size'],
                      n_channels=info_dct['n_channels'],
                      latent_dim=train_dct['latent_dim'],
                      dim=model_dct['dim'],
                      depth=model_dct['decoder_depth'],
                      kernel_size=model_dct['kernel_size'],
                      patch_size=model_dct['patch_size'],
                      )
    # Model
    model = VAE(
        encoder=encoder,
        decoder=decoder,
    ).to(train_dct['device'])

    # Loss, optimiser and scheduler
    loss_fn = KLReconLoss(reconstruction_loss=train_dct['reconstruction_loss'],
                          reduction=train_dct['reduction_loss'],
                          beta=train_dct['beta_loss'],
                          warmup_epoch=train_dct['warmup_epoch_loss'],
                          C=train_dct['C_loss'],
                          device=train_dct['device'],
                          use_amp=bool(train_dct['use_amp']),
                          )
    optimiser = torch.optim.AdamW(model.parameters(), lr=train_dct['learning_rate'], weight_decay = train_dct['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                           mode=train_dct['scheduler_mode'],
                                                           factor=train_dct['scheduler_factor'],
                                                           patience=train_dct['scheduler_patience'],
                                                           min_lr=1e-6,
                                                           )

    # Training pipeline (put benatools Fitter)
    fitter = VAEFitter(
                 model=model,
                 device=train_dct['device'],
                 loss=loss_fn,
                 optimizer=optimiser,
                 scheduler=scheduler,
                 validation_scheduler=True,
                 step_scheduler=False,
                 folder=train_dct['output_dir'],
                 verbose=True,
                 save_log=True,
                 use_amp=bool(train_dct['use_amp'])
    )

    return fitter
