# Requierments
import logging as log
import json
import os
import sys
import torch
import wandb
import fire

# Dependencies
from src.dataset import VAEDataset
from src.metrics import SSIM
from src.callbacks import wandb_update

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# Main method. Fire automatically allign method arguments with parse commands from console
def main(
        data_flag:str,
        data_config:str="input/info.json",
        kaggle_config:str="input/kaggle_config.json",
        model_config:str="input/model_config.json",
        training_config:str="input/training_config.json",
        wandb_config:str="input/wandb_config.json",
        ):

    #
    # Part I: Data Gathering
    #

    # Info dict
    with open(data_config, 'r') as f:
        info_dct = json.load(f)[data_flag]
    if not os.path.isdir(f"input/{info_dct['python_class']}"):
        # Read Kaggle dict
        with open(kaggle_config, 'r') as f:
            kaggle_dct = json.load(f)
        # Kaggle Credentials
        os.environ['KAGGLE_USERNAME'] = kaggle_dct['KAGGLE_USERNAME']
        os.environ['KAGGLE_KEY'] = kaggle_dct['KAGGLE_KEY']
        # Import setup method now that credentials are environment variables
        from src.setup import setup_kaggle_data
        setup_kaggle_data(data_flag, data_config)
    # Validations
    log.info(f"Run sanity check on dataset:")
    assert info_dct['python_class'] in os.listdir("input/"), f"Folder containing dataset is not in input directory."
    assert all(item in os.listdir(os.path.join('input',info_dct['python_class'])) for item in ['train', 'val']), f"Folders train and val not found inside {os.path.join('input',info_dct['python_class'])}"
    assert all(item in os.listdir(os.path.join('input',info_dct['python_class'],'train')) for item in info_dct['label'].values()), f"Train folder does not contain images of all classes defined in documentation. Missing ones are: {[elem for elem in os.listdir(os.path.join('input',info_dct['python_class'],'train')) if elem not in info_dct['label'].values()]}"
    assert all(item in os.listdir(os.path.join('input',info_dct['python_class'],'val')) for item in info_dct['label'].values()), f"Validation folder does not contain images of all classes defined in documentation. Missing ones are: {[elem for elem in os.listdir(os.path.join('input',info_dct['python_class'],'val')) if elem not in info_dct['label'].values()]}"

    #
    # Part II: Model Training
    #

    # Training dictionary
    with open(training_config, 'r') as f:
        train_dct = json.load(f)
    # Get data
    log.info(f"Prepare DataLoaders:")
    train_dts = VAEDataset(data_config_path=data_config,
                         data_flag=data_flag,
                         folder='train',
                        )
    train_dtl = torch.utils.data.DataLoader(train_dts, batch_size = train_dct['batch_size'], num_workers = train_dct['num_workers_dataloader'], shuffle=True)
    val_dts = VAEDataset(data_config_path=data_config,
                         data_flag=data_flag,
                         folder='val',
                         transforms=None,
                        )
    val_dtl = torch.utils.data.DataLoader(val_dts, batch_size = 2*train_dct['batch_size'], num_workers = train_dct['num_workers_dataloader'], shuffle=False)
    # Get metric
    log.info(f"Initialise SSIM metric:")
    ssim_metric = SSIM(data_range=1.)
    # Define pipeline
    log.info(f"Running setup_model method:")
    from src.setup import setup_model
    fitter = setup_model(data_flag, data_config, model_config, training_config)
    # Weights and Biases login
    with open(wandb_config, 'r') as f:
        wandb_dct = json.load(f)
    wandb.login(key=wandb_dct['WB_KEY'])
    wandb.init(project=wandb_dct['WB_PROJECT']+'_VAE', entity=wandb_dct['WB_ENTITY'], group = data_flag, config=train_dct)
    # Fitter
    log.info(f"Start fitter training:")
    _ = fitter.fit(train_loader = train_dtl,
                   val_loader = val_dtl,
                   n_epochs = train_dct['epochs'],
                   metrics = [ssim_metric],
                   early_stopping = train_dct['early_stopping'],
                   early_stopping_mode = train_dct['scheduler_mode'],
                   verbose_steps = 10,
                   callbacks = [wandb_update],
                   )
    # Move best checkpoint to Weights and Biases root directory to be saved
    log.info(f"Move best checkpoint to Weights and Biases root directory to be saved:")
    os.replace(f"{train_dct['output_dir']}/best-checkpoint.bin", f"{wandb.run.dir}/best-checkpoint.bin")
    # Finish W&B session
    log.info(f"Finish W&B session:")
    wandb.finish()

if __name__=="__main__":
    fire.Fire(main)
