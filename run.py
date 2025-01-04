import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset

from pytorch_lightning.callbacks import EarlyStopping

# Define a function to initialize weights using Xavier initialization
import torch.nn as nn
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight) 

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
model = vae_models[config['model_params']['name']](**config['model_params'])

# Apply the initialization function to all layers in the model
model.apply(init_weights)

experiment = VAEXperiment(model,
                          config['exp_params'])

data = VAEDataset(**config["data_params"], pin_memory=False)

data.setup()
early_stop_callback = EarlyStopping(
    monitor="val_loss",  # Metric to monitor
    min_delta=0.00001,  # Minimum change in the monitored quantity to qualify as an improvement
    patience=3,  # Number of epochs with no improvement after which training will be stopped
    verbose=True, 
    mode="min"  # Direction of improvement ('min' for minimizing the metric, 'max' for maximizing)
)
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     early_stop_callback,
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy="ddp",
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)