import os
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from ffcv.loader import OrderOption
from lightning.pytorch import (Trainer,
                               seed_everything)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import (LearningRateMonitor,
                                         ModelCheckpoint)

from pydeps.og_proposal.vae import (VAE,
                                    Decoder)
from pydeps.og_proposal.tasks import (SceneEmbedding,
                                      OGDecoder)
from pydeps.og_proposal.dataset import (ogvae_loader,
                                        ogdecoder_loader)

def main():
    parser = argparse.ArgumentParser(
        description='Trains components of data driven proposal',
    )
    parser.add_argument('config',
                        type = str,
                        help = 'path to the config file')

    args = parser.parse_args()
    with open(f"/project/scripts/nn/configs/{args.config}.yaml", 'r') as file:
        config = yaml.safe_load(file)


    logger = CSVLogger(save_dir=config['logging_params']['save_dir'],
                       name=config['mode'])

    # For reproducibility
    seed_everything(config['manual_seed'], True)

    if config['mode'] == 'scene_vae':
        arch = VAE(**config['model_params'])
        print(config)
        task = SceneEmbedding(arch,  **config['exp_params'])
        loader = ogvae_loader
    elif config['mode'] == 'og_decoder':
        vae = VAE(**config['model_params'])
        encoder = SceneEmbedding.load_from_checkpoint(config['vae_chkpt'],
                                                      model = vae)
        decoder = Decoder(**config['model_params'])
        task = OGDecoder(encoder, decoder, **config['exp_params'])
        loader = ogdecoder_loader
    else:
        raise ValueError(f"mode {config['mode']} not recognized")

    
    runner = Trainer(logger=logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=3,
                                         dirpath =os.path.join(logger.log_dir , "checkpoints"),
                                         monitor= "val_loss",
                                         save_last=True,
                                         every_n_epochs=10),
                     ],
                     accelerator = 'auto',
                     deterministic = False,
                     **config['trainer_params'])
    # device = runner.device_ids[0]
    device = torch.device('cuda:0')
    train_loader = loader(config['path_params']['train_path'],
                          device, 
                          **config['loader_params'],
                          order = OrderOption.RANDOM)
    test_loader = loader(config['path_params']['test_path'],
                         device, batch_size = 16)


    Path(f"{logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
    Path(f"{logger.log_dir}/reconstructions").mkdir(exist_ok=True, parents=True)
    print(f"======= Training {logger.name} =======")
    runner.fit(task, train_loader, test_loader)

if __name__ == '__main__':
    main()
