"""
test version

"""

import json
import os
import yaml
import torch
import pytorch_lightning as pl
import diffusers
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from torch.nn import functional as F
from diffusers import DDPMScheduler, UNet2DModel
from utils.utils import load_amos



class DiffusionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Define model parameters
        in_channels = 2 if config["training"]["conditional"] else 1
        out_channels = 1
        image_size = config["training"]["target_shape"][0]

        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", timestep_spacing="uniform")

    def forward(self, inputs, timesteps):
        return self.model(inputs, timesteps).sample

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)  # Ensure tensors are on the same device
        noise = torch.randn_like(images).to(self.device)  # Ensure noise tensor is on the correct device
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (images.size(0),), dtype=torch.long, device=self.device)

        noisy_images = self.scheduler.add_noise(images, noise, timesteps)

        # Prepare inputs
        inputs = torch.cat((noisy_images, labels), dim=1) if self.config["training"]["conditional"] else noisy_images
        predicted_noise = self.model(inputs, timesteps).sample

        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        if batch_idx % 100 == 0:
            self.log_images(images, noisy_images, predicted_noise, timesteps, batch_idx)
        return loss

    def log_images(self, original, noisy, predicted_noise, timesteps, batch_idx):
        original, noisy = original.to(self.device), noisy.to(self.device)

        # Reverse denoising
        reconstructed_images = noisy.clone()
        for t in reversed(range(self.scheduler.config.num_train_timesteps)):
            t_tensor = torch.full((original.size(0),), t, dtype=torch.long).to(self.device)
            predicted = self.forward(reconstructed_images, t_tensor)
            alpha_t = self.scheduler.alphas_cumprod[t].to(self.device)
            one_minus_alpha_t = 1 - alpha_t
            reconstructed_images = ((reconstructed_images - torch.sqrt(one_minus_alpha_t) * predicted) / torch.sqrt(alpha_t)).clamp(0, 1)

        # Create grids
        grid_original = make_grid(original, nrow=4, normalize=True)
        grid_noisy = make_grid(noisy, nrow=4, normalize=True)
        grid_reconstructed = make_grid(reconstructed_images, nrow=4, normalize=True)

        # Log to TensorBoard
        self.logger.experiment.add_image("Original Images", grid_original, batch_idx)
        self.logger.experiment.add_image("Noisy Images", grid_noisy, batch_idx)
        self.logger.experiment.add_image("Reconstructed Images", grid_reconstructed, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config["training"]["learning_rate"])


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # Load configuration
    config_path = "config.yaml"
    config = load_config(config_path)

    # Load dataset
    training_data, validation_data, training_loader, validation_loader = load_amos(
        config["dataset"]
    )

    # Initialize logger
    logger = TensorBoardLogger(save_dir=config["logging"]["log_dir"], name=config["logging"]["experiment_name"])

    # Initialize model and trainer
    model = DiffusionModel(config)
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        logger=logger,
        accelerator="gpu",  # Automatically handles multiple GPUs if available
        devices=torch.cuda.device_count(),
    )

    # Train the model
    trainer.fit(model, training_loader)


if __name__ == "__main__":
    main()







