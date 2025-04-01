import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from PIL import Image
import diffusers
from diffusers import DDIMScheduler, DDPMScheduler, DiffusionPipeline, UNet2DModel
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from data_loader import AMOSDataset
from tqdm import tqdm
import torchvision.utils as vutils
import json


def parse_args():
    parser = argparse.ArgumentParser(description="AMOS Dataset Diffusion Model Training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and validation")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--target_shape", type=tuple, default=(128, 128), help="Target image shape")
    parser.add_argument("--conditional", type=bool, default=True, help="Use conditional input (image + label)")
    parser.add_argument("--forward", type=bool, default=False, help="Use conditional input (image + label) in forward pass")
    parser.add_argument("--log_dir", type=str, default="diff_epoch_50_condition_v2_500_no_clamp_resume_v2", help="Directory for TensorBoard logs")
    parser.add_argument("--scheduler", type=str, choices=["ddpm", "ddim"], default="ddim", help="Choose the scheduler: 'ddpm' or 'ddim'")
    parser.add_argument("--num_inference_timesteps", type=int, default=500, help="Number of inference timesteps")
    parser.add_argument("--checkpoint_path", type=str, default="/vol/miltank/projects/practical_WS2425/diffusion/code/SGDiffusion/diff_epoch_50_condition_v2_500_no_clamp_resume/amos_diffusion_20250122_134637/model_epoch_20.pth", help="Path to a model checkpoint to continue training")
    return parser.parse_args()

# Parse arguments and print them
args = parse_args()
print(f"Batch size: {args.batch_size}")
print(f"Epochs: {args.epochs}")
print(f"Learning rate: {args.lr}")
print(f"Target image shape: {args.target_shape}")
print(f"Conditional input: {args.conditional}")
print(f"Conditional in forward pass: {args.forward}")
print(f"Scheduler: {args.scheduler}")
print(f"Number of inference timesteps: {args.num_inference_timesteps}")
print(f"Log directory: {args.log_dir}")


def main():
    args = parse_args()
    
    # Dataset directories
    root_dir = '/vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices'
    train_images_dir = os.path.join(root_dir, 'imagesTr')  # Directory for training images
    train_labels_dir = os.path.join(root_dir, 'labelsTr')  # Directory for training labels
    val_images_dir = os.path.join(root_dir, 'imagesVa')    # Directory for validation images
    val_labels_dir = os.path.join(root_dir, 'labelsVa')    # Directory for validation labels


    def collate_fn(batch):
    # Filter out None items
        batch = [item for item in batch if item is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    # Create datasets and dataloaders
    train_dataset = AMOSDataset(train_images_dir, train_labels_dir, args.target_shape)
    val_dataset = AMOSDataset(val_images_dir, val_labels_dir, args.target_shape)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model, scheduler, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model parameters (assuming 1 channel for grayscale images)
    in_channels = 2 if args.conditional else 1  # 1 for image, 2 for image + label
    out_channels = 1  # Or the number of classes if segmentation
    image_size = args.target_shape[0]  # Assuming square images
    
    # Using SGD model
    model = diffusers.UNet2DModel(
        sample_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device) 

    # Load checkpoint if provided
    if args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            print(f"Loading model from checkpoint: {args.checkpoint_path}")
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        else:
            print(f"Checkpoint path {args.checkpoint_path} does not exist.")   

    # # Path to the model directory containing the checkpoint and config
    # model_dir = '/vol/aimspace/projects/practical_WS2425/diffusion/code/SGDiffusion/pretrain_model/'

    # # Load the config file
    # # config = json.load(open(os.path.join(model_dir, "config.json")))

    # # Load the model using the checkpoint
    # model = UNet2DModel.from_pretrained(model_dir).to(device)

    # Initialize the forward (DDPM) and backward (DDIM or DDPM) schedulers
    if args.scheduler == "ddpm":
        forward_scheduler = diffusers.DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            timestep_spacing="linspace"
        )
        backward_scheduler = diffusers.DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            timestep_spacing="linspace"
        )
    else:
        forward_scheduler = diffusers.DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            timestep_spacing="linspace"
        )
        backward_scheduler = diffusers.DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            timestep_spacing="trailing",
            clip_sample=True
        )

    backward_scheduler.set_timesteps(args.num_inference_timesteps)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Set up TensorBoard
    log_dir = os.path.join(args.log_dir, f"amos_diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir)

    # Training loop
    global_step = 0  # Track the number of training steps
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0  # Track loss for the current epoch

        for batch_idx, (images, labels, label_name) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs} - Training"):
            images, labels = images.to(device), labels.to(device)

            if args.conditional:
                # Creating a second channel with constant values (e.g., zeros or ones)
                if args.forward :
                    # Concatenate images with labels
                    inputs = torch.cat((images, labels), dim=1)
                else :
                    constant_channel = torch.zeros_like(images)
                    inputs = torch.cat((images, constant_channel), dim=1)
            else:
                inputs = images

            # Add noise to the input images using the scheduler
            noise = torch.randn_like(images)  # Random noise
            timesteps = torch.randint(0, forward_scheduler.config.num_train_timesteps, (images.size(0),), device=device, dtype=torch.long)
            noisy_images = forward_scheduler.add_noise(images, noise, timesteps)

            # # Clamp noisy images between 0 and 1 to avoid out-of-bound values
            # noisy_images = noisy_images.clamp(0, 1)

            # Pass concatenated tensor with 2 channels if conditional
            if args.conditional:
                if args.forward:
                    # Concatenate images with labels
                    inputs = torch.cat((noisy_images, labels), dim=1)
                else: 
                    # Creating a second channel with constant values (e.g., zeros or ones)
                    constant_channel = torch.zeros_like(images) 
                    inputs = torch.cat((noisy_images, constant_channel), dim=1)
              
            else:
                inputs = noisy_images

            # Predict noise using the model
            predicted_noise = model(inputs, timesteps).sample

            # Compute the loss (Mean Squared Error)
            loss = F.mse_loss(predicted_noise, noise)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), global_step)

            global_step += 1

            # Visualization every 10000 batches
            if batch_idx % 1000 == 0:
                with torch.no_grad():
                    current_images = noisy_images.clone()
                    batch_size = images.size(0)

                    # Reverse denoising process
                    for t in reversed(range(args.num_inference_timesteps)):
                        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
                        
                        if args.conditional:
                            conditional_input = torch.cat((current_images, labels), dim=1)
                            noise_prediction = model(conditional_input, t_tensor).sample
                        else:
                            noise_prediction = model(current_images, t_tensor).sample
                        
                        # # Clamp the noise prediction to avoid out-of-bound values during the backward pass
                        # noise_prediction = noise_prediction.clamp(0, 1)

                        # Update current images using the scheduler
                        step_result = backward_scheduler.step(noise_prediction, t, current_images)
                        current_images = step_result.prev_sample

                        # # Clamp the denoised images between 0 and 1
                        # current_images = current_images.clamp(0, 1)


                    # Prepare images for TensorBoard
                    original_images_display = images.clamp(0, 1)
                    # noisy_images_display = noisy_images.clamp(0, 1)
                    denoised_image_display = current_images.clamp(0, 1)

                    # Create grids of images
                    grid_original = vutils.make_grid(original_images_display, nrow=8, normalize=True, scale_each=True)
                    # grid_noisy = vutils.make_grid(noisy_images_display, nrow=8, normalize=True, scale_each=True)
                    grid_denoised = vutils.make_grid(denoised_image_display, nrow=8, normalize=True, scale_each=True)

                    # Add the images to TensorBoard
                    writer.add_image("Original Image", grid_original, global_step)
                    # writer.add_image("Noisy Image", grid_noisy, global_step)
                    writer.add_image("Denoised Image", grid_denoised, global_step)

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}")

        # Save model checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(log_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()

