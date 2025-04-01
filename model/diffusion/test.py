import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from data_loader import AMOSDataset
import diffusers
from diffusers import DDIMScheduler, DDPMScheduler, DiffusionPipeline, UNet2DModel
from tqdm import tqdm
import torchvision.utils as vutils
import argparse
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description="AMOS Dataset Diffusion Model Testing")
    parser.add_argument("--checkpoint_path", type=str, default="/vol/miltank/projects/practical_WS2425/diffusion/code/SGDiffusion/diff_epoch_50_condition_forward_resume/amos_diffusion_20250202_162858/model_epoch_50.pth", help="Path to the trained model checkpoint")
    parser.add_argument("--output_image_dir", type=str, default="/vol/miltank/projects/practical_WS2425/diffusion/code/evaluation/input/diff_epoch_50_condition_forward/128/timesteps_500_ddim_trailing/generated", help="Directory for saving test results")
    parser.add_argument("--output_mask_dir", type=str, default="/vol/miltank/projects/practical_WS2425/diffusion/code/evaluation/input/diff_epoch_50_condition_forward/128/timesteps_500_ddim_trailing/masks", help="Directory for saving test results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--target_shape", type=tuple, default=(128, 128), help="Target image shape")
    parser.add_argument("--num_inference_timesteps", type=int, default=500, help="Number of inference timesteps")
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddim",help="Scheduler type for reverse diffusion")
    return parser.parse_args()


def load_model(checkpoint_path, in_channels, out_channels, image_size, device):
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels, # image + segmentation mask
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def main():
    args = parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration
    in_channels = 2
    out_channels = 1
    image_size = args.target_shape[0]  # Assuming square images

    model = load_model(args.checkpoint_path, in_channels, out_channels, image_size, device)

    # Scheduler configuration
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



    print("Starting testing...")

    # DataLoader for labels only
    def collate_fn(batch):
    # Filter out None items
        batch = [item for item in batch if item is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    test_labels_dir = '/vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices/labelsVa'
    test_images_dir = '/vol/miltank/projects/practical_WS2425/diffusion/data/amos_robert_slices/imagesVa'
    test_dataset = AMOSDataset(test_images_dir, test_labels_dir, args.target_shape)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    os.makedirs(args.output_image_dir, exist_ok=True)
    os.makedirs(args.output_mask_dir, exist_ok=True)

    generated_count = 0
    max_results = 200  # Limit to 200 generated images
    
    # Generate process
    for batch_idx, (images, labels, label_name) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Generating Images and Masks"):
        labels = labels.to(device)
        
        # Start with random noise
        batch_size = labels.size(0)
        generated_images = torch.randn((batch_size, 1, args.target_shape[0], args.target_shape[1]), device=device)
        
        # Reverse diffusion process
        # Generate image
        with torch.no_grad():
            for t in reversed(range(args.num_inference_timesteps)):
                t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
                inputs = torch.cat((generated_images, labels), dim=1) 
                noise_prediction = model(inputs, t_tensor).sample
                
                step_result = backward_scheduler.step(noise_prediction, t, generated_images)
                generated_images = step_result.prev_sample

            generated_masks = torch.randn((batch_size, 1, args.target_shape[0], args.target_shape[1]), device=device)

        # Generate mask
        generated_masks = torch.randn((batch_size, 1, args.target_shape[0], args.target_shape[1]), device=device)

        with torch.no_grad():
            for t in reversed(range(args.num_inference_timesteps)):
                t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
                inputs = torch.cat((generated_masks, labels), dim=1)  
                noise_prediction = model(inputs, t_tensor).sample
                step_result = backward_scheduler.step(noise_prediction, t, generated_masks)
                generated_masks = step_result.prev_sample

        # Binary segmentation mask
        binary_mask = (generated_masks > 0.5).float()


        # save image & segmentation mask
        for img, mask, label_name in zip(generated_images.clamp(0, 1), binary_mask.clamp(0, 1), label_name):
            if generated_count >= max_results:
                break
            
            label_name = os.path.splitext(label_name)[0]  # remove .png
            
            save_image(img, os.path.join(args.output_image_dir, f"{label_name}.png"))
            save_image(mask, os.path.join(args.output_mask_dir, f"{label_name}.png"))

            generated_count += 1
            if generated_count >= max_results:
                break


if __name__ == "__main__":
    main()