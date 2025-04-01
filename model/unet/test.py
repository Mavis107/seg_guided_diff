import torch
import argparse
import yaml
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
import os
from utils import readable_timestamp  # Assuming you have this utility function

# Dice Coefficient Calculation (for segmentation tasks)
def compute_dice_score(preds, labels):
    smooth = 1e-6  # A small constant to avoid division by zero
    preds = preds.view(-1)
    labels = labels.view(-1)
    intersection = torch.sum(preds * labels)
    union = torch.sum(preds) + torch.sum(labels)
    dice_score = (2 * intersection + smooth) / (union + smooth)
    return dice_score.item()

# Argument parser
parser = argparse.ArgumentParser()

# Hyperparameters
timestamp = readable_timestamp()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--n_classes", type=int, default=15)
parser.add_argument("--dataset", type=str, default='AMOS')  # assuming the test dataset is AMOS

# Model options
parser.add_argument("-save", action="store_true", help="Save the model")
parser.add_argument("--filename", type=str, default=timestamp)

# Paths for data and model
parser.add_argument("--test_images_path", type=str, default = "/vol/aimspace/projects/practical_WS2425/diffusion/data/amos_slices/imagesTs", required=True, help="Path to test images")
parser.add_argument("--test_labels_path", type=str, required=True, help="Path to test labels")
parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")

args = parser.parse_args()

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=1, n_classes=args.n_classes).to(device)
model.load_state_dict(torch.load(args.model_path))
model.eval()

# Prepare the Test Dataset
test_dataset = AMOSDataset(args.test_images_path, args.test_labels_path, transform=None)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Initialize variables for metrics
criterion = nn.CrossEntropyLoss()
test_loss = 0
dice_scores = []

# TensorBoard writer for logging metrics
writer = SummaryWriter(log_dir=f'runs/{args.filename}')

# Run testing
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        # Get model predictions
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Compute Dice coefficient
        preds = torch.argmax(outputs, dim=1)
        dice_score = compute_dice_score(preds, labels)
        dice_scores.append(dice_score)

        # Log batch-wise metrics
        writer.add_scalar('Loss/Test_batch', loss.item(), batch_idx)
        writer.add_scalar('Dice/Test_batch', dice_score, batch_idx)

# Average loss and dice score
avg_test_loss = test_loss / len(test_loader)
avg_dice_score = sum(dice_scores) / len(dice_scores)

print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Average Dice Score: {avg_dice_score:.4f}")

# Log epoch-wise metrics
writer.add_scalar('Loss/Test_epoch', avg_test_loss, 0)
writer.add_scalar('Dice/Test_epoch', avg_dice_score, 0)

# Optionally, visualize some results
if args.save:
    # Visualize a batch of results
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        # Visualize the first image in the batch
        image = images[0].cpu().numpy().squeeze()  # Remove batch dimension
        label = labels[0].cpu().numpy().squeeze()
        pred = preds[0].cpu().numpy().squeeze()

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Input Image")
        axes[1].imshow(label, cmap='jet')
        axes[1].set_title("Ground Truth")
        axes[2].imshow(pred, cmap='jet')
        axes[2].set_title("Predicted Mask")
        plt.show()
        break  # Remove this to visualize multiple images

# Close the writer after testing
writer.close()
