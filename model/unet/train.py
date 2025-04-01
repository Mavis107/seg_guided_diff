import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet_model import UNet
from data_preprocessing import AMOSDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision import transforms


# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Define data paths
root_dir = '/vol/aimspace/projects/practical_WS2425/diffusion/data/amos_slices/'
train_images_dir = os.path.join(root_dir, 'imagesTr')  # Directory for training images
train_labels_dir = os.path.join(root_dir, 'labelsTr')  # Directory for training labels
val_images_dir = os.path.join(root_dir, 'imagesVa')    # Directory for validation images
val_labels_dir = os.path.join(root_dir, 'labelsVa')    # Directory for validation labels

# Create dataset and dataloaders with transform
train_dataset = AMOSDataset(train_images_dir, train_labels_dir, transform=transform)
val_dataset = AMOSDataset(val_images_dir, val_labels_dir, transform=transform)

# Define dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, loss, optimizer
model = UNet(n_channels=1, n_classes=15).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# TensorBoard writer
writer = SummaryWriter(log_dir='../runs/unet_training/logs')

# Directory to save checkpoints
checkpoint_dir = '../runs/unet_training/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

num_epochs = 2
step = 0

def log_images(epoch, loader, phase="Validation"):
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1, keepdim=True)  # Get predicted classes

            # Make grids for images, labels, and predictions
            image_grid = make_grid(images, nrow=4, normalize=True)
            label_grid = make_grid(labels.unsqueeze(1).float(), nrow=4, normalize=True)  # Convert labels to float
            output_grid = make_grid(outputs.float(), nrow=4, normalize=True)  # Convert predictions to float

            # Log grids to TensorBoard
            writer.add_image(f"{phase}/Images", image_grid, epoch)
            writer.add_image(f"{phase}/Labels", label_grid, epoch)
            writer.add_image(f"{phase}/Predictions", output_grid, epoch)
            break  # Log only the first batch


for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            step += 1
            writer.add_scalar("Loss/Train_batch", loss.item(), step)
            pbar.set_postfix({"Batch Loss": loss.item()})
            pbar.update(1)

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}")
    writer.add_scalar("Loss/Train_epoch", avg_train_loss, epoch)

    # Log training images
    log_images(epoch, train_loader, phase="Training")

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"unet_epoch_{epoch+1}.pth"))

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

    # Log validation images
    log_images(epoch, val_loader, phase="Validation")

# Save final model
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "unet_model.pth"))
writer.close()



