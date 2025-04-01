import os
import re
import numpy as np
import torch.optim as optim
import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage import zoom
from PIL import Image
from tqdm import tqdm



class AMOSDataset(Dataset):
    def __init__(self, images_dir, labels_dir, target_shape=(128, 128), start_index=500, transform=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.target_shape = target_shape
        self.start_index = start_index  # Starting index for loading images
        self.transform = transform
        
        # Filter filenames
        self.image_filenames = sorted(
            [f for f in os.listdir(images_dir) if self._filter_filename(f)]
        )
        self.label_filenames = sorted(
            [f for f in os.listdir(labels_dir) if self._filter_filename(f)]
        )

    def _filter_filename(self, filename):
        """
        Filters filenames and keeps only those with indices >= start_index.
        Assumes the filename format is: "amos_XXXX_sYYY.png" (XXXX is the index)
        """
        # Extract the index part
        try:
            index = int(filename.split('_')[1])  # Extract the XXXX part
            return index <= self.start_index
        except (IndexError, ValueError):
            return False  # Skip files that don't match the format

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        label_path = os.path.join(self.labels_dir, self.label_filenames[idx])
        
        # Load image and label
        image = Image.open(image_path).convert('L')    # Convert to grayscale for image
        label = Image.open(label_path).convert('L')    # Convert to grayscale for label

                     
        # Apply transformation if defined
        # Resize image and label to target size
        if self.transform:
            image = self.transform(image)
        
        # # # Resize label using nearest-neighbor for label preservation
        # label = label.resize(self.target_shape, Image.NEAREST)
        
        # Convert label to tensor (long type for segmentation task)
        label = torch.tensor(np.array(label), dtype=torch.long)

        # Add channel dimension to image (for a grayscale image, it will be (1, H, W))
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        return image, label


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

# # Create dataset and dataloaders with transform
# train_dataset = AMOSDataset(train_images_dir, train_labels_dir, transform=transform)
# val_dataset = AMOSDataset(val_images_dir, val_labels_dir, transform=transform)

# # Define dataloaders
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# print(f"Train images directory: {train_images_dir}, contains {len(os.listdir(train_images_dir))} files.")
# print(f"Train labels directory: {train_labels_dir}, contains {len(os.listdir(train_labels_dir))} files.")
# print(f"Validation images directory: {val_images_dir}, contains {len(os.listdir(val_images_dir))} files.")
# print(f"Validation labels directory: {val_labels_dir}, contains {len(os.listdir(val_labels_dir))} files.")
# print(f"The number of train images: {len(train_dataset)}")
# print(f"The number of validation images: {len(val_dataset)}")