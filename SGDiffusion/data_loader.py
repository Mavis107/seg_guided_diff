import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class AMOSDataset(Dataset):
    def __init__(self, images_dir=None, labels_dir=None, target_shape=(128, 128), start_index=500, load_images=True, transform=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.target_shape = target_shape
        self.start_index = start_index
        self.load_images = load_images

        # Define image transformation
        self.transform = transforms.Compose([
            transforms.Resize(target_shape),
            transforms.ToTensor()
        ]) if transform else None

        # Load and filter filenames
        self.image_filenames = sorted([
            f for f in os.listdir(images_dir) if self._filter_filename(f)
        ]) if images_dir and load_images else []

        self.label_filenames = sorted([
            f for f in os.listdir(labels_dir) if self._filter_filename(f)
        ]) if labels_dir else []

        # Ensure dataset sizes match
        if self.load_images and len(self.image_filenames) != len(self.label_filenames):
            min_length = min(len(self.image_filenames), len(self.label_filenames))
            self.image_filenames = self.image_filenames[:min_length]
            self.label_filenames = self.label_filenames[:min_length]

    def _filter_filename(self, filename):
        """
        Filters filenames based on a minimum index.
        Assumes filenames are formatted like 'amos_XXXX_sYYY.png' where XXXX is an index.
        """
        try:
            index = int(filename.split('_')[1])  # Extract index (XXXX part)
            return index <= self.start_index
        except (IndexError, ValueError):
            return False  # Skip files that don't match the expected format

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Returns an image and label pair if the label exists.
        If the label does not exist, it skips the image and returns the next valid pair.
        """
        if idx >= len(self.image_filenames):
            raise IndexError(f"Index {idx} out of range for dataset length {len(self)}.")

        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        label_path = os.path.join(self.labels_dir, self.label_filenames[idx])

        # Check if the label exists
        if not os.path.exists(label_path):
            # If there's no label, skip this image and return the next valid pair
            return self.__getitem__((idx + 1) % len(self))

        image = Image.open(image_path).convert('L')
        label = Image.open(label_path).convert('L')

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        label_name = os.path.basename(self.label_filenames[idx])  # Extract label name

        return image, label, label_name


# # Define data paths
# root_dir = '/vol/aimspace/projects/practical_WS2425/diffusion/data/amos_robert_slices/'
# train_images_dir = os.path.join(root_dir, 'imagesTr')  # Directory for training images
# train_labels_dir = os.path.join(root_dir, 'labelsTr')  # Directory for training labels
# val_images_dir = os.path.join(root_dir, 'imagesVa')    # Directory for validation images
# val_labels_dir = os.path.join(root_dir, 'labelsVa')    # Directory for validation labels

# # Create dataset and dataloaders with transform
# train_dataset = AMOSDataset(train_images_dir, train_labels_dir)
# val_dataset = AMOSDataset(val_images_dir, val_labels_dir)

# # Define dataloaders
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# print(f"Train images directory: {train_images_dir}, contains {len(os.listdir(train_images_dir))} files.")
# print(f"Train labels directory: {train_labels_dir}, contains {len(os.listdir(train_labels_dir))} files.")
# print(f"Validation images directory: {val_images_dir}, contains {len(os.listdir(val_images_dir))} files.")
# print(f"Validation labels directory: {val_labels_dir}, contains {len(os.listdir(val_labels_dir))} files.")
# print(f"The number of train images: {len(train_dataset)}")
# print(f"The number of validation images: {len(val_dataset)}")


