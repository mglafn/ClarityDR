import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Define transformations globally so they can be reused if needed
IMG_SIZE = 224
# Normalization values are standard for models pre-trained on ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    normalize
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

class DRDataset(Dataset):
    """Custom PyTorch Dataset for the APTOS 2019 DR dataset."""
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.df.iloc[idx]['id_code'] + '.png')
        # PIL Image is the standard for torchvision transforms
        image = Image.open(img_name).convert('RGB')
        label = self.df.iloc[idx]['diagnosis']

        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataloaders(train_df, val_df, image_dir, batch_size, num_workers=4):
    """
    Creates training and validation dataloaders.

    Args:
      train_df: DataFrame with training data (id_code, diagnosis).
      val_df: DataFrame with validation data (id_code, diagnosis).
      image_dir: Path to the directory containing images.
      batch_size: The batch size for the dataloaders.
      num_workers: Number of subprocesses to use for data loading.

    Returns:
      A tuple of (train_loader, val_loader).
    """
    train_dataset = DRDataset(df=train_df, image_dir=image_dir, transform=train_transforms)
    val_dataset = DRDataset(df=val_df, image_dir=image_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader