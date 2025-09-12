# src/data_setup.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --- Transformations ---
IMG_SIZE = 224
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

# --- Data Splitting and Caching ---
def get_data_splits(full_df, data_dir):
    """
    Splits the dataframe into train and validation sets.
    Caches the splits as CSV files to avoid re-splitting on subsequent runs.
    """
    train_split_path = os.path.join(data_dir, 'train_split.csv')
    val_split_path = os.path.join(data_dir, 'val_split.csv')

    if os.path.exists(train_split_path) and os.path.exists(val_split_path):
        print("Loading cached data splits...")
        train_df = pd.read_csv(train_split_path)
        val_df = pd.read_csv(val_split_path)
    else:
        print("Creating new data splits...")
        X = full_df['id_code']
        y = full_df['diagnosis']
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        train_df = pd.DataFrame({'id_code': X_train, 'diagnosis': y_train})
        val_df = pd.DataFrame({'id_code': X_val, 'diagnosis': y_val})

        # Cache the splits
        train_df.to_csv(train_split_path, index=False)
        val_df.to_csv(val_split_path, index=False)
        print("Data splits cached successfully.")

    return train_df, val_df

# --- PyTorch Dataset and DataLoader ---
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
        image = Image.open(img_name).convert('RGB')
        label = self.df.iloc[idx]['diagnosis']
        if self.transform:
            image = self.transform(image)
        return image, label

def create_dataloaders(train_df, val_df, image_dir, batch_size, num_workers=4):
    """Creates training and validation dataloaders."""
    train_dataset = DRDataset(df=train_df, image_dir=image_dir, transform=train_transforms)
    val_dataset = DRDataset(df=val_df, image_dir=image_dir, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader