# src/model.py

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics import Accuracy

class DRClassifier(pl.LightningModule):
    def __init__(self, num_classes=5, learning_rate=1e-3, unfreeze_base=False):
        super().__init__()
        self.save_hyperparameters() # Saves args to self.hparams

        # Load the pretrained ResNet50 model
        self.model = models.resnet50(weights='IMAGENET1K_V2')

        # Freeze the convolutional base if specified (for Model A)
        if not unfreeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Define loss and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # The optimizer will only update the parameters that have requires_grad=True
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer