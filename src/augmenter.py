import os
import cv2
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
from inquisitor import MeteoriteCNN

logger = logging.getLogger("Andvari.Augmenter")

class FieldDataset(Dataset):
    """
    Custom PyTorch Dataset that loads cropped tiles from disk.
    Expects a directory structure like:
    dataset_dir/
       ├── positive/ (tiles containing proxies)
       └── negative/ (tiles of empty dirt/shadows)
    """
    def __init__(self, root_dir, transform=None):
        self.positive_paths = glob.glob(os.path.join(root_dir, 'positive', '*.*'))
        self.negative_paths = glob.glob(os.path.join(root_dir, 'negative', '*.*'))
        
        # Labels: 1.0 for positive (meteorite), 0.0 for negative (background)
        self.samples = [(p, 1.0) for p in self.positive_paths] + \
                       [(p, 0.0) for p in self.negative_paths]
                       
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load BGR with OpenCV, convert to RGB
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            # transforms expect PIL image or specific tensor format, 
            # our custom transform pipeline handles numpy arrays
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
            
        return img_tensor, torch.tensor([label], dtype=torch.float32)

def get_field_transforms():
    """The augmentation pipeline. Multiplies the value of sparse field data."""
    return transforms.Compose([
        transforms.ToTensor(), # Converts numpy to tensor and scales 0-255 to 0.0-1.0
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # Randomly rotate the image by up to 180 degrees
        transforms.RandomRotation(180),
        # Slightly jitter brightness and contrast to simulate passing clouds
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
    ])

def train_field_model(dataset_dir, base_weights_path, output_weights_path, epochs=10, batch_size=32):
    """
    Fine-tunes the base model on localized field data.
    """
    logger.info("Augmenter Agent online. Spinning up the training forge...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("Training on CPU. This will take a while.")

    # 1. Load the Base Model
    model = MeteoriteCNN()
    if os.path.exists(base_weights_path):
        model.load_state_dict(torch.load(base_weights_path))
        logger.info(f"Loaded base weights from {base_weights_path}")
    else:
        logger.error("Base weights not found! Cannot fine-tune an empty brain.")
        return

    # 2. Freeze the Feature Extractors (Convolutional Blocks)
    # We only want to train the Fully Connected (fc1, fc2, output) layers
    for name, param in model.named_parameters():
        if "conv" in name or "BatchNorm" in name:
            param.requires_grad = False
            
    model.to(device)
    model.train() # Set to training mode so Dropout layers become active

    # 3. Setup DataLoader
    dataset = FieldDataset(root_dir=dataset_dir, transform=get_field_transforms())
    
    # num_workers=4 uses multicore CPU processing to augment images while the GPU trains
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # 4. Optimizer and Loss Function
    # We only pass the parameters that require gradients (the un-frozen ones) to the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    # Binary Cross Entropy Loss is standard for yes/no classification
    criterion = nn.BCELoss()

    # 5. The Training Loop
    logger.info(f"Starting fine-tuning for {epochs} epochs on {len(dataset)} local samples.")
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        logger.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # 6. Save the localized model
    torch.save(model.state_dict(), output_weights_path)
    logger.info(f"Field-tuned weights forged and saved to {output_weights_path}")
