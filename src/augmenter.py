import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger("Andvari.Augmenter")

def get_field_transforms():
    """
    Standardizes input for ResNet18 and adds 'field' noise 
    to make the model more robust to lighting changes.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_field_model(dataset_dir, base_weights_path, output_weights_path, epochs=10):
    """
    Fine-tunes the ResNet18 weights for the Andvari project.
    """
    logger.info("Augmenter Agent online. Spinning up the training forge...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Data Loading (Defining 'transform' correctly this time)
    transform = get_field_transforms()
    try:
        train_dataset = datasets.ImageFolder(dataset_dir, transform=transform)
        # Using a batch size of 8 to prevent CUDA Out of Memory on RTX 2050
        dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 2. Initialize Model Architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # Binary: Dirt vs Meteorite
    
    # 3. Load Base Weights
    try:
        model.load_state_dict(torch.load(base_weights_path))
        logger.info(f"Successfully loaded base weights: {base_weights_path}")
    except Exception as e:
        logger.error(f"Base weights load failed: {e}")
        return

    # 4. Freeze Early Layers (Feature Extractors)
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
            
    model.to(device)
    model.train()

    # 5. Optimizer & Loss
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 6. Training Loop
    logger.info(f"Training for {epochs} epochs on {len(train_dataset)} field samples.")
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        logger.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # 7. Save Final Weights
    torch.save(model.state_dict(), output_weights_path)
    logger.info(f"Tuned weights saved to {output_weights_path}")
