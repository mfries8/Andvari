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
    The augmentation pipeline. Matches ResNet18 requirements 
    and simulates field conditions (lighting/rotation).
    """
    return transforms.Compose([
        transforms.Resize((224, 224)), # Standard ResNet input size
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_field_model(dataset_dir, base_weights_path, output_weights_path, epochs=10):
    """
    Fine-tunes the ResNet18 base model on localized field data.
    """
    logger.info("Augmenter Agent online. Spinning up the training forge...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("Training on CPU. This will take a while.")

    # 1. Setup Data Loading
    transform = get_field_transforms()
    try:
        train_dataset = datasets.ImageFolder(dataset_dir, transform=transform)
        # batch_size=8 is a safe middle ground for an RTX 2050
        dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_dir}: {e}")
        return

    # 2. Load and Modify the Model (ResNet18)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # [Dirt, Meteorite]
    
    try:
        model.load_state_dict(torch.load(base_weights_path))
        logger.info(f"Base weights loaded from {base_weights_path}")
    except Exception as e:
        logger.error(f"Could not load base weights: {e}")
        return

    # 3. Freeze the early layers
    # We keep the "visual" layers frozen and only train the final decision layers
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
            
    model.to(device)
    model.train()

    # 4. Optimizer and Loss Function
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 5. The Training Loop
    logger.info(f"Starting fine-tuning for {epochs} epochs on {len(train_dataset)} samples.")
    
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

    # 6. Save the localized model
    torch.save(model.state_dict(), output_weights_path)
    logger.info(f"Field-tuned weights forged and saved to {output_weights_path}")
