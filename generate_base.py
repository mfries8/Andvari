import torch
import torchvision.models as models
import os

def create_foundational_brain():
    print("Igniting the Forge...")
    
    # 1. Create the models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)
    save_path = "./models/base.pth"

    # 2. Download a world-class, pre-trained CNN (ResNet18)
    # This model has already spent weeks on a supercomputer learning to see basic shapes and textures.
    print("Downloading pre-trained ImageNet weights (ResNet18)...")
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 3. Perform the Lobotomy
    # The default model is trained to recognize 1,000 different things (cats, cars, coffee mugs).
    # We rip out that final layer and replace it with a simple 2-category output: [Dirt, Meteorite]
    print("Modifying final classification layer for binary output...")
    num_ftrs = base_model.fc.in_features
    base_model.fc = torch.nn.Linear(num_ftrs, 2)

    # 4. Save the modified brain to disk
    print(f"Saving foundational weights to {save_path}...")
    torch.save(base_model.state_dict(), save_path)
    
    print("\nSUCCESS: base.pth has been forged!")
    print("You may now run the field Augmenter to teach it what your proxies look like.")

if __name__ == "__main__":
    create_foundational_brain()
