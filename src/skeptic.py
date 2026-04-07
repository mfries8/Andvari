import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import logging
from queue import Empty
from collections import defaultdict

logger = logging.getLogger("Andvari.Skeptic")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def preprocess_for_inference(img_bgr):
    """Converts OpenCV BGR image into a normalized PyTorch tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    tensor = normalize(tensor) # CRITICAL: Added missing normalization
    return tensor.unsqueeze(0)

def skeptic_worker(candidate_queue, verified_queue, weights_path=None, threshold=0.85, density_limit=10):
    """Takes initial hits, checks density caps, rotates images, and re-evaluates."""
    logger.info("Skeptic Agent online. Ready to crush some dreams.")
    
    device = torch.device("cpu")
    
    # Instantiate the new ResNet18 architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    
    model.eval()
    density_tracker = defaultdict(int)
    
    while True:
        try:
            payload = candidate_queue.get(timeout=2)
        except Empty:
            continue
            
        if payload == "POISON_PILL":
            logger.info("Skeptic received poison pill. Shutting down.")
            break
            
        parent_img = payload.get("parent_image")
        original_confidence = payload.get("confidence")
        tile_bgr = payload.get("tile_data")
        
        density_tracker[parent_img] += 1
        if density_tracker[parent_img] > density_limit:
            logger.warning(f"Density cap exceeded for {parent_img}. Discarding candidate.")
            continue
            
        rot_90 = cv2.rotate(tile_bgr, cv2.ROTATE_90_CLOCKWISE)
        rot_180 = cv2.rotate(tile_bgr, cv2.ROTATE_180)
        rot_270 = cv2.rotate(tile_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        tensors = [
            preprocess_for_inference(rot_90),
            preprocess_for_inference(rot_180),
            preprocess_for_inference(rot_270)
        ]
        
        batch = torch.cat(tensors).to(device)
        
        with torch.no_grad():
            logits = model(batch)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predictions = probabilities[:, 1].numpy()
            
        average_confidence = (original_confidence + np.sum(predictions)) / 4.0
        
        if average_confidence >= threshold:
            logger.info(f"Candidate verified! {parent_img} survived the Skeptic with average score: {average_confidence:.3f}")
            payload["confidence"] = float(average_confidence)
            verified_queue.put(payload)
        else:
            logger.debug(f"Candidate rejected. Rotation average dropped to {average_confidence:.3f}")
