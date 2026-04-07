import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import logging
from queue import Empty

logger = logging.getLogger("Andvari.Inquisitor")

def inquisitor_worker(tile_queue, candidate_queue, weights_path=None, config=None):
logger.info("Inquisitor Agent online. Warming up the GPU...")

batch_size = config["inquisitor"]["batch_size"] if config else 32
threshold = config["inquisitor"]["confidence_threshold"] if config else 0.98

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

if weights_path:
    model.load_state_dict(torch.load(weights_path))
    logger.info(f"Loaded weights from {weights_path}")
    
model.eval()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
poisoned = False

while not poisoned:
    batch_payloads = []
    batch_tensors = []
    
    while len(batch_payloads) < batch_size:
        try:
            payload = tile_queue.get(timeout=0.1)
            
            if payload == "SHUTDOWN_COMMAND":
                logger.info("Inquisitor received shutdown command. Processing final batch...")
                poisoned = True
                break
            
            img_bgr = payload.get("tile_data")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
            tensor = normalize(tensor)
            
            batch_tensors.append(tensor)
            batch_payloads.append(payload)
            
        except Empty:
            if len(batch_payloads) > 0:
                break
            
    if len(batch_tensors) == 0:
        continue
        
    batch_stack = torch.stack(batch_tensors).to(device)
    
    with torch.no_grad():
        logits = model(batch_stack)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predictions = probabilities[:, 1] 
        
    for i, conf_score in enumerate(predictions.cpu().numpy()):
        if conf_score >= threshold:
            hit_payload = batch_payloads[i]
            hit_payload.update({"confidence": float(conf_score)})
            
            candidate_queue.put(hit_payload)
            logger.info(f"Candidate found in {hit_payload.get('parent_image')} at X:{hit_payload.get('offset_x')} Y:{hit_payload.get('offset_y')}! Score: {conf_score:.3f}")

logger.info("Inquisitor shutdown complete.")
