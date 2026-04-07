import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import logging
from queue import Empty
import time

logger = logging.getLogger("Andvari.Inquisitor")

def inquisitor_worker(tile_queue, candidate_queue, weights_path=None, batch_size=32, threshold=0.98):
    """
    The GPU worker process. Dynamically batches incoming tiles, runs inference,
    and forwards highly confident hits to the Skeptic.
    """
    logger.info("Inquisitor Agent online. Warming up the GPU...")
    
    # Force the device to CUDA if available, otherwise it will agonizingly run on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA unavailable! Inference will run on CPU. Bring a sleeping bag.")
        
    # --- NEW MODEL INSTANTIATION (ResNet18) ---
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 outputs: [Dirt, Meteorite]
    model = model.to(device)
    # ------------------------------------------
    
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
        logger.info(f"Loaded weights from {weights_path}")
    else:
        logger.warning("No weights provided! Operating with randomized untrained brain.")
        
    model.eval() # Set to evaluation mode to lock Dropout and BatchNorm layers
    
    while True:
        batch_payloads = list()
        batch_tensors = list()
        
        # Dynamic batching loop
        while len(batch_payloads) < batch_size:
            try:
                # 0.1s timeout ensures we don't hang if the queue is trickling at the end of a flight
                payload = tile_queue.get(timeout=0.1)
                
                if payload == "POISON_PILL":
                    logger.info("Inquisitor received poison pill. Shutting down.")
                    return
                
                # Extract image and convert OpenCV BGR to PyTorch RGB format
                img_bgr = payload.get("tile_data")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Convert to tensor, permute to Channel-Height-Width, normalize to 0-1 range
                tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
                
                batch_tensors.append(tensor)
                batch_payloads.append(payload)
                
            except Empty:
                # If we have tiles waiting but the queue is empty, break out and process what we have
                if len(batch_payloads) > 0:
                    break
                
        # If the outer loop triggered but we have nothing, go back to waiting
        if len(batch_tensors) == 0:
            continue
            
        # Stack the individual tensors into a single massive batched tensor block
        batch_stack = torch.stack(batch_tensors).to(device)
        
        # --- NEW INFERENCE LOGIC ---
        # Run the inference without tracking gradients to save VRAM and speed things up
        with torch.no_grad():
            logits = model(batch_stack)
            # Convert raw logits to probabilities (0.0 to 1.0)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            # Extract the probability for Class 1 (Meteorite)
            predictions = probabilities[:, 1] 
        # ---------------------------
            
        # Send successful candidates to the Skeptic Agent
        for i, conf_score in enumerate(predictions.cpu().numpy()):
            if conf_score >= threshold:
                hit_payload = batch_payloads[i]
                hit_payload.update({"confidence": float(conf_score)})
                
                candidate_queue.put(hit_payload)
                logger.info(f"Candidate found in {hit_payload.get('parent_image')} at X:{hit_payload.get('offset_x')} Y:{hit_payload.get('offset_y')}! Score: {conf_score:.3f}")
