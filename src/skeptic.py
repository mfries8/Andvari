import cv2
import torch
import numpy as np
import logging
from queue import Empty
from collections import defaultdict

# We import the model architecture from the inquisitor script
from inquisitor import MeteoriteCNN

logger = logging.getLogger("Andvari.Skeptic")

def preprocess_for_inference(img_bgr):
    """Converts a standard OpenCV BGR image into a normalized PyTorch tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0) # Add a batch dimension of 1

def skeptic_worker(candidate_queue, verified_queue, weights_path=None, threshold=0.85, density_limit=10):
    """
    The CPU worker process. Takes initial hits, checks the density cap,
    rotates the image 3 times, and re-evaluates.
    """
    logger.info("Skeptic Agent online. Ready to crush some dreams.")
    
    # We strictly use the CPU here to avoid CUDA context clashes with the Inquisitor
    # and because inferring 3 images is trivial for a modern CPU.
    device = torch.device("cpu")
    model = MeteoriteCNN().to(device)
    
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    
    model.eval()
    
    # Tracks how many candidates have come from a specific parent image
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
        
        # 1. The Density Cap Check
        density_tracker[parent_img] += 1
        if density_tracker[parent_img] > density_limit:
            # If an image has 11 meteorites in it, it's a false positive cluster (e.g., a shadow field)
            logger.warning(f"Density cap exceeded for {parent_img}. Discarding candidate.")
            continue
            
        # 2. The Rotation Filter
        # Generate the 90, 180, and 270 degree rotations
        rot_90 = cv2.rotate(tile_bgr, cv2.ROTATE_90_CLOCKWISE)
        rot_180 = cv2.rotate(tile_bgr, cv2.ROTATE_180)
        rot_270 = cv2.rotate(tile_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Preprocess all three
        tensors = [
            preprocess_for_inference(rot_90),
            preprocess_for_inference(rot_180),
            preprocess_for_inference(rot_270)
        ]
        
        # Stack into a mini-batch of 3 and run inference
        batch = torch.cat(tensors).to(device)
        
        with torch.no_grad():
            predictions = model(batch).squeeze(1).numpy()
            
        # 3. The Final Verdict
        # Average the confidence of the original orientation plus the three rotations
        average_confidence = (original_confidence + np.sum(predictions)) / 4.0
        
        if average_confidence >= threshold:
            logger.info(f"Candidate verified! {parent_img} survived the Skeptic with average score: {average_confidence:.3f}")
            
            # Update the payload with the final score and push to the Cartographer
            payload["confidence"] = float(average_confidence)
            verified_queue.put(payload)
        else:
            logger.debug(f"Candidate rejected. Rotation average dropped to {average_confidence:.3f}")
