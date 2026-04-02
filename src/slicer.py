import cv2
import os
import logging
import time
from queue import Empty

logger = logging.getLogger("Andvari.Slicer")

class TelemetryParser:
    """Base class for extracting metadata. To be expanded later."""
    def extract(self, image_path):
        # Stub: Will parse DJI EXIF or SLR flight logs
        return {"lat": 0.0, "lon": 0.0, "alt": 50.0, "pitch": -90.0, "heading": 0.0}

def slicer_worker(raw_queue, tile_queue, tile_size=512, overlap=0.2):
    """
    The worker process. Pulls raw images from the queue, extracts telemetry,
    chops them into tiles, and feeds the GPU queue.
    """
    logger.info("Slicer Agent online. Waiting for raw images...")
    parser = TelemetryParser()
    
    # Calculate stride based on overlap (e.g., 20% overlap means moving 80% of tile size)
    stride = int(tile_size * (1.0 - overlap))
    
    while True:
        try:
            # Timeout allows the process to check for shutdown signals periodically
            image_path = raw_queue.get(timeout=3)
        except Empty:
            continue
            
        if image_path == "POISON_PILL":
            # The signal to die gracefully
            logger.info("Slicer received poison pill. Shutting down.")
            break
            
        logger.info(f"Slicing: {os.path.basename(image_path)}")
        
        # 1. Extract Telemetry
        telemetry = parser.extract(image_path)
        
        # 2. Load Image (cv2 loads as BGR, but we'll let PyTorch handle the RGB flip later)
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load {image_path}. Corrupted file?")
            continue
            
        height, width, _ = img.shape
        
        # 3. Chop into tiles
        tile_count = 0
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                # Slice the numpy array
                tile = img[y:y+tile_size, x:x+tile_size]
                
                # Package the payload
                payload = {
                    "parent_image": image_path,
                    "tile_data": tile,           # The actual numpy array (image data)
                    "offset_x": x,               # Pixel X coordinate of top-left corner
                    "offset_y": y,               # Pixel Y coordinate of top-left corner
                    "telemetry": telemetry       # The parent's GPS data
                }
                
                # Push to the GPU queue. 
                # Note: If the GPU is slow, this will block until there is space, 
                # naturally throttling the CPU so we don't blow up system RAM.
                tile_queue.put(payload)
                tile_count += 1
                
        logger.debug(f"Produced {tile_count} tiles from {os.path.basename(image_path)}")
