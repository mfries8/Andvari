import cv2
import os
import glob
import random
import logging
from queue import Empty

logger = logging.getLogger("Andvari.Slicer")

class TelemetryParser:
    """Base class for extracting metadata. To be expanded later."""
    def extract(self, image_path):
        # Stub: Will parse DJI EXIF or SLR flight logs
        return {"lat": 0.0, "lon": 0.0, "alt": 50.0, "pitch": -90.0, "heading": 0.0}

def slicer_worker(raw_queue, tile_queue, tile_size=224, overlap=0.2):
    """
    The inference worker process. Pulls raw images from the queue, 
    chops them into tiles, and feeds the GPU queue.
    """
    logger.info("Slicer Agent online. Waiting for raw images...")
    parser = TelemetryParser()
    stride = int(tile_size * (1.0 - overlap))
    
    while True:
        try:
            image_path = raw_queue.get(timeout=3)
        except Empty:
            continue
            
        if image_path == "POISON_PILL":
            logger.info("Slicer received poison pill. Shutting down.")
            break
            
        # 1. Extract Telemetry
        telemetry = parser.extract(image_path)
        
        # 2. Load Image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load {image_path}. Corrupted file?")
            continue
            
        height, width, _ = img.shape
        
        # 3. Chop into tiles
        tile_count = 0
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                tile = img[y:y+tile_size, x:x+tile_size]
                payload = {
                    "parent_image": os.path.basename(image_path),
                    "tile_data": tile,           
                    "offset_x": x,               
                    "offset_y": y,               
                    "telemetry": telemetry       
                }
                tile_queue.put(payload)
                tile_count += 1

def generate_training_data(input_dir, output_dir, tile_size=224):
    """
    Interactive Slicer UI. Corrects monitor scaling math to ensure
    clicks accurately extract native 224x224 tiles from 44MP images.
    """
    pos_dir = os.path.join(output_dir, "positive")
    neg_dir = os.path.join(output_dir, "negative")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]"))
    if not image_paths:
        logger.error(f"No JPG images found in {input_dir}")
        return

    logger.info("Launching Annotation UI. Click the targets. Press SPACEBAR to advance.")

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]

        # 1. Scale down for a standard 1080p monitor UI
        ui_height = 900
        scale_factor = orig_h / ui_height
        ui_width = int(orig_w / scale_factor)

        ui_img = cv2.resize(img, (ui_width, ui_height))
        clone = ui_img.copy()
        targets = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Visual feedback on the UI
                cv2.circle(ui_img, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Andvari Annotator", ui_img)
                
                # Reverse the scaling math to get the TRUE 44MP pixel coordinates
                true_x = int(x * scale_factor)
                true_y = int(y * scale_factor)
                targets.append((true_x, true_y))

        cv2.namedWindow("Andvari Annotator")
        cv2.setMouseCallback("Andvari Annotator", click_event)
        
        while True:
            cv2.imshow("Andvari Annotator", ui_img)
            key = cv2.waitKey(1) & 0xFF
            
            # Press SPACEBAR to save targets and load next image
            if key == 32:
                break
            # Press 'r' to clear mistakes on the current image
            elif key == ord('r'):
                ui_img = clone.copy()
                targets.clear()

        # 2. Extract perfectly centered 224x224 Positive Tiles
        half_tile = tile_size // 2
        for i, (tx, ty) in enumerate(targets):
            # Strict bounds checking
            y1 = max(0, ty - half_tile)
            y2 = min(orig_h, ty + half_tile)
            x1 = max(0, tx - half_tile)
            x2 = min(orig_w, tx + half_tile)
            
            # Force exact 224x224 dimension if clicked near an edge
            if y2 - y1 < tile_size: y2 = y1 + tile_size
            if x2 - x1 < tile_size: x2 = x1 + tile_size

            positive_tile = img[y1:y2, x1:x2]
            out_path = os.path.join(pos_dir, f"{os.path.basename(img_path)}_pos_{i}.jpg")
            cv2.imwrite(out_path, positive_tile)

        # 3. Auto-Mine Random Negatives (Balance the dataset 5:1)
        num_negatives = max(10, len(targets) * 5)
        for i in range(num_negatives):
            rx = random.randint(0, orig_w - tile_size)
            ry = random.randint(0, orig_h - tile_size)
            
            # Prevent random negative box from accidentally swallowing a rock
            overlap = False
            for (tx, ty) in targets:
                if abs(rx - tx) < tile_size and abs(ry - ty) < tile_size:
                    overlap = True
                    break
            
            if not overlap:
                negative_tile = img[ry:ry+tile_size, rx:rx+tile_size]
                out_path = os.path.join(neg_dir, f"{os.path.basename(img_path)}_neg_{i}.jpg")
                cv2.imwrite(out_path, negative_tile)

        cv2.destroyAllWindows()
