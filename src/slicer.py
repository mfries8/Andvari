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
    Dual-mode Slicer. Provides a UI for positive targets, and
    silently auto-mines background noise from negative images.
    """
    pos_out = os.path.join(output_dir, "positive")
    neg_out = os.path.join(output_dir, "negative")
    os.makedirs(pos_out, exist_ok=True)
    os.makedirs(neg_out, exist_ok=True)

    pos_in = os.path.join(input_dir, "positive_raw")
    neg_in = os.path.join(input_dir, "negative_raw")

    # ==========================================
    # PHASE 1: Targeted Annotation (UI)
    # ==========================================
    if os.path.exists(pos_in):
        logger.info("Found positive_raw folder. Launching UI for targeted annotation.")
        pos_images = glob.glob(os.path.join(pos_in, "*.[jJ][pP][gG]"))
    else:
        logger.warning("No positive_raw folder found. Defaulting to all images in input directory.")
        pos_images = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]"))

    if pos_images:
        logger.info(f"Loading {len(pos_images)} positive images. Click targets, SPACEBAR to advance.")
        for img_path in pos_images:
            img = cv2.imread(img_path)
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            ui_height = 900
            scale_factor = orig_h / ui_height
            ui_width = int(orig_w / scale_factor)

            ui_img = cv2.resize(img, (ui_width, ui_height))
            clone = ui_img.copy()
            targets = []

            def click_event(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    cv2.circle(ui_img, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow("Andvari Annotator", ui_img)
                    targets.append((int(x * scale_factor), int(y * scale_factor)))

            cv2.namedWindow("Andvari Annotator")
            cv2.setMouseCallback("Andvari Annotator", click_event)
            
            while True:
                cv2.imshow("Andvari Annotator", ui_img)
                key = cv2.waitKey(1) & 0xFF
                if key == 32: # SPACEBAR
                    break
                elif key == ord('r'): # Reset drawing
                    ui_img = clone.copy()
                    targets.clear()

            half_tile = tile_size // 2
            for i, (tx, ty) in enumerate(targets):
                y1 = max(0, ty - half_tile)
                y2 = min(orig_h, ty + half_tile)
                x1 = max(0, tx - half_tile)
                x2 = min(orig_w, tx + half_tile)
                
                if y2 - y1 < tile_size: y2 = y1 + tile_size
                if x2 - x1 < tile_size: x2 = x1 + tile_size

                positive_tile = img[y1:y2, x1:x2]
                out_path = os.path.join(pos_out, f"{os.path.basename(img_path)}_pos_{i}.jpg")
                cv2.imwrite(out_path, positive_tile)
                
        cv2.destroyAllWindows()

    # ==========================================
    # PHASE 2: Silent Auto-Mining (No UI)
    # ==========================================
    if os.path.exists(neg_in):
        logger.info("Found negative_raw folder. Silently mining background noise...")
        neg_images = glob.glob(os.path.join(neg_in, "*.[jJ][pP][gG]"))
        
        for img_path in neg_images:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            orig_h, orig_w = img.shape[:2]
            
            # Auto-mine 25 random chops per empty image
            for i in range(25):
                rx = random.randint(0, orig_w - tile_size)
                ry = random.randint(0, orig_h - tile_size)
                
                negative_tile = img[ry:ry+tile_size, rx:rx+tile_size]
                out_path = os.path.join(neg_out, f"{os.path.basename(img_path)}_neg_{i}.jpg")
                cv2.imwrite(out_path, negative_tile)
                
        logger.info(f"Successfully mined negative tiles from {len(neg_images)} images.")
