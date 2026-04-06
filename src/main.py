import argparse
import logging
import sys
import os
import cv2
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# CRITICAL: This must happen before any PyTorch modules are heavily loaded.
# Forces Linux to use 'spawn' instead of 'fork' to prevent CUDA context deadlocks.
if sys.platform.startswith('linux'):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# Import the orchestrators from our agent swarm
from supervisor import Supervisor
from augmenter import train_field_model
from auditor import launch_auditor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("Andvari.Main")

logger = setup_logging()

def process_single_image(args):
    """Worker function to slice a single calibration image and triage suspects."""
    image_path, output_dir, tile_size, overlap, triage_mode = args
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to read {filename}. Skipping.")
        return 0
        
    height, width, _ = img.shape
    stride = int(tile_size * (1.0 - overlap))
    
    # Ensure directories exist
    pos_dir = os.path.join(output_dir, "positive")
    neg_dir = os.path.join(output_dir, "negative")
    suspect_dir = os.path.join(output_dir, "suspects")
    
    if triage_mode:
        os.makedirs(neg_dir, exist_ok=True)
        os.makedirs(suspect_dir, exist_ok=True)
        os.makedirs(pos_dir, exist_ok=True)
    
    tiles_saved = 0
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile = img[y:y+tile_size, x:x+tile_size]
            
            target_dir = output_dir 
            
            if triage_mode:
                gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                # Count very dark pixels
                dark_pixel_count = cv2.countNonZero((gray < 30).astype('uint8'))
                
                # If there are ANY dark pixels (even just 10), flag it as a suspect for human review.
                # Otherwise, banish it directly to the negative folder.
                if dark_pixel_count > 10:
                    target_dir = suspect_dir
                else:
                    target_dir = neg_dir
                    
            out_path = os.path.join(target_dir, f"{name}_X{x}_Y{y}.jpg")
            cv2.imwrite(out_path, tile)
            tiles_saved += 1
            
    return tiles_saved

def standalone_slice(input_dir, output_dir, tile_size=512, overlap=0.2, triage_mode=False):
    """Chops directory of high-res images into tiles for training data prep."""
    os.makedirs(output_dir, exist_ok=True)
    
    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    image_paths = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) 
        if f.lower().endswith(valid_exts)
    ]
    
    if not image_paths:
        logger.error(f"No valid images found in {input_dir}")
        return

    logger.info(f"Found {len(image_paths)} images. Slicing across {mp.cpu_count()} CPU cores...")
    logger.info(f"Triage Mode is {'ON' if triage_mode else 'OFF'}.")
    
    task_args = [(path, output_dir, tile_size, overlap, triage_mode) for path in image_paths]
    
    total_tiles = 0
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        for result in executor.map(process_single_image, task_args):
            total_tiles += result
            
    logger.info(f"Slicing complete! Generated {total_tiles} training tiles.")

def main():
    parser = argparse.ArgumentParser(
        description="Andvari: Drone-Assisted Meteorite Recovery Pipeline",
        epilog="Remember to run 'slice' and 'train' on local field data before running 'pipeline'."
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Operating Mode", required=True)
    
    # Mode 0: The Training Data Prep (Slicer)
    slice_parser = subparsers.add_parser("slice", help="Chop calibration images into tiles for training.")
    slice_parser.add_argument("--input", type=str, required=True, help="Directory containing raw
