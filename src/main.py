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
    """Worker function to slice a single calibration image and save to disk."""
    image_path, output_dir, tile_size, overlap = args
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to read {filename}. Skipping.")
        return 0
        
    height, width, _ = img.shape
    stride = int(tile_size * (1.0 - overlap))
    
    tiles_saved = 0
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile = img[y:y+tile_size, x:x+tile_size]
            out_path = os.path.join(output_dir, f"{name}_X{x}_Y{y}.jpg")
            cv2.imwrite(out_path, tile)
            tiles_saved += 1
            
    return tiles_saved

def standalone_slice(input_dir, output_dir, tile_size=512, overlap=0.2):
    """Chops directory of high-res images into tiles for training data prep."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Gather all valid images
    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    image_paths = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) 
        if f.lower().endswith(valid_exts)
    ]
    
    if not image_paths:
        logger.error(f"No valid images found in {input_dir}")
        return

    logger.info(f"Found {len(image_paths)} images. Slicing across {mp.cpu_count()} CPU cores...")
    
    # Package arguments for the worker pool
    task_args = [(path, output_dir, tile_size, overlap) for path in image_paths]
    
    total_tiles = 0
    # Use ProcessPoolExecutor to max out the CPU and chop images simultaneously
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        for result in executor.map(process_single_image, task_args):
            total_tiles += result
            
    logger.info(f"Slicing complete! Generated {total_tiles} training tiles in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Andvari: Drone-Assisted Meteorite Recovery Pipeline",
        epilog="Remember to run 'slice' and 'train' on local field data before running 'pipeline'."
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Operating Mode", required=True)
    
    # Mode 0: The Training Data Prep (Slicer)
    slice_parser = subparsers.add_parser("slice", help="Chop calibration images into tiles for training.")
    slice_parser.add_argument("--input", type=str, required=True, help="Directory containing raw calibration images.")
    slice_parser.add_argument("--output", type=str, required=True, help="Directory to save the chopped tiles.")
    slice_parser.add_argument("--tile_size", type=int, default=512, help="Size of the square tiles (default 512).")
    
    # Mode 1: The Main Swarm Pipeline
    pipe_parser = subparsers.add_parser("pipeline", help="Run the full search swarm on raw drone imagery.")
    pipe_parser.add_argument("--input", type=str, required=True, help="Directory containing raw aerial images.")
    pipe_parser.add_argument("--output", type=str, default="./data/output", help="Directory for CSV/KML and thumbnails.")
    pipe_parser.add_argument("--weights", type=str, required=True, help="Path to the trained PyTorch .pth weights file.")
    
    # Mode 2: The Field Augmenter
    train_parser = subparsers.add_parser("train", help="Fine-tune the model using local meteorite proxies.")
    train_parser.add_argument("--dataset", type=str, required=True, help="Directory containing 'positive' and 'negative' subfolders.")
    train_parser.add_argument("--base_weights", type=str, required=True, help="Path to the lab-trained base weights.")
    train_parser.add_argument("--output_weights", type=str, required=True, help="Path to save the new field-tuned weights.")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10).")
    
    # Mode 3: The Rapid Review UI
    review_parser = subparsers.add_parser("review", help="Launch the local web server to review candidates.")
    
    args = parser.parse_args()
    
    if args.mode == "slice":
        if not os.path.exists(args.input):
            logger.error(f"Input directory not found: {args.input}")
            sys.exit(1)
            
        logger.info(f"Initializing Standalone Slicer. Target: {args.input}")
        standalone_slice(args.input, args.output, args.tile_size)
        logger.info("Now manually sort the output tiles into your positive/ and negative/ folders.")

    elif args.mode == "pipeline":
        if not os.path.exists(args.input):
            logger.error(f"Input directory not found: {args.input}")
            sys.exit(1)
            
        logger.info(f"Initializing Andvari Pipeline. Target: {args.input}")
        swarm = Supervisor(raw_image_dir=args.input, output_dir=args.output)
        
        # Populate the queue
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                swarm.raw_image_queue.put(os.path.join(args.input, filename))
                
        # Poison pills for shutdown
        for _ in range(mp.cpu_count()):
            swarm.raw_image_queue.put("POISON_PILL")
            
        swarm.launch()
        
    elif args.mode == "train":
        if not os.path.exists(args.dataset):
            logger.error(f"Dataset directory not found: {args.dataset}")
            sys.exit(1)
            
        logger.info(f"Igniting the Augmenter Forge. Dataset: {args.dataset}")
        train_field_model(
            dataset_dir=args.dataset,
            base_weights_path=args.base_weights,
            output_weights_path=args.output_weights,
            epochs=args.epochs
        )
        
    elif args.mode == "review":
        logger.info("Spinning up the Auditor UI...")
        launch_auditor()

if __name__ == "__main__":
    main()
