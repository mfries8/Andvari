import argparse
import logging
import sys
import os
import multiprocessing as mp

# CRITICAL: Forces Linux to use 'spawn' instead of 'fork' to prevent CUDA context deadlocks.
if sys.platform.startswith('linux'):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# Import the orchestrators from our agent swarm
from supervisor import Supervisor
from augmenter import train_field_model
from auditor import launch_auditor
from slicer import generate_training_data  # <-- We import the correct UI Slicer here

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("Andvari.Main")

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(
        description="Andvari: Drone-Assisted Meteorite Recovery Pipeline"
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Operating Mode", required=True)
    
    # Mode 0: The Training Data Prep (Slicer)
    slice_parser = subparsers.add_parser("slice", help="Chop calibration images into tiles for training.")
    slice_parser.add_argument("--input", type=str, required=True, help="Directory containing raw calibration images.")
    slice_parser.add_argument("--output", type=str, required=True, help="Directory to save the chopped tiles.")
    slice_parser.add_argument("--tile_size", type=int, default=224, help="Size of the square tiles (default 224).")
    slice_parser.add_argument("--annotate", action="store_true", help="Launch UI to click meteorites before slicing into positive/negative folders.")
    
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
    train_parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15).")
    
    # Mode 3: The Rapid Review UI
    review_parser = subparsers.add_parser("review", help="Launch the local web server to review candidates.")
    
    args = parser.parse_args()
    
    if args.mode == "slice":
        if not os.path.exists(args.input):
            logger.error(f"Input directory not found: {args.input}")
            sys.exit(1)
            
        if args.annotate:
            # Route to the targeted Slicer that actually centers the rocks
            generate_training_data(args.input, args.output, args.tile_size)
        else:
            logger.error("Standalone raw slicing without annotation is now handled automatically by the Supervisor.")
            sys.exit(1)

    elif args.mode == "pipeline":
        if not os.path.exists(args.input):
            logger.error(f"Input directory not found: {args.input}")
            sys.exit(1)
            
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        image_files = [f for f in os.listdir(args.input) if f.lower().endswith(valid_exts)]
        total_images = len(image_files)
        
        if total_images == 0:
            logger.error(f"No valid images found in {args.input}")
            sys.exit(1)
            
        processed_counter = mp.Value('i', 0)
        logger.info(f"Initializing Andvari Pipeline. Target: {args.input} ({total_images} total images)")
        
        swarm = Supervisor(
            raw_image_dir=args.input, 
            output_dir=args.output,
            weights_path=args.weights,
            total_images=total_images,
            processed_counter=processed_counter
        )
        
        for filename in image_files:
            swarm.raw_image_queue.put(os.path.join(args.input, filename))
            
        for _ in range(mp.cpu_count()):
            swarm.raw_image_queue.put("POISON_PILL")
            
        swarm.launch()
        
    elif args.mode == "train":
        if not os.path.exists(args.dataset):
            logger.error(f"Dataset directory not found: {args.dataset}")
            sys.exit(1)
        train_field_model(
            dataset_dir=args.dataset,
            base_weights_path=args.base_weights,
            output_weights_path=args.output_weights,
            epochs=args.epochs
        )
        
    elif args.mode == "review":
        launch_auditor()

if __name__ == "__main__":
    main()
