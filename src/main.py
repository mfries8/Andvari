import argparse
import logging
import sys
import os
import multiprocessing as mp
import json

if sys.platform.startswith('linux'):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

from supervisor import Supervisor
from augmenter import train_field_model
from auditor import launch_auditor
from slicer import generate_training_data

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("Andvari.Main")

logger = setup_logging()

def load_config():
    config_path = "andvari_config.json"
    if not os.path.exists(config_path):
        logger.warning(f"Config not found at {config_path}. Using hardcoded fallbacks.")
        return None
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Andvari: Drone-Assisted Meteorite Recovery Pipeline")
    subparsers = parser.add_subparsers(dest="mode", help="Operating Mode", required=True)
    
    slice_parser = subparsers.add_parser("slice", help="Chop calibration images into tiles.")
    slice_parser.add_argument("--input", type=str, required=True)
    slice_parser.add_argument("--output", type=str, required=True)
    slice_parser.add_argument("--tile_size", type=int, default=224)
    slice_parser.add_argument("--annotate", action="store_true")
    
    pipe_parser = subparsers.add_parser("pipeline", help="Run the full search swarm.")
    pipe_parser.add_argument("--input", type=str, required=True)
    pipe_parser.add_argument("--output", type=str, default="./data/output")
    pipe_parser.add_argument("--weights", type=str, required=True)
    
    train_parser = subparsers.add_parser("train", help="Fine-tune the model.")
    train_parser.add_argument("--dataset", type=str, required=True)
    train_parser.add_argument("--base_weights", type=str, required=True)
    train_parser.add_argument("--output_weights", type=str, required=True)
    train_parser.add_argument("--epochs", type=int, default=15)
    
    review_parser = subparsers.add_parser("review", help="Launch the local web server.")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.mode == "slice":
        if args.annotate:
            generate_training_data(args.input, args.output, args.tile_size)
        else:
            logger.error("Standalone raw slicing without annotation is handled automatically by the Supervisor.")
            sys.exit(1)

    elif args.mode == "pipeline":
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        image_files = [f for f in os.listdir(args.input) if f.lower().endswith(valid_exts)]
        total_images = len(image_files)
        
        processed_counter = mp.Value('i', 0)
        logger.info(f"Initializing Andvari Pipeline. Target: {args.input} ({total_images} total images)")
        
        swarm = Supervisor(
            raw_image_dir=args.input, 
            output_dir=args.output,
            weights_path=args.weights,
            total_images=total_images,
            processed_counter=processed_counter,
            config=config
        )
        
        for filename in image_files:
            swarm.raw_image_queue.put(os.path.join(args.input, filename))
            
        for _ in range(mp.cpu_count()):
            swarm.raw_image_queue.put("SHUTDOWN_COMMAND")
            
        swarm.launch()
        
    elif args.mode == "train":
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
