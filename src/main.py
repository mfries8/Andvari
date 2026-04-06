import argparse
import logging
import sys
import os
import multiprocessing as mp

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

def main():
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Andvari: Drone-Assisted Meteorite Recovery Pipeline",
        epilog="Remember to run the 'train' mode on local field data before running the 'pipeline'."
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Operating Mode", required=True)
    
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
    # The auditor script currently hardcodes paths, but in a future refactor, 
    # we would pass the output directory argument to it here.
    
    args = parser.parse_args()
    
    if args.mode == "pipeline":
        if not os.path.exists(args.input):
            logger.error(f"Input directory not found: {args.input}")
            sys.exit(1)
            
        logger.info(f"Initializing Andvari Pipeline. Target: {args.input}")
        
        # We instantiate the Supervisor, but note that in a fully integrated script,
        # we would also pass the weights path down to the Inquisitor/Skeptic processes.
        swarm = Supervisor(raw_image_dir=args.input, output_dir=args.output)
        
        # A hacky way to populate the queue for this standalone test
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                swarm.raw_image_queue.put(os.path.join(args.input, filename))
                
        # Signal the end of the line
        for _ in range(mp.cpu_count()): # Enough poison pills for all slicer workers
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
