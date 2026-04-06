import argparse
import logging
import sys
import os
import cv2
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

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

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("Andvari.Main")

logger = setup_logging()

def interactive_annotator(image_path):
    """
    Opens an OpenCV GUI to let the user click on meteorites.
    Includes a 'Dark Pass' to draw yellow circles around dark suspects.
    Returns a list of (x, y) coordinates mapped to the original image scale.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to read {image_path} for annotation.")
        return []

    orig_h, orig_w = img.shape[:2]
    
    # Scale down for standard laptop screens (assuming max height of 800px)
    max_display_h = 800
    scale = orig_h / max_display_h
    display_w = int(orig_w / scale)
    display_h = int(orig_h / scale)
    
    display_img = cv2.resize(img, (display_w, display_h))
    ui_canvas = display_img.copy()

    # The Dark Pass: Pre-calculate suspects to guide the user's eye
    gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    dark_pixels = (gray < 30).astype('uint8') * 255
    contours, _ = cv2.findContours(dark_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 2: # Filter out single-pixel noise
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Draw yellow suspect circles
                cv2.circle(ui_canvas, (cx, cy), 15, (0, 255, 255), 1)

    clicks = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Draw a solid green dot where the user clicks
            cv2.circle(ui_canvas, (x, y), 5, (0, 255, 0), -1)
            # Map the click back to the original 44MP coordinates
            clicks.append((int(x * scale), int(y * scale)))
            cv2.imshow("Andvari Annotator", ui_canvas)

    cv2.namedWindow("Andvari Annotator")
    cv2.setMouseCallback("Andvari Annotator", mouse_callback)

    logger.info(f"Annotating: {os.path.basename(image_path)}. Click targets. Press 'N' or 'Space' for Next.")
    
    while True:
        cv2.imshow("Andvari Annotator", ui_canvas)
        key = cv2.waitKey(1) & 0xFF
        # Proceed on 'n', 'N', Spacebar, or Enter
        if key in [ord('n'), ord('N'), 32, 13]: 
            break
        elif key == 27: # Esc key to abort
            logger.warning("Annotation aborted by user.")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()
    return clicks

def process_single_image(args):
    """Worker function to slice a single image and sort based on known targets."""
    image_path, output_dir, tile_size, overlap, known_targets = args
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    img = cv2.imread(image_path)
    if img is None:
        return 0
        
    height, width, _ = img.shape
    stride = int(tile_size * (1.0 - overlap))
    
    pos_dir = os.path.join(output_dir, "positive")
    neg_dir = os.path.join(output_dir, "negative")
    
    # Only create pos/neg folders if we are actively using targets
    if known_targets is not None:
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)
    
    tiles_saved = 0
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile = img[y:y+tile_size, x:x+tile_size]
            
            target_dir = output_dir 
            
            if known_targets is not None:
                target_dir = neg_dir # Default to negative
                # Check if any clicked coordinate falls inside this tile's bounding box
                for (mx, my) in known_targets:
                    if x <= mx < x + tile_size and y <= my < y + tile_size:
                        target_dir = pos_dir
                        break # One rock is enough to make the tile positive
                    
            out_path = os.path.join(target_dir, f"{name}_X{x}_Y{y}.jpg")
            cv2.imwrite(out_path, tile)
            tiles_saved += 1
            
    return tiles_saved

def standalone_slice(input_dir, output_dir, tile_size=512, overlap=0.2, annotate_mode=False):
    """Chops directory of high-res images into tiles, utilizing human annotations if requested."""
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    
    # Build list of all valid images across directory tree
    image_paths = []
    out_abs = os.path.abspath(output_dir)
    
    for root, dirs, files in os.walk(input_dir):
        # Skip the output directory if it's nested
        if os.path.abspath(root).startswith(out_abs):
            continue
        for file in files:
            if file.lower().endswith(valid_exts):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        logger.error(f"No valid images found in {input_dir}")
        return

    # PHASE 1: Synchronous Annotation
    target_map = {} # Maps filename -> [(x1,y1), (x2,y2)]
    if annotate_mode:
        logger.info("Initializing OpenCV Annotator. Look for yellow suspect circles.")
        for img_path in image_paths:
            clicks = interactive_annotator(img_path)
            target_map[img_path] = clicks
        logger.info(f"Annotation complete. Captured targets across {len(image_paths)} images.")

    # PHASE 2: Asynchronous Slicing
    logger.info(f"Slicing {len(image_paths)} images across {mp.cpu_count()} CPU cores...")
    
    task_args = []
    for img_path in image_paths:
        # Recreate subdirectories for output
        rel_path = os.path.relpath(os.path.dirname(img_path), input_dir)
        if rel_path == ".":
            sub_output = output_dir
        else:
            sub_output = os.path.join(output_dir, rel_path)
            os.makedirs(sub_output, exist_ok=True)
            
        targets = target_map.get(img_path, None) if annotate_mode else None
        task_args.append((img_path, sub_output, tile_size, overlap, targets))
    
    total_tiles = 0
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        for result in executor.map(process_single_image, task_args):
            total_tiles += result
            
    logger.info(f"Slicing complete! Generated {total_tiles} total tiles.")

def main():
    parser = argparse.ArgumentParser(
        description="Andvari: Drone-Assisted Meteorite Recovery Pipeline"
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Operating Mode", required=True)
    
    # Mode 0: The Training Data Prep (Slicer)
    slice_parser = subparsers.add_parser("slice", help="Chop calibration images into tiles for training.")
    slice_parser.add_argument("--input", type=str, required=True, help="Directory containing raw calibration images.")
    slice_parser.add_argument("--output", type=str, required=True, help="Directory to save the chopped tiles.")
    slice_parser.add_argument("--tile_size", type=int, default=512, help="Size of the square tiles (default 512).")
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
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10).")
    
    # Mode 3: The Rapid Review UI
    review_parser = subparsers.add_parser("review", help="Launch the local web server to review candidates.")
    
    args = parser.parse_args()
    
    if args.mode == "slice":
        if not os.path.exists(args.input):
            logger.error(f"Input directory not found: {args.input}")
            sys.exit(1)
        standalone_slice(args.input, args.output, args.tile_size, annotate_mode=args.annotate)

    elif args.mode == "pipeline":
        if not os.path.exists(args.input):
            logger.error(f"Input directory not found: {args.input}")
            sys.exit(1)
        swarm = Supervisor(raw_image_dir=args.input, output_dir=args.output)
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
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
