import os
import sys
import logging
import multiprocessing as mp
import time
import cv2

# Import the GPU worker we built previously
from inquisitor import inquisitor_worker

def setup_logging():
    logger = logging.getLogger("Andvari.Supervisor")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logging()

class Supervisor:
    # Notice we added weights_path here so the boss can hand it to the GPU
    def __init__(self, raw_image_dir, output_dir, weights_path=None, total_images=0, processed_counter=None):
        """
        Orchestrates the multiprocessing swarm for field inference.
        """
        self.raw_image_dir = raw_image_dir
        self.output_dir = output_dir
        self.weights_path = weights_path
        self.total_images = total_images
        self.processed_counter = processed_counter
        
        # The main queue fed by main.py
        self.raw_image_queue = mp.Queue()
        
        # The conveyor belts between your CPU Slicers and the GPU Inquisitor
        # Capping the maxsize at 2000 so we don't accidentally fill your system RAM with waiting tiles
        self.tile_queue = mp.Queue(maxsize=2000) 
        self.candidate_queue = mp.Queue()
        
        os.makedirs(self.output_dir, exist_ok=True)

    def worker_node(self, worker_id, in_queue, tile_queue, counter, total):
        """
        The heavy lifting node. Pulls raw drone images, chops them into 224x224 bites in RAM, 
        and throws them onto the GPU conveyor belt.
        """
        logger.debug(f"Worker {worker_id} spun up and waiting for operations.")
        
        while True:
            task_path = in_queue.get()
            
            if task_path == "POISON_PILL":
                logger.debug(f"Worker {worker_id} swallowed poison pill. Shutting down.")
                break
            
            filename = os.path.basename(task_path)
            
            # ==========================================
            # ACTUAL IN-MEMORY SLICING PIPELINE
            # ==========================================
            img = cv2.imread(task_path)
            if img is not None:
                height, width, _ = img.shape
                
                # THE CRITICAL DIET FIX
                tile_size = 224  
                overlap = 0.2
                stride = int(tile_size * (1.0 - overlap))
                
                # Chop the image directly in system RAM
                for y in range(0, height - tile_size + 1, stride):
                    for x in range(0, width - tile_size + 1, stride):
                        tile = img[y:y+tile_size, x:x+tile_size]
                        
                        payload = {
                            "parent_image": filename,
                            "offset_x": x,
                            "offset_y": y,
                            "tile_data": tile
                        }
                        
                        # Put the tile on the belt for the GPU
                        tile_queue.put(payload)
            # ==========================================
            
            # --- THE HARDWARE-LOCKED PROGRESS TRACKER ---
            if counter is not None:
                with counter.get_lock():
                    counter.value += 1
                    current_n = counter.value
                    
                logger.info(f"[{current_n} out of {total}] Finished evaluating {filename}")
            else:
                logger.info(f"Finished evaluating {filename}")

    def launch(self):
        """Ignites the multiprocessing swarm."""
        num_cores = mp.cpu_count()
        
        # 1. Boot up the GPU process first
        logger.info("Spawning Inquisitor GPU process...")
        inquisitor_process = mp.Process(
            target=inquisitor_worker, 
            args=(self.tile_queue, self.candidate_queue, self.weights_path)
        )
        inquisitor_process.start()
        
        # 2. Launch the CPU Slicers
        logger.info(f"Supervisor is launching the CPU Slicer swarm across {num_cores} cores...")
        workers = []
        for i in range(num_cores):
            p = mp.Process(
                target=self.worker_node, 
                args=(
                    i, 
                    self.raw_image_queue, 
                    self.tile_queue, 
                    self.processed_counter, 
                    self.total_images
                )
            )
            workers.append(p)
            p.start()
            
        # 3. Wait for all CPU workers to finish eating the raw images
        for p in workers:
            p.join()
            
        # 4. Now that the Slicers are done, tell the GPU to shut down once the belt is empty
        self.tile_queue.put("POISON_PILL")
        inquisitor_process.join()
            
        logger.info("Swarm execution complete. Pipeline shutdown successful.")
