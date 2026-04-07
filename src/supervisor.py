import os
import sys
import logging
import multiprocessing as mp
import cv2
from PIL import Image, ExifTags

from inquisitor import inquisitor_worker
from skeptic import skeptic_worker
from cartographer import cartographer_worker

def setup_logging():
    logger = logging.getLogger("Andvari.Supervisor")
    logger.setLevel(logging.INFO)
    logger.propagate = True
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    return logger

logger = setup_logging()

def extract_dji_telemetry(image_path):
    """Intercepts the image to extract native DJI GPS data before OpenCV strips it."""
    telemetry = {"lat": 0.0, "lon": 0.0, "alt": 50.0, "heading": 0.0, "pitch": -90.0}
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if not exif:
                return telemetry
            
            gps_info = None
            for key, val in exif.items():
                if ExifTags.TAGS.get(key) == 'GPSInfo':
                    gps_info = val
                    break
                    
            if gps_info:
                # PIL GPS tags: 1:LatRef, 2:Lat, 3:LonRef, 4:Lon, 6:Alt
                def to_decimal(dms, ref):
                    deg = float(dms)
                    min = float(dms)
                    sec = float(dms)
                    decimal = deg + (min / 60.0) + (sec / 3600.0)
                    return -decimal if ref in ['S', 'W'] else decimal
                    
                if 2 in gps_info and 1 in gps_info:
                    telemetry["lat"] = to_decimal(gps_info, gps_info)
                if 4 in gps_info and 3 in gps_info:
                    telemetry["lon"] = to_decimal(gps_info, gps_info)
                if 6 in gps_info:
                    telemetry["alt"] = float(gps_info)
    except Exception as e:
        logger.debug(f"Failed to parse EXIF for {os.path.basename(image_path)}: {e}")
        
    return telemetry

class Supervisor:
    def __init__(self, raw_image_dir, output_dir, weights_path=None, total_images=0, processed_counter=None, config=None):
        self.raw_image_dir = raw_image_dir
        self.output_dir = output_dir
        self.weights_path = weights_path
        self.total_images = total_images
        self.processed_counter = processed_counter
        self.config = config
        
        self.raw_image_queue = mp.Queue()
        self.tile_queue = mp.Queue(maxsize=2000) 
        self.candidate_queue = mp.Queue()
        self.verified_queue = mp.Queue() 
        
        os.makedirs(self.output_dir, exist_ok=True)

    def worker_node(self, worker_id, in_queue, tile_queue, counter, total):
        logger.debug(f"Worker {worker_id} spun up and waiting for operations.")
        
        while True:
            task_path = in_queue.get()
            
            if task_path == "SHUTDOWN_COMMAND":
                logger.debug(f"Worker {worker_id} executing shutdown command.")
                break
            
            filename = os.path.basename(task_path)
            
            # --- THE FIX: Extract live telemetry before OpenCV ---
            live_telemetry = extract_dji_telemetry(task_path)
            
            img = cv2.imread(task_path)
            if img is not None:
                height, width, _ = img.shape
                
                tile_size = self.config["slicer"]["tile_size"] if self.config else 224
                overlap = self.config["slicer"]["overlap"] if self.config else 0.2
                stride = int(tile_size * (1.0 - overlap))
                
                for y in range(0, height - tile_size + 1, stride):
                    for x in range(0, width - tile_size + 1, stride):
                        tile = img[y:y+tile_size, x:x+tile_size]
                        
                        payload = {
                            "parent_image": filename,
                            "offset_x": x,
                            "offset_y": y,
                            "tile_data": tile,
                            "telemetry": live_telemetry  # Pass the live data
                        }
                        tile_queue.put(payload)
            
            if counter is not None:
                with counter.get_lock():
                    counter.value += 1
                    current_n = counter.value
                logger.info(f"[{current_n} out of {total}] Finished evaluating {filename}")
            else:
                logger.info(f"Finished evaluating {filename}")

    def launch(self):
        num_cores = mp.cpu_count()
        
        logger.info("Spawning Cartographer process...")
        cartographer_process = mp.Process(
            target=cartographer_worker, 
            args=(self.verified_queue, self.output_dir, self.config)
        )
        cartographer_process.start()

        logger.info("Spawning Skeptic process...")
        skeptic_process = mp.Process(
            target=skeptic_worker, 
            args=(self.candidate_queue, self.verified_queue, self.weights_path, self.config)
        )
        skeptic_process.start()
        
        logger.info("Spawning Inquisitor GPU process...")
        inquisitor_process = mp.Process(
            target=inquisitor_worker, 
            args=(self.tile_queue, self.candidate_queue, self.weights_path, self.config)
        )
        inquisitor_process.start()
        
        logger.info(f"Supervisor is launching the CPU Slicer swarm across {num_cores} cores...")
        workers = []
        for i in range(num_cores):
            p = mp.Process(
                target=self.worker_node, 
                args=(i, self.raw_image_queue, self.tile_queue, self.processed_counter, self.total_images)
            )
            workers.append(p)
            p.start()
            
        for p in workers:
            p.join()
            
        self.tile_queue.put("SHUTDOWN_COMMAND")
        inquisitor_process.join()
        
        self.candidate_queue.put("SHUTDOWN_COMMAND")
        skeptic_process.join()
        
        self.verified_queue.put("SHUTDOWN_COMMAND")
        cartographer_process.join()
            
        logger.info("Swarm execution complete. Pipeline shutdown successful.")
