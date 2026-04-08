import os
import sys
import logging
import asyncio
import multiprocessing as mp

from inquisitor import inquisitor_worker
from skeptic import skeptic_worker
from cartographer import cartographer_worker
from slicer import pool_slicer_worker, init_worker

def setup_logging():
    logger = logging.getLogger("Andvari.Supervisor")
    logger.setLevel(logging.INFO)
    logger.propagate = True
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    return logger

logger = setup_logging()

class Supervisor:
    def __init__(self, raw_image_dir, output_dir, weights_path=None, total_images=0, processed_counter=None, config=None):
        self.raw_image_dir = raw_image_dir
        self.output_dir = output_dir
        self.weights_path = weights_path
        self.total_images = total_images
        self.processed_counter = processed_counter
        self.config = config

        self.raw_image_queue = mp.Queue()
        self.tile_queue = mp.Queue(maxsize=100) 
        self.candidate_queue = mp.Queue()
        self.verified_queue = mp.Queue() 
        
        os.makedirs(self.output_dir, exist_ok=True)

    async def _async_launch(self):
        targets = []
        while not self.raw_image_queue.empty():
            path = self.raw_image_queue.get()
            if path == "SHUTDOWN_COMMAND":
                break
            targets.append(path)

        num_cores = mp.cpu_count()
        
        logger.info("Spawning Cartographer process...")
        cartographer_process = mp.Process(
            target=cartographer_worker, 
            args=(self.verified_queue, self.output_dir, self.config, self.raw_image_dir)
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
        
        logger.info(f"Supervisor is launching the CPU Slicer swarm via Pool across {num_cores} cores...")
        
        tile_size = self.config["slicer"]["tile_size"] if self.config else 224
        overlap = self.config["slicer"]["overlap"] if self.config else 0.2
        
        worker_args = [(img, tile_size, overlap) for img in targets]
        loop = asyncio.get_running_loop()
        
        with mp.Pool(processes=num_cores, initializer=init_worker, initargs=(self.tile_queue,)) as pool:
            def consume_pool():
                for _ in pool.imap_unordered(pool_slicer_worker, worker_args):
                    if self.processed_counter is not None:
                        with self.processed_counter.get_lock():
                            self.processed_counter.value += 1
                            current_n = self.processed_counter.value
                        logger.info(f"[{current_n} out of {self.total_images}] Finished evaluating image.")
                    else:
                        logger.info(f"Finished evaluating image.")

            await loop.run_in_executor(None, consume_pool)
                
        self.tile_queue.put("SHUTDOWN_COMMAND")
        
        logger.info("Waiting for Inquisitor to finish...")
        await loop.run_in_executor(None, inquisitor_process.join)
        
        self.candidate_queue.put("SHUTDOWN_COMMAND")
        logger.info("Waiting for Skeptic to finish...")
        await loop.run_in_executor(None, skeptic_process.join)
        
        self.verified_queue.put("SHUTDOWN_COMMAND")
        logger.info("Waiting for Cartographer to finish...")
        await loop.run_in_executor(None, cartographer_process.join)
            
        logger.info("Swarm execution complete. Pipeline shutdown successful.")

    def launch(self):
        asyncio.run(self._async_launch())