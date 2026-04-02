import multiprocessing as mp
import sys
import time
import logging
import asyncio

# Force 'spawn' method to prevent CUDA context corruption on Linux
if sys.platform.startswith('linux'):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Already set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Andvari.Supervisor")

class Supervisor:
    def __init__(self, raw_image_dir, output_dir):
        self.raw_image_dir = raw_image_dir
        self.output_dir = output_dir
        self.manager = mp.Manager()
        
        # Inter-Process Communication Queues
        # Manager queues allow safe data passing between isolated CPU/GPU processes
        self.raw_image_queue = self.manager.Queue()     # Fed to Slicer
        self.tile_queue = self.manager.Queue(maxsize=1000) # Slicer -> Inquisitor (VRAM buffer protection)
        self.candidate_queue = self.manager.Queue()     # Inquisitor -> Skeptic (Initial hits)
        self.verified_queue = self.manager.Queue()      # Skeptic -> Cartographer (Surviving hits)
        
        # Process handles
        self.processes = []
        
    def _start_slicer(self):
        # TODO: Import Slicer agent
        logger.info("Spawning Slicer Agent (CPU Pool)...")
        # p = mp.Process(target=slicer_worker, args=(self.raw_image_queue, self.tile_queue))
        # p.start()
        # self.processes.append(p)

    def _start_inquisitor(self):
        # TODO: Import Inquisitor agent
        logger.info("Spawning Inquisitor Agent (GPU Node)...")
        # p = mp.Process(target=inquisitor_worker, args=(self.tile_queue, self.candidate_queue))
        # p.start()
        # self.processes.append(p)

    def _start_skeptic(self):
        # TODO: Import Skeptic agent
        logger.info("Spawning Skeptic Agent (CPU Pool)...")
        # p = mp.Process(target=skeptic_worker, args=(self.candidate_queue, self.verified_queue))
        # p.start()
        # self.processes.append(p)

    def _start_cartographer(self):
        # TODO: Import Cartographer agent
        logger.info("Spawning Cartographer Agent (CPU Node)...")
        # p = mp.Process(target=cartographer_worker, args=(self.verified_queue, self.output_dir))
        # p.start()
        # self.processes.append(p)

    async def monitor_swarm(self):
        """Asynchronous loop to monitor queue depths and system health."""
        logger.info("Supervisor monitoring initiated. Type Ctrl+C to abort the mission.")
        try:
            while True:
                # Log queue sizes every 5 seconds to ensure we aren't bottlenecking
                logger.info(
                    f"Queue Status | Raw: {self.raw_image_queue.qsize()} | "
                    f"Tiles: {self.tile_queue.qsize()} | "
                    f"Candidates: {self.candidate_queue.qsize()} | "
                    f"Verified: {self.verified_queue.qsize()}"
                )
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.warning("Shutdown signal received. Killing the swarm.")
            self.shutdown()

    def launch(self):
        """Ignites the pipeline."""
        logger.info("Waking up Andvari...")
        
        # In a real run, we'd populate the raw_image_queue here by scanning the directory
        
        self._start_slicer()
        self._start_inquisitor()
        self._start_skeptic()
        self._start_cartographer()
        
        # Start the async monitoring loop on the main thread
        try:
            asyncio.run(self.monitor_swarm())
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        """Gracefully (or violently) terminate all child processes."""
        logger.info("Terminating all agent processes...")
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join()
        logger.info("All processes dead. Returning to the void.")

if __name__ == "__main__":
    # Test execution
    supervisor = Supervisor(raw_image_dir="./data/raw", output_dir="./data/output")
    supervisor.launch()
