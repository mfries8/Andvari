# Andvari Agentic Framework: agents.md

## 1. System Orchestrator: The Supervisor Agent
* **Role:** The central nervous system. Manages state, handles inter-agent communication, and ensures the pipeline doesn't trip over itself.
* **Multicore Strategy:** Operates on the main thread, utilizing `asyncio` to manage subprocesses, dispatch jobs to the worker pools, and monitor hardware thermals so your field laptop doesn't melt.

## 2. Data Ingestion & Preprocessing: The Slicer Agent
* **Role:** Reads raw high-resolution (e.g., 44MP) aerial imagery, extracts EXIF/telemetry data, and chops the images into 512x512 overlapping tiles.
* **Multicore Strategy:** Highly parallelized. Utilizes Python's `multiprocessing.Pool` to spawn worker processes across all available logical CPU cores. Each core handles a discrete raw image simultaneously, slicing it and writing the tensors to a shared memory block or fast NVMe cache.

## 3. Field Fine-Tuning: The Augmenter Agent
* **Role:** Only wakes up when you feed it new images of proxies in the specific search terrain. Generates synthetic variations (rotations, brightness tweaks) of the local proxies to retrain the final layers of the CNN.
* **Multicore Strategy:** Uses `multiprocessing` for the image augmentation pipeline (CPU-bound) before passing the synthesized dataset to the GPU for the actual weights update.

## 4. CNN Inference: The Inquisitor Agent
* **Role:** The heavy lifter. Ingests the preprocessed tiles and runs them through the Convolutional Neural Network to assign a confidence score (0 to 1) for meteorite presence.
* **Multicore/Hardware Strategy:** Strictly GPU-bound. Uses a dedicated data-loader thread to pre-fetch batches of tiles into VRAM, ensuring the GPU is running at 100% utilization without waiting on the CPU.

## 5. False Positive Elimination: The Skeptic Agent
* **Role:** Takes any tile that passes the initial confidence threshold and tries to prove it wrong. Applies the Rotation Filter (rotates 90, 180, 270 degrees and re-runs inference) and the Density Cap (flags whole images with too many "meteorites" as rabbit-poop clusters).
* **Multicore Strategy:** Uses a threaded queue. While the Inquisitor (GPU) is processing Batch N+1, the Skeptic (multicore CPU) is simultaneously applying rotation transformations to the positive hits from Batch N and firing them back to the GPU's priority queue.

## 6. GIS & Output Mapping: The Cartographer Agent
* **Role:** Takes the surviving verified hits and translates their pixel coordinates within the tile back to the parent image, and then into absolute Earth coordinates (Latitude/Longitude).
* **Multicore Strategy:** Vectorized mathematical operations using NumPy across multiple CPU cores to instantly calculate geospatial coordinates based on drone telemetry (altitude, gimbal pitch, FOV). Compiles the final `.csv` and `.kml` payload.

## 7. Human Verification UI: The Auditor Agent
* **Role:** Sp spins up a lightweight local web server (e.g., FastAPI) to present cropped thumbnails of the final candidates to the human operator for rapid "yes/no" click-through before deploying the recovery team.
* **Multicore Strategy:** Asynchronous web server handling local requests, totally detached from the heavy processing pipeline.