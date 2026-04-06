# Andvari

> "That gold will be the death of whoever possesses it." — Andvari, probably right before his rocks got stolen.

## Overview
Andvari is an automated, multi-agent machine learning pipeline designed to find freshly fallen meteorites using high-resolution drone photogrammetry. Because wandering aimlessly through a desert looking for a dark rock among millions of slightly lighter rocks is an inefficient use of human knees. 

Heavily inspired by the methodologies of the Global Fireball Network (GFN), Andvari ingests aerial imagery, runs heavily augmented Convolutional Neural Networks (CNN) to identify meteorite proxies, aggressively filters false positives (shadows, rabbit droppings, and regular terrestrial dirt), and outputs high-precision GIS coordinates to vector field recovery teams.

## The Swarm Architecture
Andvari utilizes an agentic framework to optimize hardware utilization, keeping the CPU pegged with I/O and preprocessing while feeding the GPU an endless stream of tensors.

* **Supervisor:** Orchestrates the madness.
* **Slicer:** Chops 44MP aerial images into digestible, overlapping 512x512 tiles across all available CPU cores.
* **Augmenter:** Fine-tunes the model in the field using localized background data.
* **Inquisitor:** The GPU-bound CNN inference engine.
* **Skeptic:** The false-positive filter. Rotates candidates and enforces density caps to weed out anomalies.
* **Cartographer:** Translates pixel hits back to absolute Earth coordinates (Lat/Long).
* **Auditor:** Serves up a rapid-review UI for human verification before anyone actually goes for a hike.

## Prerequisites
* Python 3.10+
* A CUDA-capable GPU (RTX 2080 Ti equivalent or better required for field processing times)
* Fast NVMe storage (Do not run this off a spinning drive unless you enjoy waiting)
* A drone capable of flying grid searches at 1.5 - 2.0 mm/pixel GSD.

## Installation
1. Clone the repository.
2. Install the required dependencies (preferably in a fresh virtual environment):

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install opencv-python numpy fastapi uvicorn jinja2 python-multipart

*(Note: Adjust the PyTorch CUDA index URL to match your specific hardware and drivers).*

## Data Management & Folder Structure
Before running any scripts, you must set up your data directories. **Crucial Rule: Do not reuse or overwrite folders between flights.** In field operations, data provenance is everything. 

Create a `data/` directory in the root of the project. For every new field site or drone flight, create a new set of appropriately numbered folders:

    Andvari/
    ├── models/
    │   └── base.pth                     <-- Your lab-trained base weights
    │
    └── data/
        ├── raw_calibration_1/           <-- Dump calibration flight here
        │
        ├── field_1_training/            <-- Training workspace
        │   ├── positive/                <-- Put sliced tiles of proxies here
        │   └── negative/                <-- Put sliced tiles of empty dirt here
        │
        ├── raw_flight_1/                <-- Dump main search grid flight here
        │
        └── flight_1_results/            <-- Pipeline outputs land here

## Field Operations (Usage)

Andvari is operated via a central command-line interface with four distinct steps for a complete field deployment.

### Step 1: Prepare Training Data (`slice`)
Before you can train the model, you need to chop your calibration flights into digestible tiles. 
1. Fly a patch seeded with 50-100 proxies and an empty native patch. 
2. Dump those raw images into `./data/raw_calibration_1/`.
3. Run the Slicer as a standalone tool to chop the raw images into 512x512 tiles:

    python main.py slice --input ./data/raw_calibration_1/ --output ./data/field_1_training/raw_tiles/

4. Open `raw_tiles/`. Manually drag any tile containing a painted rock into `positive/`, and a large sample of empty dirt tiles into `negative/`.

### Step 2: Field Fine-Tuning (`train`)
Fine-tune your lab weights to the local terrain using the data you just sorted:

    python main.py train --dataset ./data/field_1_training/ --base_weights ./models/base.pth --output_weights ./models/field_1_tuned.pth --epochs 15

### Step 3: The Main Search (`pipeline`)
Once you have flown the massive grid search over the target fall ellipse, dump the raw SD card images into your `raw_flight_1` directory and unleash the swarm:

    python main.py pipeline --input ./data/raw_flight_1/ --output ./data/flight_1_results/ --weights ./models/field_1_tuned.pth

*Note: Depending on flight size and GPU, go grab a coffee. The Supervisor will log queue depths to the console so you can monitor progress.*

### Step 4: Human Verification (`review`)
Once the pipeline finishes, review the surviving candidates before deploying field personnel on foot.

    python main.py review

This will launch a local web UI (typically at `http://127.0.0.1:8000`). Click through the cropped thumbnails to "Approve" or "Reject" hits. Approvals are automatically appended to a final deployment `final_deployment_targets.csv`.
