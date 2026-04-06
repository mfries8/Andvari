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
Before running any scripts, you must set up your data directories. **Crucial Rule: Do not reuse or overwrite folders between flights.** In field operations, data provenance is everything. If you need to go back and check why a rock was missed, you need the original data intact. 

Create a `data/` directory in the root of the project. For every new field site or drone flight, create a new set of appropriately numbered folders:

    Andvari/
    ├── models/
    │   └── base.pth                     <-- Your lab-trained base weights
    │
    └── data/
        ├── field_1_training/            <-- Create for calibration flights
        │   ├── positive/                <-- Put sliced tiles of proxies here
        │   └── negative/                <-- Put sliced tiles of empty dirt here
        │
        ├── raw_flight_1/                <-- DUMP YOUR DRONE SD CARD HERE
        │   ├── DJI_0001.JPG
        │   └── DJI_0002.JPG
        │
        └── flight_1_results/            <-- The pipeline will output maps and CSVs here

For your next flight, simply create `raw_flight_2/` and `flight_2_results/`, and update the command-line arguments accordingly. Offload old flight folders to an external SSD to save space on your processing machine.

## Field Operations (Usage)

Andvari is operated via a central command-line interface with three distinct modes: `train`, `pipeline`, and `review`.

### Step 1: Field Fine-Tuning (`train`)
Before running a search, you must train the model to ignore the local dirt and shadows.
1. Fly a calibration patch seeded with 50-100 meteorite proxies (the "positive" set).
2. Fly an empty patch (the "negative" set).
3. Chop these images into 512x512 tiles using the Slicer and place them in your `positive/` and `negative/` subfolders.
4. Run the Augmenter to fine-tune your lab weights to the local terrain:

    python main.py train --dataset ./data/field_1_training/ --base_weights ./models/base.pth --output_weights ./models/field_1_tuned.pth --epochs 15

### Step 2: The Main Search (`pipeline`)
Once you have flown the massive grid search over the target fall ellipse, dump the raw SD card images into your `raw_flight_X` directory and unleash the swarm:

    python main.py pipeline --input ./data/raw_flight_1/ --output ./data/flight_1_results/ --weights ./models/field_1_tuned.pth

*Note: Depending on flight size and GPU, go grab a coffee. The Supervisor will log queue depths to the console so you can monitor progress.*

### Step 3: Human Verification (`review`)
Once the pipeline finishes, review the surviving candidates before deploying field personnel on foot.

    python main.py review

This will launch a local web UI (typically at `http://127.0.0.1:8000`). Click through the cropped thumbnails to "Approve" or "Reject" hits. Approvals are automatically appended to a final deployment `.csv`.
