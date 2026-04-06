# Andvari

> "That gold will be the death of whoever possesses it." — Andvari, probably right before his rocks got stolen.

## Overview
Andvari is an automated, multi-agent machine learning pipeline designed to find freshly fallen meteorites using high-resolution drone photogrammetry. Because wandering aimlessly through a desert looking for a dark rock among millions of slightly lighter rocks is an inefficient use of human knees. 

Heavily inspired by the methodologies of the Global Fireball Network (GFN), Andvari ingests aerial imagery, runs heavily augmented Convolutional Neural Networks (CNN) to identify meteorite proxies, aggressively filters false positives (shadows, rabbit droppings, and regular terrestrial dirt), and outputs high-precision GIS coordinates to vector field recovery teams.

## The Swarm Architecture
Andvari utilizes an agentic framework to optimize hardware utilization, keeping the CPU pegged with I/O and preprocessing while feeding the GPU an endless stream of tensors.

* **Supervisor:** Orchestrates the madness.
* **Slicer:** Chops 44MP aerial images into digestible, overlapping 512x512 tiles across all available CPU cores. Directory-aware for batch sorting.
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
    pip install -r requirements.txt

*(Note: Adjust the PyTorch CUDA index URL to match your specific hardware and drivers. Standard `pip install` may default to CPU-only on some Linux distributions).*

## Data Management & Folder Structure
Before running any scripts, you must set up your data directories. **Crucial Rule: Do not reuse or overwrite folders between flights.** In field operations, data provenance is everything. 

Create a `data/` directory in the root of the project. Organize your calibration flights into subfolders so the Slicer can automatically process them in bulk:

    Andvari/
    ├── models/
    │   └── base.pth                     <-- Your lab-trained base weights
    │
    └── data/
        ├── field_1_training/            <-- Your raw calibration data
        │   ├── positive/                <-- Raw images containing your proxies
        │   ├── negative/                <-- Raw images of empty dirt
        │   └── test/                    <-- Raw images for holdout validation (optional)
        │
        ├── raw_flight_1/                <-- Dump main search grid flight here
        │
        └── flight_1_results/            <-- Pipeline outputs land here

## Field Operations (Usage)

Andvari is operated via a central command-line interface with four distinct steps for a complete field deployment.

### Step 1: Prepare Training Data (`slice`)
Before you can train the model, you need to chop your calibration flights into digestible tiles. The Slicer is directory-aware and will maintain your folder structure.

1. Fly your calibration patches.
2. Drop the raw, un-chopped images into their respective subfolders (`positive`, `negative`, `test`) inside your training directory.
3. Run the Slicer on the root directory. It will automatically recreate your subfolders inside a new `sliced` directory and dump the tiles appropriately:

    python main.py slice --input ./data/field_1_training/ --output ./data/field_1_training/sliced/

*(Pro-Tip: If you want the code to automatically filter out empty dirt from a folder of mixed images, append the `--triage` flag to the command. It will create a `suspects/` folder for rapid human review).*

### Step 2: Field Fine-Tuning (`train`)
Fine-tune your lab weights to the local terrain using the newly sliced data:

    python main.py train --dataset ./data/field_1_training/sliced/ --base_weights ./models/base.pth --output_weights ./models/field_1_tuned.pth --epochs 15

### Step 3: The Main Search (`pipeline`)
Once you have flown the massive grid search over the target fall ellipse, dump the raw SD card images into your `raw_flight_1` directory and unleash the swarm:

    python main.py pipeline --input ./data/raw_flight_1/ --output ./data/flight_1_results/ --weights ./models/field_1_tuned.pth

*Note: Depending on flight size and GPU, go grab a coffee. The Supervisor will log queue depths to the console so you can monitor progress.*

### Step 4: Human Verification (`review`)
Once the pipeline finishes, review the surviving candidates before deploying field personnel on foot.

    python main.py review

This will launch a local web UI (typically at `http://127.0.0.1:8000`). Click through the cropped thumbnails to "Approve" or "Reject" hits. Approvals are automatically appended to a final deployment `.csv`.
