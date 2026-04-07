# Andvari

> "That gold will be the death of whoever possesses it." — Andvari, probably right before his rocks got stolen.

## Overview
Andvari is an automated, multi-agent machine learning pipeline designed to find freshly fallen meteorites using high-resolution drone photogrammetry. Because wandering aimlessly through a desert looking for a dark rock among millions of slightly lighter rocks is an inefficient use of human knees. 

Heavily inspired by the methodologies of the Global Fireball Network (GFN), Andvari ingests aerial imagery, runs heavily augmented Convolutional Neural Networks (CNN) to identify meteorite proxies, aggressively filters false positives (shadows, rabbit droppings, and regular terrestrial dirt), and outputs high-precision GIS coordinates to vector field recovery teams.

## The Swarm Architecture
Andvari utilizes an agentic framework to optimize hardware utilization, keeping the CPU pegged with I/O and preprocessing while feeding the GPU an endless stream of tensors.

* **Supervisor:** Orchestrates the madness.
* **Slicer:** Chops 44MP aerial images into digestible, overlapping 512x512 tiles across all available CPU cores. Features a native OpenCV GUI for rapid target annotation.
* **Augmenter:** Fine-tunes a pre-trained ResNet18 model in the field using localized background data.
* **Inquisitor:** The GPU-bound CNN inference engine.
* **Skeptic:** The false-positive filter. Rotates candidates and enforces density caps to weed out anomalies.
* **Cartographer:** Translates pixel hits back to absolute Earth coordinates (Lat/Long).
* **Auditor:** Serves up a rapid-review UI for human verification before anyone actually goes for a hike.

## Prerequisites
* Python 3.10+
* A CUDA-capable NVIDIA GPU (RTX 2050/2080 Ti equivalent or better required for field processing times)
* Fast NVMe storage (Do not run this off a spinning drive unless you enjoy waiting)
* A drone capable of flying grid searches at 1.5 - 2.0 mm/pixel GSD.

## Installation
1. Clone the repository.
2. Install the required dependencies (preferably in a fresh virtual environment):

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt

*(Note: Adjust the PyTorch CUDA index URL to match your specific hardware and drivers).*

### The GPU Polygraph Test
Before running any field data, you **must** ensure Python can see your graphics card. Running this pipeline on a standard CPU will take days instead of minutes. Activate your environment and run:

    python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

If this returns `False` or `CPU`, reinstall the CUDA-enabled version of PyTorch. Do not proceed until this prints your GPU's name.

## Data Management & Folder Structure
Before running any scripts, you must set up your data directories. **Crucial Rule: Do not reuse or overwrite folders between flights.** In field operations, data provenance is everything. 

Create a `data/` directory in the root of the project. Your raw drone dumps will go here, and the code will automatically generate the required subfolders during processing:

    Andvari/
    ├── models/
    │   └── base.pth                     <-- Your lab-trained ResNet18 base weights
    │
    └── data/
        ├── raw_calibration_1/           <-- Dump raw calibration flight images here
        │
        ├── field_1_training/            <-- Pipeline auto-generates this directory
        │   ├── positive/                <-- Slicer populates this based on your clicks
        │   └── negative/                <-- Slicer populates this automatically
        │
        ├── raw_flight_1/                <-- Dump main search grid flight here
        │
        └── flight_1_results/            <-- Pipeline outputs land here

---

## Field Operations (Usage)

### ⚠️ Operational Rule of Thumb: Domain Shift
The neural network is highly specialized to the specific dirt, vegetation, and lighting it was trained on. 
* **When to REUSE weights (Skip Steps 1 & 2):** If you are flying back-to-back grids over the same terrain, on the same day, under similar weather conditions, skip the training. Proceed directly to Step 3 with your existing `.pth` file.
* **When to RETRAIN (Run Steps 1 & 2):** You MUST fly a new calibration patch and retrain the model if you change locations (e.g., dry lake to grassy field), if the lighting drastically changes (heavy overcast vs. high noon), or if the terrain gets wet (which completely alters soil albedo). 

### Step 1: Prepare Training Data (`slice`)
Before you can train the model on a new environment, you need to chop your calibration flights into digestible tiles. The Slicer includes an interactive UI so you can identify targets before the code shreds the images.

1. Fly a calibration patch seeded with your painted proxy meteorites.
2. Dump the raw images into your `./data/raw_calibration_1/` folder.
3. Run the Slicer with the annotation flag:

    python src/main.py slice --input ./data/raw_calibration_1/ --output ./data/field_1_training/ --annotate

4. **The UI Workflow:** A window will pop up showing your first image. 
    * If you see a real proxy rock, click it (a green dot will appear). 
    * Ignore shadows and dirt.
    * Press **Spacebar** or **N** to advance to the next image.
5. Once you clear the last image, the swarm will take over and instantly chop the images, routing tiles with your clicked coordinates into the `positive/` folder and everything else into the `negative/` folder.

### Step 2: Field Fine-Tuning (`train`)
Fine-tune your foundational `base.pth` weights to the local terrain using the data you just annotated:

    python src/main.py train --dataset ./data/field_1_training/ --base_weights ./models/base.pth --output_weights ./models/field_1_tuned.pth --epochs 15

### Step 3: The Main Search (`pipeline`)
Once you have flown the massive grid search over the target fall ellipse, dump the raw SD card images into your `raw_flight_1` directory and unleash the swarm:

    python src/main.py pipeline --input ./data/raw_flight_1/ --output ./data/flight_1_results/ --weights ./models/field_1_tuned.pth

*Note: The Supervisor will output `[n out of N]` progress trackers to the console so you can monitor the GPU's pace.*

### Step 4: Human Verification (`review`)
Once the pipeline finishes, review the surviving candidates before deploying field personnel on foot.

    python src/main.py review

This will launch a local web UI (typically at `http://127.0.0.1:8000`). Click through the cropped thumbnails to "Approve" or "Reject" hits. Approvals are automatically appended to a final deployment `.csv`.
