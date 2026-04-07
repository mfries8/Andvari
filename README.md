# Andvari

> "That gold will be the death of whoever possesses it." — Andvari, probably right before his rocks got stolen.

## Overview
Andvari is an automated, multi-agent machine learning pipeline designed to find freshly fallen meteorites using high-resolution drone photogrammetry. Because wandering aimlessly through a desert looking for a dark rock among millions of slightly lighter rocks is an inefficient use of human knees. 

Heavily inspired by the methodologies of the Global Fireball Network (GFN), Andvari ingests aerial imagery, runs heavily augmented Convolutional Neural Networks (CNN) to identify meteorite proxies, aggressively filters false positives (shadows, rabbit droppings, and regular terrestrial dirt), and outputs high-precision GIS coordinates to vector field recovery teams.

## The Swarm Architecture
Andvari utilizes an agentic framework to optimize hardware utilization, keeping the CPU pegged with I/O and preprocessing while feeding the GPU an endless stream of tensors.

* **Supervisor:** Orchestrates the madness.
* **Slicer:** Chops 44MP aerial images into digestible, overlapping tiles across all available CPU cores. Features a native OpenCV GUI for rapid target annotation.
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

### Dependency Installation & The CUDA Catch
Install the required dependencies, preferably in a fresh virtual environment. 

**1. Install PyTorch (Hardware Specific)**
PyTorch requires a very specific installation command depending on your graphics card. **Do not blindly copy-paste this without checking your hardware.**

```bash
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```

To find *your* exact command:
* Open your terminal or PowerShell and run `nvidia-smi`.
* Look at the top right of the output box for **CUDA Version: XX.X**. This is the *maximum* version your current NVIDIA driver supports.
* Go to the [PyTorch Local Installation Page](https://pytorch.org/get-started/locally/).
* Select your OS, `Pip`, `Python`, and a Compute Platform (CUDA version) that is **equal to or lower than** the number you just found. Copy the generated command and run it.
* *(Note: If you are running on a Mac, an Intel/AMD integrated chip, or a system without a dedicated NVIDIA GPU, select the "CPU" compute platform).*

**2. Install Andvari Requirements**
Once the correct version of PyTorch is successfully installed, install the remaining swarm dependencies:

```bash
pip install -r requirements.txt
```

### The GPU Polygraph Test
Before running any field data, you **must** ensure Python can see your graphics card. Running this pipeline on a standard CPU will take days instead of minutes. Activate your environment and run:

```bash
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If this returns `False` or `CPU`, reinstall the CUDA-enabled version of PyTorch. Do not proceed until this prints your GPU's name.

## Data Management & Folder Structure
Before running any scripts, you must set up your data directories. **Crucial Rule: Do not reuse or overwrite folders between flights.** In field operations, data provenance is everything. 

Create a `data/` directory in the root of the project. Your raw drone dumps will go here, and the code will automatically generate the required subfolders during processing:

```text
Andvari/
├── models/
│   └── base.pth                     <-- Your lab-trained ResNet18 base weights
│
└── data/
    ├── raw_training_data/           <-- Dump raw calibration flight images here
    │   ├── positive/                <-- Images with calibration target proxies
    │   └── negative/                <-- Images without proxies
    │
    ├── sliced_training_data/        <-- Pipeline auto-generates this directory
    │   ├── positive/                <-- Slicer populates this based on your clicks
    │   └── negative/                <-- Slicer populates this automatically
    │
    ├── raw_test_data/               <-- Dump main search grid flight here
    │
    └── output/                      <-- Pipeline outputs land here
```

---

## Field Operations (Usage)

### ⚠️ Operational Rule of Thumb: Domain Shift
The neural network is highly specialized to the specific dirt, vegetation, and lighting it was trained on. 
* **When to REUSE weights (Skip Steps 1 & 2):** If you are flying back-to-back grids over the same terrain, on the same day, under similar weather conditions, skip the training. Proceed directly to Step 3 with your existing `.pth` file.
* **When to RETRAIN (Run Steps 1 & 2):** You MUST fly a new calibration patch and retrain the model if you change locations (e.g., dry lake to grassy field), if the lighting drastically changes (heavy overcast vs. high noon), or if the terrain gets wet (which completely alters soil albedo). 

### Step 0: Download the Base Neural Brain
Before you process your first batch of images, you need the foundational network structure (`base.pth`). Run the built-in generator script from the project root to fetch the pre-trained ImageNet core and format it for our binary `[Dirt, Meteorite]` classifier:

```bash
python generate_base.py
```
This will automatically construct your default weights in `./models/base.pth`.

### Step 1: Prepare Training Data (`slice`)
Before you can train the model on a new environment, you need to chop your calibration flights into digestible tiles. 

*Hardware Note:* Mobile GPUs (like the RTX 2050) will run Out of Memory (OOM) if you try to train on native 512x512 tiles. You must use `--tile_size 224` to match the native ResNet18 architecture and save VRAM.

1. Fly a calibration patch seeded with your painted proxy meteorites.
2. Dump the raw images into your `./data/raw_training_data/` folder inside either `positive/` or `negative/` subfolders.
3. Run the Slicer with the annotation and tile_size flags:

```bash
python src/main.py slice --input ./data/raw_training_data/ --output ./data/sliced_training_data/ --annotate --tile_size 224
```

4. **The UI Workflow:** A window will pop up showing your first image. 
    * If you see a real proxy rock, click it (a green dot will appear). 
    * Ignore shadows and dirt.
    * Press **Spacebar** or **N** to advance to the next image.
5. Once you clear the last image, the swarm will take over and instantly chop the images, routing tiles with your clicked coordinates into the `positive/` folder and everything else into the `negative/` folder.

### Step 2: Field Fine-Tuning (`train`)
Fine-tune your foundational `base.pth` weights to the local terrain using the 224x224 data you just annotated:

```bash
python src/main.py train --dataset ./data/sliced_training_data/ --base_weights ./models/base.pth --output_weights ./models/field_1_tuned.pth --epochs 15
```

### Step 3: The Main Search (`pipeline`)
Once you have flown the massive grid search over the target fall ellipse, dump the raw SD card images into your `raw_test_data` directory and unleash the swarm:

```bash
python src/main.py pipeline --input ./data/raw_test_data/ --output ./data/output/ --weights ./models/field_1_tuned.pth
```

*Note: The Supervisor will output `[n out of N]` progress trackers to the console so you can monitor the GPU's pace.*

### Step 4: Human Verification (`review`)
Once the pipeline finishes, review the surviving candidates before deploying field personnel on foot.

```bash
python src/main.py review
```

This will launch a local web UI (typically at `http://127.0.0.1:8000`). Click through the cropped thumbnails to "Approve" or "Reject" hits. Approvals are automatically appended to a final deployment `.csv`.
