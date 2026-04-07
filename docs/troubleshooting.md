# Andvari: Field Troubleshooting & Survival Guide

### 1. The Swarm Finds "Too Many" Meteorites (False Positive Avalanches)
* **Symptoms:** The terminal shows hundreds of hits with 98%+ confidence. The Auditor UI is flooded with pictures of shadows, rabbit holes, or distinct local limestone.
* **The Cause:** Domain Shift. The model has latched onto a specific geological feature and confidently learned the wrong thing.
* **The Fix (Hard Negative Mining):**
  1. Wipe your `data/flight_X_results/` folder to prevent Auditor crashes.
  2. Re-run the `pipeline` to get a clean batch of false positives.
  3. Look at the thumbnails. Manually move the most egregious "fake meteorites" directly into your `data/field_1_training/negative/` folder.
  4. Run the `train` command again. You are explicitly forcing the Augmenter to penalize that specific terrain feature.

### 2. Auditor UI Throws an `[Internal Server Error]`
* **Symptoms:** Opening `127.0.0.1:8000` throws a 500 server error.
* **The Cause:** The `verified_candidates.csv` is pointing to thumbnail images that no longer exist, or the CSV is corrupted from a mid-flight crash.
* **The Fix:** Go to `data/output/` (or your specific flight results folder) and delete `verified_candidates.csv` and the entire `thumbnails/` directory. Run the `pipeline` again to generate clean, matched data.

### 3. CUDA Out of Memory (OOM) Errors
* **Symptoms:** The `Inquisitor` or `Augmenter` crashes instantly with a PyTorch CUDA memory allocation error.
* **The Cause:** Mobile GPUs (like the RTX 2050) have strictly limited VRAM (usually 4GB).
* **The Fix:** * *During Training:* Lower the `batch_size` in `augmenter.py` to 4 or 8.
  * *During Inference:* Lower the `batch_size` in `inquisitor.py` to 16. Ensure the `Slicer` is operating at `--tile_size 224`. Do not attempt to process 512x512 tiles on a 4GB GPU.

### 4. The Slicer is Saving "Empty Dirt" as Positive Hits
* **Symptoms:** Your `positive/` training folder is full of grass, and the meteorites are cut in half or missing entirely.
* **The Cause:** You ran the standalone `--annotate` UI on a scaled-down monitor image, but the math didn't reverse-scale the click back to the 44MP original.
* **The Fix:** Ensure you are using the updated `slicer.py` that utilizes the `scale_factor` multiplier to center the 224x224 bounding box on the true, native-resolution pixel coordinates.
