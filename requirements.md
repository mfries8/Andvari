# Drone-Assisted Meteorite Recovery ML System: Requirements Specification

## 1. Executive Summary
This document outlines the architecture and processing pipeline for an automated meteorite recovery system. The system utilizes drone-captured aerial imagery and a machine learning identification routine to locate meteorite falls. The methodology is heavily modeled on the successful framework developed by the Desert Fireball Network (DFN) at Curtin University, optimized for high-precision GIS output to vector recovery teams.

## 2. Data Ingestion & Hardware Requirements
* **Drone Specifications:** Flights must be programmed for grid searches yielding a Ground Sample Distance (GSD) of approximately **1.5 to 2.0 mm/pixel** to ensure meteorite candidates appear as 20-65 pixel objects. A minimum of **20% overlap** in both directions is required.
* **Primary Training Data:** * Isolated images of meteorite proxies (painted black rocks) to establish baseline morphology and albedo.
    * Large-scale drone survey images of the test field with scattered proxies.
* **In-Field "Local" Data Augmentation:** The system must support immediate retraining/fine-tuning in the field. Recovery teams will place known proxies (or actual fusion-crusted specimens) into the specific search terrain, fly a quick pass, and feed those localized images into the model to train it against the native background noise.
* **Compute:** The field system must be capable of processing one flight's worth of images (approx. 30 minutes of flight time) in near real-time (under 65 minutes) using an on-site GPU (e.g., RTX 2080 Ti equivalent or better).

## 3. Data Processing Strategy
* **Tiling:** Large aerial raw images (e.g., 44MP) must be sliced into smaller overlapping tiles suitable for the CNN input layer (e.g., 256x256 or 512x512 pixels).
* **Color Channel Analysis:** The processing pipeline should leverage the unique reflectance signatures of meteorites (or proxies) across RGB channels to separate them from native terrestrial rocks.
* **Coordinate Mapping:** Every tile must inherit the EXIF metadata from its parent image. The pixel coordinates of any positive detection within a tile must be mathematically mapped back to the parent image's absolute coordinate space.

## 4. Machine Learning Training & Architecture
The model will utilize a Convolutional Neural Network (CNN) structured similarly to the proven GFN architecture:
* **Input Layer:** Sized to the tile dimensions.
* **Convolutional Blocks:** Four sequential 2D Convolutional layers (e.g., 30, 60, 120, and 240 filters using 3x3 kernels). Each block must include:
    * ReLU activation.
    * Batch Normalization.
    * Max Pooling (2x2).
* **Fully Connected Layers:** * Flattening layer.
    * Dense layer (1000 nodes, ReLU, Dropout 0.5).
    * Dense layer (150 nodes, ReLU, Dropout 0.5).
* **Output Layer:** Dense layer with 1 node and a Sigmoid activation function yielding a confidence score between 0 and 1.

## 5. Meteorite Identification & False Positive Handling
False positives (shadows, rabbit droppings, dark terrestrial rocks) are the primary bottleneck. The system will implement a multi-stage elimination protocol:
* **Thresholding:** Establish a strict confidence threshold for the Sigmoid output. 
* **The Rotation Filter:** For any tile triggering a positive detection, the system will programmatically rotate the tile 90, 180, and 270 degrees and run the prediction again. If the average confidence score across all rotations drops below the threshold, the object is classified as a false positive and automatically added to the training set as background noise.
* **Density Cap:** If an individual image returns an anomalously high number of positives (e.g., >10), the entire image is flagged as a false positive cluster and fed back into the retraining loop.
* **Human-in-the-Loop Verification:** The software must generate a rapid-review UI. It will present cropped tiles of surviving candidates to a human operator to quickly discard obvious anomalies before deploying field teams.

## 6. Output Generation & GIS Integration
* **Coordinate Translation:** For every verified candidate, the system will calculate the precise latitude and longitude. This requires cross-referencing the drone's GPS telemetry, altitude, camera gimbal angle, and the specific pixel offset of the candidate from the center of the image.
* **Export Formats:** The output must be an automated generated `.csv` and `.kml` / `.geojson` file.
* **Data Fields:** * Candidate ID
    * Latitude (Decimal Degrees, at least 6 decimal places for <1m precision)
    * Longitude
    * Confidence Score (0.0 to 1.0)
    * Parent Image Filename
    * Cropped thumbnail path
* **Secondary Verification:** The KML file will be used to vector a smaller, secondary drone to fly at a lower altitude over the high-probability candidates for high-resolution visual confirmation prior to dispatching personnel on foot.
