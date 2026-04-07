# Drone-Assisted Meteorite Recovery ML System: Requirements Specification

## 1. Executive Summary
This document outlines the architecture and processing pipeline for an automated meteorite recovery system. The system utilizes drone-captured aerial imagery and a machine learning identification routine to locate meteorite falls. The methodology is heavily modeled on the successful framework developed by the Desert Fireball Network (DFN) at Curtin University, optimized for high-precision GIS output to vector recovery teams.

## 2. Data Ingestion & Hardware Requirements
* **Drone Specifications:** Flights must be programmed for grid searches yielding a Ground Sample Distance (GSD) of approximately **1.5 to 2.0 mm/pixel** to ensure meteorite candidates appear as 20-65 pixel objects. A minimum of **20% overlap** in both directions is required.
* **Site-Specific Primary Training Data (MANDATORY):** A new training dataset MUST be collected for every individual field site to capture localized geology, vegetation, and lighting.
    * **Positive Class:** A dense scatter of meteorite proxies (painted black rocks) placed in a representative patch of the target terrain and flown at search altitude.
    * **Negative Class:** Imagery of the native terrain containing no proxies, used to teach the model to ignore local false-positive hazards (shadows, distinct local rocks, flora). 
* **In-Field "Local" Data Augmentation:** The system must support immediate retraining/fine-tuning in the field utilizing the site-specific primary training data. The augmentation pipeline will synthetically multiply this local data (via rotations and flips) to train the model against the native background noise.
* **Compute:** The field system must be capable of processing one flight's worth of images (approx. 30 minutes of flight time) in near real-time (under 65 minutes) using an on-site GPU (e.g., RTX 2080 Ti equivalent or better).

## 3. Data Processing Strategy
* **Tiling:** Large aerial raw images (e.g., 44MP) must be sliced into smaller overlapping tiles suitable for the CNN input layer (e.g., 256x256 or 512x512 pixels).
* **Color Channel Analysis:** The processing pipeline should leverage the unique reflectance signatures of meteorites (or proxies) across RGB channels to separate them from native terrestrial rocks.
* **Coordinate Mapping:** Every tile must inherit the EXIF metadata from its parent image. The pixel coordinates of any positive detection within a tile must be mathematically mapped back to the parent image's absolute coordinate space.

## 4. Machine Learning Training & Architecture
The model utilizes Transfer Learning via a pre-trained ResNet18 Convolutional Neural Network (CNN) to maximize accuracy with minimal field data.
* **Input Layer:** Standardized to 224x224 pixel RGB tiles.
* **Convolutional Base:** The core 18-layer residual network architecture, pre-trained on ImageNet. During field fine-tuning, these feature-extraction layers are **frozen** to preserve the model's fundamental understanding of shapes, shadows, and textures.
* **Classification Head:** The original 1000-class output layer is replaced with a custom Fully Connected (FC) linear layer designed for binary classification (Nodes: 2).
* **Output:** A Softmax activation function yielding a confidence score (0.0 to 1.0) indicating the probability of a meteorite proxy vs. native background terrain.

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
