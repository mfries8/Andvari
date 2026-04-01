# Andvari: Drone Operations & Data Collection Protocol

## 1. The Positive Data Patch (Training)
* **Size:** A 20x20 meter patch immediately adjacent to the actual search area. It must share the exact same geological and botanical features as the target zone.
* **Proxy Count:** Scatter 50 to 100 painted black rocks (meteorite proxies) randomly across this zone. Do not organize them.
* **Imaging Schema:** * Capture 10 to 20 large-format drone images of this specific patch.
    * Fly at the exact altitude planned for the main search (yielding 1.5 - 2.0 mm/pixel GSD).
    * Capture images from multiple overlapping angles.
    * **Lighting match:** This is critical. Shoot this immediately before or after the main flight to ensure shadow lengths and lighting color temperatures perfectly match the search data.

## 2. The Negative Data Patch (Training)
* **Size:** A separate 20x20 meter patch, also representative of the search area.
* **Proxy Count:** Zero. 
* **Purpose:** This teaches the model the native background noise. Without this, the neural network will aggressively classify every rabbit turd, dark shadow, and damp clod of dirt as a pristine space rock. 
* **Imaging Schema:** Same as the positive patch (10-20 images, identical altitude, multiple angles, same lighting conditions).

## 3. The Test/Search Area (Execution)
* **Size:** Dictated by the fall ellipse, typically covering several square kilometers.
* **Imaging Schema:** Standard automated lawnmower grid search. 
* **Flight Parameters:**
    * **GSD:** 1.5 - 2.0 mm/pixel.
    * **Overlap:** Minimum 20% front and side overlap to ensure no blind spots and to provide the CNN with multiple viewing angles of the same coordinates.
    * **Speed:** Governed by the camera's shutter speed; fly slow enough to avoid motion blur at the required altitude.
