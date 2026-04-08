import cv2
import os
import glob
import random
import logging
from queue import Empty

logger = logging.getLogger("Andvari.Slicer")

class TelemetryParser:
    """Extracts metadata from image files."""
    def extract(self, image_path):
        telemetry = {"lat": 0.0, "lon": 0.0, "alt": 50.0, "pitch": -90.0, "heading": 0.0}
        try:
            from PIL import Image, ExifTags
            with Image.open(image_path) as img:
                gps_info = None
                if hasattr(img, "getexif"):
                    exif = img.getexif()
                    if hasattr(exif, "get_ifd"):
                        gps_info = exif.get_ifd(0x8825)
                
                if not gps_info and hasattr(img, '_getexif'):
                    raw_exif = img._getexif()
                    if raw_exif:
                        for k, v in raw_exif.items():
                            if ExifTags.TAGS.get(k) == 'GPSInfo':
                                gps_info = v
                                break
                                
                if gps_info:
                    def safe_float(val):
                        try: return float(val)
                        except: pass
                        if isinstance(val, (tuple, list)):
                            if len(val) == 1:
                                try: return float(val[0])
                                except: return 0.0
                            if len(val) >= 2:
                                try: return float(val[0]) / float(val[1]) if float(val[1]) != 0 else 0.0
                                except: return 0.0
                        return 0.0

                    def to_decimal(dms, ref):
                        if not dms or not ref: return 0.0
                        try:
                            deg = safe_float(dms[0]) if len(dms) > 0 else 0.0
                            minute = safe_float(dms[1]) if len(dms) > 1 else 0.0
                            sec = safe_float(dms[2]) if len(dms) > 2 else 0.0
                            
                            decimal = deg + (minute / 60.0) + (sec / 3600.0)
                            
                            ref_str = str(ref).strip().upper()
                            if 'S' in ref_str or 'W' in ref_str:
                                decimal = -decimal
                            return decimal
                        except Exception as math_e:
                            logger.warning(f"[MATH ERROR] GPS DMS format {dms}: {math_e}")
                            return 0.0
                            
                    lat = to_decimal(gps_info.get(2), gps_info.get(1))
                    lon = to_decimal(gps_info.get(4), gps_info.get(3))
                    
                    if lat != 0.0 and lon != 0.0:
                        telemetry["lat"] = lat
                        telemetry["lon"] = lon
                        
                    alt = gps_info.get(6)
                    if alt is not None:
                        telemetry["alt"] = safe_float(alt)
                else:
                    logger.warning(f"[MISSING DATA] No GPS block found in {os.path.basename(image_path)}")
            
            # Extract DJI-specific XMP metadata for exact drone heading and True AGL (RelativeAltitude)
            try:
                import re
                with open(image_path, 'rb') as f_xmp:
                    raw_data = f_xmp.read()
                    xmp_begin = raw_data.find(b'<x:xmpmeta')
                    xmp_end = raw_data.find(b'</x:xmpmeta>')
                    if xmp_begin != -1 and xmp_end != -1:
                        xmp_str = raw_data[xmp_begin:xmp_end+12].decode('utf-8', 'ignore')
                        
                        heading_match = re.search(r'FlightYawDegree="([^"]+)"', xmp_str)
                        if heading_match:
                            telemetry['heading'] = float(heading_match.group(1))
                        
                        relative_alt_match = re.search(r'RelativeAltitude="([^"]+)"', xmp_str)
                        if relative_alt_match:
                            telemetry['alt'] = float(relative_alt_match.group(1))
            except Exception as xmp_e:
                logger.warning(f"[XMP WARNING] Could not parse XMP payload for {os.path.basename(image_path)}: {xmp_e}")
                

        except Exception as e:
            logger.warning(f"[FATAL EXIF CRASH] Failed to extract GPS for {image_path}: {e}")
        return telemetry

_tile_queue = None

def init_worker(q):
    global _tile_queue
    _tile_queue = q

def pool_slicer_worker(args):
    """
    The inference worker mapped over the Pool.
    Chops one image into tiles and feeds the GPU queue.
    """
    image_path, tile_size, overlap = args
    parser = TelemetryParser()
    telemetry = parser.extract(image_path)
    
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load {image_path}. Corrupted file?")
        return 0
        
    height, width, _ = img.shape
    stride = int(tile_size * (1.0 - overlap))
    
    tile_count = 0
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile = img[y:y+tile_size, x:x+tile_size]
            payload = {
                "parent_image": os.path.basename(image_path),
                "tile_data": tile,           
                "offset_x": x,               
                "offset_y": y,               
                "telemetry": telemetry       
            }
            if _tile_queue is not None:
                _tile_queue.put(payload)
            tile_count += 1
            
    return tile_count

def generate_training_data(input_dir, output_dir, tile_size=224):
    """
    Dual-mode Slicer. Provides a UI for positive targets, and
    silently auto-mines background noise from negative images.
    """
    pos_out = os.path.join(output_dir, "positive")
    neg_out = os.path.join(output_dir, "negative")
    os.makedirs(pos_out, exist_ok=True)
    os.makedirs(neg_out, exist_ok=True)

    pos_in = os.path.join(input_dir, "positive")
    neg_in = os.path.join(input_dir, "negative")

    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    
    # ==========================================
    # PHASE 1: Targeted Annotation (UI)
    # ==========================================
    if os.path.exists(pos_in):
        logger.info("Found positive folder. Launching UI for targeted annotation.")
        pos_images = [os.path.join(pos_in, f) for f in os.listdir(pos_in) if f.lower().endswith(valid_exts)]
    else:
        logger.warning("No positive folder found. Defaulting to all images in input directory.")
        pos_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]

    if not pos_images and not os.path.exists(neg_in):
        logger.error(f"No valid images (JPG/PNG/TIF) found in {input_dir} to process. Ending early.")
        return

    if pos_images:
        total_pos = len(pos_images)
        logger.info(f"Loading {total_pos} positive images. Click targets, SPACEBAR to advance.")
        
        # We wrap the loop in enumerate to track the current index
        for current_idx, img_path in enumerate(pos_images, start=1):
            img = cv2.imread(img_path)
            if img is None:
                continue

            # --- THE NEW PROGRESS LOGGER ---
            logger.info(f"Image [{current_idx} out of {total_pos}]: Annotating {os.path.basename(img_path)}")

            orig_h, orig_w = img.shape[:2]
            ui_height = 900
            scale_factor = orig_h / ui_height
            ui_width = int(orig_w / scale_factor)

            ui_img = cv2.resize(img, (ui_width, ui_height))
            clone = ui_img.copy()
            targets = []

            def click_event(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    cv2.circle(ui_img, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow("Andvari Annotator", ui_img)
                    targets.append((int(x * scale_factor), int(y * scale_factor)))

            cv2.namedWindow("Andvari Annotator")
            cv2.setMouseCallback("Andvari Annotator", click_event)
            
            while True:
                cv2.imshow("Andvari Annotator", ui_img)
                key = cv2.waitKey(1) & 0xFF
                if key == 32: # SPACEBAR
                    break
                elif key == ord('r'): # Reset drawing
                    ui_img = clone.copy()
                    targets.clear()

            half_tile = tile_size // 2
            for i, (tx, ty) in enumerate(targets):
                y1 = max(0, ty - half_tile)
                y2 = min(orig_h, ty + half_tile)
                x1 = max(0, tx - half_tile)
                x2 = min(orig_w, tx + half_tile)
                
                if y2 - y1 < tile_size: y2 = y1 + tile_size
                if x2 - x1 < tile_size: x2 = x1 + tile_size

                positive_tile = img[y1:y2, x1:x2]
                out_path = os.path.join(pos_out, f"{os.path.basename(img_path)}_pos_{i}.jpg")
                cv2.imwrite(out_path, positive_tile)
                
        cv2.destroyAllWindows()

    # ==========================================
    # PHASE 2: Silent Auto-Mining (No UI)
    # ==========================================
    if os.path.exists(neg_in):
        logger.info("Found negative folder. Silently mining background noise...")
        neg_images = [os.path.join(neg_in, f) for f in os.listdir(neg_in) if f.lower().endswith(valid_exts)]
        
        for img_path in neg_images:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            orig_h, orig_w = img.shape[:2]
            
            # Auto-mine 25 random chops per empty image
            for i in range(25):
                rx = random.randint(0, orig_w - tile_size)
                ry = random.randint(0, orig_h - tile_size)
                
                negative_tile = img[ry:ry+tile_size, rx:rx+tile_size]
                out_path = os.path.join(neg_out, f"{os.path.basename(img_path)}_neg_{i}.jpg")
                cv2.imwrite(out_path, negative_tile)
                
        logger.info(f"Successfully mined negative tiles from {len(neg_images)} images.")
