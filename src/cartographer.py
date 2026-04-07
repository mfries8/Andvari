import os
import csv
import math
import cv2
import logging
from queue import Empty
import traceback
import numpy as np

logger = logging.getLogger("Andvari.Cartographer")

R_EARTH = 6378137.0 

def generate_kml(csv_path, kml_path):
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">\n  <Document>\n    <name>Andvari Verified Candidates</name>\n"""
    kml_footer = """  </Document>\n</kml>"""
    
    with open(kml_path, 'w') as kml_file:
        kml_file.write(kml_header)
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    placemark = f"""    <Placemark>
      <name>Candidate {row['ID']}</name>
      <description>Confidence: {row['Confidence']}</description>
      <Point>
        <coordinates>{row['Longitude']},{row['Latitude']},0</coordinates>
      </Point>
    </Placemark>\n"""
                    kml_file.write(placemark)
        kml_file.write(kml_footer)

def cartographer_worker(verified_queue, output_dir, config=None):
    logger.info("Cartographer Agent online. Mapping the treasure.")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "verified_candidates.csv")
    kml_path = os.path.join(output_dir, "verified_candidates.kml")
    thumb_dir = os.path.join(output_dir, "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)
    
    fov_h = math.radians(config["camera"]["fov_horizontal_deg"]) if config else math.radians(77.0)
    img_w = config["camera"]["image_width_px"] if config else 8192
    img_h = config["camera"]["image_height_px"] if config else 5460
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Latitude", "Longitude", "Confidence", "Parent_Image", "Thumbnail"])
            
    candidate_id = 1
    
    while True:
        try:
            payload = verified_queue.get(timeout=3)
        except Empty:
            continue
            
        if payload == "SHUTDOWN_COMMAND":
            logger.info("Cartographer received shutdown command. Finalizing KML and shutting down.")
            generate_kml(csv_path, kml_path)
            break
            
        try:
            telemetry = payload.get("telemetry")
            drone_lat = telemetry.get("lat")
            drone_lon = telemetry.get("lon")
            drone_alt = telemetry.get("alt")
            drone_heading = math.radians(telemetry.get("heading", 0.0))
            
            ground_width_m = 2.0 * drone_alt * math.tan(fov_h / 2.0)
            gsd = ground_width_m / img_w
            
            tile_height, tile_width = payload.get("tile_data").shape[:2]
            
            hit_px_x = float(payload.get("offset_x")) + (float(tile_width) / 2.0)
            hit_px_y = float(payload.get("offset_y")) + (float(tile_height) / 2.0)
            
            center_x = img_w / 2.0
            center_y = img_h / 2.0
            
            delta_px_x = hit_px_x - center_x
            delta_px_y = center_y - hit_px_y 
            
            dx_m = delta_px_x * gsd
            dy_m = delta_px_y * gsd
            
            dx_east = dx_m * math.cos(drone_heading) - dy_m * math.sin(drone_heading)
            dy_north = dx_m * math.sin(drone_heading) + dy_m * math.cos(drone_heading)
            
            delta_lat = (dy_north / R_EARTH) * (180.0 / math.pi)
            delta_lon = (dx_east / (R_EARTH * math.cos(math.pi * drone_lat / 180.0))) * (180.0 / math.pi)
            
            final_lat = drone_lat + delta_lat
            final_lon = drone_lon + delta_lon
            
            parent_name = os.path.basename(payload.get("parent_image"))
            thumb_name = f"candidate_{candidate_id:04d}_{parent_name}"
            thumb_path = os.path.join(thumb_dir, thumb_name)
            
            safe_tile = np.ascontiguousarray(payload.get("tile_data"))
            cv2.imwrite(thumb_path, safe_tile)
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    candidate_id, 
                    f"{final_lat:.7f}", 
                    f"{final_lon:.7f}", 
                    f"{payload.get('confidence'):.3f}", 
                    parent_name, 
                    thumb_path
                ])
                
            logger.info(f"Mapped Candidate {candidate_id} -> Lat: {final_lat:.7f}, Lon: {final_lon:.7f}")
            candidate_id += 1
            
        except Exception as e:
            logger.error(f"[NON-FATAL ERROR] Cartographer choked on candidate {candidate_id}: {e}")
            logger.error(traceback.format_exc())
