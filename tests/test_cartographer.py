import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def calculate_offsets(dx_m, dy_m, heading_deg):
    drone_heading = math.radians(heading_deg)
    # Testing the fixed clockwise rotation matrix applied in cartographer
    dx_east = dx_m * math.cos(drone_heading) + dy_m * math.sin(drone_heading)
    dy_north = -dx_m * math.sin(drone_heading) + dy_m * math.cos(drone_heading)
    return dx_east, dy_north

def test_heading_north_0_deg():
    # Facing North (0 deg)
    dx_east, dy_north = calculate_offsets(0, 10, 0)
    assert round(dx_east, 3) == 0.0
    assert round(dy_north, 3) == 10.0
    
    dx_east, dy_north = calculate_offsets(10, 0, 0)
    assert round(dx_east, 3) == 10.0
    assert round(dy_north, 3) == 0.0

def test_heading_east_90_deg():
    # Facing East (90 deg)
    # Forward (dy_m > 0) maps to East (+X)
    dx_east, dy_north = calculate_offsets(0, 10, 90)
    assert round(dx_east, 3) == 10.0
    assert round(dy_north, 3) == 0.0
    
    # Right wing (dx_m > 0) maps to South (-Y)
    dx_east, dy_north = calculate_offsets(10, 0, 90)
    assert round(dx_east, 3) == 0.0
    assert round(dy_north, 3) == -10.0

def test_heading_south_180_deg():
    # Facing South (180 deg)
    # Forward maps to South (-Y)
    dx_east, dy_north = calculate_offsets(0, 10, 180)
    assert round(dx_east, 3) == 0.0
    assert round(dy_north, 3) == -10.0
    
    # Right wing maps to West (-X)
    dx_east, dy_north = calculate_offsets(10, 0, 180)
    assert round(dx_east, 3) == -10.0
    assert round(dy_north, 3) == 0.0

def test_heading_west_270_deg():
    # Facing West (270 deg)
    # Forward maps to West (-X)
    dx_east, dy_north = calculate_offsets(0, 10, 270)
    assert round(dx_east, 3) == -10.0
    assert round(dy_north, 3) == 0.0
    
    # Right wing maps to North (+Y)
    dx_east, dy_north = calculate_offsets(10, 0, 270)
    assert round(dx_east, 3) == 0.0
    assert round(dy_north, 3) == 10.0
