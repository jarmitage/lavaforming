# <h1>A model workflow for MOLASSES, the lava flow simulator </h1>
# 
# MOLASSES is designed to give a flow model given a set of eruption source parameter.
# 
# Three key numerical inputs must be entered  
# 1. The lava flow residual thickness, or the thickness at rest (in meters)
# 2. The total volume of the effusive eruption (in $m^3$)
# 3. the lava flow of each pulse during the eruption (in $m^3$), which will be some fraction of the total volume.
# 
# Additionally, A DEM (Digital Elevation Model) is required for proper simulation.
# 
# Finally, an event file should be supplied.
# 
# Optionally, you can set more runs of the simulation to test/include minor variations. Keep in mind, each run will add a significant amount of runtime to the simulation.

import os
import numpy as np
import subprocess
import pandas as pd
import warnings
import sys
from tqdm import tqdm
import datetime
import geopandas as gpd
from geocube.api.core import make_geocube
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# %matplotlib inline  # This is a Jupyter notebook command, commented out for script compatibility
sys.path.insert(0, "/home/jovyan/shared/Libraries/")
import victor

# Key Parameters: Here, please input your preferred residual thickness, 
# total volume range, and pulse volume, along with the DEM filepath and events in UTM coordinates
residual = 1 #Input residual thickness (meters)

# Volume range parameters
min_volume = 1e4  # Minimum total volume (m³)
max_volume = 1e9  # Maximum total volume (m³)
num_steps = 100  # Number of steps in the volume range
volume_range = np.logspace(np.log10(min_volume), np.log10(max_volume), num_steps)
print(f"Volume range: {volume_range}")

pulse_volume = 1e1 #input pulse volume (meters, standard or scientific notation both acceptable)

#input DEM file (string, relative path to file)
dem_dir = "../../dem/"
dem_file = "eldvorp_md_2x2_trench_sketch_1"
dem = f"{dem_dir}{dem_file}.asc"

events = "326746,376249" #input UTM location of event location
parents = 1 # how many "parent" cells can distribute lava to neighboring cells
elevation_uncert = 0.5 # A small amount of elevation uncertainty can help the model capture micro-topographic features that influence flow

#OPTIONAL (default 1), add runs (integer)
runs = []

# Create output directory with datetime stamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_name = f"{dem_file}_{events.replace(',', '_')}"
output_dir = f"{timestamp}_{base_output_name}"
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

# Create events file in the output directory
coordinates = events.replace(',', ' ')
coordinates = coordinates.split(" ")
coordinates = [int(coords) for coords in coordinates]
coordinates = np.asarray(np.reshape(coordinates,(int(len(coordinates)/2),2)))
events_file_path = os.path.join(output_dir, "events.in")
with open(events_file_path, 'w') as events_file:
    events_file.write(events)

results = []

def convert_molasses(output_name, resolution=20, output_format='tif'):
    """
    Converts molasses data from a CSV file to a raster format using GeoPandas and rasterio.

    Parameters
    ----------
    output_name : str
        The name of the output raster file without the file extension.
    resolution : float, optional
        The resolution of the output raster in meters, by default 20
    output_format : str, optional
        The output format, either 'asc' or 'tif', by default 'asc'

    Returns
    -------
    None
    """
    colnames=['EAST', 'NORTH', 'THICKNESS', 'NEW_ELEV', 'ORIG_ELEV'] 
    lava = pd.read_csv('./flow_-0', skiprows=3, names=colnames, sep='\s+', header= None)
    lava.to_csv("flow.csv",header=colnames,index=False)
    df = gpd.pd.read_csv('flow.csv')
    gf = gpd.GeoDataFrame(df, 
                          geometry=gpd.points_from_xy(df.EAST, df.NORTH), 
                          crs=4326)
    geo_grid = make_geocube(vector_data=gf, measurements=['THICKNESS'], resolution=resolution)
    geo_grid = geo_grid.THICKNESS.rio.write_nodata(-9999)
    geo_grid = geo_grid.fillna(-9999)
    
    if output_format.lower() == 'tif':
        output_file = f"{output_name}.tif"
    else:
        output_file = f"{output_name}.asc"
    
    geo_grid.rio.to_raster(output_file)

# Run the model for each volume in the range with progress bar
for total_volume in tqdm(volume_range, desc="Running Molasses simulations"):
    # Create configuration file for this volume in the output directory
    config_path = os.path.join(output_dir, "custom_molasses.conf")
    with open(config_path, "w") as f:
        print(f"PARENTS = {parents}", file=f)
        print(f"ELEVATION_UNCERT = {elevation_uncert}", file=f)
        print(f"MIN_RESIDUAL = {residual}", file=f)
        print(f"MAX_RESIDUAL = {residual}", file=f)
        print(f"MIN_TOTAL_VOLUME = {total_volume}", file=f)
        print(f"MAX_TOTAL_VOLUME = {total_volume}", file=f)
        print(f"MIN_PULSE_VOLUME = {pulse_volume}", file=f)
        print(f"MAX_PULSE_VOLUME = {pulse_volume}", file=f)
        print(f"RUNS = {1 if not runs else runs}", file=f)
        print(f"ASCII_FLOW_MAP = 1", file=f)
        print(f"DEM_FILE = {dem}", file=f)
        print(f"EVENTS_FILE = {events_file_path}", file=f)
    
    # Run the model
    print(f"Running Molasses for volume: {total_volume}m^3 / {max_volume}m^3")
    subprocess.run(f"./molasses_2022 {config_path}", shell=True, stderr=subprocess.DEVNULL)
    
    # Process and save the output
    dem_asc = f"{dem_file}_{events.replace(',', '_')}_{str(int(total_volume))}"
    output_path = os.path.join(output_dir, dem_asc)
    
    # Converts molasses data from CSV to raster and save as .asc file in the output directory
    convert_molasses(f"{output_dir}/{dem_asc}", resolution=2)
    
    # Store result info
    results.append({
        'volume': total_volume,
        'output_file': output_path
    })
    print(f"Completed Molasses simulation for volume: {total_volume}m^3")

# Create a DataFrame with the results
results_df = pd.DataFrame(results)
print(f"Completed {len(results_df)} Molasses simulations with varying volumes")
print(results_df)

# Save results CSV to the output directory
results_csv_path = os.path.join(output_dir, "results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to {results_csv_path}")

