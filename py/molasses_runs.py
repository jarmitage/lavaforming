"""
A model workflow for MOLASSES, the lava flow simulator

MOLASSES is designed to give a flow model given a set of eruption source parameter.

Three key numerical inputs must be entered  
1. The lava flow residual thickness, or the thickness at rest (in meters)
2. The total volume of the effusive eruption (in $m^3$)
3. the lava flow of each pulse during the eruption (in $m^3$), which will be some fraction of the total volume.

Additionally, A DEM (Digital Elevation Model) is required for proper simulation.

Finally, an event file should be supplied.

Optionally, you can set more runs of the simulation to test/include minor variations. Keep in mind, each run will add a significant amount of runtime to the simulation.
"""

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
import shutil
import fire  # Add fire import
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/jovyan/shared/Libraries/")
import victor

def create_output_directory(dem_file, events):
    """
    Creates a timestamped output directory for the simulation run.

    Args:
        dem_file (str): The base name of the DEM file.
        events (str): The event coordinates string.

    Returns:
        str: The path to the created output directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_name = f"{dem_file}_{events[0:13].replace(',', '_')}"
    output_dir = f"{timestamp}_{base_output_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir


def create_events_file(events, output_dir):
    """
    Creates the events.in file in the specified output directory.

    Args:
        events (str): The event coordinates string (comma-separated).
        output_dir (str): The path to the output directory.

    Returns:
        str: The path to the created events.in file.
    """
    coordinates = events.replace(',', ' ')
    coordinates = coordinates.split(" ")
    coordinates = [int(coords) for coords in coordinates]
    coordinates = np.asarray(np.reshape(coordinates,(int(len(coordinates)/2),2)))
    events_file_path = os.path.join(output_dir, "events.in")
    with open(events_file_path, 'w') as events_file:
        # Writing the original events string, as the file format seems to expect comma-separated
        events_file.write(events) 
    return events_file_path


def generate_nonlinear_volume_range(min_volume, max_volume, num_steps, power=2.0):
    """
    Generates a range of volumes spaced non-linearly on a logarithmic scale.

    Args:
        min_volume (float): Minimum total volume (m³).
        max_volume (float): Maximum total volume (m³).
        num_steps (int): Number of steps in the volume range.
        power (float, optional): Controls the non-linearity of the spacing. 
                                 Must be > 1. Defaults to 2.0.

    Returns:
        np.ndarray: Array of volumes.
    """
    log_min = np.log10(min_volume)
    log_max = np.log10(max_volume)
    linear_space = np.linspace(0, 1, num_steps)
    nonlinear_space = linear_space**power
    log_volumes = log_min + (log_max - log_min) * nonlinear_space
    return 10**log_volumes


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


def run_molasses_simulations(volume_range, output_dir, parents, elevation_uncert, 
                             residual, pulse_volume, runs, dem, events_file_path, 
                             dem_file, events):
    """
    Runs Molasses simulations for a given range of volumes.

    Args:
        volume_range (np.ndarray): Array of total volumes to simulate.
        output_dir (str): Path to the output directory.
        parents (int): Number of parent cells for lava distribution.
        elevation_uncert (float): Elevation uncertainty value.
        residual (float): Residual lava thickness.
        pulse_volume (float): Volume of each pulse.
        runs (list or int): Number of runs per simulation configuration.
        dem (str): Path to the DEM file.
        events_file_path (str): Path to the events input file.
        dem_file (str): Base name of the DEM file.
        events (str): Event coordinates string.

    Returns:
        pd.DataFrame: DataFrame containing the results of the simulations.
    """
    results = []
    # Determine the maximum volume from the input range for display purposes
    max_vol_for_display = volume_range[-1] if len(volume_range) > 0 else 0 
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
        print(f"Running Molasses for volume: {total_volume}m^3 / {max_vol_for_display}m^3")
        subprocess.run(f"./molasses_2022 {config_path}", shell=True, stderr=subprocess.DEVNULL)
        
        # Process and save the output
        dem_asc = f"{dem_file}_{events[0:13].replace(',', '_')}_{str(int(total_volume)).zfill(10)}"
        output_path = os.path.join(output_dir, dem_asc) # This is just the base name for conversion
        
        # Converts molasses data from CSV to raster and save as .asc file in the output directory
        # The convert_molasses function implicitly reads './flow_-0'
        convert_molasses(f"{output_dir}/{dem_asc}", resolution=2)
        
        # Store result info
        results.append({
            'volume': total_volume,
            'output_file': f"{output_path}.asc" # Storing the actual expected output file path
        })
        print(f"Completed Molasses simulation for volume: {total_volume}m^3")

    results_df = pd.DataFrame(results)
    print(f"Completed {len(results_df)} Molasses simulations with varying volumes")
    return results_df


def finalize_results(results_df, output_dir):
    """
    Saves the results DataFrame to a CSV file and zips the output directory.

    Args:
        results_df (pd.DataFrame): DataFrame containing the simulation results.
        output_dir (str): Path to the output directory.

    Returns:
        str: The path to the created zip file, or None if zipping failed.
    """
    # Save results CSV to the output directory
    results_csv_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    # --- Zip the output directory ---
    output_folder_name = os.path.basename(output_dir)
    parent_dir = os.path.dirname(output_dir)
    # Ensure parent_dir is handled correctly if output_dir is in the current directory
    if not parent_dir:
        parent_dir = '.'
    zip_base_name = os.path.join(parent_dir, output_folder_name) # Path without extension
    zip_file_path = f"{zip_base_name}.zip"

    print(f"Zipping output folder '{output_folder_name}' to {zip_file_path} ...")
    try:
        shutil.make_archive(zip_base_name, 'zip', root_dir=parent_dir, base_dir=output_folder_name)
        print(f"Successfully created zip file: {zip_file_path}")
        return zip_file_path
    except Exception as e:
        print(f"Error creating zip file: {e}")
        return None


def main(
    dem_dir: str = "../../dem/",
    dem_file_base: str = "eldvorp_md_2x2_trench_sketch_1",
    dem_ext: str = ".asc",
    events: str = "326746,376249",
    residual: float = 1.0,
    min_volume: float = 1e4,
    max_volume: float = 1e9,
    num_steps: int = 200,
    pulse_volume: float = 1e1,
    parents: int = 1,
    elevation_uncert: float = 0.5,
    volume_exponent: float = 1.2,
    runs: int = 1
):
    """
    Runs the Molasses simulation workflow.

    This script sets up and executes a series of MOLASSES simulations across a
    range of total eruption volumes, processing the outputs into raster files.
    It utilizes the `fire` library to expose command-line arguments for customization.

    Key Steps:
    1. Generates a non-linear range of total eruption volumes.
    2. Creates a timestamped output directory for each run.
    3. Creates an `events.in` file based on provided coordinates.
    4. For each volume in the range:
        a. Creates a `custom_molasses.conf` file with simulation parameters.
        b. Executes the `molasses_2022` simulation.
        c. Converts the raw output (`flow_-0`) to a raster file (.asc or .tif).
    5. Saves a summary of results to `results.csv` in the output directory.
    6. Zips the entire output directory.

    Args:
        dem_dir (str): Directory containing the DEM file.
        dem_file_base (str): Base name of the DEM file (without extension).
        dem_ext (str): Extension of the DEM file (e.g., '.asc', '.tif').
        events (str): Event location(s) as comma-separated UTM coordinates (e.g., "X1,Y1,X2,Y2").
        residual (float): Input residual lava thickness (meters).
        min_volume (float): Minimum total eruption volume (m³).
        max_volume (float): Maximum total eruption volume (m³).
        num_steps (int): Number of volume steps for the simulation range.
        pulse_volume (float): Volume of each lava pulse (m³).
        parents (int): Number of "parent" cells distributing lava.
        elevation_uncert (float): Elevation uncertainty (meters) to account for micro-topography.
        volume_exponent (float): Exponent (>1) controlling non-linearity of volume range spacing.
        runs (int): Number of simulation runs per volume configuration.
    """
    # Construct full DEM path and base name without extension
    dem = f"{dem_dir}{dem_file_base}{dem_ext}"
    dem_file = dem_file_base

    # Validate DEM file existence
    if not os.path.exists(dem):
        print(f"Error: DEM file not found at {dem}")
        sys.exit(1)

    volume_range = generate_nonlinear_volume_range(min_volume, max_volume, num_steps, volume_exponent)
    np.set_printoptions(suppress=True, precision=2, threshold=sys.maxsize)
    print(f"Generated Volume range: {volume_range}")

    output_dir = create_output_directory(dem_file, events)

    events_file_path = create_events_file(events, output_dir)

    # Run the model for each volume in the range with progress bar
    results_df = run_molasses_simulations(
        volume_range=volume_range, 
        output_dir=output_dir, 
        parents=parents, 
        elevation_uncert=elevation_uncert, 
        residual=residual, 
        pulse_volume=pulse_volume, 
        runs=runs,  # Pass runs directly
        dem=dem, 
        events_file_path=events_file_path, 
        dem_file=dem_file, 
        events=events
    )
    print("Simulation Results Summary:")
    print(results_df)

    # Finalize results (save CSV, zip output)
    finalize_results(results_df, output_dir)

if __name__ == "__main__":
    fire.Fire(main) # Use fire to handle CLI arguments

