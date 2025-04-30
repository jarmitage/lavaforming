"""
Optimized MOLASSES Lava Flow Simulation Runner

This script runs MOLASSES simulations with predefined volume steps.
Uses Google's Fire CLI library for command line interface.
"""

import os
import numpy as np
import subprocess
import pandas as pd
import warnings
import sys
import datetime
import shutil
import time
import fire
from tqdm import tqdm
import matplotlib.pyplot as plt
import threading
sys.path.insert(0, "/home/jovyan/shared/Libraries/")
import victor

MODE_INITIAL = [1e4, 1e5, 2.5e5, 5e5, 7.5e5, 1e6]
MODE_PRIMARY = [5e5, 1e6, 2.5e6]
MODE_PRODUCTION = [5e6, 1e7, 2.5e7]
MODE_EXTENDED = [5e7, 1e8, 5e8]

def setup_environment():
    """Configure environment and suppress warnings."""
    warnings.filterwarnings("ignore")
    print("Environment configured.")

def parse_coordinates(events):
    """
    Parses the event coordinates string into a numpy array.

    Args:
        events (str): Comma-separated event coordinates string.

    Returns:
        np.ndarray: Array of coordinates, or an empty array if parsing fails.
    """
    try:
        coordinates = events.replace(',', ' ')
        coordinates = coordinates.split(" ")
        coordinates = [int(coords) for coords in coordinates]
        coordinates = np.asarray(np.reshape(coordinates,(int(len(coordinates)/2),2)))
    except (ValueError, TypeError) as e:
        print(f"Error parsing event coordinates '{events}': {e}")
        print("Expected comma-separated numbers (e.g., 'x1,y1,x2,y2,...').")
        coordinates = np.array([]) # Assign empty array if parsing fails
    return coordinates

def create_output_directory(dem_file, events):
    """
    Creates timestamped output directory for simulation results.
    
    Args:
        dem_file (str): Base name of DEM file
        events (str): Event coordinates string
        
    Returns:
        str: Path to created output directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{dem_file}_{events[0:13].replace(',', '_')}"
    output_dir = f"{timestamp}_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir

def create_events_file(events, output_dir):
    """
    Creates events.in file with vent coordinates.
    
    Args:
        events (str): Comma-separated event coordinates
        output_dir (str): Output directory path
        
    Returns:
        str: Path to created events file
    """
    events_file_path = os.path.join(output_dir, "events.in")
    with open(events_file_path, 'w') as events_file:
        events_file.write(events)
    print(f"Created events file at {events_file_path}")
    return events_file_path

def create_config_file(output_dir, parents, elevation_uncert, residual, 
                      total_volume, pulse_volume, runs, dem, events_file_path):
    """
    Creates MOLASSES configuration file for a simulation.
    
    Args:
        output_dir (str): Output directory path
        parents (int): Number of parent cells
        elevation_uncert (float): Elevation uncertainty
        residual (float): Residual lava thickness
        total_volume (float): Total eruption volume
        pulse_volume (float): Volume per pulse
        runs (int): Number of simulation runs
        dem (str): Path to DEM file
        events_file_path (str): Path to events file
        
    Returns:
        str: Path to created config file
    """
    config_path = os.path.join(output_dir, "molasses.conf")
    with open(config_path, "w") as f:
        f.write(f"PARENTS = {parents}\n")
        f.write(f"ELEVATION_UNCERT = {elevation_uncert}\n")
        f.write(f"MIN_RESIDUAL = {residual}\n")
        f.write(f"MAX_RESIDUAL = {residual}\n")
        f.write(f"MIN_TOTAL_VOLUME = {total_volume}\n")
        f.write(f"MAX_TOTAL_VOLUME = {total_volume}\n")
        f.write(f"MIN_PULSE_VOLUME = {pulse_volume}\n")
        f.write(f"MAX_PULSE_VOLUME = {pulse_volume}\n")
        f.write(f"RUNS = {runs}\n")
        f.write(f"ASCII_FLOW_MAP = 1\n")
        f.write(f"DEM_FILE = {dem}\n")
        f.write(f"EVENTS_FILE = {events_file_path}\n")
    tqdm.write(f"Created configuration file at {config_path}")
    return config_path

def run_molasses(config_path) -> tuple[bool, float]:
    """
    Executes MOLASSES simulation with the given configuration.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        bool: True if successful, False otherwise
    """
    start_time = time.time()
    tqdm.write(f"Starting MOLASSES simulation with config: {config_path}")
    result = subprocess.run(f"./molasses_2022 {config_path}", 
                          shell=True, 
                          stderr=subprocess.DEVNULL)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        tqdm.write(f"Simulation completed successfully in {elapsed_time:.2f} seconds")
        return True, elapsed_time
    else:
        tqdm.write(f"Simulation failed with return code {result.returncode}")
        return False, elapsed_time

def convert_output(output_dir, dem_file, events, total_volume, resolution=2):
    """
    Converts MOLASSES output to raster format.
    
    Args:
        output_dir (str): Output directory path
        dem_file (str): Base name of DEM file
        events (str): Event coordinates string
        total_volume (float): Total eruption volume
        resolution (int): Raster resolution in meters
        
    Returns:
        str: Path to created raster file
    """
    output_name = f"{dem_file}_{events[0:13].replace(',', '_')}_{str(int(total_volume)).zfill(10)}"
    output_path = os.path.join(output_dir, output_name)
    
    # This is a placeholder - you would replace with actual conversion code
    tqdm.write(f"Converting MOLASSES output to raster at {output_path}.asc")
    
    # Placeholder for conversion code
    victor.convert_molasses(output_path, resolution=resolution)
    
    return f"{output_path}.asc"

def run_volume_simulation(volume, output_dir, dem_file, events, 
                         parents, elevation_uncert, residual, pulse_volume, 
                         runs, dem, events_file_path, resolution):
    """
    Runs a single MOLASSES simulation for a specific volume.
    
    Args:
        volume (float): Total eruption volume
        output_dir (str): Output directory path
        dem_file (str): Base name of DEM file
        events (str): Event coordinates string
        parents (int): Number of parent cells
        elevation_uncert (float): Elevation uncertainty
        residual (float): Residual lava thickness
        pulse_volume (float): Volume per pulse
        runs (int): Number of simulation runs
        dem (str): Path to DEM file
        events_file_path (str): Path to events file
        resolution (int): Raster resolution
        
    Returns:
        dict: Simulation result information
    """
    tqdm.write(f"\nRunning simulation for volume: {volume:.2e} m³")
    
    # Create configuration file
    config_path = create_config_file(
        output_dir, parents, elevation_uncert, residual,
        volume, pulse_volume, runs, dem, events_file_path
    )
    
    # Run MOLASSES
    success, elapsed_time = run_molasses(config_path)
    
    if not success:
        return {
            'volume': volume,
            'success': False,
            'output_file': None,
            'elapsed_time': elapsed_time
        }
    
    # Convert output
    output_path = convert_output(output_dir, dem_file, events, volume, resolution)
    
    return {
        'volume': volume,
        'success': True,
        'output_file': output_path,
        'elapsed_time': elapsed_time
    }

def save_results(results, output_dir):
    """
    Saves simulation results to CSV file.
    
    Args:
        results (list): List of simulation result dictionaries
        output_dir (str): Output directory path
        
    Returns:
        str: Path to results CSV file
    """
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "simulation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
    return results_path

def archive_outputs(output_dir):
    """
    Archives output directory to ZIP file.
    
    Args:
        output_dir (str): Output directory path
        
    Returns:
        str: Path to created ZIP file
    """
    parent_dir = os.path.dirname(output_dir) or '.'
    base_name = os.path.basename(output_dir)
    zip_base = os.path.join(parent_dir, base_name)
    zip_path = f"{zip_base}.zip"
    
    print(f"Creating archive at {zip_path}")
    try:
        shutil.make_archive(zip_base, 'zip', parent_dir, base_name)
        print(f"Archive created successfully")
        return zip_path
    except Exception as e:
        print(f"Failed to create archive: {e}")
        return None

def determine_volume_list(volumes: str | None, mode: str | None) -> list[float]:
    """
    Determines the list of volumes to simulate based on input parameters.

    Args:
        volumes (str | None): Comma-separated string of volumes or None.
        mode (str | None): Predefined mode ('initial', 'primary', 'production', 'extended') or None.

    Returns:
        list[float]: The list of volumes to simulate.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    if volumes:
        try:
            volume_list = [float(v) for v in volumes.split(',')]
        except ValueError as e:
            raise ValueError(f"Error parsing volumes: {e}.\nVolumes must be a comma-separated list of numbers (e.g., '1e4,5e4,1e5'), received: {volumes}.")
    else:
        if mode is not None:
            if mode == "initial":
                volume_list = MODE_INITIAL
            elif mode == "primary":
                volume_list = MODE_PRIMARY
            elif mode == "production":
                volume_list = MODE_PRODUCTION
            elif mode == "extended":
                volume_list = MODE_EXTENDED
            else:
                raise ValueError(f"Invalid mode: {mode}. Valid modes are 'initial', 'primary', 'production', or 'extended'")
            print(f"Running in {mode} mode.")
        else:
            # Default sequence if neither volumes nor mode is specified
            volume_list = [
                1e4, 5e4, 1e5, 2.5e5,  # Initial validation
                5e5, 1e6, 2.5e6,       # Primary validation
                5e6, 1e7, 2.5e7,       # Production simulation
                5e7, 1e8, 5e8          # Extended simulation
            ]
    print(f"Volume list: {volume_list}")
    return volume_list

def run_simulations(
    volume_list: list[float],
    output_dir: str,
    dem_file: str,
    events: str,
    parents: int,
    elevation_uncert: float,
    residual: float,
    pulse_volume: float,
    runs: int,
    dem: str,
    events_file_path: str,
    resolution: int,
    coordinates: np.ndarray,
    plot: bool
):
    """
    Runs the simulation loop for all volumes, saves results, and archives outputs.

    Args:
        volume_list (list[float]): List of volumes to simulate.
        output_dir (str): Output directory path.
        dem_file (str): Base name of DEM file.
        events (str): Event coordinates string.
        parents (int): Number of parent cells.
        elevation_uncert (float): Elevation uncertainty.
        residual (float): Residual lava thickness.
        pulse_volume (float): Volume per pulse.
        runs (int): Number of simulation runs.
        dem (str): Path to DEM file.
        events_file_path (str): Path to events file.
        resolution (int): Raster resolution.
        coordinates (np.ndarray): Array of event coordinates.
        plot (bool): Whether to generate plots for each simulation step.
    """
    results = []
    plot_threads = [] # Keep track of plotting threads
    for volume in tqdm(volume_list, desc="Running simulations"):
        result = run_volume_simulation(
            volume, output_dir, dem_file, events,
            parents, elevation_uncert, residual, pulse_volume,
            runs, dem, events_file_path, resolution
        )
        results.append(result)

        # Start plotting in a separate thread
        if plot and result['success'] and result['output_file']:
            output_asc = result['output_file']
            # Create and start a new thread for plotting
            plot_thread = threading.Thread(
                target=plot_and_save_flow,
                args=(dem, output_asc, coordinates, volume),
                daemon=True # Allows main program to exit even if threads are running (optional, but can be useful)
            )
            plot_thread.start()
            plot_threads.append(plot_thread) # Add thread to list

    # Wait for all plotting threads to complete before proceeding
    if plot_threads:
        tqdm.write("Waiting for plot generation to complete...")
        for thread in plot_threads:
             thread.join()
        tqdm.write("All plots generated.")

    print("\nAll simulations completed!")
    return results
    

def plot_and_save_flow(dem, output_asc, coordinates, volume):
    """
    Generates and saves a plot for a single simulation result.

    Args:
        dem (str): Path to the DEM file.
        output_asc (str): Path to the output ASCII flow file.
        coordinates (np.ndarray): Array of event coordinates.
        volume (float): The simulation volume.
    """
    try:
        # Ensure we have coordinates to plot
        tqdm.write(f"Plotting volume: {volume:.2e} m³")
        if coordinates.size > 0:
            fig, ax = plt.subplots() # Create figure and axes
            plot_title = f"Volume: {volume:.2e} m³"
            victor.plot_flow(dem, output_asc, axes=ax, coords=coordinates, zoom=True, title=plot_title)
            
            # Construct plot filename (same base name as asc, but with .png)
            plot_filename = os.path.splitext(output_asc)[0] + ".png"
            plt.savefig(plot_filename)
            tqdm.write(f"Saved plot to {plot_filename}")
            plt.close(fig) # Close the figure to free memory
        else:
            tqdm.write("Skipping plot due to missing or invalid coordinates.")
    except Exception as e:
        tqdm.write(f"Error generating plot for volume {volume}: {e}")

def setup_simulation(dem_dir, dem_file, dem_ext, events, volumes, mode):
    """
    Sets up the simulation environment and parameters.

    Args:
        dem_dir (str): Directory containing DEM file.
        dem_file (str): Base name of DEM file.
        dem_ext (str): DEM file extension.
        events (str): Comma-separated event coordinates.
        volumes (str | None): Comma-separated string of volumes or None.
        mode (str | None): Predefined mode or None.

    Returns:
        tuple: Contains dem path, volume list, output directory,
               events file path, and coordinates array. Returns None if DEM not found.
    """
    setup_environment()
    dem = f"{dem_dir}{dem_file}{dem_ext}"
    if not os.path.exists(dem):
        print(f"Error: DEM file not found at {dem}")
        return None  # Indicate failure
    
    try:
        volume_list = determine_volume_list(volumes, mode)
    except ValueError as e:
        print(f"Error determining volumes: {e}")
        return None # Indicate failure

    output_dir = create_output_directory(dem_file, events)
    events_file_path = create_events_file(events, output_dir)
    coordinates = parse_coordinates(events)

    return dem, volume_list, output_dir, events_file_path, coordinates

def process_results(results, output_dir) -> bool:
    """
    Saves simulation results and archives the output directory.

    Args:
        results (list): List of simulation result dictionaries.
        output_dir (str): Output directory path.
    """
    try:
        save_results(results, output_dir)
        archive_outputs(output_dir)
        return True
    except Exception as e:
        print(f"Error processing results: {e}")
        return False

def main(
    dem_dir: str = "../../dem/",
    dem_file: str = None,
    dem_ext: str = ".asc",
    events: str = "326746,376249",
    residual: float = 1.0,
    pulse_volume: float = 1e1,
    parents: int = 1,
    elevation_uncert: float = 0.5,
    runs: int = 1,
    resolution: int = 10,
    volumes: str | None = None,
    mode: str | None = 'primary',
    plot: bool = True
):
    """
    Runs MOLASSES simulations for a sequence of volumes.
    
    Args:
        dem_dir (str): Directory containing DEM file
        dem_file (str): Base name of DEM file without extension
        dem_ext (str): DEM file extension
        events (str): Comma-separated event coordinates
        residual (float): Residual lava thickness (meters)
        pulse_volume (float): Volume of each lava pulse (m³)
        parents (int): Number of parent cells
        elevation_uncert (float): Elevation uncertainty (meters)
        runs (int): Number of simulation runs per volume
        resolution (int): Output raster resolution (meters)
        volumes (str): Comma-separated list of volumes to simulate
                     (in scientific notation, e.g., "1e4,5e4,1e5")
        mode (str): Volume mode to run the script in (initial, primary, production, extended)
        plot (bool): Whether to generate plots for each simulation step. Defaults to True.
    """
    setup_result = setup_simulation(dem_dir, dem_file, dem_ext, events, volumes, mode)
    if setup_result is None:
        return
    dem, volume_list, output_dir, events_file_path, coordinates = setup_result
    
    results = run_simulations(
        volume_list,
        output_dir,
        dem_file,
        events,
        parents,
        elevation_uncert,
        residual,
        pulse_volume,
        runs,
        dem,
        events_file_path,
        resolution,
        coordinates,
        plot
    )

    success = process_results(results, output_dir)
    if not success:
        print("Error processing results. Exiting.")
        return

if __name__ == "__main__":
    fire.Fire(main)
