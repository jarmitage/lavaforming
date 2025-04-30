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
import re # Added for parsing volume directories
from tqdm import tqdm
import matplotlib.pyplot as plt
import threading
import geopandas as gpd
from geocube.api.core import make_geocube
from typing import Optional, List, Dict, Tuple, Any, Union
# sys.path.insert(0, "/home/jovyan/shared/Libraries/")
# import victor

MODE_INITIAL = [1e4, 1e5, 2.5e5, 5e5, 7.5e5, 1e6]
MODE_PRIMARY = [5e5, 1e6, 2.5e6]
MODE_PRODUCTION = [5e6, 1e7, 2.5e7]
MODE_EXTENDED = [5e7, 1e8, 5e8]

def setup_environment() -> None:
    """Configure environment and suppress warnings."""
    warnings.filterwarnings("ignore")
    print("[setup_environment] Environment configured.")

def load_simulation_matrix(matrix_file: str) -> Optional[pd.DataFrame]:
    """
    Loads a simulation matrix from a CSV file defining DEM-Event pairs to test.
    
    Args:
        matrix_file (str): Path to CSV file with columns:
            - dem_file: Base name of DEM file
            - events: Comma-separated event coordinates
            - trench_id: Identifier for the trench design
            
    Returns:
        pd.DataFrame: DataFrame containing simulation matrix
    """
    try:
        matrix = pd.read_csv(matrix_file)
        required_columns = ['dem_file', 'events', 'trench_id']
        missing_columns = [col for col in required_columns if col not in matrix.columns]
        
        if missing_columns:
            print(f"[load_simulation_matrix] Error: Missing required columns: {', '.join(missing_columns)}")
            return None
            
        # Add default extensions if not provided
        if 'dem_ext' not in matrix.columns:
            matrix['dem_ext'] = '.asc'
            
        print(f"[load_simulation_matrix] Loaded {len(matrix)} DEM-Event pairs")
        return matrix
    except Exception as e:
        print(f"[load_simulation_matrix] Error loading simulation matrix: {e}")
        return None

def parse_coordinates(events: str) -> np.ndarray:
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
        print(f"[parse_coordinates] Error parsing event coordinates '{events}': {e}")
        print("[parse_coordinates] Expected comma-separated numbers (e.g., 'x1,y1,x2,y2,...').")
        coordinates = np.array([]) # Assign empty array if parsing fails
    return coordinates

def create_output_directory(dem_file: str, events: str) -> str:
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
    print(f"[create_output_directory] Created output directory: {output_dir}")
    return output_dir

def create_events_file(events: str, events_file_path: str) -> Optional[str]:
    """
    Creates events.in file with vent coordinates at the specified path.
    
    Args:
        events (str): Comma-separated event coordinates
        events_file_path (str): Full path to the events file to create
        
    Returns:
        str: Path to created events file (same as input)
    """
    try:
        with open(events_file_path, 'w') as events_file:
            events_file.write(events)
        tqdm.write(f"[create_events_file] Created events file at {events_file_path}")
        return events_file_path
    except Exception as e:
        tqdm.write(f"[create_events_file] Error creating events file {events_file_path}: {e}")
        return None

def create_config_file(
    config_path: str,
    parents: int,
    elevation_uncert: float,
    residual: float,
    total_volume: float,
    pulse_volume: float,
    runs: int,
    dem_abs_path: str,
    events_abs_path: str
) -> Optional[str]:
    """
    Creates MOLASSES configuration file for a simulation at the specified path.
    Ensures DEM_FILE and EVENTS_FILE use absolute paths.
    
    Args:
        config_path (str): Full path to the config file to create
        parents (int): Number of parent cells
        elevation_uncert (float): Elevation uncertainty
        residual (float): Residual lava thickness
        total_volume (float): Total eruption volume
        pulse_volume (float): Volume per pulse
        runs (int): Number of simulation runs
        dem_abs_path (str): Absolute path to DEM file
        events_abs_path (str): Absolute path to events file
        
    Returns:
        str: Path to created config file (same as input)
    """
    try:
        with open(config_path, "w") as f:
            f.write(f"PARENTS = {parents}\\n")
            f.write(f"ELEVATION_UNCERT = {elevation_uncert}\\n")
            f.write(f"MIN_RESIDUAL = {residual}\\n")
            f.write(f"MAX_RESIDUAL = {residual}\\n")
            f.write(f"MIN_TOTAL_VOLUME = {total_volume}\\n")
            f.write(f"MAX_TOTAL_VOLUME = {total_volume}\\n")
            f.write(f"MIN_PULSE_VOLUME = {pulse_volume}\\n")
            f.write(f"MAX_PULSE_VOLUME = {pulse_volume}\\n")
            f.write(f"RUNS = {runs}\\n")
            f.write(f"ASCII_FLOW_MAP = 1\\n")
            # Use absolute paths in config file
            f.write(f"DEM_FILE = {dem_abs_path}\\n")
            f.write(f"EVENTS_FILE = {events_abs_path}\\n")
        tqdm.write(f"[create_config_file] Created configuration file at {config_path}")
        return config_path
    except Exception as e:
        tqdm.write(f"[create_config_file] Error creating config file {config_path}: {e}")
        return None

def run_molasses(config_path: str, run_dir: str) -> tuple[bool, float]:
    """
    Executes MOLASSES simulation using the given config, running in `run_dir`.
    Captures stdout/stderr, timestamps each line, prints to console via tqdm.write,
    and saves to 'molasses_run.log' in run_dir.
    
    Args:
        config_path (str): Absolute path to configuration file
        run_dir (str): Directory to run the simulation in (where flow_-0 will be created)
        
    Returns:
        tuple[bool, float]: (True if successful, False otherwise), elapsed_time
    """
    start_time = time.time()
    # Ensure config_path is absolute for the command, as cwd changes
    abs_config_path = os.path.abspath(config_path)
    molasses_executable = os.path.abspath("./molasses_2022") # Ensure we know where the executable is
    log_file_path = os.path.join(run_dir, "molasses_run.log")

    tqdm.write(f"[run_molasses] Starting MOLASSES simulation.")
    tqdm.write(f"[run_molasses]   Config: {abs_config_path}")
    tqdm.write(f"[run_molasses]   Run directory: {run_dir}")
    tqdm.write(f"[run_molasses]   Executable: {molasses_executable}")
    tqdm.write(f"[run_molasses]   Log file: {log_file_path}")


    if not os.path.exists(molasses_executable):
        tqdm.write(f"[run_molasses] Error: molasses_2022 executable not found at {molasses_executable}")
        return False, 0.0
    if not os.path.exists(abs_config_path):
        tqdm.write(f"[run_molasses] Error: Config file not found at {abs_config_path}")
        return False, 0.0
    if not os.path.isdir(run_dir):
         tqdm.write(f"[run_molasses] Error: Run directory not found at {run_dir}")
         return False, 0.0

    command = f"{molasses_executable} {abs_config_path}"
    process = None # Initialize process variable
    try:
        # Use Popen to stream output
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirect stderr to stdout
            cwd=run_dir,
            text=True,
            bufsize=1 # Line buffered
        )

        tqdm.write(f"[run_molasses] MOLASSES process started (PID: {process.pid}). Logging output:")
        
        with open(log_file_path, 'a') as log_file: # Open log file in append mode
            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                if line: # Avoid processing empty lines if readline returns them at the end
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    timestamped_line = f"[{timestamp}] {line.strip()}"
                    tqdm.write(timestamped_line) # Write to console via tqdm
                    log_file.write(timestamped_line + '\\n') # Write to log file
        
        # Wait for the process to complete and get the return code
        return_code = process.wait()
        elapsed_time = time.time() - start_time

        if return_code == 0:
            tqdm.write(f"[run_molasses] Simulation completed successfully in {elapsed_time:.2f} seconds (in {run_dir})")
            return True, elapsed_time
        else:
            tqdm.write(f"[run_molasses] Simulation failed in {elapsed_time:.2f} seconds (in {run_dir}) with return code {return_code}")
            # Note: Stderr is already captured and logged above if redirected
            return False, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        tqdm.write(f"[run_molasses] Exception during subprocess execution: {e}")
        # Ensure process is terminated if started but failed during setup/streaming
        if process and process.poll() is None:
            process.terminate()
            process.wait() # Wait for termination
        return False, elapsed_time
    finally:
        # Ensure stdout is closed even if errors occur
        if process and process.stdout:
             process.stdout.close()

def convert_molasses(input_flow_file: str, output_raster_path: str, resolution: int = 2) -> Optional[str]:
    """
    Converts molasses data from a CSV file ('flow_-0') to a raster format.

    Args:
        input_flow_file (str): Full path to the input 'flow_-0' file.
        output_raster_path (str): Full path for the output raster file (e.g., '.tif').
        resolution (int, optional): Resolution of the output raster in meters. Defaults to 2.

    Returns:
        str | None: Full path to the created raster file, or None if an error occurred.
    """
    colnames=['EAST', 'NORTH', 'THICKNESS', 'NEW_ELEV', 'ORIG_ELEV']
    temp_csv_path = os.path.join(os.path.dirname(output_raster_path), "temp_flow.csv") # Temp CSV in output dir

    try:
        tqdm.write(f"[convert_molasses] Reading flow file: {input_flow_file}")
        if not os.path.exists(input_flow_file):
            tqdm.write(f"[convert_molasses] Error: Input flow file not found: {input_flow_file}")
            return None
            
        lava = pd.read_csv(input_flow_file, skiprows=3, names=colnames, sep='\\s+', header= None)
        # Write intermediate CSV needed for GeoPandas (consider optimizing if possible)
        tqdm.write(f"[convert_molasses] Writing temporary CSV: {temp_csv_path}")
        lava.to_csv(temp_csv_path, header=colnames, index=False)
        df = gpd.pd.read_csv(temp_csv_path)
    except Exception as e:
        tqdm.write(f"[convert_molasses] Error reading flow file or writing temp CSV: {e}")
        return None
    finally:
        # Clean up temporary CSV file
        if os.path.exists(temp_csv_path):
             try:
                 os.remove(temp_csv_path)
                 tqdm.write(f"[convert_molasses] Removed temporary CSV: {temp_csv_path}")
             except OSError as e:
                 tqdm.write(f"[convert_molasses] Warning: Could not remove temporary CSV {temp_csv_path}: {e}")

    try:
        tqdm.write(f"[convert_molasses] Creating GeoDataFrame.")
        gf = gpd.GeoDataFrame(df, 
                              geometry=gpd.points_from_xy(df.EAST, df.NORTH), 
                              crs=4326) # Assuming WGS84 for MOLASSES output coords
        
        tqdm.write(f"[convert_molasses] Creating GeoCube grid (resolution={resolution}).")
        geo_grid = make_geocube(vector_data=gf, measurements=['THICKNESS'], resolution=resolution)
        geo_grid = geo_grid.THICKNESS.rio.write_nodata(-9999)
        geo_grid = geo_grid.fillna(-9999)
    except Exception as e:
        tqdm.write(f"[convert_molasses] Error creating GeoDataFrame or GeoCube: {e}")
        return None

    try:
        tqdm.write(f"[convert_molasses] Writing raster file: {output_raster_path}")
        geo_grid.rio.to_raster(output_raster_path)
        tqdm.write(f"[convert_molasses] Successfully wrote raster: {output_raster_path}")
        return output_raster_path
    except Exception as e:
        tqdm.write(f"[convert_molasses] Error writing raster file: {e}")
        return None

def run_volume_simulation(
    volume: float,
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
    resolution: int
) -> Dict[str, Any]:
    """
    Runs a single MOLASSES simulation for a specific volume.
    
    NOTE: This function is now primarily used by the SINGLE DEM/EVENT mode.
    The multi-simulation mode incorporates this logic directly into `run_multi_simulation`.
    
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
    tqdm.write(f"\\n[run_volume_simulation] Running simulation for volume: {volume:.2e} m³ in {output_dir}")
    
    # Create configuration file (use absolute paths for safety)
    config_path = os.path.join(output_dir, "molasses.conf")
    abs_dem_path = os.path.abspath(dem)
    abs_events_path = os.path.abspath(events_file_path)
    
    config_path = create_config_file(
        config_path, parents, elevation_uncert, residual,
        volume, pulse_volume, runs, abs_dem_path, abs_events_path
    )
    if not config_path:
        return {'volume': volume, 'success': False, 'output_file': None, 'elapsed_time': 0.0}
    
    # Run MOLASSES in the output_dir
    success, elapsed_time = run_molasses(config_path, output_dir)
    
    if not success:
        return {
            'volume': volume,
            'success': False,
            'output_file': None,
            'elapsed_time': elapsed_time
        }
    
    # Convert output
    # Construct input/output paths for conversion
    input_flow_file = os.path.join(output_dir, 'flow_-0')
    output_raster_base = f"{dem_file}_{events[0:13].replace(',', '_')}_{str(int(volume)).zfill(10)}"
    output_raster_path = os.path.join(output_dir, f"{output_raster_base}.tif") # Defaulting to tif
    
    tqdm.write(f"[run_volume_simulation] Converting MOLASSES output {input_flow_file} to {output_raster_path}")
    output_file_path = convert_molasses(input_flow_file, output_raster_path, resolution=resolution)
    
    if not output_file_path:
         tqdm.write(f"[run_volume_simulation] Conversion failed for volume {volume}")
         # Mark as success=False? Or just no output file? Let's say conversion failure means no output.
         return {
             'volume': volume,
             'success': True, # Simulation itself succeeded
             'output_file': None, # But conversion failed
             'elapsed_time': elapsed_time
         }
         
    return {
        'volume': volume,
        'success': True,
        'output_file': output_file_path, # Use the path returned by convert_molasses
        'elapsed_time': elapsed_time
    }

def save_results(results: List[Dict[str, Any]], output_dir: str) -> str:
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
    print(f"[save_results] Saved results to {results_path}")
    return results_path

def archive_outputs(output_dir: str) -> Optional[str]:
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
    
    print(f"[archive_outputs] Creating archive at {zip_path}")
    try:
        shutil.make_archive(zip_base, 'zip', parent_dir, base_name)
        print(f"[archive_outputs] Archive created successfully")
        return zip_path
    except Exception as e:
        print(f"[archive_outputs] Failed to create archive: {e}")
        return None

def determine_volume_list(volumes: Optional[str], mode: Optional[str]) -> List[float]:
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
            print(f"[determine_volume_list] Running in {mode} mode.")
        else:
            # Default sequence if neither volumes nor mode is specified
            volume_list = [
                1e4, 5e4, 1e5, 2.5e5,  # Initial validation
                5e5, 1e6, 2.5e6,       # Primary validation
                5e6, 1e7, 2.5e7,       # Production simulation
                5e7, 1e8, 5e8          # Extended simulation
            ]
    print(f"[determine_volume_list] Volume list: {volume_list}")
    return volume_list

def run_simulations(
    volume_list: List[float],
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
) -> List[Dict[str, Any]]:
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
        save_results(results, output_dir)

        # Start plotting in a separate thread
        if plot and result['success'] and result['output_file']:
            tqdm.write(f"[run_simulations] Starting plotting thread for volume: {volume:.2e} m³")
            output_raster_path = result['output_file']
            # Construct the corresponding plot filename
            plot_base, _ = os.path.splitext(output_raster_path)
            plot_filename = plot_base + ".png"
            # Create and start a new thread for plotting
            plot_thread = threading.Thread(
                target=plot_and_save_flow,
                args=(dem, output_raster_path, coordinates, volume, plot_filename, result['elapsed_time']),
                daemon=True # Allows main program to exit even if threads are running (optional, but can be useful)
            )
            plot_thread.start()
            plot_threads.append(plot_thread) # Add thread to list
            tqdm.write(f"[run_simulations] Plotting threads: {len(plot_threads)}")

    # Wait for all plotting threads to complete before proceeding
    if plot_threads:
        tqdm.write("Waiting for plot generation to complete...")
        for thread in plot_threads:
             thread.join()
        tqdm.write("All plots generated.")

    print(f"\n[run_simulations] All simulations completed!\n{results.to_markdown()}")
    return results
    
def plot_and_save_flow(
    dem: str,
    output_raster_path: str,
    coordinates: np.ndarray,
    volume: float,
    plot_filename: str,
    elapsed_time: float
) -> None:
    """
    Generates and saves a plot for a single simulation result to a specific file.

    Args:
        dem (str): Path to the DEM file.
        output_raster_path (str): Path to the output raster flow file (.tif, .asc, etc.).
        coordinates (np.ndarray): Array of event coordinates.
        volume (float): The simulation volume.
        plot_filename (str): Full path to save the output plot (.png).
        elapsed_time (float): Time taken for the simulation run.
    """
    try:
        # Ensure we have coordinates to plot
        tqdm.write(f"[plot_and_save_flow] Plotting volume: {volume:.2e} m³ -> {plot_filename}")
        if not output_raster_path or not os.path.exists(output_raster_path):
            tqdm.write(f"[plot_and_save_flow] Skipping plot for volume {volume:.2e}: Raster file missing ({output_raster_path})")
            return
            
        if coordinates.size > 0:
            fig, ax = plt.subplots() # Create figure and axes
            plot_title = f"Volume: {volume:.2e} m³, Time: {elapsed_time:.2f}s"
            # victor.plot_flow likely expects raster, not just path? Check victor docs.
            # Assuming victor.plot_flow can handle the raster path:
            victor.plot_flow(dem, output_raster_path, axes=ax, coords=coordinates, zoom=True, title=plot_title)
            
            plt.savefig(plot_filename)
            tqdm.write(f"[plot_and_save_flow] Saved plot to {plot_filename}")
            plt.close(fig) # Close the figure to free memory
        else:
            tqdm.write(f"[plot_and_save_flow] Skipping plot for volume {volume:.2e}: Missing or invalid coordinates.")
    except Exception as e:
        # Catch potential errors in victor.plot_flow or saving
        tqdm.write(f"[plot_and_save_flow] Error generating plot for volume {volume:.2e} -> {plot_filename}: {e}")
        # Ensure figure is closed even if error occurs during plotting/saving
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)

def create_comparison_grid(
    results_list: List[List[Dict[str, Any]]], # Expects list of lists
    matrix_df: pd.DataFrame,
    volume_list: List[float],
    output_dir: str
) -> str:
    """
    Creates a visual comparison grid similar to the screenshot.
    
    Args:
        results_list (list): List of result dictionaries from all simulations
        matrix_df (pd.DataFrame): The simulation matrix dataframe
        volume_list (list): List of volumes simulated
        output_dir (str): Output directory path
        
    Returns:
        str: Path to comparison grid image
    """
    # Get unique trench IDs
    trench_ids = matrix_df['trench_id'].unique()
    
    # Create figure
    fig_width = 2 + len(volume_list) * 2  # 2 columns for DEM + trench ID, 2 inches per volume
    fig_height = 1.5 * len(trench_ids)  # 1.5 inches per trench
    
    fig, axes = plt.subplots(len(trench_ids), len(volume_list) + 1, 
                            figsize=(fig_width, fig_height))
    
    # Set column headers
    header_row = ['DEM'] + [f"{v:.0e}m³" for v in volume_list]
    for i, header in enumerate(header_row):
        if len(trench_ids) > 1:  # Multiple rows
            axes[0, i].set_title(header)
        else:  # Single row
            axes[i].set_title(header)
    
    # Plot each cell
    for t_idx, trench_id in enumerate(trench_ids):
        trench_rows = matrix_df[matrix_df['trench_id'] == trench_id]
        
        for v_idx, volume in enumerate(volume_list):
            # Find matching result
            for results in results_list:
                for result in results:
                    if (result['trench_id'] == trench_id and 
                        abs(result['volume'] - volume) < 0.0001):  # Compare with tolerance
                        
                        # Get the axis to plot on
                        if len(trench_ids) > 1 and len(volume_list) > 0:
                            ax = axes[t_idx, v_idx + 1]  # +1 to account for DEM column
                        elif len(trench_ids) > 1:
                            ax = axes[t_idx, v_idx]
                        elif len(volume_list) > 0:
                            ax = axes[v_idx + 1]  # +1 to account for DEM column
                        else:
                            ax = axes[v_idx]
                        
                        # Plot the result if successful
                        if result['success'] and result['output_file']:
                            # Construct the expected PNG path based on the output file
                            base_name, _ = os.path.splitext(result['output_file'])
                            plot_image_path = base_name + ".png"
                            if os.path.exists(plot_image_path):
                                img = plt.imread(plot_image_path)
                                ax.imshow(img)
                            else:
                                ax.text(0.5, 0.5, "Plot N/A", ha='center', va='center', transform=ax.transAxes)
                        else:
                            ax.text(0.5, 0.5, "Failed", ha='center', va='center', transform=ax.transAxes)
                        
                        # Remove axis ticks for cleaner look
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
        # Plot DEM in first column
        if len(trench_ids) > 1 and len(volume_list) > 0:
            ax_dem = axes[t_idx, 0]
        elif len(trench_ids) > 1:
            ax_dem = axes[t_idx]
        elif len(volume_list) > 0:
            ax_dem = axes[0]
        else:
            ax_dem = axes
            
        # Plot DEM and set row label
        dem_file = os.path.join(dem_dir, trench_rows.iloc[0]['dem_file'] + trench_rows.iloc[0]['dem_ext'])
        if os.path.exists(dem_file):
            try:
                dem_img = victor.plot_dem(dem_file, show=False)
                ax_dem.imshow(dem_img)
                ax_dem.set_ylabel(f"Trench {trench_id}")
            except:
                ax_dem.text(0.5, 0.5, f"Trench {trench_id}", 
                         ha='center', va='center', transform=ax_dem.transAxes)
        else:
            ax_dem.text(0.5, 0.5, f"Trench {trench_id}", 
                     ha='center', va='center', transform=ax_dem.transAxes)
        
        ax_dem.set_xticks([])
        ax_dem.set_yticks([])
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "comparison_grid.png")
    plt.savefig(comparison_path, dpi=300)
    plt.close(fig)
    
    print(f"[create_comparison_grid] Saved comparison grid to {comparison_path}")
    return comparison_path

def setup_simulation(
    dem_dir: str,
    dem_file: str,
    dem_ext: str,
    events: str,
    volumes: Optional[str],
    mode: Optional[str]
) -> Optional[Tuple[str, List[float], str, str, np.ndarray]]:
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
        print(f"[setup_simulation] Error: DEM file not found at {dem}")
        return None  # Indicate failure
    
    try:
        volume_list = determine_volume_list(volumes, mode)
    except ValueError as e:
        print(f"[setup_simulation] Error determining volumes: {e}")
        return None # Indicate failure

    output_dir = create_output_directory(dem_file, events)
    events_file_path = create_events_file(events, output_dir)
    coordinates = parse_coordinates(events)

    return dem, volume_list, output_dir, events_file_path, coordinates

def run_multi_simulation(
    matrix_df: Optional[pd.DataFrame], # Made optional for resume mode
    volume_list: Optional[List[float]], # Made optional for resume mode
    dem_dir: str,
    resume_dir: Optional[str] = None,
    **kwargs: Any
) -> Tuple[Optional[str], Optional[List[List[Dict[str, Any]]]]]: # Return optional in case of resume failure
    """
    Runs simulations for all DEM-Event pairs across all volumes,
    organizing outputs into volume-specific subdirectories within a master directory.
    Can resume a previous run if resume_dir is provided.
    
    Args:
        matrix_df (pd.DataFrame | None): The simulation matrix dataframe. Required if not resuming.
        volume_list (list | None): List of volumes to simulate. Required if not resuming.
        dem_dir (str): Directory containing DEM files.
        resume_dir (str | None): Path to a previous multi-simulation output directory to resume.
        **kwargs: Additional parameters for simulation (parents, elevation_uncert, 
                  residual, pulse_volume, runs, resolution, plot)
        
    Returns:
        tuple: (master_output_dir, results_list) or (None, None) if resume setup fails.
    """
    
    # --- Setup: Determine if resuming or starting new ---
    if resume_dir:
        tqdm.write(f"[run_multi_simulation] Attempting to resume from: {resume_dir}")
        if not os.path.isdir(resume_dir):
            tqdm.write(f"[run_multi_simulation] Error: Resume directory not found: {resume_dir}")
            return None, None
        master_output_dir = resume_dir
        
        # Load matrix from resume_dir
        matrix_path = os.path.join(master_output_dir, "simulation_matrix.csv")
        if not os.path.exists(matrix_path):
            tqdm.write(f"[run_multi_simulation] Error: simulation_matrix.csv not found in {master_output_dir}")
            return None, None
        try:
            matrix_df = pd.read_csv(matrix_path)
            tqdm.write(f"[run_multi_simulation] Loaded simulation matrix from {matrix_path}")
        except Exception as e:
            tqdm.write(f"[run_multi_simulation] Error loading matrix file {matrix_path}: {e}")
            return None, None
            
        # Determine volume list from existing directories
        volume_list = []
        volume_pattern = re.compile(r"^volume_(\d+\.\d+e[\+\-]\d+)$")
        try:
             for item in os.listdir(master_output_dir):
                 if os.path.isdir(os.path.join(master_output_dir, item)):
                     match = volume_pattern.match(item)
                     if match:
                         volume_list.append(float(match.group(1)))
             volume_list.sort() # Ensure consistent order
             if not volume_list:
                 tqdm.write(f"[run_multi_simulation] Warning: No volume subdirectories found in {master_output_dir}. Cannot determine volumes to resume.")
                 # Decide: Default to a standard list, or fail? Let's fail for now.
                 return None, None
             tqdm.write(f"[run_multi_simulation] Determined volumes to resume: {volume_list}")
        except Exception as e:
             tqdm.write(f"[run_multi_simulation] Error scanning for volume directories in {master_output_dir}: {e}")
             return None, None
             
        # Load previous results
        previous_results = {}
        for volume in volume_list:
            vol_dir = os.path.join(master_output_dir, f"volume_{volume:.2e}")
            results_csv = os.path.join(vol_dir, "results.csv")
            if os.path.exists(results_csv):
                try:
                    vol_df = pd.read_csv(results_csv)
                    # Convert dataframe rows to dicts for easier lookup
                    for _, row in vol_df.iterrows():
                         # Use (volume, trench_id, dem_file, events) as a unique key for robustness
                         key = (float(row['volume']), row['trench_id'], row['dem_file'], row['events'])
                         # Store the dictionary, ensuring 'success' is boolean
                         result_dict = row.to_dict()
                         result_dict['success'] = bool(row['success'])
                         previous_results[key] = result_dict
                    tqdm.write(f"[run_multi_simulation] Loaded {len(vol_df)} previous results for volume {volume:.2e}")
                except Exception as e:
                    tqdm.write(f"[run_multi_simulation] Warning: Could not load or parse {results_csv}: {e}. Will re-run simulations for this volume.")
        tqdm.write(f"[run_multi_simulation] Total previous results loaded: {len(previous_results)}")

    else:
        # New run setup
        if matrix_df is None or volume_list is None:
             tqdm.write("[run_multi_simulation] Error: matrix_df and volume_list must be provided for a new run.")
             return None, None
             
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        master_output_dir = f"{timestamp}_multi_simulation"
        os.makedirs(master_output_dir, exist_ok=True)
        tqdm.write(f"[run_multi_simulation] Created master output directory: {master_output_dir}")
        
        # Save the matrix for reference
        matrix_path = os.path.join(master_output_dir, "simulation_matrix.csv")
        try:
             matrix_df.to_csv(matrix_path, index=False)
             tqdm.write(f"[run_multi_simulation] Saved simulation matrix to {matrix_path}")
        except Exception as e:
             tqdm.write(f"[run_multi_simulation] Error saving matrix file {matrix_path}: {e}")
             # Continue? Or fail? Let's continue but log warning.

        previous_results = {} # No previous results for a new run
    
    # Get simulation parameters from kwargs or use defaults
    parents = kwargs.get('parents', 1)
    elevation_uncert = kwargs.get('elevation_uncert', 0.5)
    residual = kwargs.get('residual', 1.0)
    pulse_volume = kwargs.get('pulse_volume', 1e1)
    runs = kwargs.get('runs', 1)
    resolution = kwargs.get('resolution', 2)
    plot_flag = kwargs.get('plot', True)

    results_list = [] # This will store results lists *per volume*
    all_results_flat = list(previous_results.values()) # Keep a flat list for final summary

    # --- Loop 1: Volumes ---
    for volume in tqdm(volume_list, desc="Processing volumes"): 
        volume_output_dir = os.path.join(master_output_dir, f"volume_{volume:.2e}")
        os.makedirs(volume_output_dir, exist_ok=True)
        # For resume mode, this directory should already exist, but exist_ok=True handles it.
        tqdm.write(f"[run_multi_simulation] Processing volume directory: {volume_output_dir}")
        
        volume_results = [] # Results specifically for THIS volume loop execution
        
        # --- Loop 2: DEM-Event pairs within the current volume ---
        for idx, row in tqdm(matrix_df.iterrows(), desc=f"Volume {volume:.2e} m³", total=len(matrix_df), leave=False):
            dem_file_base = row['dem_file']
            dem_ext = row.get('dem_ext', '.asc') # Handle potential missing column
            events = row['events']
            trench_id = row['trench_id']

            # Create unique key for checking previous results
            run_key = (float(volume), trench_id, dem_file_base, events)

            # --- Check if already completed --- 
            if run_key in previous_results:
                 prev_result = previous_results[run_key]
                 # Check specifically for success (simulation AND conversion) 
                 # Also check if output file physically exists if plot is needed
                 output_file_exists = prev_result.get('output_file') and os.path.exists(str(prev_result['output_file']))
                 plot_file_exists = not plot_flag or (prev_result.get('plot_file') and os.path.exists(str(prev_result['plot_file'])))
                 
                 if prev_result.get('success') and output_file_exists and plot_file_exists:
                     tqdm.write(f"[run_multi_simulation] Skipping completed run: Vol={volume:.2e}, Trench={trench_id}, DEM={dem_file_base}")
                     volume_results.append(prev_result) # Add existing result to this volume's list
                     continue # Skip to next DEM/Event pair
                 else:
                      tqdm.write(f"[run_multi_simulation] Re-running failed/incomplete run: Vol={volume:.2e}, Trench={trench_id}, DEM={dem_file_base} (Success: {prev_result.get('success')}, Output Exists: {output_file_exists}, Plot Exists: {plot_file_exists})")
                      # Don't append here, let the simulation run and append the *new* result later

            # --- Setup for this specific DEM/Event/Volume run --- 
            tqdm.write(f"[run_multi_simulation] Setting up: Vol={volume:.2e}, DEM={dem_file_base}, Trench={trench_id}")
            
            # 1. Check DEM file existence
            dem_path = os.path.join(dem_dir, f"{dem_file_base}{dem_ext}")
            abs_dem_path = os.path.abspath(dem_path)
            if not os.path.exists(abs_dem_path):
                tqdm.write(f"[run_multi_simulation] Error: DEM file not found at {abs_dem_path}. Skipping.")
                result = {
                    'volume': volume,
                    'success': False,
                    'output_file': None,
                    'plot_file': None,
                    'config_file': None,
                    'events_file': None,
                    'elapsed_time': 0.0,
                    'trench_id': trench_id,
                    'dem_file': dem_file_base,
                    'events': events,
                    'error': 'DEM not found'
                }
                volume_results.append(result)
                # Add to flat list ONLY IF it wasn't already there as a failed run
                if run_key not in previous_results:
                     all_results_flat.append(result)
                else: # Update the existing entry in the flat list
                    existing_index = next((i for i, item in enumerate(all_results_flat) if (item['volume'], item['trench_id'], item['dem_file'], item['events']) == run_key), -1)
                    if existing_index != -1: all_results_flat[existing_index] = result
                continue # Skip to next DEM/Event pair
            
            # 2. Parse coordinates
            coordinates = parse_coordinates(events)
            if coordinates.size == 0:
                 tqdm.write(f"[run_multi_simulation] Error: Failed to parse coordinates '{events}'. Skipping.")
                 result = {
                    'volume': volume,
                    'success': False,
                    'output_file': None,
                    'plot_file': None,
                    'config_file': None,
                    'events_file': None,
                    'elapsed_time': 0.0,
                    'trench_id': trench_id,
                    'dem_file': dem_file_base,
                    'events': events,
                    'error': 'Coordinate parsing failed'
                 }
                 volume_results.append(result)
                 if run_key not in previous_results: all_results_flat.append(result)
                 else: # Update the existing entry in the flat list
                     existing_index = next((i for i, item in enumerate(all_results_flat) if (item['volume'], item['trench_id'], item['dem_file'], item['events']) == run_key), -1)
                     if existing_index != -1: all_results_flat[existing_index] = result
                 continue
                 
            # 3. Define output filenames within the volume_output_dir
            events_short = events[0:13].replace(',', '_') # Short identifier for filenames
            volume_str = str(int(volume)).zfill(10) # Padded volume string
            # Ensure base name includes trench_id for uniqueness across different events/dems with same coords maybe?
            # Let's add trench_id to the base name for clarity and safety.
            base_output_name = f"trench_{trench_id}_{dem_file_base}_{events_short}_{volume_str}" 
            
            events_file_path = os.path.abspath(os.path.join(volume_output_dir, f"{base_output_name}_events.in"))
            config_path = os.path.abspath(os.path.join(volume_output_dir, f"{base_output_name}_molasses.conf"))
            # Molasses *always* writes to 'flow_-0' in the CWD (which is volume_output_dir here)
            input_flow_file = os.path.join(volume_output_dir, 'flow_-0') 
            output_raster_path = os.path.join(volume_output_dir, f"{base_output_name}.tif") # Output raster path
            plot_filename = os.path.join(volume_output_dir, f"{base_output_name}.png") # Output plot path

            # --- Clean up potentially stale intermediate files from previous failed run --- 
            # (e.g., flow_-0, partial .tif, .png)
            stale_files = [input_flow_file, output_raster_path, plot_filename, events_file_path, config_path]
            for f_path in stale_files:
                 if os.path.exists(f_path):
                     try:
                         os.remove(f_path)
                         tqdm.write(f"[run_multi_simulation] Removed stale file: {f_path}")
                     except OSError as e:
                          tqdm.write(f"[run_multi_simulation] Warning: Could not remove stale file {f_path}: {e}")

            # 4. Create events file
            created_events_path = create_events_file(events, events_file_path)
            if not created_events_path:
                tqdm.write(f"[run_multi_simulation] Error: Failed to create events file '{events_file_path}'. Skipping.")
                result = {'volume': volume, 'success': False, 'output_file': None, 'plot_file': None, 'config_file': config_path, 'events_file': None, 'elapsed_time': 0.0, 'trench_id': trench_id, 'dem_file': dem_file_base, 'events': events, 'error': 'Event file creation failed'}
                volume_results.append(result)
                if run_key not in previous_results: all_results_flat.append(result)
                else: # Update the existing entry in the flat list
                     existing_index = next((i for i, item in enumerate(all_results_flat) if (item['volume'], item['trench_id'], item['dem_file'], item['events']) == run_key), -1)
                     if existing_index != -1: all_results_flat[existing_index] = result
                continue
                
            # --- Run the simulation for this specific DEM/Event/Volume --- 
            tqdm.write(f"[run_multi_simulation] Starting run: Vol={volume:.2e}, DEM={dem_file_base}, Trench={trench_id}")

            # 1. Create config file
            created_config_path = create_config_file(
                config_path, parents, elevation_uncert, residual,
                volume, pulse_volume, runs, abs_dem_path, events_file_path # Pass abs event path
            )
            if not created_config_path:
                 tqdm.write(f"[run_multi_simulation] Error: Failed to create config file '{config_path}'. Skipping.")
                 result = {'volume': volume, 'success': False, 'output_file': None, 'plot_file': None, 'config_file': None, 'events_file': events_file_path, 'elapsed_time': 0.0, 'trench_id': trench_id, 'dem_file': dem_file_base, 'events': events, 'error': 'Config file creation failed'}
                 volume_results.append(result)
                 if run_key not in previous_results: all_results_flat.append(result)
                 else: # Update the existing entry in the flat list
                     existing_index = next((i for i, item in enumerate(all_results_flat) if (item['volume'], item['trench_id'], item['dem_file'], item['events']) == run_key), -1)
                     if existing_index != -1: all_results_flat[existing_index] = result
                 continue

            # 2. Run MOLASSES simulation (runs in volume_output_dir)
            success, elapsed_time = run_molasses(config_path, volume_output_dir)
            final_output_file = None # Initialize output file path
            error_msg = None

            if success:
                # Check if flow_-0 was actually created
                if not os.path.exists(input_flow_file):
                     tqdm.write(f"[run_multi_simulation] Error: MOLASSES reported success but flow file {input_flow_file} not found.")
                     success = False # Mark as failed if output is missing
                     error_msg = 'MOLASSES output missing'
                else:
                     # 3. Convert MOLASSES output ('flow_-0' in volume_output_dir) to raster
                     final_output_file = convert_molasses(input_flow_file, output_raster_path, resolution=resolution)
                     if not final_output_file:
                         tqdm.write(f"[run_multi_simulation] Warning: Simulation succeeded but conversion failed for {base_output_name}")
                         error_msg = 'Conversion failed'
                         # Simulation technically succeeded, but no raster output, so overall success=False for our purpose
                         success = False 
                     else:
                          # 4. Plotting (if conversion succeeded and flag is set)
                          if plot_flag:
                              tqdm.write(f"[run_multi_simulation] Plotting: {base_output_name}")
                              # Ensure plot function gets the absolute DEM path
                              plot_and_save_flow(abs_dem_path, final_output_file, coordinates, volume, plot_filename, elapsed_time)
                          # Cleanup the raw flow file now that conversion/plotting is done (optional)
                          if os.path.exists(input_flow_file):
                              try: 
                                   os.remove(input_flow_file)
                                   tqdm.write(f"[run_multi_simulation] Cleaned up raw flow file: {input_flow_file}")
                              except OSError as e: 
                                   tqdm.write(f"[run_multi_simulation] Warning: Could not remove raw flow file {input_flow_file}: {e}")
            else:
                 error_msg = 'MOLASSES execution failed'
                 # Ensure flow_-0 doesn't linger if molasses failed partway
                 if os.path.exists(input_flow_file):
                      try: os.remove(input_flow_file)
                      except OSError as e: tqdm.write(f"[run_multi_simulation] Warning: Could not remove potentially partial flow file {input_flow_file} after failure: {e}")
            
            # --- Collect results for this run --- 
            run_success = success and (final_output_file is not None) # Define overall success
            result = {
                'volume': volume,
                'success': run_success, 
                'output_file': final_output_file if run_success else None,
                'plot_file': plot_filename if plot_flag and run_success else None,
                'config_file': config_path,
                'events_file': events_file_path,
                'elapsed_time': elapsed_time,
                'trench_id': trench_id,
                'dem_file': dem_file_base,
                'events': events,
                'error': error_msg if not run_success else None
            }
            volume_results.append(result)
            # Update or add to the flat list
            if run_key not in previous_results:
                 all_results_flat.append(result)
            else:
                 existing_index = next((i for i, item in enumerate(all_results_flat) if (item['volume'], item['trench_id'], item['dem_file'], item['events']) == run_key), -1)
                 if existing_index != -1: all_results_flat[existing_index] = result
                 else: # Should not happen if run_key was in previous_results, but defensively add
                      all_results_flat.append(result)
                      tqdm.write(f"[run_multi_simulation] Warning: run_key {run_key} was in previous_results but not found in all_results_flat for update.")

            tqdm.write(f"[run_multi_simulation] Finished run: Vol={volume:.2e}, DEM={dem_file_base}, Trench={trench_id}, Success={run_success}")
        
        # --- Finished all DEM/Event pairs for this volume --- 
        # results_list.append(volume_results) # Don't append here, we build the final list at the end
        
        # Save intermediate results CSV for this volume (contains only runs processed in *this* execution + skipped runs for this vol)
        volume_results_df = pd.DataFrame(volume_results)
        volume_csv_path = os.path.join(volume_output_dir, "results.csv")
        try:
            volume_results_df.to_csv(volume_csv_path, index=False)
            tqdm.write(f"[run_multi_simulation] Saved/Updated results for volume {volume:.2e} to {volume_csv_path}")
        except Exception as e:
             tqdm.write(f"[run_multi_simulation] Error saving results CSV {volume_csv_path}: {e}")

        # --- Progressively update overall summary table --- 
        # Use the *flat* list containing all results (old and new)
        if all_results_flat:
             # We need results_list (list of lists) for create_summary_table signature. Rebuild it.
             temp_results_list_for_summary = []
             grouped_results = pd.DataFrame(all_results_flat).groupby('volume')
             for vol, group in grouped_results:
                  temp_results_list_for_summary.append(group.to_dict('records'))
                  
             if temp_results_list_for_summary:
                  create_summary_table(temp_results_list_for_summary, master_output_dir)
                  tqdm.write(f"[run_multi_simulation] Progressively updated summary tables in {master_output_dir}")
             else:
                  tqdm.write(f"[run_multi_simulation] No results yet to create summary table.")
        # --- End progressive update ---
        
    # --- Finished all volumes --- 
    tqdm.write("[run_multi_simulation] All volumes processed. Creating final summaries.")

    # Final Summary Generation using all combined results
    final_results_list_structured = []
    if all_results_flat:
        df_all_results = pd.DataFrame(all_results_flat)
        # Ensure volume column is float for correct grouping
        df_all_results['volume'] = df_all_results['volume'].astype(float)
        grouped_results = df_all_results.groupby('volume')
        for vol, group in grouped_results:
            # Ensure volume order matches original volume_list if possible
            final_results_list_structured.append(group.to_dict('records'))
            
        # Sort the structured list based on the volume_list order
        final_results_list_structured.sort(key=lambda x: volume_list.index(x[0]['volume']))

        create_summary_table(final_results_list_structured, master_output_dir)
        # Pass necessary info to comparison grid
        if plot_flag: # Only create grid if plots were generated
             # Ensure matrix_df is available (it should be, loaded/passed earlier)
             if matrix_df is not None:
                  create_comparison_grid(final_results_list_structured, matrix_df, volume_list, master_output_dir)
             else:
                  tqdm.write("[run_multi_simulation] Warning: Cannot create comparison grid because matrix_df is missing.")
        else:
             tqdm.write("[run_multi_simulation] Skipping comparison grid generation as plotting was disabled.")
    else:
        tqdm.write("[run_multi_simulation] No results generated or found. Skipping final summary generation.")

    
    tqdm.write(f"[run_multi_simulation] Multi-simulation process complete. Master output: {master_output_dir}")
    return master_output_dir, final_results_list_structured # Return the structured list

def process_results(results: List[Dict[str, Any]], output_dir: str) -> bool:
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
        print(f"[process_results] Error processing results: {e}")
        return False

def create_summary_table(results_list: List[List[Dict[str, Any]]], output_dir: str) -> str:
    """
    Creates a summary table from all simulation results.
    
    Args:
        results_list (list): List of result dictionaries from all simulations
        output_dir (str): Output directory path
        
    Returns:
        str: Path to summary CSV file
    """
    # Flatten the list of result lists
    all_results = []
    for results in results_list:
        for result in results:
            all_results.append(result)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)
    
    # Add success rate statistics
    success_by_volume = summary_df.groupby('volume')['success'].mean().reset_index()
    success_by_volume.columns = ['volume', 'success_rate']
    
    # Add timing statistics
    timing_by_volume = summary_df.groupby('volume')['elapsed_time'].mean().reset_index()
    timing_by_volume.columns = ['volume', 'avg_elapsed_time']
    
    # Save summary
    summary_path = os.path.join(output_dir, "simulation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Save statistics
    stats_path = os.path.join(output_dir, "simulation_statistics.csv")
    stats_df = pd.merge(success_by_volume, timing_by_volume, on='volume')
    stats_df.to_csv(stats_path, index=False)
    
    print(f"[create_summary_table] Saved summary to {summary_path}")
    print(f"[create_summary_table] Saved statistics to {stats_path}")
    
    return summary_path

def generate_logarithmic_samples(
    min_volume: Optional[float] = None,
    max_volume: Optional[float] = None,
    num_samples: Optional[int] = None
) -> np.ndarray:
    """
    Generates a specific, non-uniform sequence of 40 lava flow volumes.
    
    This sequence is designed with specific steps relevant to flow behavior analysis:
    - 5k to 20k: step 2.5k
    - 25k to 50k: step 5k
    - 60k to 100k: step 10k
    - 120k to 200k: step 20k
    - 225k to 300k: step 25k
    - 350k to 900k: step 50k
    - Includes 1M
    
    Ignores min_volume, max_volume, and num_samples parameters.

    Returns:
    --------
    numpy.ndarray
        Array of 40 specific volumes in cubic meters (dtype=float).
    """
    # Define segments with varying steps
    volumes_part1 = np.arange(5000, 20000 + 2500, 2500)
    volumes_part2 = np.arange(25000, 50000 + 5000, 5000)
    volumes_part3 = np.arange(60000, 100000 + 10000, 10000)
    volumes_part4 = np.arange(120000, 200000 + 20000, 20000)
    volumes_part5 = np.arange(225000, 300000 + 25000, 25000)
    volumes_part6 = np.arange(350000, 900000 + 50000, 50000)
    volumes_part7 = np.array([1000000])
    
    # Concatenate all parts
    volumes = np.concatenate([
        volumes_part1, volumes_part2, volumes_part3, volumes_part4, 
        volumes_part5, volumes_part6, volumes_part7
    ]).astype(float)
    
    # Verify the length (should be 40)
    if len(volumes) != 40:
        print(f"[generate_logarithmic_samples] Warning: Generated {len(volumes)} volumes, expected 40. Check logic.")

    return volumes

def estimate_runtime(volumes: np.ndarray, k: float = 0.0036, alpha: float = 1.3) -> None:
    times_seconds = k * (volumes ** alpha)
    times_minutes = times_seconds / 60
    
    cumulative_minutes = np.cumsum(times_minutes)
    
    print("\nEstimated run times:")
    print(f"Total time: {cumulative_minutes[-1]:.2f} minutes ({cumulative_minutes[-1]/60:.2f} hours)")
    
    # Print table header
    print("\n{:<3} {:<10} {:<15} {:<20}".format("#", "Volume (m³)", "Est. Time (min)", "Cumulative Time (min)"))
    print("-" * 50)
    
    # Print each row
    for i, (vol, time, cum_time) in enumerate(zip(volumes, times_minutes, cumulative_minutes), 1):
        print("{:<3} {:<10.0f} {:<15.2f} {:<20.2f}".format(i, vol, time, cum_time))
