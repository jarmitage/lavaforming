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
import geopandas as gpd
from geocube.api.core import make_geocube
sys.path.insert(0, "/home/jovyan/shared/Libraries/")
import victor

MODE_INITIAL = [1e4, 1e5, 2.5e5, 5e5, 7.5e5, 1e6]
MODE_PRIMARY = [5e5, 1e6, 2.5e6]
MODE_PRODUCTION = [5e6, 1e7, 2.5e7]
MODE_EXTENDED = [5e7, 1e8, 5e8]

def setup_environment():
    """Configure environment and suppress warnings."""
    warnings.filterwarnings("ignore")
    print("[setup_environment] Environment configured.")

def load_simulation_matrix(matrix_file):
    """
    Loads a simulation matrix from a CSV file defining DEM-Event pairs to test.
    
    Args:
        matrix_file (str): Path to CSV file with columns:
            - dem_file: Base name of DEM file
            - dem_ext: DEM file extension
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
        print(f"[parse_coordinates] Error parsing event coordinates '{events}': {e}")
        print("[parse_coordinates] Expected comma-separated numbers (e.g., 'x1,y1,x2,y2,...').")
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
    print(f"[create_output_directory] Created output directory: {output_dir}")
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
    print(f"[create_events_file] Created events file at {events_file_path}")
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
    tqdm.write(f"[create_config_file] Created configuration file at {config_path}")
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
    tqdm.write(f"[run_molasses] Starting MOLASSES simulation with config: {config_path}")
    result = subprocess.run(f"./molasses_2022 {config_path}", 
                          shell=True, 
                          stderr=subprocess.DEVNULL)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        tqdm.write(f"[run_molasses] Simulation completed successfully in {elapsed_time:.2f} seconds")
        return True, elapsed_time
    else:
        tqdm.write(f"[run_molasses] Simulation failed with return code {result.returncode}")
        return False, elapsed_time

def convert_molasses(output_name, resolution=2, output_format='tif'):
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
    try:
        print(f"[convert_molasses] Reading flow files.")
        lava = pd.read_csv('./flow_-0', skiprows=3, names=colnames, sep='\s+', header= None)
        lava.to_csv("flow.csv",header=colnames,index=False)
        df = gpd.pd.read_csv('flow.csv')
    except Exception as e:
        print(f"[convert_molasses] Error reading flow files: {e}")
        return None
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
    
    try:
        geo_grid.rio.to_raster(output_file)
    except Exception as e:
        print(f"[convert_molasses] Error writing raster file: {e}")
        return None

    return output_file

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
    
    tqdm.write(f"[convert_output] Converting MOLASSES output to raster at: {output_path}")
    
    output_file = convert_molasses(output_path, resolution=resolution)
    
    return output_file

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
    tqdm.write(f"\n[run_volume_simulation] Running simulation for volume: {volume:.2e} m³")
    
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
    print(f"[save_results] Saved results to {results_path}")
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
    
    print(f"[archive_outputs] Creating archive at {zip_path}")
    try:
        shutil.make_archive(zip_base, 'zip', parent_dir, base_name)
        print(f"[archive_outputs] Archive created successfully")
        return zip_path
    except Exception as e:
        print(f"[archive_outputs] Failed to create archive: {e}")
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
        tqdm.write(f"[run_simulations] Results:\n{results.to_markdown()}")

        # Start plotting in a separate thread
        if plot and result['success'] and result['output_file']:
            tqdm.write(f"[run_simulations] Starting plotting thread for volume: {volume:.2e} m³")
            output_asc = result['output_file']
            # Create and start a new thread for plotting
            plot_thread = threading.Thread(
                target=plot_and_save_flow,
                args=(dem, output_asc, coordinates, volume),
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
        tqdm.write(f"[plot_and_save_flow] Plotting volume: {volume:.2e} m³")
        if coordinates.size > 0:
            fig, ax = plt.subplots() # Create figure and axes
            plot_title = f"Volume: {volume:.2e} m³"
            victor.plot_flow(dem, output_asc, axes=ax, coords=coordinates, zoom=True, title=plot_title)
            
            # Construct plot filename (same base name as asc, but with .png)
            plot_filename = os.path.splitext(output_asc)[0] + ".png"
            plt.savefig(plot_filename)
            tqdm.write(f"[plot_and_save_flow] Saved plot to {plot_filename}")
            plt.close(fig) # Close the figure to free memory
        else:
            tqdm.write("[plot_and_save_flow] Skipping plot due to missing or invalid coordinates.")
    except Exception as e:
        tqdm.write(f"[plot_and_save_flow] Error generating plot for volume {volume}: {e}")

def create_comparison_grid(results_list, matrix_df, volume_list, output_dir):
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

def run_multi_simulation(matrix_df, volume_list, dem_dir, **kwargs):
    """
    Runs simulations for all DEM-Event pairs across all volumes.
    
    Args:
        matrix_df (pd.DataFrame): The simulation matrix dataframe
        volume_list (list): List of volumes to simulate
        dem_dir (str): Directory containing DEM files
        **kwargs: Additional parameters for simulation
        
    Returns:
        tuple: (master_output_dir, results_list)
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    master_output_dir = f"{timestamp}_multi_simulation"
    os.makedirs(master_output_dir, exist_ok=True)
    
    # Save the matrix for reference
    matrix_df.to_csv(os.path.join(master_output_dir, "simulation_matrix.csv"), index=False)
    
    # Set up progress bar for volumes
    results_list = []
    
    # For each volume, run all DEM-Event pairs
    for volume in tqdm(volume_list, desc="Simulating volumes"):
        volume_results = []
        
        # For each DEM-Event pair
        for idx, row in tqdm(matrix_df.iterrows(), desc=f"Volume {volume:.2e} m³", total=len(matrix_df)):
            dem_file = row['dem_file']
            dem_ext = row['dem_ext']
            events = row['events']
            trench_id = row['trench_id']
            
            # Setup individual simulation
            setup_result = setup_simulation(dem_dir, dem_file, dem_ext, events, None, None)
            if setup_result is None:
                continue
                
            dem, _, output_dir, events_file_path, coordinates = setup_result
            
            # Run single volume simulation
            result = run_volume_simulation(
                volume,
                output_dir,
                dem_file,
                events,
                kwargs.get('parents', 1),
                kwargs.get('elevation_uncert', 0.5),
                kwargs.get('residual', 1.0),
                kwargs.get('pulse_volume', 1e1),
                kwargs.get('runs', 1),
                dem,
                events_file_path,
                kwargs.get('resolution', 2)
            )
            
            # Add trench_id to result
            result['trench_id'] = trench_id
            result['dem_file'] = dem_file
            result['events'] = events
            
            volume_results.append(result)
            
            # Generate plot if needed
            if kwargs.get('plot', True) and result['success'] and result['output_file']:
                plot_and_save_flow(dem, result['output_file'], coordinates, volume)
        
        results_list.append(volume_results)
        
        # Save intermediate results for this volume
        volume_output_dir = os.path.join(master_output_dir, f"volume_{volume:.2e}")
        os.makedirs(volume_output_dir, exist_ok=True)
        volume_results_df = pd.DataFrame(volume_results)
        volume_results_df.to_csv(os.path.join(volume_output_dir, "results.csv"), index=False)
    
    # Create summary table and comparison grid
    create_summary_table(results_list, master_output_dir)
    create_comparison_grid(results_list, matrix_df, volume_list, master_output_dir)
    
    return master_output_dir, results_list

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
        print(f"[process_results] Error processing results: {e}")
        return False

def create_summary_table(results_list, output_dir):
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

def main(
    dem_dir: str = "../../dem/",
    matrix_file: str = None,
    dem_file: str = None,
    dem_ext: str = ".asc",
    events: str = "326746,376249",
    residual: float = 1.0,
    pulse_volume: float = 1e1,
    parents: int = 1,
    elevation_uncert: float = 0.5,
    runs: int = 1,
    resolution: int = 2,
    volumes: str | None = None,
    mode: str | None = 'initial',
    plot: bool = True
):
    """
    Runs MOLASSES simulations for a sequence of volumes across multiple DEM-Event pairs.
    
    Args:
        dem_dir (str): Directory containing DEM files. Default is "../../dem/".
        matrix_file (str): Path to CSV file containing simulation matrix. If provided,
                         this takes precedence over individual dem_file/events parameters.
        dem_file (str): Base name of DEM file without extension.
        dem_ext (str): DEM file extension. Default is ".asc".
        events (str): Comma-separated event coordinates.
        residual (float): Residual lava thickness (meters). Default is 1.0.
        pulse_volume (float): Volume of each lava pulse (m³). Default is 1e1.
        parents (int): Number of parent cells. Default is 1.
        elevation_uncert (float): Elevation uncertainty (meters). Default is 0.5.
        runs (int): Number of simulation runs per volume. Default is 1.
        resolution (int): Output raster resolution (meters). Default is 2.
        volumes (str): Comma-separated list of volumes to simulate
                     (in scientific notation, e.g., "1e4,5e4,1e5"). Default is None.
        mode (str): Volume mode to run the script in (initial, primary, production, extended). Default is 'initial'.
        plot (bool): Whether to generate plots for each simulation step. Defaults to True.
    """
    setup_environment()
    
    try:
        volume_list = determine_volume_list(volumes, mode)
    except ValueError as e:
        print(f"[main] Error determining volumes: {e}")
        return
    
    # Determine if we're running in multi-mode or single-mode
    if matrix_file:
        # Multi-DEM/Event mode
        matrix_df = load_simulation_matrix(matrix_file)
        if matrix_df is None:
            return
            
        master_output_dir, results_list = run_multi_simulation(
            matrix_df,
            volume_list,
            dem_dir,
            residual=residual,
            pulse_volume=pulse_volume,
            parents=parents,
            elevation_uncert=elevation_uncert,
            runs=runs,
            resolution=resolution,
            plot=plot
        )
        
        print(f"[main] Multi-simulation completed! Results saved to {master_output_dir}")
        
    else:
        # Single DEM/Event mode - original functionality
        if dem_file is None:
            print("[main] Error: dem_file must be provided when not using a matrix file")
            return
            
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
            print("[process_results] Error processing results. Exiting.")
            return

if __name__ == "__main__":
    fire.Fire(main)
