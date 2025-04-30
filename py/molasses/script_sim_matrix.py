"""
Command Line Interface for running MOLASSES simulations using the molasses_lib library.
"""

import fire
import sys

# Assuming molasses_lib.py is in the same directory or PYTHONPATH is set correctly
from .molasses_lib import (
    setup_environment,
    determine_volume_list,
    load_simulation_matrix,
    run_multi_simulation,
    setup_simulation,
    run_simulations,
    process_results,
)

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
    
    Uses functions imported from the molasses_lib module.

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
        print(f"[main] Error determining volumes: {e}", file=sys.stderr)
        return
    
    # Determine if we're running in multi-mode or single-mode
    if matrix_file:
        # Multi-DEM/Event mode
        print(f"[main] Running in multi-DEM/Event mode using matrix: {matrix_file}")
        matrix_df = load_simulation_matrix(matrix_file)
        if matrix_df is None:
            print(f"[main] Failed to load matrix file: {matrix_file}", file=sys.stderr)
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
        print(f"[main] Running in single DEM/Event mode.")
        if dem_file is None:
            print("[main] Error: dem_file must be provided when not using a matrix file", file=sys.stderr)
            return
            
        print(f"[main]   DEM: {dem_dir}{dem_file}{dem_ext}")
        print(f"[main]   Events: {events}")
        
        setup_result = setup_simulation(dem_dir, dem_file, dem_ext, events, volumes, mode)
        if setup_result is None:
            print(f"[main] Failed to setup simulation environment.", file=sys.stderr)
            return
        dem, volume_list, output_dir, events_file_path, coordinates = setup_result
        
        print(f"[main] Starting simulation runs...")
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
        
        print(f"[main] Processing results for output directory: {output_dir}")
        success = process_results(results, output_dir)
        if not success:
            print(f"[main] Error processing results in {output_dir}.", file=sys.stderr)
            return
        print(f"[main] Single simulation completed! Results saved and archived for {output_dir}")

if __name__ == "__main__":
    fire.Fire(main) 