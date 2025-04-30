"""
Command Line Interface for running MOLASSES simulations using the molasses_lib library.

Example output directory structure:

20240430_123045_multi_simulation/  # Master output directory with timestamp
│
├── simulation_matrix.csv  # Copy of the input simulation matrix
├── simulation_summary.csv  # Combined results from all simulations
├── simulation_statistics.csv  # Success rates and timing stats by volume
├── comparison_grid.png  # Visual overview of all simulations
│
├── volume_1.00e+04/  # Results for 10,000 m³ simulations
│   ├── results.csv  # Results for all trenches at this volume
│   ├── trench_1_326746_376249_0000010000.png  # Flow visualization
│   ├── trench_1_326746_376249_0000010000.tif  # Flow raster data
│   ├── trench_2_326923_376301_0000010000.png
│   ├── trench_2_326923_376301_0000010000.tif
│   └── ...
│
├── volume_1.00e+05/  # Results for 100,000 m³ simulations
│   ├── results.csv
│   ├── trench_1_326746_376249_0000100000.png
│   ├── trench_1_326746_376249_0000100000.tif
│   ├── trench_2_326923_376301_0000100000.png
│   ├── trench_2_326923_376301_0000100000.tif
│   └── ...
│
├── volume_2.50e+05/  # Results for 250,000 m³ simulations
│   ├── results.csv
│   ├── trench_1_326746_376249_0000250000.png
│   └── ...
│
├── volume_5.00e+05/  # Results for 500,000 m³ simulations
│   ├── results.csv
│   └── ...
│
├── volume_7.50e+05/  # Results for 750,000 m³ simulations
│   ├── results.csv
│   └── ...
│
└── volume_1.00e+06/  # Results for 1,000,000 m³ simulations
    ├── results.csv
    └── ...
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
    plot: bool = True,
    resume_dir: str | None = None,
    debug_mode: bool = False
):
    """
    Runs MOLASSES simulations for a sequence of volumes across multiple DEM-Event pairs
    or resumes a previous multi-simulation run.
    
    Uses functions imported from the molasses_lib module.

    Args:
        dem_dir (str): Directory containing DEM files. Default is "../../dem/".
        matrix_file (str): Path to CSV file containing simulation matrix. Required if not resuming.
                         If provided when resuming, it's ignored (matrix loaded from resume_dir).
        dem_file (str): Base name of DEM file without extension (for single mode).
        dem_ext (str): DEM file extension (for single mode). Default is ".asc".
        events (str): Comma-separated event coordinates (for single mode).
        residual (float): Residual lava thickness (meters). Default is 1.0.
        pulse_volume (float): Volume of each lava pulse (m³). Default is 1e1.
        parents (int): Number of parent cells. Default is 1.
        elevation_uncert (float): Elevation uncertainty (meters). Default is 0.5.
        runs (int): Number of simulation runs per volume. Default is 1.
        resolution (int): Output raster resolution (meters). Default is 2.
        volumes (str): Comma-separated list of volumes to simulate
                     (in scientific notation, e.g., "1e4,5e4,1e5"). Ignored if resuming. Default is None.
        mode (str): Volume mode to run the script in (initial, primary, production, extended). Ignored if resuming. Default is 'initial'.
        plot (bool): Whether to generate plots for each simulation step. Defaults to True.
        resume_dir (str | None): Path to a previous multi-simulation output directory to resume.
                                If provided, matrix_file, volumes, and mode are ignored. Default is None.
        debug_mode (bool): If True, skips actual MOLASSES execution and file conversion/plotting,
                           useful for checking setup and directory structure. Defaults to False.
    """
    setup_environment()
    
    # --- Mode Determination ---
    is_multi_mode = matrix_file is not None or resume_dir is not None
    is_resume_mode = resume_dir is not None
    
    if is_resume_mode:
        print(f"[main] Running in RESUME mode for directory: {resume_dir}")
        # Volume list and matrix are determined within run_multi_simulation when resuming
        master_output_dir, results_list = run_multi_simulation(
            # Pass None for matrix_df and volume_list, they will be loaded/determined inside
            matrix_df=None,
            volume_list=None,
            dem_dir=dem_dir,
            resume_dir=resume_dir, # Pass the resume directory
            # Pass other simulation parameters
            residual=residual,
            pulse_volume=pulse_volume,
            parents=parents,
            elevation_uncert=elevation_uncert,
            runs=runs,
            resolution=resolution,
            plot=plot,
            debug_mode=debug_mode
        )
        if master_output_dir:
             print(f"[main] Resumed multi-simulation completed! Results updated in {master_output_dir}")
        else:
             print(f"[main] Failed to resume simulation from {resume_dir}.", file=sys.stderr)
             return

    elif is_multi_mode:
        # Multi-DEM/Event mode (New Run)
        print(f"[main] Running in NEW multi-DEM/Event mode using matrix: {matrix_file}")
        if not matrix_file:
             print("[main] Error: matrix_file must be provided for a new multi-simulation run.", file=sys.stderr)
             return

        try:
            volume_list = determine_volume_list(volumes, mode)
        except ValueError as e:
            print(f"[main] Error determining volumes: {e}", file=sys.stderr)
            return

        matrix_df = load_simulation_matrix(matrix_file)
        if matrix_df is None:
            print(f"[main] Failed to load matrix file: {matrix_file}", file=sys.stderr)
            return
            
        master_output_dir, results_list = run_multi_simulation(
            matrix_df=matrix_df, # Pass the loaded matrix
            volume_list=volume_list, # Pass the determined volumes
            dem_dir=dem_dir,
            resume_dir=None, # Explicitly None for new run
            # Pass other simulation parameters
            residual=residual,
            pulse_volume=pulse_volume,
            parents=parents,
            elevation_uncert=elevation_uncert,
            runs=runs,
            resolution=resolution,
            plot=plot,
            debug_mode=debug_mode
        )
        
        if master_output_dir:
            print(f"[main] New multi-simulation completed! Results saved to {master_output_dir}")
        else:
            print(f"[main] Multi-simulation run failed.", file=sys.stderr)
            return
        
    else:
        # Single DEM/Event mode - original functionality (No resume for single mode)
        print(f"[main] Running in single DEM/Event mode.")
        if dem_file is None:
            print("[main] Error: dem_file must be provided when not using a matrix file or resuming.", file=sys.stderr)
            return

        try:
            # Volume list is needed for single mode setup
            volume_list_single = determine_volume_list(volumes, mode)
        except ValueError as e:
            print(f"[main] Error determining volumes: {e}", file=sys.stderr)
            return
            
        print(f"[main]   DEM: {dem_dir}{dem_file}{dem_ext}")
        print(f"[main]   Events: {events}")
        
        # Pass volumes=None and mode=None to setup_simulation as volume_list is already determined
        setup_result = setup_simulation(dem_dir, dem_file, dem_ext, events, None, None)
        if setup_result is None:
            print(f"[main] Failed to setup simulation environment.", file=sys.stderr)
            return
        # Unpack results, but ignore the volume_list from setup_simulation as we use volume_list_single
        dem, _, output_dir, events_file_path, coordinates = setup_result
        
        print(f"[main] Starting simulation runs...")
        results = run_simulations(
            volume_list_single, # Use the list determined for single mode
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
        success = process_results(results, output_dir, debug_mode=debug_mode)
        if not success:
            print(f"[main] Error processing results in {output_dir}.", file=sys.stderr)
            return
        print(f"[main] Single simulation completed! Results saved and archived for {output_dir}")

if __name__ == "__main__":
    fire.Fire(main) 