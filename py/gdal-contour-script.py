import fire
import os
import subprocess
import matplotlib.pyplot as plt
import geopandas as gpd
from osgeo import gdal
import numpy as np
import rasterio
from rasterio.plot import show
from tqdm import tqdm

def create_contours_with_gdal(dem_path, output_vector, interval=10, attribute_name='elevation'):
    """
    Create contour lines from a DEM file using gdal_contour
    
    Parameters:
    dem_path (str): Path to the input DEM file (GeoTIFF format)
    output_vector (str): Path to save the output contour shapefile
    interval (int): Contour interval in meters
    attribute_name (str): Name of the attribute to store elevation values
    
    Returns:
    str: Path to the output contour shapefile
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_vector)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build the gdal_contour command
    cmd = [
        'gdal_contour',
        '-a', attribute_name,  # attribute name for elevation
        '-i', str(interval),   # contour interval
        '-f', 'ESRI Shapefile',  # output format
        dem_path,              # input DEM
        output_vector          # output shapefile
    ]
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"Contour lines generated successfully: {output_vector}")
        return output_vector
    except subprocess.CalledProcessError as e:
        print(f"Error generating contour lines: {e}")
        return None

def load_raster_file(raster_path):
    """
    Load a raster file (ASC, TIF, etc.) into a numpy array with its georeferencing information
    
    Parameters:
    raster_path (str): Path to the raster file
    
    Returns:
    tuple: (data array, transform, nodata value)
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata
    return data, transform, nodata

def plot_dem_with_contours(dem_path, contour_shapefile, lava_path=None, output_path=None, figsize=(12, 10), interval=None, lava_zoom=1.0, vent=None, vent_zoom=1.0, cmap='terrain', show_window=True, show_elevation_bar=False, show_thickness_bar=False):
    """
    Plot DEM with overlay of contour lines and optional lava flow data
    
    Parameters:
    dem_path (str): Path to the DEM file
    contour_shapefile (str): Path to the contour shapefile
    lava_path (str): Path to the lava flow ASC file (optional)
    output_path (str): Path to save the output map (if None, won't save)
    figsize (tuple): Figure size in inches
    interval (int): Contour interval in meters
    lava_zoom (float): Zoom level for the lava extent
    vent (tuple): (x, y) coordinates of the vent location to center the plot
    vent_zoom (float): Zoom level for the vent-centered view (1.0 = no zoom)
    cmap (str): Matplotlib colormap name for DEM visualization (default: 'terrain')
    show_window (bool): Whether to display the plot window (default: True)
    show_elevation_bar (bool): Whether to show the elevation colorbar (default: False)
    show_thickness_bar (bool): Whether to show the lava thickness colorbar (default: False)
    
    Returns:
    fig, ax: The matplotlib figure and axis objects
    """
    # Read the DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform
        extent = [transform[2], transform[2] + transform[0] * src.width,
                 transform[5] + transform[4] * src.height, transform[5]]
        dem = np.ma.masked_values(dem, src.nodata) if src.nodata else dem
    
    # Read the contour lines
    contours = gpd.read_file(contour_shapefile)
    
    # Create the figure with constrained layout
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Plot the DEM as background
    show(dem, ax=ax, cmap=cmap, title='Topographic Map with Contour Lines and Lava Flow',
         extent=extent)
    
    # If lava data is provided, plot it as an overlay and zoom to its extent
    if lava_path:
        lava_data, lava_transform, lava_nodata = load_raster_file(lava_path)
        # Calculate the lava extent from its transform
        lava_extent = [
            lava_transform[2],  # left
            lava_transform[2] + lava_transform[0] * lava_data.shape[1],  # right
            lava_transform[5] + lava_transform[4] * lava_data.shape[0],  # bottom
            lava_transform[5]   # top
        ]
        # Mask nodata values
        lava_masked = np.ma.masked_values(lava_data, lava_nodata)
        # Normalize lava data to prevent overflow
        if not np.all(lava_masked.mask):  # If we have any valid data
            lava_valid = lava_masked.compressed()  # Get non-masked values
            vmin, vmax = np.percentile(lava_valid, [2, 98])  # Robust min/max
            lava_masked = np.clip(lava_masked, vmin, vmax)
        # Plot lava with transparency where there is no lava
        lava_img = ax.imshow(lava_masked, extent=lava_extent, cmap='hot', alpha=0.7)
        # Add colorbar for lava - adjust position and size with fixed number of ticks
        if show_thickness_bar:
            lava_cbar = fig.colorbar(lava_img, ax=ax, label='Lava Flow Thickness (m)', 
                                    location='right', shrink=0.8, pad=0.02)
            # Set fixed number of ticks (e.g., 5 ticks)
            tick_count = 5
            tick_values = np.linspace(lava_masked.min(), lava_masked.max(), tick_count)
            lava_cbar.set_ticks(tick_values)
            lava_cbar.set_ticklabels([f'{val:.1f}' for val in tick_values])
        # Zoom out by the specified lava_zoom factor
        ax.set_xlim(lava_extent[0] - (lava_extent[1] - lava_extent[0]) * (lava_zoom - 1),
                    lava_extent[1] + (lava_extent[1] - lava_extent[0]) * (lava_zoom - 1))
        ax.set_ylim(lava_extent[2] - (lava_extent[3] - lava_extent[2]) * (lava_zoom - 1),
                    lava_extent[3] + (lava_extent[3] - lava_extent[2]) * (lava_zoom - 1))
    
    # Plot the contour lines
    contours.plot(ax=ax, color='black', linewidth=0.5)
    
    # Add a colorbar for the DEM - adjust position and size
    if show_elevation_bar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=dem.min(), vmax=dem.max()))
        dem_cbar = fig.colorbar(sm, ax=ax, location='left', shrink=0.8, pad=0.02)
        dem_cbar.set_label('Elevation (meters)')
    
    # Add some map elements
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    if interval:
        ax.text(0.05, 0.95, f'Contour Interval: {interval}m', 
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7),
                verticalalignment='top')
    
    # If vent location is provided, center the plot around it
    if vent:
        x, y = vent
        # Calculate plot extent based on the DEM dimensions and vent zoom
        width = (extent[1] - extent[0]) / vent_zoom
        height = (extent[3] - extent[2]) / vent_zoom
        ax.set_xlim(x - width/2, x + width/2)
        ax.set_ylim(y - height/2, y + height/2)
        
        # Plot vent location
        ax.plot(x, y, 'r^', markersize=3, label='Vent')
        ax.legend()
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show the plot only if show_window is True
    if show_window:
        plt.show()
    
    return fig, ax

def main(input_file: str, output_path: str | None = None, interval: int = 10, 
         lava_file: str | None = None, lava_folder: str | None = None, lava_zoom: float = 1.0, 
         vent: tuple | None = None, vent_zoom: float = 1.0, cmap: str = 'terrain', 
         show_window: bool = True, show_elevation_bar: bool = False, show_thickness_bar: bool = False):
    """Generate a contour map from a DEM file with optional lava flow overlay.
    
    Args:
        input_file: Path to input DEM file
        output_path: Path for the output map image (optional)
        interval: Contour interval in meters
        lava_file: Path to single lava flow ASC file (optional)
        lava_folder: Path to folder containing multiple lava flow ASC files (optional)
        lava_zoom: Zoom level for the lava extent
        vent: Vent location as (x,y) coordinates tuple
        vent_zoom: Zoom level for the vent-centered view (default=1.0, higher values = more zoom)
        cmap: Matplotlib colormap name for DEM visualization (default: 'terrain')
        show_window: Whether to display the plot window (default: True)
        show_elevation_bar: Whether to show the elevation colorbar (default: False)
        show_thickness_bar: Whether to show the lava thickness colorbar (default: False)
    """
    # Parse vent coordinates if provided
    vent_coords = None
    if vent:
        try:
            x, y = float(vent[0]), float(vent[1])
            vent_coords = (x, y)
        except (ValueError, IndexError):
            print("Error: Vent coordinates must be provided as two numbers (e.g., --vent 326746 376249)")
            return

    # Create output directory if needed
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Generate contours in the same directory as the output file, or current directory if no output specified
    output_dir = os.path.dirname(output_path) if output_path else "."
    contour_shapefile = os.path.join(output_dir, "contours.shp")
    
    contour_file = create_contours_with_gdal(
        input_file,
        contour_shapefile,
        interval=interval
    )
    
    if not contour_file:
        return

    if lava_folder:
        # Process all ASC and TIF files in the lava folder
        tqdm.write(f"Scanning folder: {lava_folder}")
        lava_files = [f for f in os.listdir(lava_folder) if f.lower().endswith(('.asc', '.tif', '.tiff'))]
        tqdm.write(f"Found {len(lava_files)} raster files\n")
        
        # Add progress bar
        for lava_filename in tqdm(lava_files, desc="Processing lava files"):
            try:
                lava_path = os.path.join(lava_folder, lava_filename)
                
                # Generate output filename based on the lava filename
                if output_path:
                    base_name = os.path.splitext(output_path)[0]
                    ext = os.path.splitext(output_path)[1]
                    current_output = f"{base_name}_{os.path.splitext(lava_filename)[0]}{ext}"
                else:
                    current_output = None
                
                # Close any existing figures to prevent memory issues
                plt.close('all')
                
                plot_dem_with_contours(
                    input_file,
                    contour_file,
                    lava_path=lava_path,
                    output_path=current_output,
                    figsize=(12, 10),
                    interval=interval,
                    lava_zoom=lava_zoom,
                    vent=vent_coords,
                    vent_zoom=vent_zoom,
                    cmap=cmap,
                    show_window=show_window,
                    show_elevation_bar=show_elevation_bar,
                    show_thickness_bar=show_thickness_bar
                )
                if current_output:
                    tqdm.write(f"Saved: {os.path.basename(current_output)}")
            except Exception as e:
                tqdm.write(f"\nError processing {lava_filename}: {str(e)}")
                continue
    elif lava_file:
        # Process single lava file as before
        plot_dem_with_contours(
            input_file,
            contour_file,
            lava_path=lava_file,
            output_path=output_path,
            figsize=(12, 10),
            interval=interval,
            lava_zoom=lava_zoom,
            vent=vent_coords,
            vent_zoom=vent_zoom,
            cmap=cmap,
            show_window=show_window,
            show_elevation_bar=show_elevation_bar,
            show_thickness_bar=show_thickness_bar
        )

if __name__ == "__main__":
    fire.Fire(main) 