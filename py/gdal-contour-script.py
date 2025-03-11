import fire
import os
import subprocess
import matplotlib.pyplot as plt
import geopandas as gpd
from osgeo import gdal
import numpy as np
import rasterio
from rasterio.plot import show

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

def load_asc_file(asc_path):
    """
    Load an ASC file into a numpy array with its georeferencing information
    
    Parameters:
    asc_path (str): Path to the ASC file
    
    Returns:
    tuple: (data array, transform, nodata value)
    """
    with rasterio.open(asc_path) as src:
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata
    return data, transform, nodata

def plot_dem_with_contours(dem_path, contour_shapefile, lava_path=None, output_path=None, figsize=(12, 10), interval=None, lava_zoom=1.0):
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
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the DEM as background
    show(dem, ax=ax, cmap='terrain', title='Topographic Map with Contour Lines and Lava Flow',
         extent=extent)
    
    # If lava data is provided, plot it as an overlay and zoom to its extent
    if lava_path:
        lava_data, lava_transform, lava_nodata = load_asc_file(lava_path)
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
        lava_img = ax.imshow(lava_masked, extent=lava_extent, cmap='magma', alpha=0.7)
        # Add colorbar for lava
        lava_cbar = fig.colorbar(lava_img, ax=ax, label='Lava Flow Thickness (m)')
        # Zoom out by the specified lava_zoom factor
        ax.set_xlim(lava_extent[0] - (lava_extent[1] - lava_extent[0]) * (lava_zoom - 1),
                    lava_extent[1] + (lava_extent[1] - lava_extent[0]) * (lava_zoom - 1))
        ax.set_ylim(lava_extent[2] - (lava_extent[3] - lava_extent[2]) * (lava_zoom - 1),
                    lava_extent[3] + (lava_extent[3] - lava_extent[2]) * (lava_zoom - 1))
    
    # Plot the contour lines
    contours.plot(ax=ax, color='black', linewidth=0.5)
    
    # Add a colorbar for the DEM
    sm = plt.cm.ScalarMappable(cmap='terrain', norm=plt.Normalize(vmin=dem.min(), vmax=dem.max()))
    dem_cbar = fig.colorbar(sm, ax=ax)
    dem_cbar.set_label('Elevation (meters)')
    
    # Add some map elements
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    if interval:
        ax.text(0.05, 0.95, f'Contour Interval: {interval}m', 
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7),
                verticalalignment='top')
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Map saved to {output_path}")
    
    # Show the plot
    plt.show()
    
    return fig, ax

def main(input_file: str, output_path: str | None = None, interval: int = 10, lava_file: str | None = None, lava_zoom: float = 1.0):
    """Generate a contour map from a DEM file with optional lava flow overlay.
    
    Args:
        input_file: Path to input DEM file
        output_path: Path for the output map image (optional)
        interval: Contour interval in meters
        lava_file: Path to lava flow ASC file (optional)
        lava_zoom: Zoom level for the lava extent
    """
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
    
    if contour_file:
        # Plot the contours over the DEM with optional lava flow
        plot_dem_with_contours(
            input_file,
            contour_file,
            lava_path=lava_file,
            output_path=output_path,
            figsize=(12, 10),
            interval=interval,
            lava_zoom=lava_zoom
        )

if __name__ == "__main__":
    fire.Fire(main) 