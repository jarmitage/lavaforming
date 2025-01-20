#!/usr/bin/env python3

import rasterio
from rasterio.windows import from_bounds
import fire
import sys
from pathlib import Path
from pyproj import Transformer, CRS

def transform_coords(bounds_wgs84):
    """Transform WGS84 coordinates to ISN93."""
    # Create precise transformer for Iceland's ISN93
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),  # WGS84
        CRS.from_epsg(3057),  # ISN93
        always_xy=True
    )
    
    # Input is (lon,lat), output will be (x,y) in meters
    # For Iceland, longitude is negative, but we pass it directly
    xmin, ymin = transformer.transform(bounds_wgs84[1], bounds_wgs84[0])
    xmax, ymax = transformer.transform(bounds_wgs84[3], bounds_wgs84[2])
    
    # Make sure min/max are in correct order
    return (
        min(xmin, xmax),
        min(ymin, ymax),
        max(xmin, xmax),
        max(ymin, ymax)
    )

def crop_dem(input_path: str, 
             bounds: tuple,
             output_path: str = None,
             is_wgs84: bool = True) -> None:
    """
    Crop a DEM using specified bounds.
    
    Args:
        input_path: Path to input DEM file
        bounds: Tuple of coordinates (lon_min,lat_min,lon_max,lat_max) or (xmin,ymin,xmax,ymax)
        output_path: Path for cropped output. If None, will generate automatically
        is_wgs84: If True, assumes bounds are in WGS84 (lat/lon) and converts to ISN93
    """
    try:
        # Validate input path
        if not Path(input_path).exists():
            print(f"Error: Input file '{input_path}' does not exist")
            sys.exit(1)
            
        if len(bounds) != 4:
            raise ValueError("Bounds must have exactly 4 values")
                
        print(f"Input bounds (lon,lat): {bounds}")
            
        # Transform coordinates if they're in WGS84
        if is_wgs84:
            bounds_tuple = transform_coords(bounds)
            print(f"Transformed bounds (ISN93 meters): {bounds_tuple}")
        else:
            bounds_tuple = bounds
        
        # Open the DEM to check bounds
        with rasterio.open(input_path) as src:
            dem_bounds = src.bounds
            print(f"Original DEM bounds (ISN93): {dem_bounds}")
            print(f"Original size: {src.width}x{src.height} pixels")
            
            # Check if requested bounds are within DEM extent
            if (bounds_tuple[0] < dem_bounds.left or 
                bounds_tuple[1] < dem_bounds.bottom or
                bounds_tuple[2] > dem_bounds.right or
                bounds_tuple[3] > dem_bounds.top):
                print("\nWarning: Requested bounds extend outside DEM coverage!")
                print("Cropping will be limited to DEM extent.")
                
                # Clip bounds to DEM extent
                bounds_tuple = (
                    max(bounds_tuple[0], dem_bounds.left),
                    max(bounds_tuple[1], dem_bounds.bottom),
                    min(bounds_tuple[2], dem_bounds.right),
                    min(bounds_tuple[3], dem_bounds.top)
                )
                print(f"Adjusted bounds: {bounds_tuple}")
            
            # Generate output path if not provided
            if output_path is None:
                input_stem = Path(input_path).stem
                output_path = f"{input_stem}_cropped.tif"
            
            # Create window from bounds
            window = from_bounds(*bounds_tuple, src.transform)
            
            # Get window-specific transform
            transform = rasterio.windows.transform(window, src.transform)
            
            # Read the data in the window
            data = src.read(1, window=window)
            
            # Update profile for new data
            profile = src.profile.copy()
            profile.update({
                'height': window.height,
                'width': window.width,
                'transform': transform
            })
            
            # Write cropped data
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
                
            print(f"\nSuccessfully cropped DEM")
            print(f"New size: {window.width}x{window.height} pixels")
            print(f"Output saved to: {output_path}")
            
            # Calculate and print size reduction
            input_size = Path(input_path).stat().st_size / (1024 * 1024)  # MB
            output_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
            print(f"\nFile sizes:")
            print(f"Input:  {input_size:.1f} MB")
            print(f"Output: {output_size:.1f} MB")

    except rasterio.errors.RasterioIOError as e:
        print(f"Error reading/writing raster: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        print(f"Invalid bounds: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    fire.Fire(crop_dem)