#!/usr/bin/env python3

import rasterio
from rasterio.enums import Resampling
import fire
import sys
from pathlib import Path

def resample_dem(input_path: str, resolution: float = 20.0, output_path: str = None) -> None:
    """
    Resample a DEM to a coarser resolution using bilinear interpolation.
    
    Args:
        input_path: Path to input DEM file
        resolution: Desired pixel resolution in meters (default: 20.0)
        output_path: Path where resampled DEM will be saved. If not provided, 
                    will generate based on input filename
    """
    try:
        # Validate input path
        if not Path(input_path).exists():
            print(f"Error: Input file '{input_path}' does not exist")
            sys.exit(1)
        
        # Generate output path if not provided
        if output_path is None:
            input_stem = Path(input_path).stem
            output_path = f"{input_stem}_{int(resolution)}x{int(resolution)}m.tif"
        
        # Open and process the DEM
        with rasterio.open(input_path) as src:
            # Calculate new dimensions
            scale_factor = resolution / src.res[0]
            new_width = int(src.width / scale_factor)
            new_height = int(src.height / scale_factor)
            
            # Create transform for new resolution
            transform = rasterio.transform.from_bounds(
                src.bounds.left, 
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
                new_width,
                new_height
            )
            
            # Set up profile for output file
            profile = src.profile.copy()
            profile.update({
                'width': new_width,
                'height': new_height,
                'transform': transform
            })
            
            # Perform resampling
            data = src.read(
                1,  # First band
                out_shape=(new_height, new_width),
                resampling=Resampling.bilinear
            )
            
            # Write resampled data
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
            
            print(f"Successfully resampled DEM to {resolution}x{resolution}m resolution")
            print(f"Output saved to: {output_path}")
            print(f"New dimensions: {new_width}x{new_height} pixels")
            
    except rasterio.errors.RasterioError as e:
        print(f"Error processing raster: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    fire.Fire(resample_dem)