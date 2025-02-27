import os
import subprocess

# Math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.interpolate import griddata

# GIS
import rasterio as rio
import rioxarray as rxr
import xarray as xr
import utm
from pyproj import Transformer
from xrspatial import hillshade
from rasterio.enums import Resampling

tif_sq_img = "../dem/Reykjanes_Square.tif"
tif_sq_raw = "../dem/Reykjanes_Square_Raw.tif"
tif_sq_geo = "../dem/Reykjanes_Square_Geo.tif"
tif_sq_geo_lg = "../dem/Reykjanes_Square_Geo_Lg.tif"
tif_paths = [tif_sq_img, tif_sq_raw, tif_sq_geo, tif_sq_geo_lg]

def print_info(tif_path):
    with rio.open(tif_path) as src:
        print("Metadata for:", tif_path)
        print("\nBounds:", src.bounds)
        print("Width:", src.width)
        print("Height:", src.height)
        print("CRS:", src.crs)
        print("Transform:", src.transform)
        print("Count:", src.count, "(number of bands)")
        print("Dtype:", src.dtypes[0])
        print("Nodata value:", src.nodata)
        print("\nTags:", src.tags())

def print_bands_info(tif_path):
    with rio.open(tif_path) as src:
        print(f"File: {tif_path}")
        for i in range(src.count):
            print(f"  Band {i+1}:")
            print(f"    Nodata value: {src.nodata}")
            print(f"    Color interpretation: {src.colorinterp[i]}")
            # Additional useful band information
            band_stats = src.statistics(i + 1)
            if band_stats:
                print(f"    Min: {band_stats.min}")
                print(f"    Max: {band_stats.max}")

def print_info_all():
    print_info(tif_sq_img)
    print_info(tif_sq_raw)
    print_info(tif_sq_geo)
    print_info(tif_sq_geo_lg)

def print_bands_info_all():
    print_bands_info(tif_sq_img)
    print_bands_info(tif_sq_raw)
    print_bands_info(tif_sq_geo)
    print_bands_info(tif_sq_geo_lg)

def copy_info(src_tif, dst_tif):
    with rio.open(src_tif) as src:
        with rio.open(dst_tif, 'w', **src.meta) as dst:
            dst.write(src.read())

def geotiff_from_band(band, meta, tif_path):
    with rio.open(tif_path, 'w', **meta) as dst:
        dst.write(band,1)

def read_band(src, band_idx):  
    """Read a band from an opened rasterio dataset
    
    Args:
        src: Opened rasterio dataset
        band_idx: Band index to read (1-based)
    """
    return src.read(band_idx)

def print_band_info(band):
    print(f"    Shape: {band.shape}")
    print(f"    Dtype: {band.dtype}")
    print(f"    Min: {band.min()}")
    print(f"    Max: {band.max()}")

def plot_band(src_or_band, band_idx=1):
    if isinstance(src_or_band, rio.DatasetReader):
        img = read_band(src_or_band, band_idx)
    else:
        img = src_or_band
    plt.imshow(img)
    plt.colorbar()
    plt.show()

def plot_bands(tif_paths, normalize=False):
    """Plot all bands of all tif files in the list as a subplot
    
    Args:
        tif_paths (list): List of paths to TIF files to plot
    """
    # Count total number of bands across all files
    total_bands = 0
    bands_per_file = []
    for tif_path in tif_paths:
        with rio.open(tif_path) as src:
            total_bands += src.count
            bands_per_file.append(src.count)
    
    # Create subplot grid
    rows = len(tif_paths)
    cols = max(bands_per_file)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Make axes 2D if only one row
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each band
    for file_idx, tif_path in enumerate(tif_paths):
        with rio.open(tif_path) as src:
            # Plot each band for this file
            for band_idx in range(src.count):
                ax = axes[file_idx, band_idx]
                img = src.read(band_idx + 1)
                if normalize:
                    img = normalize_band(img)
                
                # Create image plot
                im = ax.imshow(img)
                plt.colorbar(im, ax=ax)
                
                # Set title
                ax.set_title(f"{os.path.basename(tif_path)}\nBand {band_idx + 1}")
            
            # Hide unused axes
            for band_idx in range(src.count, cols):
                axes[file_idx, band_idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
def normalize_band(band, range_min=0, range_max=255):
    """Normalize the band to the range 0-255"""
    return (band - band.min()) / (band.max() - band.min()) * (range_max - range_min) + range_min

# def rescale_band(band, old_min, old_max, new_min, new_max):
#     """Rescale the band to the new range"""
#     return (band - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

def rescale_band(target_band, reference_band):
    target_min = target_band.min()
    target_max = target_band.max()
    reference_min = reference_band.min()
    reference_max = reference_band.max()

    # Normalize target band to 0-1 range
    normalized_target = (target_band - target_min) / (target_max - target_min)

    # Rescale normalized target band to reference band's range
    rescaled_target = normalized_target * (reference_max - reference_min) + reference_min

    return rescaled_target

def rescale_band_preserve_zero(band, old_min, old_max, new_min, new_max):
    """Rescale the band to the new range while preserving zero values.
    
    Args:
        band: Input band data (numpy array)
        old_min: Minimum value of the input range
        old_max: Maximum value of the input range
        new_min: Minimum value of the target range
        new_max: Maximum value of the target range
        
    Returns:
        Rescaled band with zero values preserved
    """
    # Create a mask for non-zero values
    non_zero_mask = band != 0
    
    # Create output array, initially all zeros
    result = np.zeros_like(band)
    
    # Only rescale non-zero values
    result[non_zero_mask] = (band[non_zero_mask] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    
    return result

def bands_diff(band_1, band_2, normalize=False):
    """Calculate the difference between two bands
    
    Args:
        band_1: First band as numpy array
        band_2: Second band as numpy array
        normalize: Whether to normalize bands before differencing
        
    Returns:
        Difference between the two bands as numpy array
    """
    if normalize:
        band_1 = normalize_band(band_1)
        band_2 = normalize_band(band_2)
    return band_1 - band_2

def plot_bands_diff(band_1, band_2, normalize=False):
    """Plot the difference between two bands"""
    diff = bands_diff(band_1, band_2, normalize)
    plt.imshow(diff)
    plt.colorbar()
    plt.show()

def bands_blend_add(band_1, band_2, rescale=False, preserve_zero=False):
    """Blend two bands by adding them together
    
    Args:
        band_1: First band as numpy array
        band_2: Second band as numpy array
        rescale: Whether to rescale band_2 to match band_1's range before adding
        preserve_zero: Whether to preserve zero values when rescaling
        
    Returns:
        Sum of the two bands as numpy array
    """
    if rescale:
        b1_min, b1_max = band_1.min(), band_1.max()
        b2_min, b2_max = band_2.min(), band_2.max()
        if preserve_zero:
            band_2 = rescale_band_preserve_zero(band_2, b2_min, b2_max, b1_min, b1_max)
        else:
            band_2 = rescale_band(band_2, b2_min, b2_max, b1_min, b1_max)
    return band_1 + band_2


def bands_blend_max(band_1, band_2, rescale=False, preserve_zero=False):
    """Blend two bands by taking the maximum value"""
    if rescale:
        b1_min, b1_max = band_1.min(), band_1.max()
        b2_min, b2_max = band_2.min(), band_2.max()
        if preserve_zero:
            band_2 = rescale_band_preserve_zero(band_2, b2_min, b2_max, b1_min, b1_max)
        else:
            band_2 = rescale_band(band_2, b2_min, b2_max, b1_min, b1_max)
    return np.maximum(band_1, band_2)

def plot_bands_blend(src1, band_idx_1, src2, band_idx_2, rescale=False, preserve_zero=False):
    """Plot the blended sum of two bands"""
    band_1 = read_band(src1, band_idx_1)
    band_2 = read_band(src2, band_idx_2)
    blended = bands_blend_add(band_1, band_2, rescale, preserve_zero)
    plt.imshow(blended)
    plt.colorbar()
    plt.show()

def read_src(tif_path):
    """Open a rasterio dataset from a file path
    
    Args:
        tif_path: Path to the GeoTIFF file
        
    Returns:
        Opened rasterio dataset
    """
    return rio.open(tif_path)

def read_tif_srcs(tif_paths=tif_paths):
    """Read all tif files in the list"""
    return [read_src(tif_path) for tif_path in tif_paths]

