#!/usr/bin/env python3

"""
# Invert the mask (keep outside instead of inside geometry)
python dem_overlay.py --dem_path=dem.tif --geometry_path=geom.geojson --invert=True

# Specify output path
python dem_overlay.py --dem_path=dem.tif --geometry_path=geom.geojson --output_path=output.tif

# If your geometry is in a different coordinate system
python dem_overlay.py --dem_path=dem.tif --geometry_path=geom.geojson --geometry_crs="EPSG:4326"
"""

import numpy as np
import rasterio
from rasterio.mask import mask
import json
import fire
from pathlib import Path
import geopandas as gpd
from shapely.geometry import shape, mapping
from pyproj import Transformer

def transform_coords(geometry, source_crs="EPSG:4326", target_crs="EPSG:3057"):
    """Transform geometry coordinates between coordinate systems."""
    # Create GeoDataFrame with the geometry
    gdf = gpd.GeoDataFrame(geometry=[shape(geometry)], crs=source_crs)
    # Reproject to target CRS
    gdf_transformed = gdf.to_crs(target_crs)
    return mapping(gdf_transformed.geometry.iloc[0])

def overlay_geometry(dem_path: str, 
                    geometry_path: str,
                    output_path: str = None,
                    geometry_crs: str = "EPSG:4326",
                    invert: bool = False) -> None:
    """
    Overlay a geometry onto a DEM by masking.
    
    Args:
        dem_path: Path to input DEM file
        geometry_path: Path to geometry file (GeoJSON)
        output_path: Path for output DEM
        geometry_crs: CRS of input geometry (default: WGS84)
        invert: If True, mask outside the geometry instead of inside
    """
    # Read geometry
    with open(geometry_path) as f:
        geometry = json.load(f)
        
    # Handle both single geometry and feature collection
    if geometry["type"] == "FeatureCollection":
        geometries = [feat["geometry"] for feat in geometry["features"]]
    elif geometry["type"] == "Feature":
        geometries = [geometry["geometry"]]
    else:
        geometries = [geometry]
    
    # Transform geometries to DEM's CRS if needed
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs.to_string()
        
        if geometry_crs != dem_crs:
            print(f"Converting geometry from {geometry_crs} to {dem_crs}")
            geometries = [transform_coords(geom, geometry_crs, dem_crs) 
                         for geom in geometries]
    
    # Generate output path if not provided
    if output_path is None:
        dem_stem = Path(dem_path).stem
        geom_stem = Path(geometry_path).stem
        output_path = f"{dem_stem}_{geom_stem}_overlay.tif"
    
    # Mask DEM with geometry
    with rasterio.open(dem_path) as src:
        out_image, out_transform = mask(src, geometries, 
                                      crop=True, 
                                      invert=invert,
                                      all_touched=True)
        
        # Update metadata for output
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # Write output
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
            
        print(f"Successfully created overlay at: {output_path}")

if __name__ == "__main__":
    fire.Fire(overlay_geometry)