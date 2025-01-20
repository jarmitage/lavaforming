"""
First, get DEM info using this Python script (outside Blender):
"""
import rasterio

def get_dem_info(dem_path):
    """Get spatial information from DEM file."""
    with rasterio.open(dem_path) as src:
        bounds = src.bounds
        transform = src.transform
        resolution = src.res
        
        print(f"DEM Bounds (ISN93):")
        print(f"Left:   {bounds.left:.2f}")
        print(f"Right:  {bounds.right:.2f}")
        print(f"Bottom: {bounds.bottom:.2f}")
        print(f"Top:    {bounds.top:.2f}")
        print(f"\nResolution:")
        print(f"X: {resolution[0]:.2f}m")
        print(f"Y: {resolution[1]:.2f}m")
        print(f"\nSize: {src.width}x{src.height} pixels")
        
        return bounds, resolution

# Usage:
# bounds, resolution = get_dem_info("your_dem.tif")

"""
Then in Blender, use this script to scale and position your object:
"""
import bpy
import bmesh
from mathutils import Vector, Matrix
import math

def scale_to_dem(obj, dem_bounds, dem_resolution, margin_percent=5):
    """
    Scale and position a Blender object to match DEM coordinates.
    
    Args:
        obj: Blender object
        dem_bounds: DEM bounds (left, bottom, right, top)
        dem_resolution: DEM resolution (x_res, y_res)
        margin_percent: Extra space around object as percentage
    """
    # Get object dimensions and bounds
    bbox = obj.bound_box
    bbox_vectors = [Vector(corner) for corner in bbox]
    
    # Calculate current object bounds
    obj_min = Vector((min(v.x for v in bbox_vectors),
                     min(v.y for v in bbox_vectors),
                     min(v.z for v in bbox_vectors)))
    obj_max = Vector((max(v.x for v in bbox_vectors),
                     max(v.y for v in bbox_vectors),
                     max(v.z for v in bbox_vectors)))
    
    # Calculate DEM dimensions
    dem_width = dem_bounds[2] - dem_bounds[0]   # right - left
    dem_height = dem_bounds[3] - dem_bounds[1]  # top - bottom
    
    # Calculate desired scale with margin
    margin = max(dem_width, dem_height) * (margin_percent / 100)
    target_width = dem_width - (2 * margin)
    target_height = dem_height - (2 * margin)
    
    # Calculate scale factors
    current_width = obj_max.x - obj_min.x
    current_height = obj_max.y - obj_min.y
    
    scale_x = target_width / current_width if current_width != 0 else 1
    scale_y = target_height / current_height if current_height != 0 else 1
    
    # Use uniform scaling based on the smaller scale factor
    uniform_scale = min(scale_x, scale_y)
    
    # Apply scale
    obj.scale = Vector((uniform_scale, uniform_scale, uniform_scale))
    
    # Apply scale to make it permanent
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # Position object to match DEM coordinates
    obj.location = Vector((
        dem_bounds[0] + margin,  # left + margin
        dem_bounds[1] + margin,  # bottom + margin
        0  # Z at 0
    ))
    
    print(f"\nObject scaled and positioned:")
    print(f"Scale factor: {uniform_scale:.4f}")
    print(f"New location: ({obj.location.x:.2f}, {obj.location.y:.2f}, {obj.location.z:.2f})")
    
    # Update scene
    bpy.context.view_layer.update()

def align_to_dem():
    """Scale and position active object to match DEM bounds."""
    # Example DEM bounds and resolution (replace with your values)
    dem_bounds = (289998.0, 329998.0, 340000.0, 380000.0)  # left, bottom, right, top
    dem_resolution = (2.0, 2.0)  # x_res, y_res in meters
    
    obj = bpy.context.active_object
    if obj is None:
        print("No active object selected!")
        return
        
    if obj.type != 'MESH':
        print("Selected object must be a mesh!")
        return
    
    # Scale and position object
    scale_to_dem(obj, dem_bounds, dem_resolution)

# Usage in Blender:
# align_to_dem()