import rasterio
from rasterio.transform import from_origin
import numpy as np
from fire import Fire
import os

def analyze_geotiff(tiff_path):
    """
    Analyze a GeoTIFF file and return its metadata, coordinates, and origins
    
    Parameters:
    tiff_path (str): Path to the GeoTIFF file
    
    Returns:
    dict: Dictionary containing metadata and coordinate information
    """
    with rasterio.open(tiff_path) as src:
        # Get basic metadata
        metadata = {
            'driver': src.driver,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtypes': [str(dt) for dt in src.dtypes],  # Get dtypes for all bands
            'crs': src.crs.to_string() if src.crs else None,
            'transform': src.transform,
            'bounds': src.bounds,
            'res': src.res,
            'nodata': src.nodata,
            'indexes': src.indexes,
            'descriptions': src.descriptions,
            'colorinterp': [str(ci) for ci in src.colorinterp]
        }
        
        # Get coordinates for corners
        corners = {
            'upper_left': src.transform * (0, 0),
            'upper_right': src.transform * (src.width, 0),
            'lower_left': src.transform * (0, src.height),
            'lower_right': src.transform * (src.width, src.height),
            'center': src.transform * (src.width/2, src.height/2)
        }
        
        # Get tags if any
        tags = src.tags()
        
        # Get projection info directly from rasterio
        projection_info = {
            'crs_wkt': src.crs.wkt if src.crs else None,
            'epsg_code': src.crs.to_epsg() if src.crs else None,
            'proj_factor': src.transform.a,  # scale factor in x direction
            'is_geographic': src.crs.is_geographic if src.crs else None,
            'is_projected': src.crs.is_projected if src.crs else None
        }
        
        # Get band-specific metadata
        band_metadata = {}
        for idx in src.indexes:
            band_metadata[f'band_{idx}'] = {
                'dtype': str(src.dtypes[idx-1]),
                'nodata': src.nodatavals[idx-1] if src.nodatavals else None,
                'description': src.descriptions[idx-1] if src.descriptions else None,
                'units': src.units[idx-1] if hasattr(src, 'units') else None
            }
        
        return {
            'metadata': metadata,
            'corners': corners,
            'projection': projection_info,
            'tags': tags,
            'band_metadata': band_metadata
        }

class GeoTiffAnalyzer:
    """CLI tool for analyzing GeoTIFF files"""
    
    def info(self, tiff_path):
        """
        Print comprehensive information about a GeoTIFF file
        
        Args:
            tiff_path (str): Path to the GeoTIFF file
        """
        try:
            analysis = analyze_geotiff(tiff_path)
            
            print(f"=== GeoTIFF Analysis for {os.path.basename(tiff_path)} ===\n")
            
            print("Basic Metadata:")
            print("--------------")
            for key, value in analysis['metadata'].items():
                print(f"{key}: {value}")
            
            print("\nCorner Coordinates:")
            print("-----------------")
            for corner, coords in analysis['corners'].items():
                print(f"{corner}: {coords}")
            
            print("\nProjection Information:")
            print("---------------------")
            for key, value in analysis['projection'].items():
                print(f"{key}: {value}")
            
            print("\nBand Metadata:")
            print("-------------")
            for band, meta in analysis['band_metadata'].items():
                print(f"\n{band}:")
                for key, value in meta.items():
                    print(f"  {key}: {value}")
            
            if analysis['tags']:
                print("\nTags:")
                print("-----")
                for key, value in analysis['tags'].items():
                    print(f"{key}: {value}")
                    
        except rasterio.errors.RasterioIOError as e:
            print(f"Error: Could not open {tiff_path}. {str(e)}")
        except Exception as e:
            print(f"Error analyzing file: {str(e)}")

    def metadata(self, tiff_path):
        """
        Get only metadata information
        
        Args:
            tiff_path (str): Path to the GeoTIFF file
        """
        return analyze_geotiff(tiff_path)['metadata']

    def corners(self, tiff_path):
        """
        Get only corner coordinates
        
        Args:
            tiff_path (str): Path to the GeoTIFF file
        """
        return analyze_geotiff(tiff_path)['corners']

    def projection(self, tiff_path):
        """
        Get only projection information
        
        Args:
            tiff_path (str): Path to the GeoTIFF file
        """
        return analyze_geotiff(tiff_path)['projection']

    def tags(self, tiff_path):
        """
        Get only tags/metadata
        
        Args:
            tiff_path (str): Path to the GeoTIFF file
        """
        return analyze_geotiff(tiff_path)['tags']
        
    def stats(self, tiff_path):
        """
        Get basic statistics for each band
        
        Args:
            tiff_path (str): Path to the GeoTIFF file
        """
        with rasterio.open(tiff_path) as src:
            stats = {}
            for band in src.indexes:
                data = src.read(band)
                data = data[data != src.nodatavals[band-1]] if src.nodatavals and src.nodatavals[band-1] is not None else data
                stats[f'band_{band}'] = {
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data))
                }
        return stats

def main():
    """Entry point for the CLI"""
    Fire(GeoTiffAnalyzer)

if __name__ == "__main__":
    main()