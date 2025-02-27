import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import json

class GeoJSON3DRenderer:
    def __init__(self, figsize=(10, 10), projection_angle=(45, 35, 0)):
        """
        Initialize renderer with projection angles in degrees.
        Args:
            figsize: Tuple of (width, height) for the figure
            projection_angle: Tuple of (alpha, beta, gamma) rotation angles in degrees
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect('equal')
        
        # Convert angles to radians for projection matrix
        self.alpha, self.beta, self.gamma = np.radians(projection_angle)
        self.projection_matrix = self._create_projection_matrix()
        
    def _create_projection_matrix(self):
        """Create 3D to 2D projection matrix using rotation angles."""
        # Rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.alpha), -np.sin(self.alpha)],
            [0, np.sin(self.alpha), np.cos(self.alpha)]
        ])
        
        Ry = np.array([
            [np.cos(self.beta), 0, np.sin(self.beta)],
            [0, 1, 0],
            [-np.sin(self.beta), 0, np.cos(self.beta)]
        ])
        
        Rz = np.array([
            [np.cos(self.gamma), -np.sin(self.gamma), 0],
            [np.sin(self.gamma), np.cos(self.gamma), 0],
            [0, 0, 1]
        ])
        
        # Combine rotation matrices
        R = Rz @ Ry @ Rx
        
        # Orthographic projection matrix (drops z-coordinate)
        P = np.array([[1, 0, 0],
                     [0, 1, 0]])
        
        return P @ R
    
    def project_point(self, point):
        """Project a 3D point to 2D using the projection matrix."""
        # Handle both 2D and 3D coordinates
        if len(point) == 2:
            point = [point[0], point[1], 0]
        
        point = np.array(point)
        projected = self.projection_matrix @ point
        return projected[0], projected[1]
    
    def extract_coordinates(self, geometry):
        """Extract coordinates from a GeoJSON geometry."""
        if geometry['type'] == 'Point':
            return [geometry['coordinates']]
        elif geometry['type'] == 'LineString':
            return geometry['coordinates']
        elif geometry['type'] == 'Polygon':
            return geometry['coordinates'][0]  # Outer ring only
        elif geometry['type'] == 'MultiPolygon':
            return [coord for polygon in geometry['coordinates'] for coord in polygon[0]]
        return []

    def render_feature(self, feature, **style_kwargs):
        """Render a single GeoJSON feature with 3D projection."""
        geometry = feature['geometry']
        coords = self.extract_coordinates(geometry)
        
        # Project all coordinates to 2D
        projected_coords = [self.project_point(coord) for coord in coords]
        
        if geometry['type'] == 'Point':
            x, y = projected_coords[0]
            self.ax.scatter(x, y, **style_kwargs)
            
        elif geometry['type'] == 'LineString':
            x, y = zip(*projected_coords)
            self.ax.plot(x, y, **style_kwargs)
            
        elif geometry['type'] in ['Polygon', 'MultiPolygon']:
            patches = []
            if geometry['type'] == 'Polygon':
                patches.append(Polygon(projected_coords))
            else:
                for polygon in geometry['coordinates']:
                    projected = [self.project_point(coord) for coord in polygon[0]]
                    patches.append(Polygon(projected))
                    
            collection = PatchCollection(patches, **style_kwargs)
            self.ax.add_collection(collection)

    def render_geojson(self, geojson_data, style_map=None):
        """
        Render 3D GeoJSON data with optional styling.
        
        Args:
            geojson_data: Dict or str (path to GeoJSON file)
            style_map: Dict mapping feature types to style parameters
        """
        if isinstance(geojson_data, str):
            with open(geojson_data, 'r') as f:
                geojson_data = json.load(f)
                
        if style_map is None:
            style_map = {
                'Point': {'color': 'red', 'marker': 'o', 's': 100},
                'LineString': {'color': 'blue', 'linewidth': 2},
                'Polygon': {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'},
                'MultiPolygon': {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
            }

        # Initialize bounds for projected coordinates
        bounds = {'minx': float('inf'), 'miny': float('inf'), 
                 'maxx': float('-inf'), 'maxy': float('-inf')}

        # Render features
        features = geojson_data['features'] if 'features' in geojson_data else [geojson_data]
        for feature in features:
            geometry = feature['geometry']
            coords = self.extract_coordinates(geometry)
            
            # Update bounds using projected coordinates
            for coord in coords:
                x, y = self.project_point(coord)
                bounds['minx'] = min(bounds['minx'], x)
                bounds['miny'] = min(bounds['miny'], y)
                bounds['maxx'] = max(bounds['maxx'], x)
                bounds['maxy'] = max(bounds['maxy'], y)
            
            # Apply style based on geometry type
            style = style_map.get(geometry['type'], {})
            self.render_feature(feature, **style)

        # Set plot bounds with padding
        padding = 0.1
        width = bounds['maxx'] - bounds['minx']
        height = bounds['maxy'] - bounds['miny']
        self.ax.set_xlim(bounds['minx'] - width * padding, 
                        bounds['maxx'] + width * padding)
        self.ax.set_ylim(bounds['miny'] - height * padding, 
                        bounds['maxy'] + height * padding)

    def show(self):
        """Display the plot."""
        plt.show()

    def save(self, filename):
        """Save the plot to a file."""
        plt.savefig(filename, bbox_inches='tight', dpi=300)

# Example usage
if __name__ == "__main__":
    # Example GeoJSON data with elevation (z-coordinate)
    example_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 0, 0]
                    ]]
                }
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [0.5, 0.5, 0.5]
                }
            }
        ]
    }

    # Create renderer with custom projection angles
    renderer = GeoJSON3DRenderer(projection_angle=(45, 35, 0))
    renderer.render_geojson(example_geojson)
    renderer.show()