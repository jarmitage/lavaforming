import numpy as np
import pandas as pd
import fire
from typing import Tuple, Dict, Optional

class ASCGridParser:
    """
    Parser for ESRI ASCII grid format (.asc) files.
    Handles parsing, data extraction, and basic analysis of grid data.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the parser with a file path.
        
        Args:
            filepath (str): Path to the .asc file
        """
        self.filepath = filepath
        self.header = {}
        self.data = None
        self._parse_file()
    
    def _parse_header(self, header_lines: list) -> Dict:
        """
        Parse the six-line header of the ASC file.
        
        Args:
            header_lines (list): First six lines of the ASC file
            
        Returns:
            dict: Header information
        """
        header = {}
        expected_keys = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 
                        'cellsize', 'NODATA_value']
        
        for line, key in zip(header_lines, expected_keys):
            value = line.split()[1]
            # Convert numeric values to appropriate type
            if key != 'NODATA_value':
                header[key] = float(value)
                if key in ['ncols', 'nrows']:
                    header[key] = int(header[key])
            else:
                try:
                    header[key] = float(value)
                except ValueError:
                    header[key] = value
                    
        return header
    
    def _parse_file(self):
        """Parse the entire ASC file, storing header and data."""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
            
        # Parse header (first 6 lines)
        self.header = self._parse_header(lines[:6])
        
        # Parse data
        data_rows = []
        for line in lines[6:]:
            # Split line and convert values to float, replacing 'nan' with np.nan
            row = [float(x) if x.lower() != 'nan' else np.nan 
                  for x in line.strip().split()]
            if row:  # Skip empty lines
                data_rows.append(row)
        
        self.data = np.array(data_rows)
    
    def get_statistics(self) -> Dict:
        """
        Calculate basic statistics of the grid data.
        
        Returns:
            dict: Statistical measures of the data
        """
        valid_data = self.data[~np.isnan(self.data)]
        
        return {
            'total_cells': self.data.size,
            'valid_cells': valid_data.size,
            'min_value': np.min(valid_data),
            'max_value': np.max(valid_data),
            'mean_value': np.mean(valid_data),
            'median_value': np.median(valid_data),
            'std_dev': np.std(valid_data)
        }
    
    def get_xy_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate X and Y coordinates for each grid cell.
        
        Returns:
            tuple: Arrays of X and Y coordinates
        """
        x_coords = np.arange(
            self.header['xllcorner'],
            self.header['xllcorner'] + self.header['ncols'] * self.header['cellsize'],
            self.header['cellsize']
        )
        
        y_coords = np.arange(
            self.header['yllcorner'],
            self.header['yllcorner'] + self.header['nrows'] * self.header['cellsize'],
            self.header['cellsize']
        )
        
        return x_coords, y_coords
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the grid data to a pandas DataFrame with coordinates.
        
        Returns:
            pd.DataFrame: DataFrame containing the grid data and coordinates
        """
        x_coords, y_coords = self.get_xy_coords()
        
        # Create meshgrid of coordinates
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Create DataFrame
        df = pd.DataFrame({
            'X': X.flatten(),
            'Y': Y.flatten(),
            'Value': self.data.flatten()
        })
        
        return df
    
    def get_contour_data(self, min_value: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for contour plotting.
        
        Args:
            min_value (float, optional): Minimum value to include in contour data
            
        Returns:
            tuple: X coordinates, Y coordinates, and Z values for contour plotting
        """
        x_coords, y_coords = self.get_xy_coords()
        X, Y = np.meshgrid(x_coords, y_coords)
        
        Z = self.data.copy()
        if min_value is not None:
            Z[Z < min_value] = np.nan
            
        return X, Y, Z

def convert_asc_for_qgis(input_file, output_file):
    """
    Convert ASC file to QGIS-compatible format.
    
    Args:
        input_file (str): Path to input ASC file
        output_file (str): Path to output ASC file
    """
    # Read the original file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Fix header formatting and NODATA value
    header = {
        'ncols': int(lines[0].split()[1]),
        'nrows': int(lines[1].split()[1]),
        'xllcorner': float(lines[2].split()[1]),
        'yllcorner': float(lines[3].split()[1]),
        'cellsize': float(lines[4].split()[1])
    }
    
    # Create new header with proper formatting
    new_header = [
        f"ncols {header['ncols']}",
        f"nrows {header['nrows']}",
        f"xllcorner {header['xllcorner']}",
        f"yllcorner {header['yllcorner']}",
        f"cellsize {header['cellsize']}",
        "NODATA_value -9999"
    ]
    
    # Process data lines
    data_lines = []
    for line in lines[6:]:
        if line.strip():
            values = line.split()
            # Replace 'nan' with -9999
            values = ['-9999' if v.lower() == 'nan' else v for v in values]
            data_lines.append(' '.join(values))
    
    # Write the new file
    with open(output_file, 'w') as f:
        # Write header
        f.write('\n'.join(new_header) + '\n')
        # Write data
        f.write('\n'.join(data_lines))

def main(**kwargs):
    # Initialize parser with file path
    parser = ASCGridParser(kwargs.get("i"))
    
    # Print header information
    print("Header Information:")
    for key, value in parser.header.items():
        print(f"{key}: {value}")
    
    # Print basic statistics
    print("\nData Statistics:")
    stats = parser.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Convert to DataFrame
    df = parser.to_dataframe()
    print("\nDataFrame Preview:")
    print(df.head())
    
    # Get contour data
    X, Y, Z = parser.get_contour_data(min_value=0.0)

    if kwargs.get("fix"):
        convert_asc_for_qgis(kwargs.get("i"), kwargs.get("o"))

    return stats, df, X, Y, Z

if __name__ == "__main__":
    fire.Fire(main)
