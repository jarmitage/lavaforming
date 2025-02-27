import numpy as np

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

# Example usage:
# convert_asc_for_qgis('molasses.asc', 'molasses_qgis.asc')
