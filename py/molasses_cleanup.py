#!/usr/bin/env python3
import os
import numpy as np
import fire

def cleanup_asc_file(input_file, output_file):
    """
    Clean up an ASC file with nan values, replacing them with a standard NODATA value
    and ensuring proper formatting.
    
    Args:
        input_file (str): Path to the input ASC file
        output_file (str): Path to the output cleaned ASC file
    """
    print(f"Cleaning up {input_file}...")
    
    # Read the header and data
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    ncols = int(lines[0].split()[1])
    nrows = int(lines[1].split()[1])
    xllcorner = float(lines[2].split()[1])
    yllcorner = float(lines[3].split()[1])
    cellsize = float(lines[4].split()[1])
    
    # Change NODATA value from 'nan' to a standard value (-9999)
    nodata_value = -9999
    
    # Create the new header
    header = [
        f"ncols        {ncols}\n",
        f"nrows        {nrows}\n",
        f"xllcorner    {xllcorner}\n",
        f"yllcorner    {yllcorner}\n",
        f"cellsize     {cellsize}\n",
        f"NODATA_value {nodata_value}\n"
    ]
    
    # Process data lines
    data_lines = []
    for i in range(6, len(lines)):
        values = lines[i].strip().split()
        # Replace 'nan' with the nodata_value
        values = [str(nodata_value) if val == 'nan' else val for val in values]
        data_lines.append(' '.join(values) + '\n')
    
    # Write the new file
    with open(output_file, 'w') as f:
        f.writelines(header)
        f.writelines(data_lines)
    
    print(f"Cleaned file saved to {output_file}")
    print(f"NODATA values (nan) were replaced with {nodata_value}")

def main(input_file=None, output_file=None):
    """
    Clean up an ASC file with nan values to make it compatible with QGIS.
    
    Args:
        input_file (str, optional): Path to the input ASC file. If not provided, 
                                    defaults to 'dem/molasses (1).asc' in the repo.
        output_file (str, optional): Path to the output cleaned ASC file. If not provided,
                                    defaults to adding '_cleaned' to the input filename.
    """
    # If input_file not provided, use default
    if input_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 
                                 "dem", "molasses (1).asc")
    
    # If output_file not provided, create default name
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_cleaned{ext}"
    
    cleanup_asc_file(input_file, output_file)

if __name__ == "__main__":
    fire.Fire(main)
