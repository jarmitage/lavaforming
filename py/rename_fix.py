import os
import re

# Set the directory containing your files
directory = "/Users/jackarmitage/Dropbox/downloads/lavadata/20250414_124137_eldvorp_md_2x2_trench_sketch_3_326699_378345/"  # Update this to your directory path

# Define the padding length (how many digits you want in total)
padding_length = 10  # Adjust this based on your largest number

# Regex pattern to match the volume number at the end before the file extension
pattern = re.compile(r'(eldvorp_md_2x2_trench_sketch_3_326699_378345_)(\d+)(\.tif)')

# Get list of files in the directory
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Track renames for reporting
renamed_files = []
skipped_files = []

# Process each file
for filename in files:
    match = pattern.match(filename)
    if match:
        # Extract the parts
        prefix = match.group(1)
        volume_str = match.group(2)
        extension = match.group(3)
        
        # Zero pad the volume number
        padded_volume = volume_str.zfill(padding_length)
        
        # Create the new filename
        new_filename = f"{prefix}{padded_volume}{extension}"
        
        # Only rename if the filename actually changes
        if new_filename != filename:
            original_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            # Perform the rename
            os.rename(original_path, new_path)
            renamed_files.append((filename, new_filename))
        else:
            skipped_files.append(filename)
    else:
        skipped_files.append(filename)

# Print summary
print(f"Renamed {len(renamed_files)} files:")
for old, new in renamed_files:
    print(f"  {old} -> {new}")

print(f"\nSkipped {len(skipped_files)} files that didn't match the pattern or already had correct padding.")