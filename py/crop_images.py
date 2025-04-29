import os
import fire
from PIL import Image
from glob import glob
from tqdm import tqdm

def crop_images(
    input_folder, 
    output_folder=None,
    x=0, 
    y=0, 
    width=None, 
    height=None,
    extensions=('jpg', 'jpeg', 'png', 'bmp', 'tiff')
):
    """
    Crop all images in a folder to the specified dimensions.
    
    Args:
        input_folder (str): Path to the folder containing images to crop
        output_folder (str, optional): Path to save cropped images. If None, overwrites original images
        x (int): X-coordinate of the top-left corner of the crop box
        y (int): Y-coordinate of the top-left corner of the crop box
        width (int, optional): Width of the crop. If None, crops to the image width
        height (int, optional): Height of the crop. If None, crops to the image height
        extensions (tuple): Image file extensions to process
    """
    # Create output folder if specified and doesn't exist
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Find all image files with the specified extensions
    image_files = []
    for ext in extensions:
        pattern = os.path.join(input_folder, f'*.{ext}')
        image_files.extend(glob(pattern))
        # Add uppercase extensions too
        pattern = os.path.join(input_folder, f'*.{ext.upper()}')
        image_files.extend(glob(pattern))
    
    tqdm.write(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_path in tqdm(image_files, desc="Cropping images"):
        try:
            # Open the image
            with Image.open(img_path) as img:
                # Get image dimensions
                img_width, img_height = img.size
                
                # Set default width and height if not specified
                crop_width = width if width is not None else img_width - x
                crop_height = height if height is not None else img_height - y
                
                # Calculate right and bottom coordinates
                right = x + crop_width
                bottom = y + crop_height
                
                # Ensure crop dimensions are valid
                if x >= img_width or y >= img_height or right > img_width or bottom > img_height:
                    tqdm.write(f"Invalid crop dimensions for {img_path}. Skipping.")
                    continue
                
                # Crop the image
                cropped_img = img.crop((x, y, right, bottom))
                
                # Determine output path
                if output_folder:
                    base_name = os.path.basename(img_path)
                    output_path = os.path.join(output_folder, base_name)
                else:
                    output_path = img_path
                
                # Save cropped image
                cropped_img.save(output_path)
                # tqdm.write(f"Cropped: {img_path} -> {output_path}")
                
        except Exception as e:
            tqdm.write(f"Error processing {img_path}: {e}")
    
    tqdm.write("Crop operation completed")

if __name__ == "__main__":
    fire.Fire(crop_images)