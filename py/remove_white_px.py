import numpy as np
from PIL import Image
import os
import fire

def make_white_transparent(input_path, output_path):
    """
    Opens a PNG image, makes all pure white pixels transparent,
    and saves the result as a new PNG file.

    Args:
        input_path (str): Path to the input PNG image.
        output_path (str): Path where the modified PNG image will be saved.
    """
    try:
        # Open the image using Pillow
        img = Image.open(input_path)

        # Ensure the image has an alpha channel (RGBA)
        # If it's RGB, it will be converted, adding an alpha channel
        # fully opaque (255) by default.
        img = img.convert("RGBA")

        # Convert the image data to a NumPy array
        data = np.array(img)

        # --- Identify White Pixels (R=255, G=255, B=255) ---
        # Extract the RGB channels (first 3 values)
        rgb_channels = data[:, :, :3]

        # Create a boolean mask where all RGB channels are 255
        # This mask will be True for white pixels, False otherwise
        white_pixels_mask = np.all(rgb_channels == [255, 255, 255], axis=2)

        # --- Modify Alpha Channel ---
        # Use the mask to set the alpha channel (4th value) to 0 (transparent)
        # for the pixels identified as white.
        data[white_pixels_mask, 3] = 0

        # --- Create and Save New Image ---
        # Convert the modified NumPy array back to a Pillow Image object
        new_img = Image.fromarray(data)

        # Save the new image as a PNG (to preserve transparency)
        new_img.save(output_path, "PNG")

        print(f"Successfully processed '{input_path}'")
        print(f"Output saved as '{output_path}'")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
    except Exception as e:
        print(f"An unexpected error occurred processing '{input_path}': {e}")


if __name__ == '__main__':
  fire.Fire(make_white_transparent)
