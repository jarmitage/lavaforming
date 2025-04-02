import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
import fire

def images_to_video_with_crossfade(
    image_folder, 
    video_name="output_video.avi", 
    fps=30, 
    crossfade_duration=1.0,
    display_duration=3.0,
    image_pattern="*.jpg"
):
    """
    Creates a video from a sequence of images with crossfade transitions.
    
    Parameters:
        image_folder (str): Path to the folder containing images
        video_name (str): Name of the output video file
        fps (int): Frames per second for the video
        crossfade_duration (float): Duration of crossfade in seconds
        display_duration (float): Duration each image is displayed in seconds
        image_pattern (str): Pattern to match image files (e.g., "*.jpg", "*.png")
    """
    # Get list of images
    images = sorted(glob.glob(os.path.join(image_folder, image_pattern)))
    
    # If no images found
    if not images:
        tqdm.write(f"No images found in '{image_folder}' with pattern '{image_pattern}'.")
        return
    
    tqdm.write(f"Found {len(images)} images.")
    
    # Calculate frames per transition based on FPS and crossfade duration
    transition_frames = int(fps * crossfade_duration)
    display_frames = int(fps * display_duration)
    
    # Read the first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    
    # Calculate total frames for progress bar
    total_frames = (len(images) - 1) * (display_frames + transition_frames) + display_frames
    
    with tqdm(total=total_frames, desc="Creating video") as pbar:
        # Process each pair of consecutive images
        for i in range(len(images) - 1):
            # Read the current and next image
            img1 = cv2.imread(images[i])
            img2 = cv2.imread(images[i + 1])
            
            # Write the current image for its display duration
            for _ in range(display_frames):
                video.write(img1)
                pbar.update(1)
            
            # Create the transition frames
            for j in range(transition_frames):
                # Calculate the blend factor (0 to 1)
                alpha = j / transition_frames
                
                # Blend the images
                blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
                
                # Write the blended frame
                video.write(blended)
                pbar.update(1)
        
        # Add the last image for its display duration
        last_img = cv2.imread(images[-1])
        for _ in range(display_frames):
            video.write(last_img)
            pbar.update(1)
    
    # Release everything when job is finished
    video.release()
    tqdm.write(f"Video created successfully: {video_name}")
    
    # Return stats about the created video
    return {
        "images_processed": len(images),
        "video_frames": total_frames,
        "video_duration_seconds": total_frames / fps,
        "output_file": video_name
    }

def main(
    image_folder, 
    output="output_video.avi", 
    fps=30, 
    crossfade=1.0,
    display=3.0,
    pattern="*.jpg"
):
    """
    Command-line interface for creating videos from images with crossfade transitions.
    
    Args:
        image_folder: Path to the folder containing images
        output: Name of the output video file (default: output_video.avi)
        fps: Frames per second for the video (default: 30)
        crossfade: Duration of crossfade in seconds (default: 1.0)
        display: Duration each image is displayed in seconds (default: 3.0)
        pattern: Pattern to match image files (default: *.jpg)
    
    Returns:
        Dictionary with stats about the created video
    """
    return images_to_video_with_crossfade(
        image_folder=image_folder,
        video_name=output,
        fps=fps,
        crossfade_duration=crossfade,
        display_duration=display,
        image_pattern=pattern
    )

if __name__ == "__main__":
    fire.Fire(main)