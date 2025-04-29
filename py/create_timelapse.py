import os
import fire
import gdal_contour_script as gcs 
import crop_images as ci
import video_maker as vm
import shutil
def main(
    dem_path: str | None = None,
    output_dir: str | None = None,
    contour_interval: int = 10,
    lava_dir: str | None = None,
    vent: tuple[int, int] = (326746, 376249),
    lava_zoom: float = 1.0,
    vent_zoom: float = 1.0,
    cmap: str = 'bone',
    show_window: bool = False,
    show_elevation_bar: bool = False,
    show_thickness_bar: bool = False,
    cleanup: bool = False
):
    """
    Generates contour lines from a DEM, processes lava flow raster data,
    and creates timelapse plots overlaying lava flow onto the DEM with contours.
    """
    model_name = None
    if 'molasses' in lava_dir:
        model_name = 'molasses'
    elif 'lava2d' in lava_dir:
        model_name = 'lava2d'
    else:
        raise ValueError(f"Unknown model name: {lava_dir}")

    lava_dir_basename = f"{os.path.basename(lava_dir)}_{model_name}"
    img_dir = f"{output_dir}/{lava_dir_basename}/images"
    cropped_dir = f"{output_dir}/{lava_dir_basename}/cropped"
    contour_dir = f"{output_dir}/{lava_dir_basename}/contours"
    contour_shapefile_path = os.path.join(contour_dir, 'generated_contours.shp')
    timelapse_path = f"{output_dir}/{lava_dir_basename}/{lava_dir_basename}.mp4"

    skip_create_contours = False
    skip_process_lava = False
    skip_crop_images = False
    skip_video_maker = False

    if os.path.exists(contour_shapefile_path):
        skip_create_contours = True
        print(f"Skipping generation. Contour shapefile already exists at {contour_shapefile_path}.")

    if os.path.exists(img_dir):
        skip_process_lava = True
        print(f"Skipping processing. Images already exist at {img_dir}.")

    if os.path.exists(cropped_dir):
        skip_crop_images = True
        print(f"Skipping cropping. Cropped images already exist at {cropped_dir}.")

    if os.path.exists(timelapse_path):
        skip_video_maker = True
        print(f"Skipping video creation. Timelapse video already exists at {timelapse_path}.")

    os.makedirs(output_dir, exist_ok=True)

    if not skip_create_contours:
        gcs.create_contours_with_gdal(
            dem_path=dem_path,
            output_vector=contour_shapefile_path,
            interval=contour_interval
        )

    if not skip_process_lava:
        os.makedirs(img_dir, exist_ok=True)
        gcs.process_lava_folder(
            lava_folder=lava_dir,
            output_path=img_dir,
            input_file=dem_path,
            contour_file=contour_shapefile_path,
            interval=contour_interval,
            lava_zoom=lava_zoom,
            vent_coords=vent,
            vent_zoom=vent_zoom,
            cmap=cmap,
            show_window=show_window,
            show_elevation_bar=show_elevation_bar,
            show_thickness_bar=show_thickness_bar
        )

    if not skip_crop_images:
        os.makedirs(cropped_dir, exist_ok=True)
        ci.crop_images(
            input_folder=img_dir,
            output_folder=cropped_dir,
            x=450,
            y=650,
            width=3150,
            height=2000
        )

    if not skip_video_maker:
        vm.images_to_video_with_crossfade(
            image_folder=cropped_dir,
            video_name=timelapse_path,
            fps=60,
            crossfade_duration=1.0,
            display_duration=0.5,
            image_pattern="*.png"
        )

    if cleanup:
        if os.path.exists(timelapse_path):
            os.rename(timelapse_path, os.path.join(output_dir, f"{lava_dir_basename}.mp4"))
        shutil.rmtree(img_dir)
        shutil.rmtree(cropped_dir)
        shutil.rmtree(contour_dir)
        shutil.rmtree(f"{output_dir}/{lava_dir_basename}")

if __name__ == "__main__":
    fire.Fire(main)
