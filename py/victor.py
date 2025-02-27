import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rxr
import numpy as np
import cartopy.crs as ccrs
import rasterio as rio
import imageio
import h5py
import pandas as pd
import boto3
from google.cloud import storage
from botocore.exceptions import ClientError
import xrspatial as xrs
# from azure.storage.blob import BlobServiceClient
from datetime import datetime
import urllib
import os
import utm
import pyproj
import matplotlib.colors as colors
import geopandas as gpd
from geocube.api.core import make_geocube
import subprocess
import requests

def hillshade(array, azimuth, angle_altitude):
    """
    Shades a raster with a given azimuth and angle for clearer visuals.

    Parameters
    ----------
    array : ndarray
        Raster data
    azimuth : float
        Horizontal angle from north (degrees)
    angle_altitude : float
        Angle of "sun" shading (degrees)

    Returns
    -------
    ndarray
        Shaded raster data
    """
    # Subtract azimuth from 360 to get azimuth in Cartesian coordinates
    azimuth = 360.0 - azimuth 
    
    # Calculate gradient in x and y directions
    x, y = np.gradient(array)
    
    # Calculate slope and aspect
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    
    # Convert azimuth and angle_altitude to radians
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.
    
    # Calculate shaded raster data
    shaded = np.sin(altituderad)*np.sin(slope) + \
             np.cos(altituderad)*np.cos(slope)* \
             np.cos((azimuthrad - np.pi/2.) - aspect)
    
    # Scale shaded raster data to range [0, 255]
    return 255*(shaded + 1)/2

def plot_dem(dem, markercoords=np.array([]), title=None):
    """
    Experimental raster plotting function that does not require figure and axes to be passed.

    Parameters
    ----------
    dem : str
        Relative or absolute path to raster (can be either ASCII or TIFF)
    coords : ndarray, optional
        Optional ndarray to plot points for coordinates
    title : str, optional
        Optional string to title plotted raster

    Returns
    -------
    None
    """
    # Open raster file using rasterio
    raster = rxr.open_rasterio(dem)
    # Select the first band of the raster
    raster = raster.sel({"band": 1})
    # Apply hillshade to the raster and plot it
    render = xrs.hillshade(raster, 45, 30)
    render.plot(cmap="gray",add_colorbar=False)
    # If coordinates are provided, scatter plot them
    if markercoords.any():
        if markercoords.ndim == 2:
            plt.scatter(markercoords[:, 0], markercoords[:, 1], color="red", marker="^")
        elif markercoords.ndim == 1:
            plt.scatter(markercoords[0], markercoords[1], color="red", marker="^")
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    # Set title of the plot
    if title is None:
        plt.title(dem)
    else:
        plt.title(title)
    
def plot_flow(dem, flow, coords=np.array([]), zoom=True, label="Thickness (m)", title=None, lognorm=False,axes=None):
    """
    Experimental raster+flow plotting, does not require figure and axes to be passed.

    Parameters
    ----------
    dem : str
        Relative or absolute path to raster (can be either ASCII or TIFF)
    flow : str
        Relative or absolute path to flow data (can be ASCII, TIFF, or CSV)
    coords : ndarray, optional
         ndarray to plot points for coordinates
    zoom : bool, optional
        Flag to display a section more tightly around AOI.
    label : str, optional
        String to label flow colorbar descriptor

    Returns
    -------
    None
    """
    if axes:
        ax = axes
    else:
        # Create a new axes object
        ax = plt.axes()

    raster = rxr.open_rasterio(dem)
    # Select the first band of the raster
    raster = raster.sel({"band": 1})
    # Apply hillshade to the raster and plot it
    render = xrs.hillshade(raster, 45, 30)
    render.plot(ax=ax,cmap="gray",add_colorbar=False)
    # If coordinates are provided, scatter plot them
    if coords.any():
        if coords.ndim == 2:
            plt.scatter(coords[:, 0], coords[:, 1], color="red", marker="^")
        elif coords.ndim == 1:
            plt.scatter(coords[0], coords[1], color="red", marker="^")
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    # Set title of the plot
    if title is None:
        plt.title(dem)
    else:
        plt.title(title)
    
    # Check the file extension of the flow data
    filename, file_extension = os.path.splitext(flow)
    if file_extension == ".csv":
        # If the flow data is in CSV format, read it into a pandas DataFrame
        flow_data = pd.read_csv(flow)
        # Plot the flow data with color mapping to the 'THICKNESS' column
        sc = plt.plot(x=flow_data["EAST"], y=flow_data["NORTH"], c=flow_data["THICKNESS"], cmap="hot")
        plt.colorbar(sc, label=label, shrink=.6)
        
        # If zoom is True, set the x and y limits of the plot to the min and max values of the 'EAST' and 'NORTH' columns
        if zoom:
            ax.set_xlim(flow_data["EAST"].min(), flow_data["EAST"].max())
            ax.set_ylim(flow_data["NORTH"].min(), flow_data["NORTH"].max())
    else:
        # If the flow data is in raster format, open it using rasterio and select the first band
        
        flow_raster = rxr.open_rasterio(flow,masked=True)
        flow_raster = flow_raster.sel({"band": 1})
        flow_clipped = flow_raster.dropna('x',how='all')
        flow_clipped = flow_clipped.dropna('y',how='all')
        
        if lognorm:
            flow_render = flow_raster.plot(ax=ax, cmap="hot", norm=colors.LogNorm(vmin=flow_thickness.min(), vmax=flow_thickness.max()), cbar_kwargs={'label': label, 'shrink': .6})
        else:
            flow_render = flow_raster.plot(ax=ax, cmap="hot", cbar_kwargs={'label': label, 'shrink': .6})
        
        # If zoom is True, set the x and y limits of the plot to the min and max values of the x and y coordinates
        if zoom:
            padding_x = (flow_clipped.x.max() - flow_clipped.x.min())/4
            padding_y = (flow_clipped.y.max() - flow_clipped.y.min())/4
            ax.set_xlim(flow_clipped.x.min()-padding_x, flow_clipped.x.max()+padding_x)
            ax.set_ylim(flow_clipped.y.min()-padding_y, flow_clipped.y.max()+padding_y)
            
    if title is None:
        plt.title(dem)
    else:
        plt.title(title)
    
def plot_dem_deprecated(dem, fig, ax, coords=np.array([]), epsg=32628):
    """
    Standard raster plotting, requires geoaxes and figure to be passed.

    Parameters
    ----------
    dem : str
        Relative or absolute path to raster (can be either ASCII or TIFF)
    fig : matplotlib.figure.Figure
        Matplotlib figure to assign raster
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Matplotlib geoaxes to assign raster
    coords : numpy.ndarray, optional 
        ndarray to plot points for coordinates. Defaults to an empty array.
    epsg : int, optional
        EPSG code for projection, recommended to include same input as geoaxes. Defaults to 32628.

    Returns
    -------
    None
    """
    # Flag for UTM coordinates
    utm = False

    # Check the file extension of the DEM
    filename, file_extension = os.path.splitext(dem)

    if file_extension in [".geotiff", ".tiff", ".tif"]:
        # If the DEM is in TIFF format, read it using rasterio and select the first band
        raster = rio.open(dem)
        read_raster = raster.read()

        # Plot the raster with hillshade and specified color map and extent
        ax.imshow(hillshade(read_raster[0,:,:],120,30), cmap='Greys', vmin=0, vmax=300,
                  transform=ccrs.epsg(epsg),
                  extent=(raster.bounds.left, raster.bounds.right, raster.bounds.bottom, raster.bounds.top))

    elif file_extension in [".asc", ".ascii"]:
        # If the DEM is in ASCII format, open it using rasterio and select the first band
        dem_check = rxr.open_rasterio(dem).drop('band')[0]

        # Check if the coordinates are in UTM format
        if dem_check.x.max() > 180 or dem_check.y.max() > 90 or dem_check.x.min() < -180 or dem_check.y.min() < -90:
            dem_check = dem_check.rename({'x':'easting', 'y':'northing'})
            utm = True
        else:
            dem_check = dem_check.rename({'x':'long', 'y':'lat'})

        # Set nodata value to NaN
        if "_FillValue" in dem_check.attrs:
            nodata = dem_check.attrs["_FillValue"]
            dem_check = dem_check.where(dem_check!=nodata, np.nan)

        # Plot the raster with hillshade and specified color map and extent
        if utm:
            ax.imshow(hillshade(dem_check,120,30), cmap="gray", transform=ccrs.epsg(epsg),
                      extent=(min(dem_check["easting"]), max(dem_check["easting"]), min(dem_check["northing"]), max(dem_check["northing"])))
            ax.set_xticks([min(dem_check["easting"]), max(dem_check["easting"])])
            ax.set_yticks([min(dem_check["northing"]), max(dem_check["northing"])])
        else:
            ax.imshow(hillshade(dem_check,120,30), cmap="gray", transform=ccrs.epsg(epsg))
            ax.set_xticks([min(dem_check["long"]), max(dem_check["long"])])
            ax.set_yticks([min(dem_check["lat"]), max(dem_check["lat"])])

        # Plot points for coordinates if provided
        if not coords.any():
            pass
        elif coords.ndim == 2:
            ax.scatter(coords[:,0], coords[:,1], marker="^", c="red")
        else:
            ax.scatter(coords[0], coords[1], marker="^", c="red")

    # Show the plot
    plt.show()
    
def plot_flow_deprecated(dem, flow, fig, ax, coords, zoom=True, model=None, epsg=32628, label="Thickness of residual (m)", lognorm=False):
    """
    Standard raster+flow plotting, requires figure and geoaxes variables to be passed.

    Parameters
    ----------
    dem : str
        Relative or absolute path to raster (can be either ASCII or TIFF)
    flow : str
        Relative or absolute path to flow data (can be ASCII, TIFF, or CSV)
    fig : Matplotlib figure
        Matplotlib figure to assign raster
    ax : Matplotlib geoaxes
        Matplotlib geoaxes to assign raster
    coords : ndarray or None
        ndarray to plot points for coordinates (optional)
    zoom : bool, optional
        Flag to display a section more tightly around AOI.
    model : str, optional
        String specifically used to specify Mr Lava Loba due to unique output format
    epsg : int, optional
        Projection, recommended to include same input as geoaxes.
    label : str, optional
        String to label flow colorbar descriptor
    lognorm : bool, optional
        Flag to apply logarithmic normalization to flow colorbar

    Returns
    -------
    None
    """
    # Convert coordinates to numpy array
    coords = np.array(coords)

    # Extract file extension
    filename, file_extension = os.path.splitext(dem)

    # Plot raster data based on file extension
    if file_extension == ".tif" or file_extension == ".geotiff":
        raster = rio.open(dem)
        read_raster = raster.read()
        ax.imshow(hillshade(read_raster[0,:,:],120,30),cmap='Greys',vmin=0,vmax=300,transform=ccrs.epsg(epsg),
                 extent=(raster.bounds.left, raster.bounds.right, raster.bounds.bottom, raster.bounds.top))
        ct = rxr.open_rasterio(dem).drop('band')[0].plot.contour(ax=ax,
                                                                 cmap=plt.cm.copper,
                                                                 transform=ccrs.epsg(epsg))
        x_full_min, x_full_max = raster.bounds.left, raster.bounds.right
        y_full_min, y_full_max = raster.bounds.bottom, raster.bounds.top
    elif file_extension == ".asc" or file_extension == ".ascii":
        raster = rxr.open_rasterio(dem).drop('band')[0].rename({'x':'easting', 'y':'northing'})
        if "_FillValue" in raster.attrs:
            nodata = raster.attrs["_FillValue"]
            raster = raster.where(raster!=nodata, np.nan)
        ax.imshow(hillshade(raster,120,30),cmap="gray",transform=ccrs.epsg(epsg), 
                 extent=(min(raster["easting"]), max(raster["easting"]), min(raster["northing"]), max(raster["northing"])))
        ct = raster.plot.contour(ax=ax,
                                 cmap=plt.cm.copper,
                                 transform=ccrs.epsg(epsg))
        x_full_min, x_full_max = min(raster["easting"]).values, max(raster["easting"]).values
        y_full_min, y_full_max = min(raster["northing"]).values, max(raster["northing"]).values

    # Add contour lines to plot
    ax.clabel(ct, ct.levels, inline=True, fontsize=9,colors="red")

    # Plot flow thickness data
    flow_thickness = rxr.open_rasterio(flow).drop('band')[0].rename({'x':'easting', 'y':'northing'})
    nodata = flow_thickness.attrs["_FillValue"]
    flow_thickness = flow_thickness.where(flow_thickness!=nodata, np.nan)

    # Calculate zoom extent based on flow data
    x_zoom_min, x_zoom_max = min(flow_thickness["easting"]).values-1000, max(flow_thickness["easting"]).values+1000
    y_zoom_min, y_zoom_max = min(flow_thickness["northing"]).values-1000, max(flow_thickness["northing"]).values+1000

    # Apply logarithmic normalization if specified
    if lognorm:
        flow_thickness.plot(ax=ax,
                           cmap=plt.cm.hot,
                           cbar_kwargs={'label': label, 'shrink': .6},
                           norm=colors.LogNorm(vmin=flow_thickness.min(), vmax=flow_thickness.max()),
                           transform=ccrs.epsg(epsg))
    else:
        flow_thickness.plot(ax=ax,
                           cmap=plt.cm.hot,
                           cbar_kwargs={'label': label, 'shrink': .6},
                           transform=ccrs.epsg(epsg))

    # Plot coordinates if provided
    if coords.ndim == 2:
        x,y = coords[:,0], coords[:,1]
    else:
        x,y = coords[0],coords[1]
    ax.scatter(x,y,marker="^",c="black")

    # Set plot extent based on zoom flag or model
    if not zoom:
        if file_extension in [".asc", ".ascii"]:
            x_min, x_max = x_full_min,x_full_max
            y_min, y_max = y_full_min,y_full_max
        elif file_extension in [".tif", ".geotiff", ".tiff"]:
            temp = rio.open(dem)
            left,bottom,right,top = temp.bounds
            x_min, x_max = left, right
            y_min, y_max = bottom, top
    elif model=="mrlavaloba":
        y1 = flow_thickness.idxmin(dim="northing")
        x1 = flow_thickness.idxmin(dim="easting")
        y2 = flow_thickness.idxmax(dim="northing")
        x2 = flow_thickness.idxmax(dim="easting")
        x_min = min(x1[~np.isnan(x1)]) - 1000
        x_max = max(x2[~np.isnan(x2)]) + 1000
        y_min = min(y1[~np.isnan(y1)]) - 1000
        y_max = max(y2[~np.isnan(y2)]) + 1000
    else:
        x_min, x_max = x_zoom_min,x_zoom_max
        y_min, y_max = y_zoom_min,y_zoom_max
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_xticks(np.linspace(x_min, x_max,5))
    ax.set_yticks(np.linspace(y_min, y_max,10))
    ax.set_title('Lava Flow', fontsize=16)
      
def plot_titan(dem, step, fig, ax, coords, diter, zoom=True, epsg=32628, save_csv=True, sim_dir = '.'):
    """
    Specialized plotting function for TITAN2D.

    Parameters
    ----------
    dem : str
        Relative or absolute path to raster (can be either ASCII or TIFF)
    step : int
        Integer to specify step number of titan output.
    fig : matplotlib.figure.Figure
        Matplotlib figure to assign raster
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Matplotlib geoaxes to assign raster
    coords : numpy.ndarray or None
        Optional ndarray to plot points for coordinates
    diter : int
        Number of iteration intervals for Titan2D output
    zoom : bool, optional
        Flag to display a section more tightly around AOI.
    epsg : int, optional
        Projection, recommended to include same input as geoaxes.
    save_csv : bool, optional
        Flag to save output data to CSV format (more readable than xdmf)
    sim_dir : str, optional
        Parent directory of vizout

    Returns
    -------
    None
    """

    # Format step number
    filled = str(step*diter).zfill(8)

    # Read xdmf file and extract relevant data
    height = []
    filename = "".join([sim_dir,"/vizout/","xdmf_p0000_i",filled,".h5"])
    with h5py.File(filename, 'r')  as h5f: # file will be closed when we exit from WITH scope
        connections = h5f.get("Mesh/Connections")
        points = h5f.get("Mesh/Points")
        height = h5f.get("Properties/PILE_HEIGHT")[:]
        centers = []
        for i in range(connections.shape[0]):
            midpoint = points[np.sort(connections[i,:])]
            out = np.mean(midpoint,axis=0)
            centers = np.append(centers,out,axis=0)
        centers = np.reshape(centers,(connections.shape[0],3))
    height = np.ndarray.flatten(height)
    df = {"X_CENTER": centers[:,0], 'Y_CENTER': centers[:,1], 'Z_CENTER': centers[:,2],'PILE_HEIGHT': height}
    df = pd.DataFrame(df)

    # Save data to CSV if save_csv flag is True
    if save_csv:
        df.to_csv("./titandata.csv")

    # Filter data based on nonzero pile height
    nonzero = df["PILE_HEIGHT"] > 0
    flow = df[nonzero]

    # Plot raster data
    raster = rio.open(dem)
    read_raster = raster.read()
    ax.imshow(hillshade(read_raster[0,:,:],120,30),cmap='Greys',vmin=0,vmax=300,transform=ccrs.epsg(epsg),
                 extent=(raster.bounds.left, raster.bounds.right, raster.bounds.bottom, raster.bounds.top))
    x_full_min, x_full_max = raster.bounds.left, raster.bounds.right
    y_full_min, y_full_max = raster.bounds.bottom, raster.bounds.top

    # Plot scatter plot of flow data
    sc = ax.scatter(x=flow["X_CENTER"], y=flow["Y_CENTER"], c=flow["PILE_HEIGHT"], s=1,
    cmap=plt.cm.hot)

    # Add colorbar
    cb = plt.colorbar(sc, shrink=.6)
    cb.set_label('Pile Height (m)', rotation=90)
    plt.title("Flow")

    # Set plot extent based on zoom flag
    x_zoom_min, x_zoom_max = min(flow["X_CENTER"])-1000, max(flow["X_CENTER"])+1000
    y_zoom_min, y_zoom_max = min(flow["Y_CENTER"])-1000, max(flow["Y_CENTER"])+1000
    ct = ax.contour(read_raster[0],
        cmap=plt.cm.copper,
        transform=ccrs.epsg(epsg)
    )
    ax.clabel(ct, ct.levels, inline=True, fontsize=9,colors="red")

    # Plot coordinates if provided
    x,y = coords[0],coords[1]
    ax.scatter(x,y,marker="^",c="black")

    # Set plot extent based on zoom flag or model
    if not zoom:
        x_min, x_max = x_full_min,x_full_max
        y_min, y_max = y_full_min,y_full_max
    else:
        x_min, x_max = x_zoom_min,x_zoom_max
        y_min, y_max = y_zoom_min,y_zoom_max
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_xticks(np.linspace(x_min, x_max,5))
    ax.set_yticks(np.linspace(y_min, y_max,10))
    ax.clabel(ct, ct.levels, inline=True, fontsize=9,colors="white")
    ax.set_title(f'''TITAN2D @ iteration {diter}''', fontsize=16)

    return None
    
def plot_benchmark(dem, flow, fig, ax, coords=None, zoom=True, model=None, label="Thickness of residual (m)", vmax=None, epsg=32628, outline=None):
    """
    Plots benchmark data and flow data on a geoaxes.

    Requires geoaxes and figure to be passed along with simulation step number.

    Parameters
    ----------
    dem : str
        Relative or absolute path to raster (can be either ASCII or TIFF)
    flow : str
        Relative or absolute path to flow data (can be ASCII, TIFF, or CSV)
    fig : matplotlib.figure.Figure
        Matplotlib figure to assign raster
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Matplotlib geoaxes to assign raster
    coords : numpy.ndarray or None, optional
        Optional ndarray to plot points for coordinates
    zoom : bool, optional
        Flag to display a section more tightly around AOI.
    model : str or None, optional
        String specifically used to specify Mr Lava Loba due to unique output format
    label : str, optional
        Optional string to label flow colorbar descriptor
    vmax : int or None, optional
        Optional integer to limit/scale maximum value
    epsg : int, optional
        Projection, recommended to include same input as geoaxes.
    outline : str or None, optional
        Relative or absolute path to raster to be used for contour outline

    Returns
    -------
    flow_plotted : matplotlib.collections.QuadMesh
        Plot object
    maxval : float
        Maximum value in flow data
    """
    # Extract file extension and open raster
    filename, file_extension = os.path.splitext(dem)
    if file_extension.lower() in [".tif", ".tiff", ".geotiff"]:
        raster = rio.open(dem)
        read_raster = raster.read()
    elif file_extension in [".asc", ".ascii"]:
        raster = rxr.open_rasterio(dem).drop('band')[0].rename({'x':'easting', 'y':'northing'})
        nodata = raster.attrs["_FillValue"]
        raster = raster.where(raster!=nodata, np.nan)

    # Plot raster data
    ax.imshow(hillshade(read_raster[0,:,:] if file_extension.lower() in [".tif", ".tiff", ".geotiff"] else raster,
                       225,25),cmap='Greys',vmin=0,transform=ccrs.epsg(epsg),
              extent=(raster.bounds.left if file_extension.lower() in [".tif", ".tiff", ".geotiff"] else min(raster["easting"]),
                      raster.bounds.right if file_extension.lower() in [".tif", ".tiff", ".geotiff"] else max(raster["easting"]),
                      raster.bounds.bottom if file_extension.lower() in [".tif", ".tiff", ".geotiff"] else min(raster["northing"]),
                      raster.bounds.top if file_extension.lower() in [".tif", ".tiff", ".geotiff"] else max(raster["northing"])))

    # Plot outline if specified
    if outline is not None:
        ct = rxr.open_rasterio(outline).drop('band')[0].plot.contour(ax=ax,
                                                                     cmap="white",
                                                                     transform=ccrs.epsg(epsg))

    # Plot flow data
    flow_thickness = rxr.open_rasterio(flow).drop('band')[0].rename({'x':'easting', 'y':'northing'})
    nodata = flow_thickness.attrs["_FillValue"]
    flow_thickness = flow_thickness.where(flow_thickness!=nodata, np.nan)
    maxval = float(flow_thickness.max())
    x_zoom_min, x_zoom_max = min(flow_thickness["easting"]).values-1000, max(flow_thickness["easting"]).values+1000
    y_zoom_min, y_zoom_max = min(flow_thickness["northing"]).values-1000, max(flow_thickness["northing"]).values+1000

    # Plot flow data with or without specified maximum value
    if vmax is None:
        flow_plotted = flow_thickness.plot(ax=ax,
                                           cmap=plt.cm.Wistia,
                                           add_colorbar=False,
                                           vmin=0, transform=ccrs.epsg(epsg))
    else:
        flow_plotted = flow_thickness.plot(ax=ax,
                                           cmap=plt.cm.Wistia,
                                           add_colorbar=False,
                                           vmin=0, vmax=vmax,
                                           transform=ccrs.epsg(epsg))

    # Plot coordinates if provided
    if coords is not None:
        if coords.ndim == 2:
             x,y = coords[:,0], coords[:,1]
        elif coords.ndim == 1:
            x,y = coords[0],coords[1]
    ax.scatter(x,y,marker="^",c="black")

    # Set plot extent based on zoom flag or model
    if not zoom:
        x_min, x_max = raster.bounds.left if file_extension.lower() in [".tif", ".tiff", ".geotiff"] else min(flow_thickness["easting"]), raster.bounds.right if file_extension.lower() in [".tif", ".tiff", ".geotiff"] else max(flow_thickness["easting"])
        y_min, y_max = raster.bounds.bottom if file_extension.lower() in [".tif", ".tiff", ".geotiff"] else min(flow_thickness["northing"]), raster.bounds.top if file_extension.lower() in [".tif", ".tiff", ".geotiff"] else max(flow_thickness["northing"])
    elif model == "mrlavaloba":
        y1 = flow_thickness.idxmin(dim="northing")
        x1 = flow_thickness.idxmin(dim="easting")
        y2 = flow_thickness.idxmax(dim="northing")
        x2 = flow_thickness.idxmax(dim="easting")
        x_min = min(x1[~np.isnan(x1)]) - 1000
        x_max = max(x2[~np.isnan(x2)]) + 1000
        y_min = min(y1[~np.isnan(y1)]) - 1000
        y_max = max(y2[~np.isnan(y2)]) + 1000
    else:
        x_min, x_max = x_zoom_min,x_zoom_max
        y_min, y_max = y_zoom_min,y_zoom_max
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_xticks(np.linspace(x_min, x_max,5))
    ax.set_yticks(np.linspace(y_min, y_max,5))

    return flow_plotted,maxval
    
def make_titan_gif(dem, fig, ax, coords, max_iter, diter, gif_name, epsg=32628, sim_dir='.'):
    """
    Creates a gif from all TITAN2D steps.

    Requires geoaxes and figure to be passed along with simulation step number.

    Parameters
    ----------
    dem : str
        Relative or absolute path to raster (can be either ASCII or TIFF)
    fig : matplotlib.figure.Figure
        Matplotlib figure to assign raster
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Matplotlib geoaxes to assign raster
    coords : numpy.ndarray
        Array to plot points for coordinates
    max_iter : int
        Total iterations TITAN2D output (for calculations)
    diter : int
        Time interval chosen for TITAN2D output
    gif_name : str
        Name of output gif
    epsg : int, optional
        Projection, recommended to include same input as geoaxes. Defaults to 32628.
    sim_dir : str, optional
        Parent directory of vizout. Defaults to '.'.
    """

    # Generate step numbers
    num_steps = np.arange(int(np.ceil(max_iter/diter))+1)

    # Generate frames for gif
    for step in range(int(np.ceil(max_iter/diter))+1):
        # Initialize figure and axes for each step
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.epsg(epsg))

        # Plot TITAN output for current step
        plot_titan(dem, step, fig, ax, coords, diter=diter, zoom=False, sim_dir=sim_dir)

        # Save figure as png
        outname = "".join(("gif_files/flow_", str(num_steps[step])))
        plt.savefig(outname)
        plt.close()

    # Read in frames for gif
    frames = []
    for i in range(int(np.ceil(max_iter/diter))+1):
        image = imageio.v2.imread(f'gif_files/flow_{i}.png')
        frames.append(image)

    # Create gif
    imageio.mimsave(gif_name,  # output gif
                    frames,  # array of input frames
                    duration=100)
    
def download_file_gcp(bucket_name, source_blob_name, destination_file_name, api_creds_json):
    """
    Downloads a blob from the specified Google Cloud bucket.
    
    Parameters
    ----------
    bucket_name : str
        The ID of your Google Cloud bucket.
    source_blob_name : str
        The ID of your Google Cloud object.
    destination_file_name : str
        The path to which the file should be downloaded.
    api_creds_json : str
        The relative or absolute path to the JSON file containing the Google Cloud API credentials.

    Returns
    -------
    None
    """

    # Set the environment variable for the Google Cloud API credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_creds_json

    # Create a storage client
    storage_client = storage.Client()

    # Get the bucket using the bucket name
    bucket = storage_client.bucket(bucket_name)

    # Get the blob using the source blob name
    blob = bucket.blob(source_blob_name)

    # Download the blob to the destination file name
    blob.download_to_filename(destination_file_name)

    # Print a confirmation message
    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )
    
def upload_file_gcp(bucket_name, source_file_name, destination_blob_name, api_cred_json):
    """
    Uploads a file to the specified Google Cloud bucket.

    Parameters
    ----------
    bucket_name : str
        The ID of your Google Cloud bucket.
    source_file_name : str
        The path to your file to upload.
    destination_blob_name : str
        The ID of your GCS object.
    api_cred_json : str
        The relative or absolute path to the JSON file containing the Google Cloud API credentials.

    Returns
    -------
    None
    """
    # Set the environment variable for the Google Cloud API credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_cred_json

    # Create a storage client
    storage_client = storage.Client()

    # Get the bucket using the bucket name
    bucket = storage_client.bucket(bucket_name)

    # Get the blob using the destination blob name
    blob = bucket.blob(destination_blob_name)

    # Upload the file from the source file name to the blob
    blob.upload_from_filename(source_file_name)

    # Print a confirmation message
    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

def download_file_aws(access_key, secret_access_key, bucket_name, blob_name, file_name, session_token=None):
    """Downloads a file from AWS S3 Bucket.

    Parameters
    ----------
    access_key : str
        Public access key for bucket.
    secret_access_key : str
        Private IAM access key for bucket.
    bucket_name : str:
        The ID of your AWS S3 bucket.
    blob_name:
        The ID of your AWS S3 object.
    file_name : str
        The path to which the file should be downloaded.
    session_token: str, optional
        Optional string to continue an existing S3 connection. Defaults to None.

    Returns
    -------
    None
    """
    # Create an S3 client with the provided access key and secret access key.
    # If a session token is provided, use it to establish the connection.
    if session_token == None:
        client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_access_key,
        )
    else:
        client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token
        )

    try:
        # Download the file from the bucket to the specified file path.
        response = client.download_file(bucket_name, blob_name, file_name)
    except ClientError as e:
        # Print an error message if the access keys are incorrect.
        print("Incorrect access keys: please enter valid credentials")

    # Print a confirmation message indicating the file was successfully downloaded.
    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            blob_name, bucket_name, file_name
        )
    )

def upload_file_aws(access_key, secret_access_key, bucket_name, blob_name, file_name, session_token=None):
    """Uploads a file to AWS S3 Bucket.

    Parameters
    ----------
    access_key : str
        Public access key for bucket.
    secret_access_key : str
        Private IAM access key for bucket.
    bucket_name : str:
        The ID of your AWS S3 bucket.
    blob_name:
        The ID of your AWS S3 object.
    file_name : str
        The path to the file going to be uploaded.
    session_token: str, optional
        Optional string to continue an existing S3 connection. Defaults to None.

    Returns
    -------
    None
    """
    # Create an S3 client with the provided access key and secret access key.
    # If a session token is provided, use it to establish the connection.
    if session_token is None:
        client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_access_key,
        )
    else:
        client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token
        )

    try:
        # Upload the file from the specified file path to the bucket with the specified object name.
        response = client.upload_file(file_name, bucket_name, blob_name)
    except ClientError as e:
        # Print an error message if the access keys are incorrect.
        print("Incorrect access keys: please enter valid credentials")

    # Print a confirmation message indicating the file was successfully uploaded.
    print(
        f"File {file_name} uploaded to {bucket_name} as {blob_name}."
    )

def download_from_azure(conn_string, container_name, blob_name, local_file_name):
    """
    Downloads a file from Azure container service

    Args:
        conn_string (str): Connection string to create session with Azure
        container_name (str): The ID of the Azure bucket
        blob_name (str): The ID of the file to download
        local_file_name (str): The name to assign once the file is downloaded
    """
    # Create a blob service client from the connection string
    blob_service_client = BlobServiceClient.from_connection_string(conn_string)
    
    # Get a blob client for the specified container and blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    # Download the blob and write it to the specified local file
    with open(file=local_file_name, mode="wb") as sample_blob:
        download_stream = blob_client.download_blob()
        sample_blob.write(download_stream.readall())
    
    # Print a confirmation message
    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            blob_name, container_name, local_file_name
        )
    )
        
def upload_to_azure(conn_string, container_name, blob_name, local_file_name):
    """
    Uploads a file to Azure container service.

    Args:
        conn_string (str): Connection string to create session with Azure.
        container_name (str): The ID of the Azure bucket.
        blob_name (str): The ID of the file once uploaded to the container.
        local_file_name (str): The name/path of the local file to upload.
    """
    # Create a blob service client from the connection string
    blob_service_client = BlobServiceClient.from_connection_string(conn_string)
    
    # Get a container client for the specified container
    container_client = blob_service_client.get_container_client(container=container_name)
    
    # Open the local file in binary mode and upload it to the container
    with open(file=local_file_name, mode="rb") as data:
        blob_client = container_client.upload_blob(name=blob_name, data=data, overwrite=True)
    
    # Print a confirmation message
    print(
        f"File {local_file_name} uploaded to {container_name} as {blob_name}."
    )

def download_dem(north, south, east, west, outputFormat, dataset,filename=""):
    """Download a DEM from the OpenTopography API based on latitude and longitude bounds.

    Parameters:
        north : float
            North latitude bound of the Area of Interest (AOI).
        south : float
            South latitude bound of the AOI.
        east : float
            East longitude bound of the AOI.
        west : float
            West longitude bound of the AOI.
        outputFormat : str
            Format of the output file. Choose between 'ascii' and 'tiff'.
        dataset : str
            Satellite dataset to pull data from.
            Available datasets:
                SRTMGL3 (SRTM GL3 90m)
                SRTMGL1 (SRTM GL1 30m)
                SRTMGL1_E (SRTM GL1 Ellipsoidal 30m)
                AW3D30 (ALOS World 3D 30m)
                AW3D30_E (ALOS World 3D Ellipsoidal, 30m)
                SRTM15Plus (Global Bathymetry SRTM15+ V2.1)
                NASADEM (NASADEM Global DEM)
                COP30 (Copernicus Global DSM 30m)
                COP90 (Copernicus Global DSM 90m)
                EU_DTM (DTM 30m)

    Returns
    -------
    File name on success
    File name if exact file already exists
    -1 on failure
    """
    # Check if latitude and longitude bounds are valid
    if north < south:
        print("Invalid latitude range")
        return -1
    elif east < west:
        print("Invalid longitude range")
        return -1

    # Set the output format
    if outputFormat == "ascii":
        out = "AAIGrid"
    elif outputFormat == "tiff":
        out = "GTiff"
    else:
        print("Invalid format. Choose from ['ascii', 'tiff']")
        return -2

    # Construct the URL for the API request
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y")
    name = f"{int(north)}N_{int(south)}S_{int(east)}W_{int(west)}E_{dataset}_{dt_string}"
    completedName = f"DEM_{name}"
    if outputFormat == "ascii":
        name = f"{name}.asc"
        completedName = f"{completedName}.asc"
    elif outputFormat == "tif" or outputFormat == "tiff":
        name = f"{name}.tiff"
        completedName = f"{completedName}.tiff"

    # Check if the file already exists
    if os.path.isfile(completedName):
        print(f"File already exists with name {completedName}, exiting.")
        return completedName

    # Download the DEM file
    url = (
        f"https://portal.opentopography.org/API/globaldem?demtype={dataset}&south={south}&north={north}&west={west}&east={east}&outputFormat={outputFormat}&API_Key=3ac3c07f20ee63fd3babe7884f24e2c3"
    )
    urllib.request.urlretrieve(url, name)

    # Read the downloaded file
    with rio.open(name) as src:
        profile = src.profile
        data = src.read()

    # Set the new nodata value in the profile
    profile.update(nodata=0)

    if filename == "":
        # Write the output raster with the updated nodata value
        with rio.open(completedName, "w", **profile) as dst:
            dst.write(data)
        os.remove(name)
        print(f"DEM downloaded as {completedName}")
        return completedName
    else:
        with rio.open(filename, "w", **profile) as dst:
            dst.write(data)
        os.remove(name)
        print(f"DEM downloaded as {filename}")
        return filename

def search_opentopo(minx, maxx, miny, maxy, detail = False, federated = True):
    """
    Allows user to search for available datasets in the OpenTopography library.

    Parameters
    ----------
    minx : float
        Leftmost longitude bound of AOI
    maxx : float
        Rightmost longitude bound of AOI
    miny : float
        Lowest latitude bound of AOI
    maxy : float
        Highest latitude bound of API
    detail : bool, optional
        Toggle to show detailed metadata. Default is False.
    federated : bool, optional
        Toggle to ignore non federated datasets, such as USGS. Default is True.

    Returns
    -------
    list of dict
        Datasets available in the specified AOI and API range.
    """
    if miny > maxy:
        print("Invalid latitude range")
        return []
    elif minx > maxx:
        print("Invalid longitude range")
        return []
    include_federated = "true" if federated else "false"
    output_format = "json"
    url = (
        f"https://portal.opentopography.org/API/otCatalog?minx={minx}&miny={miny}"
        f"&maxx={maxx}&maxy={maxy}&detail={repr(detail)}&outputFormat={output_format}"
        f"&include_federated={include_federated}"
    )
    response = requests.get(url)
    jason = response.json()
    return jason.get("Datasets", [])

def download_dem_usgs(north: float, south: float, east: float, west: float,
                      outputFormat: str, res: str, filename=""):
    """Download USGS DEM using OpenTopography API.

    Parameters
    ----------
    north : float
        North latitude bound of Area of Interest (AOI).
    south : float
        South latitude bound of AOI.
    east : float
        East longitude bound of AOI.
    west : float
        West longitude bound of AOI.
    outputFormat : str
        Output format of the DEM file. Choose between 'ascii' and 'tif'.
    res : str
        Resolution of the DEM. Choose between '1m', '10m', and '30m'.

    Returns
    -------
    None on success
    -1 if latitude or longitude range is invalid
    1 if file already exists,
    """
    # Check if latitude and longitude bounds are valid
    if north < south:
        print("Invalid latitude range")
        return -1
    elif east < west:
        print("Invalid longitude range")
        return -1

    # Set output format
    if outputFormat == "ascii":
        out = "AAIGrid"
    elif outputFormat == "tif":
        out = "GTiff"

    # Construct URL for API request
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y")
    name = f"{int(north)}N_{int(south)}S_{int(east)}W_{int(west)}E_usgs{res}_{dt_string}"
    completedName = f"DEM_{name}"
    if outputFormat == "ascii":
        name = f"{name}.asc"
        completedName = f"{completedName}.asc"
    elif outputFormat == "tif":
        name = f"{name}.geotiff"
        completedName = f"{completedName}.geotiff"

    # Download DEM if it does not already exist
    if os.path.isfile(completedName):
        print(f"File already exists with name {completedName}, exiting.")
        return 1
    url = f"https://portal.opentopography.org/API/usgsdem?datasetName=USGS{res}&south={south}&north={north}&west={west}&east={east}&outputFormat={outputFormat}&API_Key=3ac3c07f20ee63fd3babe7884f24e2c3"
    urllib.request.urlretrieve(url, name)

    # Set nodata value to 0 and export the DEM
    with rio.open(name) as src:
        profile = src.profile
        data = src.read()

        # Set the new nodata value in the profile
        profile.update(nodata=0)

        # Write the output raster with the updated nodata value
    if filename == "":
        # Write the output raster with the updated nodata value
        with rio.open(completedName, "w", **profile) as dst:
            dst.write(data)
        os.remove(name)
        print(f"DEM downloaded as {completedName}")
        return completedName
    else:
        with rio.open(filename, "w", **profile) as dst:
            dst.write(data)
        os.remove(name)
        print(f"DEM downloaded as {filename}")
        return filename

    # Remove temporary file
    os.remove(name)
    
def download_dem_utm(north, south, east, west, nw_zone, se_zone, outputFormat, dataset,filename=""):
    """
    Download a DEM from the OpenTopography API based on UTM coordinates and desrpective zones.

    Parameters
    ----------
    north : float
        North bound of the Area of Interest (AOI).
    south : float
        South bound of the AOI.
    east : float
        East bound of the AOI.
    west : float
        West bound of the AOI.
    nw_zone : str
        Zone of the top left corner of the AOI.
    se_zone : str
        Zone of the bottom right corner of the AOI.
    outputFormat : str
        Format of the output file. Choose between 'ascii' and 'tif'.
    dataset : str
        Satellite dataset to pull data from.
        Available datasets:
            SRTMGL3 (SRTM GL3 90m)
            SRTMGL1 (SRTM GL1 30m)
            SRTMGL1_E (SRTM GL1 Ellipsoidal 30m)
            AW3D30 (ALOS World 3D 30m)
            AW3D30_E (ALOS World 3D Ellipsoidal, 30m)
            SRTM15Plus (Global Bathymetry SRTM15+ V2.1)
            NASADEM (NASADEM Global DEM)
            COP30 (Copernicus Global DSM 30m)
            COP90 (Copernicus Global DSM 90m)
            EU_DTM (DTM 30m)

    Returns
    -------
    None on success
    -1 if latitude or longitude range is invalid
    1 if file already exists,
    """
    # Convert UTM coordinates to lat/lon
    north, west = utm.to_latlon(west, north, int(nw_zone[:-1]), nw_zone[-1])
    south, east = utm.to_latlon(east, south, int(se_zone[:-1]), se_zone[-1])

    # Check if latitude and longitude bounds are valid
    if north < south:
        print("Invalid latitude range")
        return -1
    elif east < west:
        print("Invalid longitude range")
        return -1

    # Construct URL for API request
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y")
    name = f"DEM_{int(north)}N_{int(south)}S_{int(east)}W_{int(west)}E_{dataset}_{dt_string}" if filename == "" else filename 
    if outputFormat == "AAIGrid":
        name = f"{name}.asc"
    elif outputFormat == "GTiff":
        name = f"{name}.geotiff"

    # Download DEM if it does not already exist
    if os.path.isfile(name):
        print(f"File already exists with name {name}, exiting.")
        return 1
    url = (f"https://portal.opentopography.org/API/globaldem?demtype={dataset}&south={south}"
           f"&north={north}&west={west}&east={east}&outputFormat={outputFormat}&API_Key=3ac3c07f20ee63fd3babe7884f24e2c3")
    urllib.request.urlretrieve(url, name)
    print(f"DEM downloaded as {name}")
    
def dem_to_utm(infile, outfile=None):
    """
    Converts an ascii file in lat/lon to UTM.

    Parameters
    ----------
    infile : str
        Relative or absolute path to DEM.
    outfile : str, optional
        Optional string to save new file to, will overwrite original otherwise.
    
    Returns
    -------
    None
    """
    # Read DEM file
    f = open(infile,'r').readlines()
    
    # Extract coordinates and spacing
    xll = f[2]
    long = float(xll.split()[1])
    yll = f[3]
    lat = float(yll.split()[1])
    spacing = f[4]
    
    # Convert dx and dy to UTM coordinates
    if spacing.split()[0] == "dx":
        convert_dx = float(spacing.split()[1])*111319.48
        new_dx = "".join(("dx","    ",str(convert_dx),"\n"))
        dy = f[5]
        convert_dy = float(dy.split()[1])*111319.48
        new_dy = "".join(("dy","    ",str(convert_dy),"\n"))
        f[4] = new_dx
        f[5] = new_dy
    else:
        convert_spacing = float(spacing.split()[1])*111319.48
        new_spacing = "".join(("cellsize","    ",str(convert_spacing),"\n"))
        f[4] = new_spacing
    
    # Convert latitude and longitude to UTM coordinates
    translated = utm.from_latlon(lat,long)
    new_xll = "".join(("xllcorner","    ",str(translated[0]),"\n"))
    new_yll = "".join(("yllcorner","    ",str(translated[1]),"\n"))
    f[2] = new_xll
    f[3] = new_yll
    
    # Write new file
    if outfile == None:
        outfile = infile
    f2 = open(outfile,"w")
    f2.writelines(f)

def dem_to_latlong(infile, utm_zone, outfile=None):
    """Converts an ascii file in UTM to lat/lon

    Parameters
    ----------
    infile : str
        Relative or aboslute path to DEM
    utm_zone : str
        Required string in number-letter format to specify area
    outfile : str, optional
        Optional string to save new file to, will overwrite original otherwise.

    Returns
    -------
    None
    """
    # Read DEM file
    f = open(infile,'r').readlines()
    
    # Extract coordinates and spacing
    xll = f[2]
    long = float(xll.split()[1])
    yll = f[3]
    lat = float(yll.split()[1])
    spacing = f[4]
    
    # Convert dx and dy to UTM coordinates
    if spacing.split()[0] == "dx":
        convert_dx = float(spacing.split()[1])/111319.48
        new_dx = "".join(("dx","    ",str(convert_dx),"\n"))
        dy = f[5]
        convert_dy = float(dy.split()[1])/111319.48
        new_dy = "".join(("dy","    ",str(convert_dy),"\n"))
        f[4] = new_dx
        f[5] = new_dy
    else:
        convert_spacing = float(spacing.split()[1])/111319.48
        new_spacing = "".join(("cellsize","    ",str(convert_spacing),"\n"))
        f[4] = new_spacing
    
    # Convert latitude and longitude to UTM coordinates
    if len(utm_zone) == 2:
        translated = utm.to_latlon(long,lat,int(utm_zone[0]),utm_zone[1])
    elif len(utm_zone) == 3:
        translated = utm.to_latlon(long,lat,int(utm_zone[0:2]),utm_zone[2])
    new_xll = "".join(("xllcorner", "    ", str(translated[0]),"\n"))
    new_yll = "".join(("yllcorner", "    ", str(translated[1]),"\n"))
    f[2] = new_xll
    f[3] = new_yll
    
    # Write new file
    if outfile == None:
        outfile = infile
    f2 = open(outfile,"w")
    f2.writelines(f)

def find_arctic_cell(lat, lon):
    """
    Find the Arctic cell coordinates based on the given latitude and longitude.

    Parameters
    ----------
    lat : float
        The latitude of the point.
    lon : float
        The longitude of the point.

    Returns
    -------
    list : str
        A list containing the northing, easting, northing_subtile, and easting_subtile coordinates.
    """
    easting = int(np.floor((lon+4000000)/100000))+1
    northing = int(np.floor((lat+4000000)/100000))+1
    easting_subtile = 1 if (((lon+4000000)/100000)+1 - easting) < .5 else 2
    northing_subtile = 1 if (((lat+4000000)/100000)+1 - northing) < .5 else 2
    return [northing, easting, northing_subtile, easting_subtile]

def download_arcticdem(lat, lon):
    """
    Downloads Arctic DEM tiles based on the given latitude and longitude.

    Parameters:
        lat (float): The latitude of the point.
        lon (float): The longitude of the point.

    Returns
    -------
    None

    Example usage:
        download_arcticdem(60.0, -10.0)
    """
    myProj = pyproj.Proj("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    transformed = myProj.transform(lat, lon)
    tile_data= np.array([find_arctic_cell(transformed[0], transformed[1])])
    boundaries = np.array([[transformed[0] + 5000, transformed[1] + 5000],[transformed[0] - 5000, transformed[1] + 5000], [transformed[0] - 5000, transformed[1] - 5000], [transformed[0] + 5000, transformed[1] - 5000]])
    for edge in boundaries:
        cell = find_arctic_cell(edge[0],edge[1])
        if (cell not in tile_data.tolist()):
            tile_data = np.vstack([tile_data,cell])
    for i in range(len(tile_data)):
        filename = f"""{tile_data[i,0]}_{tile_data[i,1]}/{tile_data[i,0]}_{tile_data[i,1]}_{tile_data[i,2]}_{tile_data[i,3]}_2m_v3.0.tar.gz"""
        string = f"""wget -r -N -nH -np -R index.html* --cut-dirs=6 https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/2m/{filename}"""
        subprocess.run(string,shell=True)
        subprocess.run(f"tar -xvzf {filename}",shell=True)

def search_volcano(name):
    """
    Searches for a volcano with a given name in the Excel file "/home/jovyan/shared/Libraries/GVP_Volcano_List_Holocene.xlsx" and returns an array of volcano information.

    Parameters
    ----------
    name : str
        The name of the volcano to search for.

    Returns
    -------
    vals: numpy.ndarray
        An array of volcano information, with each row containing the volcano name, latitude, longitude, and elevation (in meters).

    """
    # Read the Excel file containing volcano information
    df = pd.read_excel("/home/jovyan/shared/Libraries/GVP_Volcano_List_Holocene.xlsx", header=1)

    # Search for a volcano with a name matching the given input
    parsed = df.loc[df['Volcano Name'].str.contains(name, case=False)]

    # Convert the filtered data to a numpy array
    vals = np.array((parsed["Volcano Name"].values, parsed["Latitude"].values, parsed["Longitude"].values, parsed["Elevation (m)"].values))

    return vals

# def download_volcano_dem(name):

def convert_molasses(output_name):
    """
    Converts molasses data from a CSV file to a raster format using GeoPandas and rasterio.

    Parameters
    ----------
    output_name : str
        The name of the output raster file without the file extension.

    Returns
    -------
    None

    """
    colnames=['EAST', 'NORTH', 'THICKNESS', 'NEW_ELEV', 'ORIG_ELEV'] 
    lava = pd.read_csv('./flow_-0', skiprows=3, names=colnames, sep='\s+', header= None)
    lava.to_csv("flow.csv",header=colnames,index=False)
    df = gpd.pd.read_csv('flow.csv')
    gf = gpd.GeoDataFrame(df, 
                          geometry=gpd.points_from_xy(df.EAST, df.NORTH), 
                          crs=4326)
    geo_grid = make_geocube(vector_data=gf, measurements=['THICKNESS'], resolution=20)
    geo_grid.rio.to_raster("".join([output_name,".asc"]))

def jaccard_similarity(observed, simulated):
    """
    Calculate the Jaccard similarity coefficient between two rasters.

    Parameters
    ----------
    observed: str
        The path to the observed raster.
    simulated : str
        The path to the simulated raster.

    Returns
    -------
    Rj: float
        The Jaccard similarity coefficient between the two rasters.
    """
    with rio.open(observed) as src:
        profile = src.profile
        transform = src.transform
        observed_feature = src.read(1)
        print(f"Observed Feature Raster Geographic Coordinates --> {src.crs}") 
        # implicitely prints the coordinate system and explicitely prints
        # the coordinates

    # Open and read the simulated feature raster
    with rio.open(simulated) as src: # input your simulater raster 
        # GeoTIFF file here
        profile = src.profile
        transform = src.transform
        simulated_feature = src.read(1)
        print(f"Simulated Feature Raster Geographic Coordinates --> {src.crs}") 
        # implicitely prints the coordinate system and explicitely prints
        # the coordinates

    # Calculate the areas inundated: True Positives (TP), False Positives (FP), 
    # and False Negatives (FN)
    
    TP = np.sum(np.logical_and(observed_feature > 0, simulated_feature > 0))
    FP = np.sum(np.logical_and(observed_feature == 0, simulated_feature > 0)) 
    FN = np.sum(np.logical_and(observed_feature > 0, simulated_feature == 0)) 

    # Calculate the Bayesian Metrics: Jaccard similarity coefficient (Rj), 
    # model precision (Rmp), and model sensitivity (Rms)
    Rj = (np.sum(TP) / (np.sum(TP) + np.sum(FN) + np.sum(FP))) * 100
    return Rj