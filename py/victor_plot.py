import matplotlib.pyplot as plt
import rioxarray as rxr
import numpy as np
import pandas as pd
import os
import matplotlib.colors as colors
import xrspatial as xrs

def plot_dem(dem, markercoords=np.array([]), title=None, save=None, zoom=None):
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
    save : str, optional
        If provided, save the plot to this filename
    zoom : float, optional
        If provided, zoom in by this factor around markercoords or image center

    Returns
    -------
    None
    """
    # Open raster file using rasterio
    raster = rxr.open_rasterio(dem)
    bounds = raster.rio.bounds()
    # Select the first band of the raster
    raster = raster.sel({"band": 1})
    # Apply hillshade to the raster and plot it
    render = xrs.hillshade(raster, 45, 30)
    render.plot(cmap="gray",add_colorbar=False)
    
    # If coordinates are provided, scatter plot them
    if markercoords.any():
        if markercoords.ndim == 2:
            plt.scatter(markercoords[:, 0], markercoords[:, 1], color="red", marker="^")
            center_x = np.mean(markercoords[:, 0])
            center_y = np.mean(markercoords[:, 1])
        elif markercoords.ndim == 1:
            plt.scatter(markercoords[0], markercoords[1], color="red", marker="^")
            center_x = markercoords[0]
            center_y = markercoords[1]
    else:
        # If no markers, use center of image
        center_x = (bounds[2] + bounds[0]) / 2
        center_y = (bounds[3] + bounds[1]) / 2
    
    # Apply zoom if specified
    if zoom is not None and zoom > 0:
        # Calculate the current view extent
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        # Calculate new extents based on zoom factor
        half_width = width / (2 * zoom)
        half_height = height / (2 * zoom)
        
        # Set limits centered on marker or image center
        plt.xlim(center_x - half_width, center_x + half_width)
        plt.ylim(center_y - half_height, center_y + half_height)
    
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    # Set title of the plot
    if title is None:
        plt.title(dem)
    else:
        plt.title(title)
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        
def plot_flow(dem, flow, coords=np.array([]), zoom=True, label="Thickness (m)", title=None, lognorm=False,axes=None,scale=None,save=None):
    """
    Raster+flow plotting, does not require figure and axes to be passed.

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
    save : str, optional
        If provided, save the plot to this filename

    Returns
    -------
    maxval : float
        Maximum depth of a flow
    """
    if axes:
        ax = axes
    else:
        # Create a new axes object
        ax = plt.axes()

    raster = rxr.open_rasterio(dem)
    bounds = raster.rio.bounds()
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
    ax.axis('equal')
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
        maxval = flow_raster.max()
        flow_raster = flow_raster.sel({"band": 1})
        flow_clipped = flow_raster.dropna('x',how='all')
        flow_clipped = flow_clipped.dropna('y',how='all')
        
        if lognorm:
            flow_render = flow_raster.plot(ax=ax, cmap="hot", norm=colors.LogNorm(vmin=flow_thickness.min(), vmax=flow_thickness.max()), cbar_kwargs={'label': label, 'shrink': .6})
        elif scale:
            flow_render = flow_raster.plot(ax=ax, cmap="hot", norm=colors.LogNorm(vmin=.1, vmax=scale), add_colorbar=False)
        else:
            flow_render = flow_raster.plot(ax=ax, cmap="hot", cbar_kwargs={'label': label, 'shrink': .6})
        
        # If zoom is True, set the x and y limits of the plot to the min and max values of the x and y coordinates
        if zoom:
            padding_x = (flow_clipped.x.max() - flow_clipped.x.min())/4
            padding_y = (flow_clipped.y.max() - flow_clipped.y.min())/4
            ax.set_xlim(flow_clipped.x.min()-padding_x, flow_clipped.x.max()+padding_x)
            ax.set_ylim(flow_clipped.y.min()-padding_y, flow_clipped.y.max()+padding_y)
        else:
            ax.set_xlim(bounds[0],bounds[2])
            ax.set_ylim(bounds[1],bounds[3])
    if title is None:
        plt.title(dem)
    else:
        plt.title(title)
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        
    return flow_render