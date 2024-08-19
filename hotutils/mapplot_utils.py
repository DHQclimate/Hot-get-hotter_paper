import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
import cmaps
import matplotlib as mpl


def cmap(name):
    """
    Parameters
    ----------
    name : A str, names in the NCL colormaps.

    Returns
    -------
    cmap<cmaps.colormap.Colormap object>.

    """
    try:
        s='cmaps.'+name
        cmap=eval(s)
    except ValueError:
        cmap=mpl.cm.get_cmap(name)
    return cmap

def axtp_changes_plot(fig,ts,p_t,mycolor,level,labeloc,tlabel,title1,title2,title3,if_point,ft,x1,x2,y1,y2,aspect=2.6):
    proj = ccrs.PlateCarree(central_longitude=0)
    leftlon, rightlon, lowerlat, upperlat = (-120, 160, -30, 30)
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()

    ax1 = fig.add_axes([x1, y1, x2-x1, y2-y1],projection = proj)
    ax1.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=proj)
    ax1.add_feature(cfeature.COASTLINE.with_scale('110m'))
    ax1.add_feature(cfeature.OCEAN.with_scale('110m'),color='white',zorder=2)

    ax1.set_xticks(np.arange(leftlon,rightlon+1,40), crs=ccrs.PlateCarree())  
    ax1.set_yticks(np.arange(lowerlat, upperlat + 1, 10), crs=proj)  
    ax1.xaxis.set_major_formatter(lon_formatter)  
    ax1.yaxis.set_major_formatter(lat_formatter)  

    ax1.minorticks_on()  
    ax1.xaxis.set_minor_locator(MultipleLocator(20))
    ax1.yaxis.set_minor_locator(MultipleLocator(5))

    ax1.tick_params(length=3, width=1.1, labelsize=ft)  
    ax1.tick_params(which='minor', length=2)

    ax1.set_title(title2,loc='center',fontsize=ft+7,fontweight='bold')
    ax1.set_title(title1, loc='left', fontsize=ft+7,fontweight='bold') 
    ax1.set_title(title3, loc='right', fontsize=ft+7,fontweight='bold')  
    ax1.text(-120 - (280)*labeloc[0],30 + (60)*labeloc[1], tlabel, fontsize=ft+9, color='k', fontweight='bold')

    cycle_ts, cycle_lon = add_cyclic_point(ts, coord=ts.lon)
    cycle_LON, cycle_LAT = np.meshgrid(cycle_lon, ts.lat)

    c1 = ax1.contourf(cycle_LON, cycle_LAT, cycle_ts, levels=level,
                      zorder=0, transform=ccrs.PlateCarree(), cmap=mycolor, extend='both') 
    if if_point:
        cycle_sdm_sig, _ = add_cyclic_point(p_t, coord=ts.lon)
        ax1.contourf(cycle_LON, cycle_LAT,cycle_sdm_sig,levels=[0,0.05,1],colors='none',hatches=[None,'..'],transform=ccrs.PlateCarree(),zorder=1)
    ax1.set_aspect(aspect)

    return ax1,c1


def location_map(fig,extent,box,x11,x22,y11,y22):
    proj = ccrs.PlateCarree(central_longitude=0)
    ax1 = fig.add_axes([x11, y11, x22-x11, y22-y11],projection = proj)
    ax1.plot([-180, 180], [0, 0], transform=ccrs.PlateCarree(), color='white', linewidth=1, linestyle='--')
    ax1.set_extent(extent, crs=proj)
    ax1.stock_img()
    Amazon = patches.Rectangle((box[0], box[2]), box[1]-box[0], box[3]-box[2], linewidth=1.2, linestyle='-', zorder=4, edgecolor='k', facecolor='none',
                              transform=ccrs.PlateCarree())
    ax1.axis('off')
    ax1.add_patch(Amazon)