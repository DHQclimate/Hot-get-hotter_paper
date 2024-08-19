# -*- coding: utf-8 -*-
"""
@author: Hongqiang Dong
"""

import numpy as np
import xarray as xr 
import copy
import regionmask

def ram(ds,region='all',lat_name='lat',lon_name='lon',precision='10'):
    """
    latitude of the weighted and calculate the Area mean value with latitude weighted
    Parameters
    ----------
    ds : Xarray.Dataarray(*,lat,lon)
        input: data with dims of lat and lon

    region : regions for calculating area mean ---- 'all' or 'land' or 'ocean'
        'all'  : calculate area mean without masking
        'land' : calculate area mean with ocean masking (only save the values on land)
        'ocean': calculate area mean with land masking (only save the values on ocean)

    lat_name : 'lat' or 'latitude'
        input: the name of latitude dim

    lon_name : 'lon' or 'longitude'
        input: the name of longitude dim

    precision: the precision for landmask ---- '10' or '50' or '110'
        '10' : using natural_earth_v5_0_0.land_10
        '50' : using natural_earth_v5_0_0.land_50
        '110': using natural_earth_v5_0_0.land_110

    Returns
    -------
    ds_am : Xarray.Dataarray(*)
        output: Area mean value
    """
    weights = np.cos(np.deg2rad(ds[lat_name]))
    if region == 'all':
        ds_am   = ds.weighted(weights).mean(dim=(lat_name,lon_name))
    elif region == 'land':
        if precision == '10':
            mask  = regionmask.defined_regions.natural_earth_v5_0_0.land_10.mask(ds[lon_name],ds[lat_name])
        elif precision == '50':
            mask  = regionmask.defined_regions.natural_earth_v5_0_0.land_50.mask(ds[lon_name],ds[lat_name])
        else:
            mask  = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(ds[lon_name],ds[lat_name])
        ds_mask = ds.where(mask==0)
        ds_am   = ds_mask.weighted(weights).mean(dim=(lat_name,lon_name))
    elif region == 'ocean':
        if precision == '10':
            mask  = regionmask.defined_regions.natural_earth_v5_0_0.land_10.mask(ds[lon_name],ds[lat_name])
        elif precision == '50':
            mask  = regionmask.defined_regions.natural_earth_v5_0_0.land_50.mask(ds[lon_name],ds[lat_name])
        else:
            mask  = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(ds[lon_name],ds[lat_name])
        ds_mask = ds.where(mask)
        ds_am   = ds_mask.weighted(weights).mean(dim=(lat_name,lon_name))
    return ds_am
