import numpy as np
import xarray as xr
from scipy.stats import linregress
from global_land_mask import globe

def hemisphere_change(ds):
    ds[12]  = np.nan
    ds[:12] = -ds[:12]
    return ds 

def ns_vstack(ds_sh,ds_nh):
    ds_ns = xr.concat([ds_sh,ds_nh[:,1:,:]],dim='lat')
    ds_ns[:,12,:] = np.nan
    return ds_ns

def ns_vstack2D(ds_sh,ds_nh):
    ds_ns = xr.concat([ds_sh,ds_nh[1:,:]],dim='lat')
    ds_ns[12,:] = np.nan
    return ds_ns

def ns_vstackobs(ds_sh,ds_nh,order=True):
    ds_eq = xr.DataArray(np.full([1,len(ds_nh.lon)],np.nan),coords={'lat':np.array([0]),'lon':ds_nh.lon.data})
    if order:
        ds_ns = xr.concat([ds_sh,ds_eq,ds_nh],dim='lat')
    else:
        ds_ns = xr.concat([ds_nh,ds_eq,ds_sh],dim='lat')
    # ds_ns[29:31] = np.nan
    return ds_ns

def percentile_pattern(ds,pt):
    model_number = len(ds.model)
    ds_sign      = np.sign(ds)
    ds_sign_sum  = ds_sign.sum('model')
    pt_result    = xr.where(abs(ds_sign_sum) > (2 * pt -1) * model_number,1,0)
    # pt_xr_array  = xr.DataArray(pt_result, dims={'lat','lon'}, coords={'lat':ds.lat,'lon':ds.lon})
    return pt_result

def pattern_trend(ds,threhold=2,lat_name='lat',lon_name='lon',xrdata=True):
    """
    Usage: to calculate the linear squres trend of the 3D data
    Author: Hongqiang Dong from Ocean University of China
    Created on Mon Jan 29 2024
    Parameters
    ----------
    ds : xarray.DataArray(year,lat,lon)
        input data
    threhold : int
        input: the threhold of nan values numbers on each grid
    xrdata : Ture or False
        input: output type is xr.DataArray (True) or np.array (False)
    Returns
    -------
    slope : xarray.DataArray(lat,lon) or numpy.array
        output: the regression coef of the inputdata's trend
    pvalue :xarray.DataArray(lat,lon) or numpy.array
        output: p test_check
    """
    nlat,nlon = len(ds[lat_name]), len(ds[lon_name]) 
    sdata = np.zeros((nlat,nlon))
    pdata = np.zeros((nlat,nlon))
    x     = ds.year
    nyear = len(x)
    for i in range(nlat):
        for j in range(nlon):
            y = ds[:,i,j].data
            nan_count = np.isnan(y).sum()
            if nan_count == 0:
                sdata[i,j], _, _,pdata[i,j],_ = linregress(x,y)
            elif nan_count < nyear/threhold:
                valid_indices = ~np.isnan(x) & ~np.isnan(y)
                x_valid = x[valid_indices]
                y_valid = y[valid_indices]
                sdata[i,j], _, _,pdata[i,j],_ = linregress(x_valid,y_valid)
            else:
                sdata[i,j] = np.nan
                pdata[i,j] = np.nan
    if xrdata:
        slope  = xr.DataArray(sdata,dims=('lat','lon'),coords={'lat':ds[lat_name],'lon':ds[lon_name]})
        pvalue = xr.DataArray(pdata,dims=('lat','lon'),coords={'lat':ds[lat_name],'lon':ds[lon_name]})
    else:
        slope  = sdata
        pvalue = pdata
    return slope,pvalue

def change_lon(ds,lon_name):
# 将经度0~360 改变成为 -180~180
    ds['longitude_adjusted'] = xr.where(
        ds[lon_name] > 180,
        ds[lon_name] - 360,
        ds[lon_name])
    ds = (
        ds
        .swap_dims({lon_name: 'longitude_adjusted'})
        .sel(**{'longitude_adjusted': sorted(ds.longitude_adjusted)})
        .drop(lon_name))
    ds = ds.rename({'longitude_adjusted': lon_name})
    return ds

def mask_land(ds, label, lonname):
    '''
    Parameters
    ----------
    ds : xarray.DataArray
        input1.
    label : str
        "land" or "ocean".
    lonname : str
        "lon" or "longitude"
    Returns
    -------
    ds : xarray.DataArray
        output data land-only or ocean-only
    '''
    if lonname == 'lon':
        lat = ds.lat.data
        lon = ds.lon.data
        if np.any(lon > 180):
            lon = lon - 180
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
            temp = []
            temp = mask[:, 0:(len(lon) // 2)].copy()
            mask[:, 0:(len(lon) // 2)] = mask[:, (len(lon) // 2):]
            mask[:, (len(lon) // 2):] = temp
        else:
            lons, lats = np.meshgrid(lon, lat)# Make a grid
            mask = globe.is_ocean(lats, lons)# Get whether the points are on ocean.
        ds.coords['mask'] = (('lat', 'lon'), mask)
    elif lonname == 'longitude':
        lat = ds.latitude.data
        lon = ds.longitude.data
        if np.any(lon > 180):
            lon = lon - 180
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
            temp = []
            temp = mask[:, 0:(len(lon) // 2)].copy()
            mask[:, 0:(len(lon) // 2)] = mask[:, (len(lon) // 2):]
            mask[:, (len(lon) // 2):] = temp
        else:
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
        lons, lats = np.meshgrid(lon, lat)
        mask = globe.is_ocean(lats, lons)
        ds.coords['mask'] = (('latitude', 'longitude'), mask)
    if label == 'land':
        ds = ds.where(ds.mask == True)
    elif label == 'ocean':
        ds = ds.where(ds.mask == False)
    return ds

def count_decimal_places(number):
    # 将数字转换为字符串
    number_str = str(number)
    
    # 判断是否包含小数点
    if '.' in number_str:
        # 获取小数点后的部分，并返回其长度
        return len(number_str.split('.')[1])
    else:
        # 如果没有小数点，则返回 0
        return 0
    
def save_point(number,k):
    number_str = round(float(number), k)
    n = count_decimal_places(number_str)
    if n == k:
        result = str(number_str)
    else:
        result = str(number_str) + '0'
    return result