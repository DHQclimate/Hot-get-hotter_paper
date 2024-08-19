import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from scipy import stats
from scipy.stats import linregress
from scipy.stats.mstats import ttest_ind

def select_amj(ds):
    ds_amj = ds.sel(time=ds['time'].dt.month.isin([4,5,6])).groupby('time.year').mean('time')
    return ds_amj

def select_son(ds):
    ds_son = ds.sel(time=ds['time.season']=='SON').groupby('time.year').mean('time') 
    return ds_son

def annual_mean(ds):
    ds_am = ds.groupby('time.year').mean('time')
    return ds_am

def amj_change(dsh,dss):
    dsh_spr = select_amj(dsh).mean('year') - annual_mean(dsh)
    dss_spr = select_amj(dss).mean('year') - annual_mean(dss)
    spr_cha = dss_spr.mean('year') - dsh_spr.mean('year')
    return spr_cha    

def son_change(dsh,dss):
    dsh_spr = select_son(dsh).mean('year') - annual_mean(dsh)
    dss_spr = select_son(dss).mean('year') - annual_mean(dss)
    spr_cha = dss_spr.mean('year') - dsh_spr.mean('year')
    return spr_cha

def avg_amj(ds1,ds2):
    ds1_scm = ds1.mean('time')
    ds2_scm = ds2.mean('time')
    ds1_amj = ds1.sel(time=ds1['time'].dt.month.isin([4,5,6])).mean('time') - ds1_scm
    ds2_amj = ds2.sel(time=ds2['time'].dt.month.isin([4,5,6])).mean('time') - ds2_scm
    dds_amj = ds2_amj - ds1_amj
    return dds_amj

def avg_son(ds1,ds2):
    ds1_scm = ds1.mean('time')
    ds2_scm = ds2.mean('time')
    ds1_mam = ds1.sel(time=ds1['time.season']=='SON').mean('time') - ds1_scm
    ds2_mam = ds2.sel(time=ds2['time.season']=='SON').mean('time') - ds2_scm
    dds_mam = ds2_mam - ds1_mam
    return dds_mam

def dps(ds,dc):
    '''
    Usage: calculate the changes of seasonal cycle remove annual mean
    Author: Hongqiang Dong from Ocean University of China
    Created on Sat Jan 27 2024
    :param ds: model
    :param dc: Ctrl
    :return: t-test p values 
    '''
    ds['time'] = dc['time']
    ddm = (ds - dc).groupby("time.month").mean("time")
    dcm = dc.groupby('time.month').mean('time')
    ds_ym = ds.groupby('time.year').mean('time')
    dc_ym = dc.groupby('time.year').mean('time')
    p = []
    for i in range(12):
        #tp
        ds_mt = ds.sel(time=ds['time.month']==i+1) - ds_ym.data
        dc_mt = dc.sel(time=dc['time.month']==i+1) - dc_ym.data
        _,pi = ttest_ind(ds_mt, dc_mt, 0, equal_var=True)
        p.append(pi)
    dd_sc  = ddm - ddm.mean()
    P = np.array(p)
    return dd_sc,P,ddm.mean(),dcm

def dps_err(ds,dc):
    '''
    Usage: calculate error bar values based on quartile for changes of seasonal cycle
    Parameters
    ----------
    ds    : xarray.DataArray(time)
    dc    : xarray.DataArray(time)
    Author: Hongqiang Dong from Ocean University of China
    Created on Sat Jan 27 2024
    Returns
    -------
    yerr  : error bar values based on quartile
    '''
    ds['time'] = dc['time']
    dd  = (ds-dc).groupby('time.month').mean('time')
    ddm = dd  - dd.mean('month')
    dd_err  = ddm.quantile([0.25,0.75],dim='model')
    dd_yerr = [abs(ddm.mean('model').data - dd_err[0].data),abs(dd_err[1].data - ddm.mean('model').data)]
    # dd_yerr = np.std(dd,axis=0)
    return dd_yerr

def dps_err_MM(ds1,ds2,dh1,dh2):
    ds1['time'] = dh1['time']
    ds2['time'] = dh2['time']
    dd1 = (ds1-dh1).groupby('time.month').mean('time')
    dd2 = (ds2-dh2).groupby('time.month').mean('time')
    dd  = (dd1 - dd1.mean('month')) - (dd2 - dd2.mean('month'))
    dd_err  = dd.quantile([0.25,0.75],dim='model')
    # dd_err  = np.percentile(dd,[25,75],axis=0)
    # dd_yerr = [dd.mean(axis=0) - dd_err[0],dd_err[1] - dd.mean(axis=0)]
    dd_yerr = [abs(dd.mean('model').data - dd_err[0].data),abs(dd_err[1].data - dd.mean('model').data)]
    return dd_yerr

def spring_trend(ds_mty,myear=1):
    td_parameter = ds_mty.polyfit(dim='year',deg=1,full=False)
    n            = len(ds_mty.year)
    y_tmd        = td_parameter.polyfit_coefficients[0] * ds_mty.year + td_parameter.polyfit_coefficients[1]
    Regression   = sum((y_tmd - np.mean(ds_mty))**2) # U 
    Residual     = sum((ds_mty - y_tmd)**2)          # Q
    trend        = td_parameter.polyfit_coefficients[0]
    F            = (Regression / 1) / (Residual / ( n - 2 ))  # F number F = U/1 / Q/(n-2)
    p            = stats.f.sf(F, 1, n-2)                      # get the P values for F
    Trend  = trend * myear
    return Trend.mean(), p

def mm_trend(ds,myear=1):
    trend_mm = xr.zeros_like(ds.mean('year'))
    p_mm     = xr.zeros_like(ds.mean('year'))
    for i in range(len(ds.model)):
        trend_mm[i],p_mm[i] = spring_trend(ds[i],myear)
    return trend_mm,p_mm

def relative_spring(ds,hemisphere='N'):
    if hemisphere == 'N':
        ds_spr = select_amj(ds)
    else:
        ds_spr = select_son(ds)
    ds_ann = annual_mean(ds)
    ds_rsp = ds_spr - ds_ann
    return ds_rsp,ds_spr,ds_ann