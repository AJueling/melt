import numpy as np
import xarray as xr
import pyproj
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean as cmo
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs

def add_lonlat(ds):
    project = pyproj.Proj("epsg:3031")
    xx, yy = np.meshgrid(ds.x, ds.y)
    lons, lats = project(xx, yy, inverse=True)
    dims = ['y','x']
    ds = ds.assign_coords({'lat':(dims,lats), 'lon':(dims,lons)})    
    return ds

def makebackground(ax):
    cmap = plt.get_cmap('ocean')
    ds = xr.open_dataset('../../data/BedMachineAntarctica_2020-07-15_v02.nc')
    ds = ds.isel(x=slice(3000,4000),y=slice(6500,9000))
    mask = xr.where(ds.mask==1,2,ds.mask)
    #mask[:] = xr.where(mask==1,2,mask)
    ds = add_lonlat(ds)
    ax.pcolormesh(ds.lon,ds.lat,mask,cmap=cmap,vmin=-4,vmax=3,transform=ccrs.PlateCarree(),shading='auto')
    ax.set_extent([246,261,-75.3,-74.2],crs=ccrs.PlateCarree())
    gl = ax.gridlines()
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180,179,5))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(-89,89))
    
def makebackground2(ax):
    cmap = plt.get_cmap('ocean')
    ds = xr.open_dataset('../../data/BedMachineAntarctica_2020-07-15_v02.nc')
    ds = ds.isel(x=slice(3000,4500),y=slice(6500,9500))
    mask = xr.where(ds.mask==1,2,ds.mask)
    #mask[:] = xr.where(mask==1,2,mask)
    ds = add_lonlat(ds)
    ax.pcolormesh(ds.lon,ds.lat,mask,cmap=cmap,vmin=-4,vmax=3,transform=ccrs.PlateCarree(),shading='auto')
    ax.set_extent([227,261,-75.3,-73.1],crs=ccrs.PlateCarree())
    gl = ax.gridlines()
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180,179,5))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(-89,89))    
    
def plotmelt(ax,lon,lat,melt):
    cmap = plt.get_cmap('inferno')
    melt = np.where(melt>0,melt,np.nan)
    IM = ax.pcolormesh(lon,lat,melt,vmin=1,vmax=200,norm=mpl.colors.LogNorm(),cmap=cmap,transform=ccrs.PlateCarree(),shading='nearest')
    return IM

def plotnormmelt(ax,lon,lat,melt):
    cmap = plt.get_cmap('cmo.curl')
    melt = melt-np.nanmean(melt)
    melt = melt/np.nanstd(melt)
    IM = ax.pcolormesh(lon,lat,melt,vmin=-2,vmax=2,cmap=cmap,transform=ccrs.PlateCarree(),shading='nearest')
    return IM    

def plotdiffmelt(ax,lon,lat,melt):
    cmap = plt.get_cmap('cmo.curl')
    IM = ax.pcolormesh(lon,lat,melt,vmin=-50,vmax=50,cmap=cmap,transform=ccrs.PlateCarree())
    return IM

def prettyplot(dsav,figsize=(10,10)):
 
    try:
        ds = xr.open_dataset(f"{dsav['filename'].values}.nc")
    except:
        print('No output saved yet, cannot plot')
        return

    fig,ax = plt.subplots(1,1,figsize=figsize)            
    ax.set_aspect('equal', adjustable='box') 
    
    x = ds['x'].values
    y = ds['y'].values
    melt  = ds['melt'].values
    mask  = ds['mask'].values
    
    xx,yy = np.meshgrid(x,y)
    
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    x_ = np.append(x,x[-1]+dx)-dx/2
    y_ = np.append(y,y[-1]+dy)-dy/2

    ax.pcolormesh(x_,y_,mask,cmap=plt.get_cmap('ocean'),vmin=-4,vmax=3)
    
    cmap = plt.get_cmap('inferno')
    
    melt = np.where(melt>1,melt,1)
    levs = np.power(10, np.arange(0,np.log10(200),.01))
    IM = ax.contourf(xx,yy,np.where(mask==3,melt,np.nan),levs,cmap=cmap,norm=mpl.colors.LogNorm())
    #IM = ax.pcolormesh(xx,yy,np.where(mask==3,melt,np.nan),norm=mpl.colors.LogNorm(vmin=1,vmax=200),cmap=cmap,shading='nearest')

    the_divider = make_axes_locatable(ax)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(IM, cax=color_axis)
    cbar.set_ticks([1,10,100])
    cbar.set_ticklabels([1,10,100])
    cbar.ax.tick_params(labelsize=21)
    cbar.set_label('Melt [m/yr]', fontsize=21, labelpad=-2)
    
    if ds['name_model'] == 'Layer':
        U = ds['U'].values*ds['D'].values
        V = ds['V'].values*ds['D'].values
        
        spd = (U**2 + V**2)**.5
        lw = .04*spd
        strm = ax.streamplot(x,y,U,V,linewidth=lw,color='w',density=8,arrowsize=0)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    fname = f"../../results/{ds['name_model'].values}/figures/{ds['name_geo'].values}_{ds.attrs['name_forcing']}__{ds['tend'].values:.3f}"

    plt.savefig(f"{fname}.png")
    plt.show()
    
def plotamundsen(Tdeep,ztcl,flow=False):
    fig,ax = plt.subplots(1,1,figsize=(7.5,12))            
    ax.set_aspect('equal', adjustable='box') 

    ds = xr.open_dataset('../../data/BedMachineAntarctica_2020-07-15_v02.nc')
    ds = ds.isel(x=slice(3290,3700),y=slice(7170,8060))

    x = ds['x'].values
    y = ds['y'].values
    mask  = ds['mask'].values
    mask[:] = xr.where(mask==1,2,mask)

    xx,yy = np.meshgrid(x,y)

    dx = x[1]-x[0]
    dy = y[1]-y[0]
    x_ = np.append(x,x[-1]+dx)-dx/2
    y_ = np.append(y,y[-1]+dy)-dy/2

    ax.pcolormesh(x_,y_,mask,cmap=plt.get_cmap('cmo.diff'),vmin=-.3,vmax=2.3)

    cmap = plt.get_cmap('inferno')

    for geom in ['Thwaites','PineIsland','CrossDots']:
        ds = xr.open_dataset(f'../../results/Layer_{geom}_tanh_Tdeep{Tdeep}_ztcl{ztcl}.nc')

        x = ds['x'].values
        y = ds['y'].values
        melt  = ds['melt'].values
        mask  = ds['mask'].values

        xx,yy = np.meshgrid(x,y)

        melt = np.where(melt>1,melt,1)
        levs = np.power(10, np.arange(0,2.51,.01))
        IM = ax.contourf(xx,yy,np.where(mask==3,melt,np.nan),levs,cmap=cmap,norm=mpl.colors.LogNorm())      
    if flow:
        U = ds['U'].values
        V = ds['V'].values

        spd = (U**2 + V**2)**.5
        lw = 4*spd/spd.max()
        strm = ax.streamplot(x,y,U,V,linewidth=lw,color='w',density=5,arrowsize=0)


    the_divider = make_axes_locatable(ax)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(IM, cax=color_axis)
    cbar.set_ticks([1,10,100])
    cbar.set_ticklabels([1,10,100])

    cbar.ax.tick_params(labelsize=21)
    cbar.set_label('Melt [m/yr]', fontsize=21, labelpad=-2)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"../../results/Amundsen_{Tdeep}_{ztcl}.png")
    plt.show()
    
#Quick and dirty function to read geom for Layer model
#Has the option for smoothing (ns) or coarsening (nc) the grid
def quickread(name,nc=1,ns=0):
    if name=='Thwaites_e':
        x0,x1,y0,y1 = 3460,3640,7425,7700
    elif name=='PineIsland':
        x0,x1,y0,y1 = 3290,3550,7170,7400
    elif name=='CrossDots':
        x0,x1,y0,y1 = 3445,3700,7730,8060
    elif name=='Getz':
        x0,x1,y0,y1 = 3510,4330,8080,9050
    elif name=='Cosgrove':
        x0,x1,y0,y1 = 3070,3190,7210,7420  
    elif name=='TottenMU':
        x0,x1,y0,y1 = 10960,11315,8665,9420
    elif name=='Amery':
        x0,x1,y0,y1 = 10010,11160,4975,5450
    elif name=='FRIS':
        x0,x1,y0,y1 = 3600,5630,4560,6420    
    
    """ds: input; nc: factor to reduce resolution; ns: number of times to apply smoothing"""
    
    ds = xr.open_dataset('../../data/BedMachineAntarctica_2020-07-15_v02.nc')
    ds = ds.isel(x=slice(x0,x1),y=slice(y0,y1))
    mask = ds.mask
    mask[:] = xr.where(mask==1,2,mask)
    draft = (ds.surface-ds.thickness).astype('float64')

    ds.coarsen(x=nc,y=nc)
    mask = mask[::nc,::nc]
    draft = draft[::nc,::nc]

    for n in range(0,ns):
        draft = .5*draft + .125*(np.roll(draft,-1,axis=0)+np.roll(draft,1,axis=0)+np.roll(draft,-1,axis=1)+np.roll(draft,1,axis=1))

    draft = xr.DataArray(draft,name='draft')
    
    geom = xr.merge([mask,draft])
    geom['name_geo'] = name
    return geom