import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean as cmo
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plotmelt(ds,figsize=(10,10)):
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

    ax.pcolormesh(x_,y_,mask,cmap=plt.get_cmap('cmo.diff'),vmin=-1,vmax=3)
    
    cmap = plt.get_cmap('inferno')
    
    melt = np.where(melt>.1,melt,.1)
    levs = np.power(10, np.arange(-1,2.51,.01))
    IM = ax.contourf(xx,yy,np.where(mask==3,melt,np.nan),levs,cmap=cmap,norm=mpl.colors.LogNorm())      

    the_divider = make_axes_locatable(ax)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(IM, cax=color_axis)
    cbar.set_ticks([.1,1,10,100])
    cbar.set_ticklabels([.1,1,10,100])
    cbar.ax.tick_params(labelsize=21)
    cbar.set_label('Melt [m/yr]', fontsize=21, labelpad=-2)
    
    if ds['name_model'] == 'Layer':
        U = ds['U'].values
        V = ds['V'].values
        
        spd = (U**2 + V**2)**.5
        lw = 4*spd/spd.max()
        strm = ax.streamplot(x,y,U,V,linewidth=lw,color='w',density=5,arrowsize=.5)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{ds['filename'].values}.png")
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