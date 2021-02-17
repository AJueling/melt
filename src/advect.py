import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm


zgl = r'$z_{gl}(x,y)'

def advect_grl(ds, eps, T, verbose=True, plots=True):
    """ function to advect grounding line depth as described in Pelle at al (2019) 
    using centered finite different in space and RK4 in time with backward derivatives
    (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
    
    input:
    ds  .. [xr.DataSet]     containing
                            x/y   .. grid coordinates of size nx/ny [m]
                            grl   .. binary mask identifying grounding line 
                            draft .. includes grl depth at grl mask [m]
                            u/v   .. x/y velocities  [m/yr]
    T   .. [int]            number of years to run
    eps .. [float]          diffusion constant epsilon scaled with dx**2,
                            decent values around 1/25
    
    output:
    evo .. [xr.DataArray]  (ny,nx,nt) evolution of grounding line depths $z_{gl}$ in [m]
    """
    dx = (ds.x[1]-ds.x[0]).values
    dy = (ds.y[1]-ds.y[0]).values
    if dx<0:
        if verbose:  print('inverting x-coordinates')
        ds = ds.reindex(x=list(reversed(ds.x)))
        dx = -dx
    if dy<0:
        if verbose:   print('inverting y-coordinates')
        ds = ds.reindex(y=list(reversed(ds.y)))
        dy = -dy
    eps *= dx**2
    maxvel = max([-ds.u.min().values, ds.u.max().values,
                  -ds.v.min().values, ds.v.max().values])
    draftmin = ds.draft.min().values
    draftmax = ds.draft.max().values
    if maxvel==0:  dt = .1  # for test cases with diffusion only
    else:          dt = dx/maxvel/2  # time step in [years]
    Nt = int(T/dt)+1
    if verbose:
        print(f'dx = {dx} m;  dy = {dy} m;  maxvel = {maxvel} m/yr')
        print(f'min(draft) = {draftmin:.2f} m;  max(draft) = {draftmax:.2f} m')
        print(f'dt = {dt} yr;  Nt = {Nt}; T = {T} yr')
    
    ds = ds.pad(x=1, mode='edge')
    ds = ds.pad(y=1, mode='edge')
    evo = xr.DataArray(data=np.zeros((len(ds.y), len(ds.x), Nt)),
                       dims=['y','x','time'],
                       coords={'y':ds.y, 'x':ds.x, 'time':np.arange(0,dt*Nt-1e-10,dt)}
                      )
    # domain mask of points inside ice shelf, but outside of grl
    mask_ = ds.mask.where(ds.grl==0)
    ds['u'] = xr.where(ds.mask, ds.u, 0)
    ds['v'] = xr.where(ds.mask, ds.v, 0)
    
    # initial conditions
    # in ice shelf set depth to minimum draft depth
    evo[:,:,0] = xr.where(ds.mask==1, draftmin, 0)
    #  at grl set depth to draft
    evo[:,:,0] = xr.where(ds.grl==1, ds.draft, evo[:,:,0])  
    
    def reset_pads(z):
        """ resets padding values
        so that 1st&2nd order derivatives = 0 on boundary
        input:
        z .. 2D xr.DataArray
        """
        z[ 0, :] = z.isel(y= 1)
        z[-1, :] = z.isel(y=-2)
        z[ :, 0] = z.isel(x= 1)
        z[ :,-1] = z.isel(x=-2)
        return z

    def rhs(z, eps):
        """ evaluates right hand side function
        with space centered, 2nd order accurate method
        """
        ip1 = z.roll(x=-1, roll_coords=False)
        im1 = z.roll(x= 1, roll_coords=False)
        jp1 = z.roll(y=-1, roll_coords=False)
        jm1 = z.roll(y= 1, roll_coords=False)
        adv = ds.u*(ip1-im1)/2/dx + ds.v*(jp1-jm1)/2/dy
        dif = (ip1-2*z+im1)/dx**2 + (jp1-2*z+jm1)/dy**2
        return -adv+eps*dif

    for t in tqdm(range(1,Nt)):  # explicit time evolution
        evo_ = evo.isel(time=t-1).copy()
        k1 = rhs(evo_        , eps)
        k2 = rhs(evo_+dt*k1/2, eps)
        k3 = rhs(evo_+dt*k2/2, eps)
        k4 = rhs(evo_+dt*k3  , eps)
        evo[:,:,t] = evo_ + dt*(k1+2*(k2+k3)+k4)/6       
        
        # update boundary conditions
        evo[:,:,t] = xr.where(ds.grl==1 , ds.draft, evo[:,:,t])  # grl depth constant
        evo[:,:,t] = reset_pads(evo[:,:,t])
        evo[:,:,t] = xr.where(evo[:,:,t]<draftmin, draftmin, evo[:,:,t])  #
        evo[:,:,t] = xr.where(evo[:,:,t]>draftmax, draftmax, evo[:,:,t])  #
    evo = evo.where(ds.mask==1)  # mask out everything outside ice shelf
    
    if plots:
        # convergence plot
        plt.figure(figsize=(6.4,4), constrained_layout=True)
        (np.sqrt((evo-evo.shift(time=-1))**2)/ds.mask.sum().values)\
        .sum(dim=['x','y']).plot()
        plt.axhline(0, c='k', lw=.5)
        #plt.yscale('log')
        plt.ylabel(r'$\frac{1}{N}\Sigma\sqrt{(z_t-z_{t-1})^2}$  [m]')
        plt.xlabel('time [yr]')
        plt.show()

        # final z_{gl} plot
        plt.figure(figsize=(6.4,4), constrained_layout=True)
        evo.isel(time=-1).plot(label=zgl)
        plt.title(f'final {zgl} at time={evo.time[-1].values} years')
        plt.show()
    return evo