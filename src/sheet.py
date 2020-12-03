import numpy as np
import xarray as xr
#import matplotlib.pyplot as plt
#import cmocean as cmo
#from IPython.display import clear_output

from constants import ModelConstants

import sheet_utils as su

class SheetModel(ModelConstants):
    """ 2D sheet model based on Jenkins (1991) and Lazeroms et al. (2018)
    
        input:
        ds including:
            x      ..  [m]     x coordinate
            y      ..  [m]     y coordinate

            2D [y,x] fields:
            mask   ..  [bin]   mask identifying ocean (0), grounded ice (2), ice shelf (3)
            draft  ..  [m]     ice shelf draft
            Ta     ..  [degC]  ambient temperature
            Sa     ..  [psu]   ambient salinity

        output:  [calling `.compute()`]
        ds  ..  xarray Dataset holding all quantities with their coordinates
    """
    
    def __init__(self, ds):
        #Read input
        self.dx = ds.x[1]-ds.x[0]
        self.dy = ds.y[1]-ds.y[0]
        if self.dx<0:
            print('inverting x-coordinates')
            ds = ds.reindex(x=list(reversed(ds.x)))
            self.dx = -self.dx
        if self.dy<0:
            print('inverting y-coordinates')
            ds = ds.reindex(y=list(reversed(ds.y)))
            self.dy = -self.dy
            
        self.x    = ds.x
        self.y    = ds.y        
        self.mask = ds.mask
        self.zb   = ds.draft
        self.z    = ds.z.values
        self.Tz   = ds.Tz.values
        self.Sz   = ds.Sz.values

        #Physical parameters
        ModelConstants.__init__(self)
        self.f = -1.37e-4     # Coriolis parameter [1/s]
        
        #Some input params
        self.days = 6         # Total runtime in days
        self.nu = .5          # Nondimensional factor for Robert Asselin time filter
        self.slip = 2         # Nondimensional factor Free slip: 0, no slip: 2, partial no slip: [0..2]  
        self.Ah = 150         # Laplacian viscosity [m^2/s]
        self.dt = 30          # Time step [s]
        self.Kh = 50          # Diffusivity [m^2/s]
        
        #Some parameters for displaying output
        self.diagint = 100    # Timestep at which to print diagnostics
        self.figsize = (15,10)
        self.verbose = True
    
    def integrate(self):
        """Integration of 2 time steps, now-centered Leapfrog scheme"""
        su.intD(self,2*self.dt)
        su.intu(self,2*self.dt)
        su.intv(self,2*self.dt)
        su.intT(self,2*self.dt)
        su.intS(self,2*self.dt)        
        return
    
    def timefilter(self):
        """Time filter, Robert Asselin scheme"""
        self.D[1,:,:] += self.nu/2 * (self.D[0,:,:]+self.D[2,:,:]-2*self.D[1,:,:]) * self.tmask
        self.u[1,:,:] += self.nu/2 * (self.u[0,:,:]+self.u[2,:,:]-2*self.u[1,:,:]) * self.umask
        self.v[1,:,:] += self.nu/2 * (self.v[0,:,:]+self.v[2,:,:]-2*self.v[1,:,:]) * self.vmask
        self.T[1,:,:] += self.nu/2 * (self.T[0,:,:]+self.T[2,:,:]-2*self.T[1,:,:]) * self.tmask
        self.S[1,:,:] += self.nu/2 * (self.S[0,:,:]+self.S[2,:,:]-2*self.S[1,:,:]) * self.tmask
        return
                
    def updatevars(self):
        """Update temporary variables"""
        self.D = np.roll(self.D,-1,axis=0)
        self.u = np.roll(self.u,-1,axis=0)
        self.v = np.roll(self.v,-1,axis=0)
        self.T = np.roll(self.T,-1,axis=0)
        self.S = np.roll(self.S,-1,axis=0)
        su.updatesecondary(self)
        return
    
    def plotfields(self):
        su.plotpanels(self)
        return

    def plotdiags(self):
        su.plotdDdt(self)
        su.plotdudt(self)
        su.plotdTdt(self)
        su.plotdSdt(self)
        return
    
    def plotmelt(self,figsize=(15,15)):
        su.plotmelt(self,figsize=figsize)
        return
    
    def compute(self):
        su.create_mask(self)
        su.create_grid(self)
        su.initialize_vars(self)

        for self.t in range(self.nt):
            self.updatevars()
            self.integrate()
            self.timefilter()
            self.D = np.where(self.D<1,1,self.D)
            if self.t in np.arange(self.diagint,self.nt,self.diagint):
                su.printdiags(self)
        if self.verbose:
            print('-----------------------------')
            print(f'Run completed, final values:')
            su.printdiags(self)
            print('-----------------------------')
            print('-----------------------------')
        
        #Output

        u = xr.DataArray(self.u[1,:,:],dims=['y','x'],coords={'y':self.y,'x':self.x},name='U')
        v = xr.DataArray(self.u[1,:,:],dims=['y','x'],coords={'y':self.y,'x':self.x},name='V')
        D = xr.DataArray(self.u[1,:,:],dims=['y','x'],coords={'y':self.y,'x':self.x},name='D')
        T = xr.DataArray(self.u[1,:,:],dims=['y','x'],coords={'y':self.y,'x':self.x},name='T')
        S = xr.DataArray(self.u[1,:,:],dims=['y','x'],coords={'y':self.y,'x':self.x},name='S')
        
        melt = xr.DataArray(self.melt,dims=['y','x'],coords={'y':self.y,'x':self.x},name='melt')
        entr = xr.DataArray(self.entr,dims=['y','x'],coords={'y':self.y,'x':self.x},name='entr')
        mav  = xr.DataArray(3600*24*365.25*(self.melt*self.dx*self.dy).sum()/(self.tmask*self.dx*self.dy).sum(),name='mav')
        mmax = xr.DataArray(3600*24*365.25*self.melt.max(),name='mmax')
        
        ds = xr.merge([u,v,D,T,S,melt,entr,mav,mmax])
    
        return ds