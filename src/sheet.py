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
        self.Ta   = ds.Ta
        self.Sa   = ds.Sa

        #Physical parameters
        ModelConstants.__init__(self)
        self.f = -1.37e-4     # Coriolis parameter [1/s]
        
        #Some input params
        self.days = 6         # Total runtime in days
        self.nu = .5          # Nondimensional factor for Robert Asselin time filter
        self.slip = 2         # Nondimensional factor Free slip: 0, no slip: 2, partial no slip: [0..2]  
        
        self.Ah = 500         # Laplacian viscosity [m^2/s]
        self.dt = 150         # Time step [s]
        self.Kh = 500         # Diffusivity [m^2/s]
        
        #Parameters to stabilise plume thickness
        self.minD = 0.10      # Thickness below which restoring is activated
        self.tres = 1e10      # Time scale of restoring. Set to ~ 900sec to use to prevent emptying of cell
        self.Ddiff = 0        # Diffusion in thickness. Set to ~ 20 m2/s to use
        
        #Some parameters for displaying output
        self.diagint = 30     # Timestep at which to print diagnostics
        self.figsize = (15,10)
    
    def integrate(self):
        """Integration of 2 time steps, now-centered Leapfrog scheme"""
        self.D[2,:,:] = self.D[0,:,:] + 2*self.dt * su.rhsD(self)
        self.u[2,:,:] = self.u[0,:,:] + 2*self.dt * su.rhsu(self)
        self.v[2,:,:] = self.v[0,:,:] + 2*self.dt * su.rhsv(self)
        self.T[2,:,:] = self.T[0,:,:] + 2*self.dt * su.rhsT(self)
        self.S[2,:,:] = self.S[0,:,:] + 2*self.dt * su.rhsS(self)         
      
        return
    
    def timefilter(self):
        """Time filter, Robert Asselin scheme"""
        self.D[1,:,:] += self.nu/2 * (self.D[0,:,:]+self.D[2,:,:]-2*self.D[1,:,:]) * self.tmask
        self.u[1,:,:] += self.nu/2 * (self.u[0,:,:]+self.u[2,:,:]-2*self.u[1,:,:]) * self.umask
        self.v[1,:,:] += self.nu/2 * (self.v[0,:,:]+self.v[2,:,:]-2*self.v[1,:,:]) * self.vmask
        self.T[1,:,:] += self.nu/2 * (self.T[0,:,:]+self.T[2,:,:]-2*self.T[1,:,:]) * self.tmask
        self.S[1,:,:] += self.nu/2 * (self.S[0,:,:]+self.S[2,:,:]-2*self.S[1,:,:]) * self.tmask
        return
    
    def printdiags(self):
        """Print diagnostics at given intervals as defined below"""
        #Maximum thickness
        diag0 = (self.D[1,:,:]*self.tmask).max().values
        #Average thickness [m]
        diag1 = ((self.D[1,:,:]*self.tmask*self.dx*self.dy).sum()/(self.tmask*self.dx*self.dy).sum()).values
        #Maximum melt rate [m/yr]
        diag2 = 3600*24*365.25*su.melt(self).max().values
        #Average melt rate [m/yr]
        diag3 = 3600*24*365.25*((su.melt(self)*self.dx*self.dy).sum()/(self.tmask*self.dx*self.dy).sum()).values
        #Minimum thickness
        diag4 = xr.where(self.tmask==0,100,self.D[1,:,:]).min().values
        #Volume tendency [Sv]
        #diag5 = 1e-6*((self.D[2,:,:]-self.D[0,:,:])*self.tmask*self.dx*self.dy).sum()/2/self.dt.values
        #Integrated melt flux [Sv]
        #diag6 = 1e-6*(self.melt()*self.dx*self.dy).sum().values
        #Integrated entrainment [Sv]
        #diag7 = 1e-6*(self.entr()*self.dx*self.dy).sum().values
        #Integrated volume thickness convergence == net in/outflow [Sv]
        diag8 = 1e-6*(su.convTups(self,self.D[1,:,:])*self.tmask*self.dx*self.dy).sum().values
        #Maximum velocity [m/s]
        diag9 = ((su.im(self.u[1,:,:])**2 + su.jm(self.v[1,:,:])**2)**.5).max().values
        
        print(f'{self.time[self.t]:.03f} days | D_av: {diag1:.03f}m | D_max: {diag0:.03f}m | D_min: {diag4:.03f}m | M_av: {diag3:.03f} m/yr | M_max: {diag2:.03f} m/yr | In/out: {diag8:.03f} Sv | Max. vel: {diag9:.03f} m/s')
                  
        return
                
    def updatevars(self):
        """Update temporary variables"""
        su.updatevar(self,self.D)
        su.updatevar(self,self.u)
        su.updatevar(self,self.v)
        su.updatevar(self,self.T)
        su.updatevar(self,self.S) 
        return
    
    def plotfields(self):
        su.plotpanels(self)
        return

    def plotdiags(self):
        su.plotdudt(self)
        #su.plotdvdt(self)
        su.plotdSdt(self)
        su.plotdDdt(self)
        return
    
    def compute(self):
        su.create_mask(self)
        su.create_grid(self)
        su.initialize_vars(self)

        for self.t in range(self.nt):
            self.updatevars()
            self.integrate()
            self.timefilter()
            if self.t in np.arange(self.diagint,self.nt,self.diagint):
                self.printdiags()
    
        print('-----------------------------')
        print(f'Run completed, final values:')
        self.printdiags()
        print('-----------------------------')
        print('-----------------------------')
        #Output
        melt = xr.DataArray(su.melt(self),dims=['y','x'],coords={'y':self.y,'x':self.x},name='melt')
        entr = xr.DataArray(su.entr(self),dims=['y','x'],coords={'y':self.y,'x':self.x},name='entr')
        mav  = xr.DataArray(3600*24*365.25*((su.melt(self)*self.dx*self.dy).sum()/(self.tmask*self.dx*self.dy).sum()).values,name='mav')
        
        ds = xr.merge([self.D[1,:,:],self.u[1,:,:],self.v[1,:,:],self.T[1,:,:],self.S[1,:,:],melt,entr,mav])
    
        return ds