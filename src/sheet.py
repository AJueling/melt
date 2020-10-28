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
            x   ..  [m]     x coordinate
            y   ..  [m]     y coordinate

            mask ..  [bin]   mask identifying ocean (0), grounded ice (1), ice shelf (3)

            zb  ..  [m]     ice shelf draft
            Ta  ..  [degC]  ambient temperature
            Sa  ..  [psu]   ambient salinity

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
        self.zb   = ds.zb
        self.Ta   = ds.Ta
        self.Sa   = ds.Sa

        #Physical parameters
        ModelConstants.__init__(self)
        self.f = -1.37e-4     # Coriolis parameter [1/s]
        
        #Some input params
        self.diagint = 100    # Timestep at which to print diagnostics
        self.Ddiff = .01      # Factor to reduce diffusion in thickness
        self.days = 5         # Total runtime in days
        self.maxvel = 3       # Maximum velocity to define time step and diffusivity [m/s]
        self.nu = .5          # Nondimensional factor for Robert Asselin time filter
        self.slip = 2         # Nondimensional factor Free slip: 0, no slip: 2, partial no slip: [0..2]
        self.nfs  = -1
        
    def drho(self):
        """Linear equation of state. delta rho/rho0"""
        return self.beta*(self.Sa-self.S[1,:,:]) - self.alpha*(self.Ta-self.T[1,:,:])
    
    def entr(self):
        """Entrainment """   
        return self.E0*(np.abs(su.im(self.u[1,:,:])*self.dzdx + su.jm(self.v[1,:,:])*self.dzdy))*self.tmask

    def melt(self):
        """Melt"""       
        return self.cp/self.L*self.CG*(su.im(self.u[1,:,:])**2+su.jm(self.v[1,:,:])**2)**.5*(self.T[1,:,:]-self.Tf)*self.tmask
    
    def rhsD(self):
        """right hand side of d/dt D"""
        t1 = su.convT(self,self.D[1,:,:])
        t2 = self.melt()
        t3 = self.entr()
        t4 = self.Ddiff*self.Ah*su.lap(self,self.D[0,:,:])
        
        return (t1+t2+t3+t4) * self.tmask

    def rhsu(self):
        """right hand side of d/dt u"""
        
        t1 = -self.u[1,:,:] * su.ip((self.D[2,:,:]-self.D[0,:,:]))/(2*self.dt)        
        t2 =  (self.D[1,:,:]*su.im(self.u[1,:,:])**2 - self.D[1,:,:].roll(x=-1,roll_coords=False)*su.ip(self.u[1,:,:])**2)/self.dx  # * self.tmask.roll(x=-1,roll_coords=False)
        t3 =  (su.jm(su.ip(self.D[1,:,:]))*su.jm(self.u[1,:,:])*su.ip(self.v[1,:,:].roll(y=-1,roll_coords=False)) - su.jp(su.ip(self.D[1,:,:]))*su.jp(self.u[1,:,:])*su.ip(self.v[1,:,:]))/self.dy * self.vmask # * self.tmask.roll(y=-1,roll_coords=False)

        #Equations from Hewitt 2020:
        #t4 =  su.ip(self.drho())*self.g*su.ip(self.D[1,:,:]) * self.dzdx * self.tmask
        #t5 = -.5*self.g*((self.drho()*self.D[1,:,:]**2).roll(x=-1,roll_coords=False) - self.drho()*self.D[1,:,:]**2)/self.dx * self.tmask * self.tmask.roll(x=-1,roll_coords=False)

        #Equations from Holland et al 2006:
        t4 = -self.g*su.ip(self.drho()*self.D[1,:,:])*((self.D[1,:,:]-self.zb).roll(x=-1,roll_coords=False) - (self.D[1,:,:]-self.zb))/self.dx * self.tmask# * self.tmask.roll(x=-1,roll_coords=False)
        t5 = .5*self.g*su.ip(self.D[1,:,:])**2*(self.drho().roll(x=-1,roll_coords=False)-self.drho())/self.dx * self.tmask# * self.tmask.roll(x=-1,roll_coords=False)
        
        t6 =  su.ip(self.D[1,:,:])*self.f*su.ip(su.jm(self.v[1,:,:]))
        t7 = -self.Cd*self.u[1,:,:]*np.abs(self.u[1,:,:])
        
        t8 = self.Ah*su.lapu(self,self.u[0,:,:])

        return (t1+t2+t3+t4+t5+t6+t7+t8)/su.ip(self.D[1,:,:]) * self.umask

    def rhsv(self):
        """right hand side of d/dt v"""

        t1 = -self.v[1,:,:] * su.jp((self.D[2,:,:]-self.D[0,:,:]))/(2*self.dt) 
        t2 =  (su.im(su.jp(self.D[1,:,:]))*su.jp(self.u[1,:,:].roll(x=-1,roll_coords=False))*su.im(self.v[1,:,:]) - su.jp(su.ip(self.D[1,:,:]))*su.jp(self.u[1,:,:])*su.ip(self.v[1,:,:]))/self.dx * self.umask # * self.tmask.roll(x=-1,roll_coords=False)
        t3 =  (self.D[1,:,:]*su.jm(self.v[1,:,:])**2 - self.D[1,:,:].roll(y=-1,roll_coords=False)*su.jp(self.v[1,:,:])**2)/self.dy * self.vmask  # * self.tmask.roll(y=-1,roll_coords=False)

        #Equations from Hewitt 2020:
        #t4 =  su.jp(self.drho())*self.g*su.jp(self.D[1,:,:]) * self.dzdy * self.tmask
        #t5 = -.5*self.g*((self.drho()*self.D[1,:,:]**2).roll(y=-1,roll_coords=False) - self.drho()*self.D[1,:,:]**2)/self.dy * self.tmask * self.tmask.roll(y=-1,roll_coords=False)

        #Equations from Holland et al 2006:
        t4 = -self.g*su.jp(self.drho()*self.D[1,:,:])*((self.D[1,:,:]-self.zb).roll(y=-1,roll_coords=False) - (self.D[1,:,:]-self.zb))/self.dy * self.tmask# * self.tmask.roll(y=-1,roll_coords=False)
        t5 = .5*self.g*su.jp(self.D[1,:,:])**2*(self.drho().roll(y=-1,roll_coords=False)-self.drho())/self.dy * self.tmask #* self.tmask.roll(y=-1,roll_coords=False)

        t6 = -su.jp(self.D[1,:,:])*self.f*su.jp(su.im(self.u[1,:,:])) 
        t7 = -self.Cd*self.v[1,:,:]*np.abs(self.v[1,:,:])
        
        t8 = self.Ah*su.lapv(self,self.v[0,:,:])
            
        return (t1+t2+t3+t4+t5+t6+t7+t8)/su.jp(self.D[1,:,:]) * self.vmask
    
    def rhsT(self):
        """right hand side of d/dt T"""
        
        t1 = -self.T[1,:,:] * (self.D[2,:,:]-self.D[0,:,:])/(2*self.dt)
        t2 =  su.convT(self,self.D[1,:,:]*self.T[1,:,:])
        t3 =  self.entr()*self.Ta
        t4 =  self.melt()*(self.Tf - self.L/self.cp)
        t5 =  self.Ah*su.lapT(self,self.T[0,:,:])

        return (t1+t2+t3+t4+t5)/self.D[1,:,:] * self.tmask

    def rhsS(self):
        """right hand side of d/dt S"""
        
        t1 = -self.S[1,:,:] * (self.D[2,:,:]-self.D[0,:,:])/(2*self.dt)
        t2 =  su.convT(self,self.D[1,:,:]*self.S[1,:,:])
        t3 =  self.entr()*self.Sa
        t4 =  self.Ah*su.lapT(self,self.S[0,:,:])
        
        return (t1+t2+t3+t4)/self.D[1,:,:] * self.tmask
    
    def integrate(self):
        """Integration of 2 time steps, now-centered Leapfrog scheme"""
        self.D[2,:,:] = self.D[0,:,:] + 2*self.dt * self.rhsD()
        self.u[2,:,:] = self.u[0,:,:] + 2*self.dt * self.rhsu()
        self.v[2,:,:] = self.v[0,:,:] + 2*self.dt * self.rhsv()
        self.T[2,:,:] = self.T[0,:,:] + 2*self.dt * self.rhsT()
        self.S[2,:,:] = self.S[0,:,:] + 2*self.dt * self.rhsS()         
      
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
        diag2 = 3600*24*365.25*self.melt().max().values
        
        #Average melt rate [m/yr]
        diag3 = 3600*24*365.25*((self.melt()*self.dx*self.dy).sum()/(self.tmask*self.dx*self.dy).sum()).values
        
        #Minimum thickness
        diag4 = xr.where(self.tmask==0,100,self.D[1,:,:]).min().values
        
        #Volume tendency [Sv]
        #diag4 = 1e-6*((self.D[2,:,:]-self.D[0,:,:])*self.tmask*self.dx*self.dy).sum()/2/self.dt.values
        
        #Integrated melt flux [Sv]
        #diag5 = 1e-6*(self.melt()*self.dx*self.dy).sum().values
        
        #Integrated entrainment [Sv]
        #diag6 = 1e-6*(self.entr()*self.dx*self.dy).sum().values
        
        #Integrated volume thickness convergence == net in/outflow [Sv]
        diag7 = 1e-6*(su.convT(self,self.D[1,:,:])*self.tmask*self.dx*self.dy).sum().values
        
        print(f'{self.time[self.t]:.03f} days | D_av: {diag1:.03f}m | D_max: {diag0:.03f}m | D_min: {diag4:.03f}m | M_av: {diag3:.03f} m/yr | M_max: {diag2:.03f} m/yr | In/out: {diag7:.03f} Sv')
            
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

    def compute(self): 
        su.create_mask(self)
        su.create_grid(self)
        su.initialize_vars(self)

        for self.t in range(self.nt):      
            self.updatevars()
            self.integrate()
            self.timefilter()
            su.boundaries(self)
            if self.t in np.arange(1,self.nt,self.diagint):
                self.printdiags()
    
        print('-----------------------------')
        print(f'Run completed, final values:')
        self.printdiags()
        melt = xr.DataArray(self.melt(),dims=['y','x'],coords={'y':self.y,'x':self.x},name='melt')
        entr = xr.DataArray(self.entr(),dims=['y','x'],coords={'y':self.y,'x':self.x},name='entr')
        
        ds = xr.merge([self.D[1,:,:],self.u[1,:,:],self.v[1,:,:],self.T[1,:,:],self.S[1,:,:],melt,entr])
    
        return ds