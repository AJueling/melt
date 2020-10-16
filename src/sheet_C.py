import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean as cmo
from IPython.display import clear_output

from constants import ModelConstants


class SheetModel(ModelConstants):
    """ sheet model based on Jenkins (1991) and Lazeroms et al. (2018)
        On an A grid
    
        input:
        ds including:
            x   ..  [m]     x coordinate
            y   ..  [m]     y coordinate

            mask ..  [bin]   mask identifying ice shelf cavity
            grl ..  [bin]   mask identifying grounding line; separate for 4 orientations N,S,W,E
            isf ..  [bin]   mask identifying ice shelf front

            zb  ..  [m]     ice shelf draft
            Ta  ..  [degC]  ambient temperature
            Sa  ..  [psu]   ambient salinity

        output:  [calling `.compute()`]
        ds  ..  xarray Dataset holding all quantities with their coordinates
    """
    
    def __init__(self, ds, years=1,maxvel=5,plotint=30):
        
        self.plotint = plotint
        
        #Read input
        self.x  = ds.x
        self.y  = ds.y
        self.mask= ds.mask
        self.grl= ds.grl
        self.isf= ds.isf
        self.zb = ds.zb
        self.Ta = ds.Ta
        self.Sa = ds.Sa

        self.years = years
        self.maxvel = maxvel
        
        self.create_mask()
        self.create_grid()
        
        #Physical parameters
        ModelConstants.__init__(self)
        self.f = -1.37e-4 # Coriolis parameter [1/s]
        self.nfs = -1 #Free slip: 1, no slip: -1
        
        self.initialize_vars()
        
        #Initial values
        self.D += .1#.1+.0001*self.x
        self.T += self.Tf
        self.S += 20#self.Sa.max().values*self.x/self.x.max().values
            
        return

    def create_mask(self):
        #assert sum(sum(self.mask[0,:])+sum(self.mask[-1,:])+sum(self.mask[:,0])+sum(self.mask[:,-1])) == 0
        #assert sum(self.mask*self.grl) == 0
        #assert sum(self.mask*self.isf) == 0
        
        self.isfE = self.isf*self.mask.roll(x= 1,roll_coords=False)
        self.isfN = self.isf*self.mask.roll(y= 1,roll_coords=False)
        self.isfW = self.isf*self.mask.roll(x=-1,roll_coords=False)
        self.isfS = self.isf*self.mask.roll(y=-1,roll_coords=False)
        
        self.grlE = self.grl*self.mask.roll(x= 1,roll_coords=False)
        self.grlN = self.grl*self.mask.roll(y= 1,roll_coords=False)
        self.grlW = self.grl*self.mask.roll(x=-1,roll_coords=False)
        self.grlS = self.grl*self.mask.roll(y=-1,roll_coords=False)        
        
        self.umask = (self.mask+self.isf)*(1-self.grl)*(1-self.grlE.roll(x=-1,roll_coords=False))
        self.vmask = (self.mask+self.isf)*(1-self.grl)*(1-self.grlN.roll(y=-1,roll_coords=False))
        
        return
    
    def create_grid(self):
        #Spatial
        self.nx = len(self.x)
        self.ny = len(self.y)
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        self.xu = self.x+self.dx
        self.yv = self.y+self.dy
        #Temporal
        self.Ah = .5*self.maxvel*self.dx.values
        self.dt = min(self.dx/2/self.maxvel,self.dx**2/self.Ah/8)        
        self.nt = int(self.years*365*24*3600/self.dt)+1
        self.time = np.linspace(0,self.years,self.nt) #Time in years      
        
    def initialize_vars(self):
        self.u = xr.DataArray(np.zeros((self.ny,self.nx)),dims=['y','x'],coords={'y':self.y,'x':self.x})
        self.v = xr.DataArray(np.zeros((self.ny,self.nx)),dims=['y','x'],coords={'y':self.y,'x':self.x})
        self.D = xr.DataArray(np.zeros((self.ny,self.nx)),dims=['y','x'],coords={'y':self.y,'x':self.x})
        self.T = xr.DataArray(np.zeros((self.ny,self.nx)),dims=['y','x'],coords={'y':self.y,'x':self.x})
        self.S = xr.DataArray(np.zeros((self.ny,self.nx)),dims=['y','x'],coords={'y':self.y,'x':self.x})
        
        self.Tf = self.l1*self.Sa+self.l2+self.l3*self.zb # Local freezing point [degC]
        
        #Remove this, rewrite dzdx, dzdy in entrainment
        vp = self.zb.roll(x=-1, roll_coords=False)
        vm = self.zb.roll(x= 1, roll_coords=False)
        self.dzdx = ((vp-vm)/2/self.dx)
        self.dzdx[:,0] = self.dzdx[:,1]
        self.dzdx[:,-1] = self.dzdx[:,-2]        
        
        vp = self.zb.roll(y=-1, roll_coords=False)
        vm = self.zb.roll(y= 1, roll_coords=False)        
        self.dzdy = ((vp-vm)/2/self.dy)
        self.dzdy[0,:] = self.dzdy[1,:]
        self.dzdy[-1,:] = self.dzdy[-2,:]
        return
    
    def im(self,var):
        return .5*(var+var.roll(x=1,roll_coords=False))
    
    def ip(self,var):
        return .5*(var+var.roll(x=-1,roll_coords=False))
    
    def jm(self,var):
        return .5*(var+var.roll(y=1,roll_coords=False))
    
    def jp(self,var):
        return .5*(var+var.roll(y=-1,roll_coords=False))
    
    def lap(self,var):
        return (self.ip(var)+self.im(var)-2*var)/self.dx**2 + (self.jp(var)+self.jm(var)-2*var)/self.dy**2
        
    def convT(self,var):
        t1 = (self.im(var)*self.u.roll(x=1,roll_coords=False) - self.ip(var)*self.u)/self.dx * self.umask
        t2 = (self.jm(var)*self.v.roll(y=1,roll_coords=False) - self.jp(var)*self.v)/self.dy * self.vmask
        
        return t1+t2
        
    def drho(self):
        """Linear equation of state. delta rho/rho0"""
        return self.beta*(self.Sa-self.S) - self.alpha*(self.Ta-self.T)
    
    def entr(self):
        """Entrainment """
        return self.E0*(np.abs(self.im(self.u)*self.dzdx + self.jm(self.v)*self.dzdy))*self.mask
        #return self.E0*(np.abs(self.im(self.u)*self.dzdx + self.jm(self.v)*self.dzdy))*self.mask
        #return np.maximum(0,self.E0*(self.im(self.u)*self.dzdx + self.jm(self.v)*self.dzdy).values)*self.mask

    def melt(self,T):
        """Melt"""       
        return self.cp/self.L*self.CG*(self.im(self.u)**2+self.jm(self.v)**2)**.5*(T-self.Tf)*self.mask
    
    def rhsD(self,D,plot=False):
        """right hand side of d/dt D"""    
        
        t1 = self.convT(D)*self.mask
        t2 = self.melt(self.T)
        t3 = self.entr()
        t4 = self.Ah*self.lap(D)*self.mask
        
        if plot and self.t in np.arange(0,self.nt,self.plotint):
            fig,ax = plt.subplots(1,9,figsize=(30,4),sharex=True,sharey=True)
            vmax = np.max(np.abs(t1+t2+t3+t4))

            im = ax[0].pcolormesh(t1+t2+t3+t4,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[0],orientation='horizontal')
            ax[0].set_title('dDdt')            
            im = ax[1].pcolormesh(t1,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[1],orientation='horizontal')
            ax[1].set_title('conv')
            im = ax[2].pcolormesh(t2,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[2],orientation='horizontal')
            ax[2].set_title('melt')
            im = ax[3].pcolormesh(t3,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[3],orientation='horizontal')
            ax[3].set_title('entr')
            im = ax[4].pcolormesh(t4,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[4],orientation='horizontal')
            ax[4].set_title('diff')
            plt.show();
        
        return (t1+t2+t3+t4)*self.mask

    def rhsT(self,T,plot=False):
        """right hand side of d/dt T"""
        
        t1 = -T*self.dDdt
        t2 =  self.convT(self.D*T)*self.mask
        t3 =  self.entr()*self.Ta
        t4 =  self.melt(T)*(self.Tf - self.L/self.cp)
        t5 =  self.D*self.Ah*self.lap(T)*self.mask
        
        if plot and self.t in np.arange(0,self.nt,self.plotint):
            fig,ax = plt.subplots(1,9,figsize=(30,4),sharex=True,sharey=True)
            vmax = np.max(np.abs(t1+t2+t3+t4+t5))

            im = ax[0].pcolormesh(t1+t2+t3+t4+t5,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[0],orientation='horizontal')
            ax[0].set_title('dTdt')            
            im = ax[1].pcolormesh(t1,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[1],orientation='horizontal')
            ax[1].set_title('T dDdt')
            im = ax[2].pcolormesh(t2,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[2],orientation='horizontal')
            ax[2].set_title('conv')
            im = ax[3].pcolormesh(t3,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[3],orientation='horizontal')
            ax[3].set_title('entr')
            im = ax[4].pcolormesh(t4,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[4],orientation='horizontal')
            ax[4].set_title('melt')
            im = ax[5].pcolormesh(t5,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[5],orientation='horizontal')
            ax[5].set_title('diff')

            plt.show();

        return (t1+t2+t3+t4+t5)/self.D*self.mask

    def rhsS(self,S,plot=False):
        """right hand side of d/dt S"""
        
        t1 = -S*self.dDdt
        t2 =  self.convT(self.D*S)*self.mask
        t3 =  self.entr()*self.Sa
        t4 =  self.D*self.Ah*self.lap(S)*self.mask

        if plot and self.t in np.arange(0,self.nt,self.plotint):
            fig,ax = plt.subplots(1,9,figsize=(30,4),sharex=True,sharey=True)
            vmax = np.max(np.abs(t1+t2+t3+t4))

            im = ax[0].pcolormesh(t1+t2+t3+t4,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[0],orientation='horizontal')
            ax[0].set_title('dSdt')            
            im = ax[1].pcolormesh(t1,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[1],orientation='horizontal')
            ax[1].set_title('S dDdt')
            im = ax[2].pcolormesh(t2,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[2],orientation='horizontal')
            ax[2].set_title('conv')
            im = ax[3].pcolormesh(t3,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[3],orientation='horizontal')
            ax[3].set_title('entr')
            im = ax[4].pcolormesh(t4,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[4],orientation='horizontal')
            ax[4].set_title('diff')

            plt.show();
        
        return (t1+t2+t3+t4)/self.D*self.mask
    
    def rhsu(self,u,plot=False):
        """right hand side of d/dt u"""
        
        t1 = -u*self.ip(self.dDdt)        
        t2 =  (self.D*self.im(u)**2 - self.D.roll(x=-1,roll_coords=False)*self.ip(u)**2)/self.dx * self.umask        
        t3 =  (self.jm(self.ip(self.D))*self.jm(u)*self.ip(self.v.roll(y=-1,roll_coords=False)) - self.jp(self.ip(self.D))*self.jp(u)*self.ip(self.v))/self.dy * self.vmask
        t4 =  self.ip(self.drho())*self.g*self.ip(self.D) * (self.zb.roll(x=-1,roll_coords=False) - self.zb)/self.dx * self.mask
        t5 = -.5*self.g*((self.drho()*self.D*self.D).roll(x=-1,roll_coords=False) - self.drho()*self.D*self.D)/self.dx*self.mask
        t6 =  self.ip(self.D)*self.f*self.ip(self.jm(self.v))
        t7 = -self.Cd*u*np.abs(u)
        t8 =  self.ip(self.D)*self.Ah*self.lap(u)
        
        if plot and self.t in np.arange(0,self.nt,self.plotint):
            fig,ax = plt.subplots(1,9,figsize=(30,4),sharex=True,sharey=True)
            vmax = np.max(np.abs(t1+t2+t3+t4+t5+t6+t7+t8))

            im = ax[0].pcolormesh(t1+t2+t3+t4+t5+t6+t7+t8,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[0],orientation='horizontal')
            ax[0].set_title('dudt')            
            im = ax[1].pcolormesh(t1,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[1],orientation='horizontal')
            ax[1].set_title('u dDdt')
            im = ax[2].pcolormesh(t2,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[2],orientation='horizontal')
            ax[2].set_title('u ddx')
            im = ax[3].pcolormesh(t3,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[3],orientation='horizontal')
            ax[3].set_title('v ddy')
            im = ax[4].pcolormesh(t4,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[4],orientation='horizontal')
            ax[4].set_title('dzb/dx')
            im = ax[5].pcolormesh(t5,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[5],orientation='horizontal')
            ax[5].set_title('dDrho/dx')            
            im = ax[6].pcolormesh(t6,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[6],orientation='horizontal')
            ax[6].set_title('corio')
            im = ax[7].pcolormesh(t7,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[7],orientation='horizontal')
            ax[7].set_title('drag')
            im = ax[8].pcolormesh(t8,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[8],orientation='horizontal')
            ax[8].set_title('diff')

            plt.show();
        
        return (t1+t2+t3+t4+t5+t6+t7+t8)/self.ip(self.D) * self.mask * self.umask

    def rhsv(self,v,plot=False):
        """right hand side of d/dt v"""

        t1 = -v*self.jp(self.dDdt)
        t2 =  (self.im(self.jp(self.D))*self.jp(self.u.roll(x=-1,roll_coords=False))*self.im(v) - self.jp(self.ip(self.D))*self.jp(self.u)*self.ip(v))/self.dx * self.umask
        t3 =  (self.D*self.jm(v)**2 - self.D.roll(y=-1,roll_coords=False)*self.jp(v)**2)/self.dy * self.vmask
        t4 =  self.jp(self.drho())*self.g*self.jp(self.D) * (self.zb.roll(y=-1,roll_coords=False) - self.zb)/self.dy * self.mask
        t5 = -.5*self.g*((self.drho()*self.D*self.D).roll(y=-1,roll_coords=False) - self.drho()*self.D*self.D)/self.dy * self.mask
        t6 = -self.jp(self.D)*self.f*self.jp(self.im(self.u)) 
        t7 = -self.Cd*v*np.abs(v)
        t8 =  self.jp(self.D)*self.Ah*self.lap(v)
        
        if plot and self.t in np.arange(0,self.nt,self.plotint):
            fig,ax = plt.subplots(1,9,figsize=(30,4),sharex=True,sharey=True)
            vmax = np.max(np.abs(t1+t2+t3+t4+t5+t6+t7))

            im = ax[0].pcolormesh(t1+t2+t3+t4+t5+t6+t7,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[0],orientation='horizontal')
            ax[0].set_title('dvdt')            
            im = ax[1].pcolormesh(t1,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[1],orientation='horizontal')
            ax[1].set_title('v dDdt')
            im = ax[2].pcolormesh(t2,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[2],orientation='horizontal')
            ax[2].set_title('u ddx')
            im = ax[3].pcolormesh(t3,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[3],orientation='horizontal')
            ax[3].set_title('v ddy')
            im = ax[4].pcolormesh(t4,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[4],orientation='horizontal')
            ax[4].set_title('dzb/dx')
            im = ax[5].pcolormesh(t5,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[5],orientation='horizontal')
            ax[5].set_title('dDrho/dx')            
            im = ax[6].pcolormesh(t6,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[6],orientation='horizontal')
            ax[6].set_title('corio')
            im = ax[7].pcolormesh(t7,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[7],orientation='horizontal')
            ax[7].set_title('drag')
            im = ax[8].pcolormesh(t8,cmap=plt.get_cmap('cmo.balance'),vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax[8],orientation='horizontal')
            ax[8].set_title('diff')

            plt.show();
            
        return (t1+t2+t3+t4+t5+t6+t7+t8)/self.ip(self.D) * self.mask * self.vmask
    
    def intD(self):
        """integrate d/dt D"""
        k1 = self.rhsD(self.D)
        k2 = self.rhsD(self.D+self.dt*k1/2)
        k3 = self.rhsD(self.D+self.dt*k2/2)
        k4 = self.rhsD(self.D+self.dt*k3,plot=False)

        self.dDdt = (k1+2*(k2+k3)+k4)/6
        return 

    def intu(self):
        """integrate d/dt u"""
        k1 = self.rhsu(self.u)
        k2 = self.rhsu(self.u+self.dt*k1/2)
        k3 = self.rhsu(self.u+self.dt*k2/2)
        k4 = self.rhsu(self.u+self.dt*k3,plot=False)
    
        self.dudt = (k1+2*(k2+k3)+k4)/6
        return 

    def intv(self):
        """integrate d/dt v"""
        k1 = self.rhsv(self.v)
        k2 = self.rhsv(self.v+self.dt*k1/2)
        k3 = self.rhsv(self.v+self.dt*k2/2)
        k4 = self.rhsv(self.v+self.dt*k3,plot=False)
        
        self.dvdt = (k1+2*(k2+k3)+k4)/6 
        return  

    def intT(self):
        """integrate d/dt T"""
        k1 = self.rhsT(self.T)
        k2 = self.rhsT(self.T+self.dt*k1/2)
        k3 = self.rhsT(self.T+self.dt*k2/2)
        k4 = self.rhsT(self.T+self.dt*k3,plot=False)
        
        self.dTdt = (k1+2*(k2+k3)+k4)/6
        return          

    def intS(self):
        """integrate d/dt S"""
        k1 = self.rhsS(self.S)
        k2 = self.rhsS(self.S+self.dt*k1/2)
        k3 = self.rhsS(self.S+self.dt*k2/2)
        k4 = self.rhsS(self.S+self.dt*k3,plot=False)
        
        self.dSdt = (k1+2*(k2+k3)+k4)/6  
        return

    def boundaries(self):          
               
        self.D = xr.where(self.grlN+self.isfN,self.D.roll(y= 1,roll_coords=False),self.D)
        self.D = xr.where(self.grlS+self.isfS,self.D.roll(y=-1,roll_coords=False),self.D)        
        self.D = xr.where(self.grlE+self.isfE,self.D.roll(x= 1,roll_coords=False),self.D)        
        self.D = xr.where(self.grlW+self.isfW,self.D.roll(x=-1,roll_coords=False),self.D)        
        
        self.T = xr.where(self.grlN+self.isfN,self.T.roll(y= 1,roll_coords=False),self.T)
        self.T = xr.where(self.grlS+self.isfS,self.T.roll(y=-1,roll_coords=False),self.T)
        self.T = xr.where(self.grlE+self.isfE,self.T.roll(x= 1,roll_coords=False),self.T)
        self.T = xr.where(self.grlW+self.isfW,self.T.roll(x=-1,roll_coords=False),self.T)
        
        self.S = xr.where(self.grlN+self.isfN,self.S.roll(y= 1,roll_coords=False),self.S)
        self.S = xr.where(self.grlS+self.isfS,self.S.roll(y=-1,roll_coords=False),self.S)        
        self.S = xr.where(self.grlE+self.isfE,self.S.roll(x= 1,roll_coords=False),self.S)
        self.S = xr.where(self.grlW+self.isfW,self.S.roll(x=-1,roll_coords=False),self.S)
        
        self.u = xr.where(self.grlN,self.nfs*self.u.roll(y= 1,roll_coords=False),self.u)
        self.u = xr.where(self.grlS,self.nfs*self.u.roll(y=-1,roll_coords=False),self.u)    
        
        self.v = xr.where(self.grlE,self.nfs*self.v.roll(x= 1,roll_coords=False),self.v)
        self.v = xr.where(self.grlW,self.nfs*self.v.roll(x=-1,roll_coords=False),self.v)

        self.u = xr.where(self.isfN,self.u.roll(y= 1,roll_coords=False),self.u)
        self.u = xr.where(self.isfS,self.u.roll(y=-1,roll_coords=False),self.u)        
        self.u = xr.where(self.isfE,self.u.roll(x= 1,roll_coords=False),self.u)
        self.u = xr.where(self.isfW,self.u.roll(x=-1,roll_coords=False),self.u)
        
        self.v = xr.where(self.isfN,self.v.roll(y= 1,roll_coords=False),self.v)
        self.v = xr.where(self.isfS,self.v.roll(y=-1,roll_coords=False),self.v)         
        self.v = xr.where(self.isfE,self.v.roll(x= 1,roll_coords=False),self.v)
        self.v = xr.where(self.isfW,self.v.roll(x=-1,roll_coords=False),self.v) 
        
        return
            
    def timestep(self):

        self.intD()
        self.intu()
        self.intv()
        self.intT()
        self.intS()
        
        self.D += self.dt*self.dDdt
        self.u += self.dt*self.dudt
        self.v += self.dt*self.dvdt
        self.T += self.dt*self.dTdt
        self.S += self.dt*self.dSdt
        
        return
        
    def plotpanel(self,dax,var,cmap,title,symm=True,strm=False):
        x = np.append(self.x.values,self.x[-1].values+self.dx.values)-self.dx.values/2
        y = np.append(self.y.values,self.y[-1].values+self.dy.values)-self.dy.values/2
        if symm:
            im = dax.pcolormesh(x,y,xr.where(self.mask,var,np.nan).values,cmap=plt.get_cmap(cmap),vmax=np.max(np.abs(var)),vmin=-np.max(np.abs(var)))
        else:
            im = dax.pcolormesh(x,y,xr.where(self.mask,var,np.nan).values,cmap=plt.get_cmap(cmap))
        if strm:
            dax.streamplot(self.x.values,self.y.values,self.u.values*self.umask.values*self.mask.values,self.v.values*self.vmask.values*self.mask.values,density=[.5,1],color='tab:orange')
            
        plt.colorbar(im,ax=dax,orientation='horizontal')
        dax.set_title(title)
        return
    
    def printdiags(self):
        print(self.time[self.t]*365,(self.D*self.dx+self.dy).sum().values,(self.entr()*self.dx*self.dy).sum().values,(self.melt(self.T)*self.dx*self.dy).sum().values)
        
    def compute(self):
        
        for self.t in range(self.nt):           
            if self.t in np.arange(self.plotint,self.nt,self.plotint):     
                clear_output(wait=True)
                fig,ax = plt.subplots(2,6,figsize=(20,10),sharex=True,sharey=True)            

                ax[0,0].set_ylabel(f'day {self.time[self.t]*365:.2f}')

        #        self.plotpanel(ax[0,0],(self.grlN+self.isfN)>0,'cmo.speed','grlN',symm=False)
        #        self.plotpanel(ax[1,0],self.isfN,'cmo.speed','isfN',symm=False)
        #        self.plotpanel(ax[0,1],self.grl,'cmo.speed','grl',symm=False)
        #        self.plotpanel(ax[1,1],(self.mask+self.isf)*(1-self.grl),'cmo.speed','umask',symm=False)
     
                self.plotpanel(ax[0,0],self.u*self.umask,'cmo.curl','u')
                self.plotpanel(ax[1,0],self.dudt,'cmo.balance','du/dt')

                self.plotpanel(ax[0,1],self.v*self.vmask,'cmo.curl','v')
                self.plotpanel(ax[1,1],self.dvdt,'cmo.balance','dv/dt')            

                self.plotpanel(ax[0,2],self.D,'cmo.deep','D',symm=False,strm=True)
                self.plotpanel(ax[1,2],self.dDdt,'cmo.balance','dD/dt')

                self.plotpanel(ax[0,3],self.T,'cmo.thermal','T',symm=False)
                self.plotpanel(ax[1,3],self.dTdt,'cmo.balance','dT/dt')

                self.plotpanel(ax[0,4],self.S,'cmo.haline','S',symm=False)
                self.plotpanel(ax[1,4],self.dSdt,'cmo.balance','dS/dt')

                self.plotpanel(ax[0,5],self.melt(self.T),'cmo.matter','melt',symm=False)
                self.plotpanel(ax[1,5],self.entr(),'cmo.turbid','entr',symm=False)

                plt.show();
            
            self.timestep()
            self.boundaries()
            #self.printdiags()        
        return