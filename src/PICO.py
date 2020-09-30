import numpy as np
import xarray as xr

from geometry import ModelGeometry
from constants import ModelConstants


class PicoModel(ModelConstants, ModelGeometry):
    """ 2D melt model
        based on Reese et al. (2018), doi: 10.5194/tc-12-1969-2018
        melt plume driven by the ice pump circulation
        assumes: small angle
        equation and table numbers refer to publication (doi: 10.1175/JPO-D-18-0131.1)
        
        caution:
        - x/y are used for the spatial dimensions/coordinates in the paper and for T/S differences in appendix 1
        - at the very end of appendix A1, there is a sign error in the equation for T_1
        
        input:
        name  ..  (str)     name of ice shelf
        Ta    ..  (float)   ambient temperature in [degC]
        Sa    ..  (float)   ambient salinity in [psu]
        
        output:  [calling `.compute()`]
        ds  ..  xr.Dataset holding all quantities with their coordinates
            .  draft .. (float) ocean-ice interface depth in [m]
            .  mask  .. (bool)  ice shelf mask
            .  grl   .. (bool)  grounding line mask
            .  isf   .. (bool)  ice shelf front mask
    """
    
    def __init__(self, name, Ta, Sa, n):
        """ initialize model class
        define parameters
        geometry: calc. distances, define boxes, calc. area
        
        """
        ModelConstants.__init__(self)
        ModelGeometry.PICO(self, name=name, n=n)
        # self.ds = da.copy()
        
        # geometry
        self.n = n  # number of boxes (int; <=5)
        
        # hydrostatic pressure at each location(x,y)
        # assuming constant density
        self.p = self.ds.draft*self.rho0*self.g
        
        self.nulambda = self.rhoi/self.rhow*self.L/self.cp
        self.A = np.zeros((self.n+1))
        self.g1 = np.zeros((self.n+1))
        self.g2 = np.zeros((self.n+1))
        self.pk = np.zeros((self.n+1))
        self.find_distances()
        self.define_boxes()
        self.calc_area()
        print('area', self.A)
        
        for k in np.arange(1,n+1):
            self.g1[k] = self.A[k]*self.gammae  # defined just above (A6)
            self.g2[k] = self.g1[k]/self.nulambda
            self.pk[k] = self.p.where(self.ds.box==k).mean(['x','y'])
        
        self.T = np.zeros((self.n+1))
        self.S = np.zeros((self.n+1))
        self.m = np.zeros((self.n+1))
        self.T[0] = Ta
        self.S[0] = Sa
        
        return
    
    def T_s(self, k):
        """ spatially explicit T^\star(x,y) temperature defined just above (A6) """
        return self.a*self.S[k-1]+self.b-self.c*self.pk[k]-self.T[k-1]
    
    def solve_B_1(self):
        """ solve for the T1, S1, m1, circulation q, appendix A1 """
        # T1, S1
        s = self.S[0]/self.nulambda
        Crbsa = self.C*self.rho0*(self.beta*s-self.alpha)
        x = -self.g1[1]/(2*Crbsa) + np.sqrt((self.g1[1]/2/Crbsa)**2-(self.g1[1]*self.T_s(1))/(Crbsa))
        # assert x>0  # argued at the end of appendix A1
        self.T[1] = self.T[0] - x
        y = x*self.S[0]/(self.nulambda)
        self.S[1] = self.S[0] - y
        
        # overturning eqn. (A9)
        adT = self.alpha*(self.T[0]-self.T[1])
        bdS = self.beta *(self.S[0]-self.S[1])
        self.q = self.C*self.rho0*(bdS-adT) 
        
        # melt
        self.melt_k(1)
        return
        
    def solve_B_k(self):
        """ solving iteratively for boxes k in [2,n], appendix A2 """
        for k in np.arange(2,self.n+1):
            x = -self.g1[k]*self.T_s(k)/(self.q+self.g1[k]-self.g2[k]*self.a*self.S[k-1])
            self.T[k] = self.T[k-1]-x
            y = self.S[k-1]*x/self.nulambda  # (A8)
            self.S[k] = self.S[k-1]-y
            self.melt_k(k)
        return
    
    def melt_k(self, k):
        """ melt in box k, eqn (13)
        1. solve melt for all grid cells in box k, each with pressure
        2. sum to total melt in box k
        """
        # speatially explicit melt with T_k, S_k  ambient temperatures
        # also computes melt outside of box k, but that is masked out in the following step
        # melt at each location
        mk = -self.gammae/self.nulambda*(self.a*self.S[k]+self.b-self.c*self.p-self.T[k])
        # total melt of box k
        self.m[k] = (mk*self.ds.area).where(self.ds.box==k).sum()
        return
        
    def compute_pico(self):
        """ """
        self.solve_B_1()
        if self.n>1:
            self.solve_B_k()
        kwargs = {'dims':'boxnr', 'coords':{'boxnr':np.arange(self.n+1)}}
        T = xr.DataArray(data=self.T, name='Tk', **kwargs)
        S = xr.DataArray(data=self.S, name='Sk', **kwargs)
        m = xr.DataArray(data=self.m, name='mk', **kwargs)
        q = xr.DataArray(data=self.q, name='q')
        
        return xr.merge([self.ds, T, S, m, q])