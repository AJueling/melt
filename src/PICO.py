import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

class PicoModel(object):
    """ 2D melt model
    based on Reese et al. (2018), doi: 10.5194/tc-12-1969-2018
    melt plume driven by the ice pump circulation
    assumes: small angle
    equation and table numbers refer to publication (doi: 10.1175/JPO-D-18-0131.1)
    
    caution:
    - x/y are used for the spatial dimensions/coordinates in the paper and for T/S differences in appendix 1
    - at the very end of appendix A1, there is a sign error in the equation for T_1
    
    input:
    da  ..  (xr.DataSet)
         .  draft .. [m]       ocen ice interface depth
         .  mask  .. [binary]  ice shelf mask
         .  grl   .. [binary]  grounding line mask
         .  isf   .. [binary]  ice shelf freint ask
    Ta  ..  (float)        [degC]  ambient temperature
    Sa  ..  (float)        [psu]   ambient salinity
    
    output:  [calling `.compute()`]
    ds  ..  xarray Dataset holding all quantities with their coordinates
    """
    
    def __init__(self, da, Ta, Sa, n):
        """ initialize model class
        define parameters
        geometry: calc. distances, define boxes, calc. area
        
        """
        self.ds = da.copy()
        
        # geometry
        self.n = n  # number of cells
        
        # parameters
        self.a      = -0.0572   # [degC /psu]
        self.b      =  0.0788   # [degC]
        self.c      =  7.77e-8  # [degC/Pa]
        self.alpha  =  7.5e-5   # [1/degC]
        self.beta   =  7.7e-4   # [1/psu]
        self.rho0   =  1033     # [kg/m^3]
        self.L      =  3.34e5   # [J/kg]
        self.cp     =  3974     # [J/kg/degC]
        self.rhoi   =   910     # [kg/m^3]
        self.rhow   =  1028     # [kg/m^3]
        self.gammaS =  2e-6     # [m/s]
        self.gammaT =  5e-5     # [m/s]
        self.gammae =  2e-5     # [m/s]
        self.C      =  1e6      # [m^6/s/kg]
        self.g      =  9.81     # [m/s^2]

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
    
    # geometry
    def distance_to_line(self, mask_a, mask_b):
        """ calculate minimum distance from all points in mask_a to points in mask_b
        input:  (all 2D arrays)
        x, y    ..  x/y coordinate xr.DataArrays
        mask_a  ..  mask of points for which minimum distance to mask_b is determined
        mask_b  ..  mask of line (grounding line/ ice shelf front)

        output:  reconstructed xr.DataArray with distances
        """
        X, Y = np.meshgrid(mask_a.x.values, mask_a.y.values)
        x = xr.DataArray(data=X, dims=['y','x'], coords={'y':mask_a.y,'x':mask_a.x})
        y = xr.DataArray(data=Y, dims=['y','x'], coords={'y':mask_a.y,'x':mask_a.x})
        
        # stacking into single dimension
        stackkws = {'all_points':['x','y']}
        x_ = x.stack(**stackkws)
        y_ = y.stack(**stackkws)
        mask_a_ = mask_a.stack(**stackkws)
        mask_b_ = mask_b.stack(**stackkws)

        # masking both x,y by both masks to reduce computational load
        ma_x = x_.where(mask_a_).dropna(dim='all_points')
        ma_y = y_.where(mask_a_).dropna(dim='all_points')
        mb_x = x_.where(mask_b_).dropna(dim='all_points')
        mb_y = y_.where(mask_b_).dropna(dim='all_points')
        index = pd.MultiIndex.from_tuples(list(zip(*[ma_y.values,ma_x.values])),names=['y','x'])
        Na, Nb = len(ma_x.values), len(mb_x.values)
        # to indicate cost savings
        print(f'number of points in mask_a: {Na:6d} ;\
                percentage of total array points: {Na/len(x_)*100:5.2f} %')
        print(f'number of points in mask_b: {Nb:6d} ;\
                percentage of total array points: {Nb/len(x_)*100:5.2f} %')

        # calculate euclidean distance and find minimum                   
        dist = np.min(np.sqrt((np.tile(ma_x.values,(Nb,1)) - np.tile(mb_x.values.reshape(-1,1),Na))**2 + 
                              (np.tile(ma_y.values,(Nb,1)) - np.tile(mb_y.values.reshape(-1,1),Na))**2), axis=0)
        s = pd.Series(dist, index=index)
        return xr.DataArray.from_series(s)
    
    def find_distances(self):
        """ calculate minimum distances to ice shelf front / grounding line """
        self.ds['dgrl'] = self.distance_to_line(self.ds.mask, self.ds.grl)
        self.ds['disf'] = self.distance_to_line(self.ds.mask, self.ds.isf)
        return
    
    def define_boxes(self):
        """ boxes based on total and relative distance to grl and isf
        output:
        ds.rd   ..        relative distance field
        ds.box  .. [int]  mask number field
        """
        self.ds['rd'] = self.ds.dgrl/(self.ds.dgrl+self.ds.disf)
        self.ds['box'] = xr.DataArray(data=np.zeros(np.shape(self.ds.mask)), name='box',
                           dims=['y','x'], coords=self.ds.coords)
        for k in np.arange(1,self.n+1):
            lb = self.ds.mask.where(self.ds.rd>=1-np.sqrt((self.n-k+1)/self.n))
            ub = self.ds.mask.where(self.ds.rd<=1-np.sqrt((self.n-k)/self.n))
            self.ds.box.values += xr.where(ub*lb==1, k*self.ds.mask, 0).values
        return
    
    def calc_area(self):
        """ calculate area of box k
        assumes regularly space coordinates named x and y
        """
        dx = self.ds.x[1]-self.ds.x[0]
        dy = self.ds.y[1]-self.ds.y[0]
        self.ds['area'] = dx*dy*self.ds.mask
        for k in np.arange(1,self.n+1):
            self.A[k] = self.ds.area.where(self.ds.box==k).sum(['x','y'])  
        return
    
    # physics
    def T_s(self, k):
        """ spatially explicit T^\star(x,y) temperature defined just above (A6) """
        return self.a*self.S[k-1]+self.b-self.c*self.pk[k]-self.T[k-1]
    
    def solve_B_1(self):
        """ solve for the T1, S1, m1, circulation q, appendix A1 """
        # T1, S1
        s = self.S[0]/self.nulambda
        Crbsa = self.C*self.rho0*(self.beta*s-self.alpha)
        x = -self.g1[1]/(2*Crbsa) + np.sqrt((self.g1[1]/2/Crbsa)**2-(self.g1[1]*self.T_s(1))/(Crbsa))
        assert x>0  # argued at the end of appendix A1
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
        
    def compute(self):
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
