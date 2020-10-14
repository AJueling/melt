import sys
import numpy as np
import pandas as pd
import xarray as xr

from geometry import ModelGeometry, glaciers, path_results
from constants import ModelConstants

# ice shelves not listes in PICO publication
# n from Fig. 3, Ta/Sa from Fig. 2;                             # drainage basin
noPICO = {'Lambert'         : {'n':3, 'Ta':-1.73, 'Sa':34.70},  #  6
          'MoscowUniversity': {'n':2, 'Ta':-0.73, 'Sa':34.73},  #  8
          'Dotson'          : {'n':2, 'Ta':+0.47, 'Sa':34.73},  # 14
         }
# Table 2 of Reese et al. (2018)
df = pd.read_csv('../../doc/Reese2018/Table2.csv', index_col=0)

class PicoModel(ModelConstants, ModelGeometry):
    """ 2D melt model by Reese et al. (2018), doi: 10.5194/tc-12-1969-2018
        melt plume driven by the ice pump circulation
        assumes: small angle
        equation and table numbers refer to publication (doi: 10.1175/JPO-D-18-0131.1)
        
        caution:
        - x/y are used for the spatial dimensions/coordinates in the paper and for T/S differences in appendix 1
        - at the very end of appendix A1, there is a sign error in the equation for T_1
        
        input:
        name  ..  (str)     name of ice shelf (stored in geometry.py `glaciers` list)
        Ta    ..  (float)   ambient temperature in [degC]
        Sa    ..  (float)   ambient salinity in [psu]
        n     ..  (int)     number of boxes
        
        output:  [calling `.compute()`]
        ds  ..  xr.Dataset holding all quantities with their coordinates
            [from ModelGeometry]
            .  draft .. (float) ocean-ice interface depth in [m]
            .  mask  .. (bool)  ice shelf mask
            .  grl   .. (bool)  grounding line mask
            .  isf   .. (bool)  ice shelf front mask
    """
    
    def __init__(self, name, Ta=None, Sa=None, n=None, ds=None):
        """ initialize model class
        1. load constants from ModelConstant class
        2. load geometry from ModelGeometry class, or use custom geometry dataset `ds`
        3. create arrays holding (intermediate) data

        boxnr 0 rfers to ambient (temperature/salinity) or total (area/melt)
        """
        assert name in glaciers or name=='test'
        assert type(Ta) is float or Ta is None
        assert type(Sa) is float or Sa is None
        assert type(n)==int and n>0 and n<10 or n is None
        ModelConstants.__init__(self)
        self.name = name
        if n is None:  n = self.find('n')
        if ds is None:
            ModelGeometry.__init__(self, name=name, n=n)
            self.ds = self.PICO().drop(['mapping', 'spatial_ref'])
        else:
            assert name=='test'
            self.ds = ds
        if Ta is None:  Ta = self.find('Ta')
        assert Ta>-3 and Ta<10
        if Sa is None:  Sa = self.find('Sa')
        assert Sa>0 and Sa<50
        self.n = n
        self.fn_PICO_output = f'{path_results}/PICO/{name}_n{n}_Ta{Ta}_Sa{Sa}.nc'

        # hydrostatic pressure at each location(x,y)
        # assuming constant density
        self.p = abs(self.ds.draft)*self.rho0*self.g
        self.p.name = 'pressure'
        
        self.nulambda = self.rhoi/self.rhow*self.L/self.cp
        # intermediate constants for each box
        self.g1 = np.zeros((self.n+1))    # A_k*gamma^\star_T
        self.g2 = np.zeros((self.n+1))    # g1/nu/lambda
        self.pk = np.zeros((self.n+1))    # average pressure of box k
        
        for k in np.arange(1,n+1):
            self.g1[k] = abs(self.ds.area_k[k])*self.gammae  # defined just above (A6)
            self.g2[k] = self.g1[k]/self.nulambda
            self.pk[k] = self.p.where(self.ds.box==k).mean(['x','y'])
        
        self.M = xr.zeros_like(self.ds.draft)  # melt(x,y)
        self.M.name = 'melt'
        self.T = np.zeros((self.n+1))          # box avg ambient temp
        self.S = np.zeros((self.n+1))          # 
        self.m = np.zeros((self.n+1))
        self.T[0] = Ta
        self.S[0] = Sa
        return

    def find(self, q):
        """ find quantitity `q` either from PICO publication or dict above """
        if q=='n':     nPn, dfn = 'n', 'bn'
        elif q=='Ta':  nPn, dfn = 'Ta', 'T0'
        elif q=='Sa':  nPn, dfn = 'Sa', 'S0'
        else:          raise ValueError('argument `q` needs to be `n`. `Ta` or `Sa`')
        if self.name in noPICO:
            Q = noPICO[self.name][nPn]
        else:
            
            Q = df[dfn].loc[self.name]
        if q=='n':  Q = int(Q)
        else:       Q = float(Q)
        return Q
    
    def T_s(self, k):
        """ spatially explicit $T^\star(x,y)$ temperature defined just above (A6) """
        return self.a*self.S[k-1]+self.b-self.c*self.pk[k]-self.T[k-1]
    
    def solve_B_1(self):
        """ solve for the T1, S1, m1, circulation q, appendix A1 """
        # T1, S1
        s = self.S[0]/self.nulambda
        Crbsa = self.C*self.rho0*(self.beta*s-self.alpha)
        x = -self.g1[1]/(2*Crbsa) + np.sqrt((self.g1[1]/2/Crbsa)**2-(self.g1[1]*self.T_s(1))/(Crbsa))
        print(f'{x=}')
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
        1. solve melt for all grid cells in box k, each with pressure(x,y)
        2. sum to total melt in box k
        """
        # speatially explicit melt with T_k, S_k  ambient temperatures
        # also computes melt outside of box k, but that is masked out in the following step
        # melt at each location
        temp = self.a*self.S[k]+self.b-self.c*self.p-self.T[k]  # Eqn. (A5)
        mk = (-self.gammae/self.nulambda*temp).where(self.ds.box==k)
        mk *= 3600*24*365  # [m/s] -> [m/yr]
        self.m[k] = mk.mean()
        self.M += mk.fillna(0)
        print(self.m)
        return
        
    def compute_pico(self):
        """ """
        self.solve_B_1()
        if self.n>1:
            self.solve_B_k()
        self.m[0] = self.M.where(self.ds.mask).mean()
        kwargs = {'dims':'boxnr', 'coords':{'boxnr':np.arange(self.n+1)}}
        T = xr.DataArray(data=self.T, name='Tk', **kwargs)
        S = xr.DataArray(data=self.S, name='Sk', **kwargs)
        m = xr.DataArray(data=self.m, name='mk', **kwargs)
        q = xr.DataArray(data=self.q, name='q')
        print(self.m)
        
        ds = xr.merge([self.p, self.M, T, S, m, q])
        # ds.to_netcdf(self.fn_PICO_output)
        return self.ds, ds  # geometry dataset and PICO output dataset

    def compare_Reese(self):
        """ compares melt rates to the ones in original publication and obs. """
        


if __name__=='__main__':
    """ calculate PICO model
    called as `python PICO.py {glacier_name} {Ta} {Sa} {n}`
    """
    assert len(sys.argv) in [4,5]
    name = sys.argv[1]
    assert name in glaciers, f'input {glacier} not recognized, must be in {glaciers}'
    Ta = float(sys.argv[2])
    Sa = float(sys.argv[3])
    if len(sys.argv)==5:  n = int(sys.argv[4])
    else:                 n = 3
    PicoModel(name=name, Ta=Ta, Sa=Sa, n=n).compute_pico()