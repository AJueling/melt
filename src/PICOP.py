import numpy as np
import xarray as xr
import pandas as pd

from PICO import PicoModel
from Plume import PlumeModel
from real_geometry import RealGeometry, path, glaciers
from ideal_geometry import IdealGeometry, cases


class PicoPlumeModel(PicoModel, PlumeModel, RealGeometry):
    """ PICOP model by Pelle et al. (2019)
        combines PICO model (Reese et al. 2018) and plume model (Lazeroms et al. 2018)
    """
    def __init__(self, ds):
        """ create geometry """
        PicoModel.__init__(self, ds=ds)

        # self.compute_pico()
        # also initiates ModelConstants
        # loads 

        # if n is None:  n = self.find('n')
        # if ds is None:  # load PICOP geometry file    
        #     RealGeometry.__init__(self, name=name, n=n)
        #     self.ds_geometry = self.PICOP_geometry().drop(['mapping', 'spatial_ref'])
        # else:
        #     assert name=='test'
        #     self.ds_geometry = ds
        # if Ta is None:  Ta = self.find('Ta')
        # assert Ta>-3 and Ta<10
        # if Sa is None:  Sa = self.find('Sa')
        # assert Sa>0 and Sa<50
        # self.n = n
        # del self.ds
        # self.fn_PICOP_output = f'{path}/results/PICOP/{name}_n{n}_Ta{Ta}_Sa{Sa}.nc'

        # self.melt = xr.zeros_like(self.ds_geometry.draft)  # melt(x,y)
        # self.melt.name = 'melt'

        return
    
    def integrate_melt(self):
        """ calculate melt rate integral """
        return
        
    def calc_nondim_locations(self):
        """ """
        return self.ds

    def compute_picop(self):
        """ compute melt rates of the PICOP model """
        # PICO model provides box temperature and salinity
        self.ds_geo, self.ds_PICO = self.compute_pico()
        dp = xr.merge([self.ds_geo, self.ds_PICO])
        # print(dp)
        for i in range(1,len(dp.boxnr)):
            # replace ambient conditions with PICO results
            dp['Ta'] = xr.where(dp.box==i, dp.Tk[i], dp.Ta)
            dp['Sa'] = xr.where(dp.box==i, dp.Sk[i], dp.Sa)
        # print(dp)
        PlumeModel.__init__(self, dp=dp)
        self.ds = self.compute_plume()
        return self.ds_geo, self.ds_PICO, self.ds