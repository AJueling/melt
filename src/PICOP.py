import numpy as np
import xarray as xr
import pandas as pd

from PICO import PicoModel
from Plume import PlumeModel
from geometry import ModelGeometry, path


class PicoPlumeModel(PicoModel, PlumeModel, ModelGeometry):
    """ PICOP model by Pelle et al. (2019)
        combines PICO model (Reese et al. 2018) and plume model (Lazeroms et al. 2018)
    """
    def __init__(self, name, Ta=None, Sa=None, n=None, ds=None):
        """ create geometry """
        assert name in glaciers or name=='test'
        assert type(Ta) is float or Ta is None
        assert type(Sa) is float or Sa is None
        assert type(n)==int and n>0 and n<10 or n is None

        PicoModel.__init__(self, name=name)  # also initiates ModelConstants
        if n is None:  n = self.find('n')
        if ds is None:  # load PICOP geometry file
            ModelGeometry.__init__(self, name=name, n=n)
            self.ds_geometry = self.PICOP_geometry().drop(['mapping', 'spatial_ref'])
        else:
            assert name=='test'
            self.ds_geometry = ds
        if Ta is None:  Ta = self.find('Ta')
        assert Ta>-3 and Ta<10
        if Sa is None:  Sa = self.find('Sa')
        assert Sa>0 and Sa<50
        self.n = n
        del self.ds
        self.fn_PICOP_output = f'{path}/results/PICOP/{name}_n{n}_Ta{Ta}_Sa{Sa}.nc'

        self.melt = xr.zeros_like(self.ds_geometry.draft)  # melt(x,y)
        self.melt.name = 'melt'

        return
    
    def integrate_melt(self):
        """ calculate melt rate integral """
        return
        
    def calc_nondim_locations(self):
        """ """
        return self.ds

    def compute(self):
        """ compute melt rates of the PICOP model """
        # PICO model provides box temperature and salinity
        self.ds_PICO = xr.open_dataset(self.fn_PICO_output)
        
        PlumeModel.__init__(self, dp=self.prepare_for_plume())
        self.ds = self.compute_plume()

        self.ds = 1
        return self.ds_geometry, self.ds_PICO, self.ds