import numpy as np
import xarray as xr
import pandas as pd

from PICO import PicoModel
from Plume import PlumeModel
from geometry import ModelGeometry


class PicoPlumeModel(PicoModel, PlumeModel, ModelGeometry):
    """ PICOP model by Pelle et al. (2019)
        combines PICO model (Reese et al. 2018) and plume model (Lazeroms et al. 2018)
    """
    def __init__(self, name, Ta, Sa, n=None):
        """ create geometry """
        ModelGeometry.PICOP(self, name=name, n=n)
        PicoModel.__init__(self, name=name, Ta=Ta, Sa=Sa, n=n)
        return
    
    def integrate_melt(self):
        """ calculate melt rate integral """
        return
        
    def compute(self):
        """ compute melt rates of the PICOP model """
        # PICO model provides box temperature and salinity
        self.compute_pico()
#         self.calc_melt_rate()
        PlumeModel.__init__(self, X=np.zeros((10)), zb=np.zeros((10)), Ta=Ta, Sa=Sa)
#         self.integrate_melt()
        print(self.l2)
        return
#         return self.ds