""" Simple Melt Models """
import numpy as np
import xarray as xr

from constants import ModelConstants
from real_geometry import RealGeometry, glaciers
from ideal_geometry import IdealGeometry, cases

class SimpleModels(ModelConstants):
    """ two simple quadratic melt models from Favier et al. (2019) (doi:10.5194/gmd-2019-26)
    
        input:
        name  ..  (str)         testcase / name of real iceshelf
        ds    ..  (xr.Dataset)  contains
              . draft  (x,y)
              . Ta     (x,y)
              . mask   (x,y)

        output:
        ds    ..  (xr.Dataset)  contains additionally
              . Ml     (x,y)    melt with local linear parametrization
              . Mq     (x,y)    melt with local quadratic parametrization
              . Mp     (x,y)    melt with non-local quadratic parametrization


    """

    def __init__(self, ds):
        ModelConstants.__init__(self)
        self.ds = ds
        for q in ['draft','Ta','Sa','p','Tf']:
            assert q in self.ds, f'{q} missing'
        return

    def compute(self):
        """ computes both local (Eq. 4) and non-local (Eq. 5) melt rates [m/yr] """
        pf = self.rhow*self.cp/self.rhoi/self.L
        dT = (self.ds.Ta-self.ds.Tf).where(self.ds.mask==3)
        self.ds['dT'] = dT
        self.ds['Ml'] = self.spy*self.gT_Ml*pf*dT
        self.ds['Mq'] = self.spy*self.gT_Mq*pf**2*dT**2
        self.ds['Mp'] = self.spy*self.gT_Mp*pf**2*dT*dT.mean(['x','y'])  # assumes equal grid spacing

        self.ds.dT.attrs = {'long_name':'thermal forcing', 'units':'degC'}
        self.ds.Ml.attrs = {'long_name':'local linear melt param.', 'units':'m/yr', 'from':'Eq. 2 of Favier19'}
        self.ds.Mq.attrs = {'long_name':'local quadratic melt param.', 'units':'m/yr', 'from':'Eq. 4 of Favier19'}
        self.ds.Mp.attrs = {'long_name':'non-local quadratic melt param.', 'units':'m/yr', 'from':'Eq. 5 of Favier19'}
        return self.ds