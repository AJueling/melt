import numpy as np
import xarray as xr

cases = ['plumeref',  # reference 1D case for Plume model
         'plume1D',   # any 1D domain for plume model
         'plume2D',   # a quasi-1D 2D domain for plume model
         'test1',     # constant slope, constant u, v=0
         'test2',     # same as test 1 but in y direction
         'test3',     # sinusoidal grounding line
         ]


class IdealGeometry(object):
    """ creating idealized geometries for comparing melt models; illustrated in `Geometry.ipynb` """
    def __init__(self, name, pdict=None):
        """ 
        input:
        name  .. (str) 
        pdict .. (dict)  parameter dictionary
        """
        assert name in cases
        self.name = name
        self.pdict = pdict
        return

    def test_geometry(self, case, Ta=0, Sa=34, n=3):
        """ creates standard test case geometries
        format designed to be compatible with Bedmachine data
        
        input:   case (int)

        output:  ds (xr.Dataset) contains the following fields
        draft   ..  
        mask    ..  [0] grounding line, [1] ice shelf, [2] ice shelf front
        u/v     ..  
        Ta/Sa   ..  constant, ambient temperature/salinity
        angle   ..  local angle, needed for Plume and PICOP
        grl_adv ..  advected grounding line depth, needed for Plume and PICOP
        """

        assert type(case)==int

        nx, ny = 30,30
        mask = np.ones((ny,nx))
        x = np.linspace(0,1e5,nx)
        y = np.linspace(0,1e5,ny)
        dx, dy = x[1]-x[0], y[1]-y[0]
        d_x, d_y = np.meshgrid(x, y)

        if case==1:
            mask[:,0] = 0
            mask[-1,:] = 0
            mask[0,:] = 0
            mask[:,-1] = 2
            draft, _ = np.meshgrid(np.linspace(-1000,-500,nx), np.ones((ny)))
            dgrl = d_x
            alpha = np.arctan(np.gradient(draft, axis=1)/dx)
            grl_adv = np.full_like(draft, -1000)
        elif case==2:
            mask[:,0] = 0
            mask[0,:] = 0
            mask[:,-1] = 0
            mask[-1,:] = 2
            _, draft = np.meshgrid( np.ones((nx)), np.linspace(-1000,-500,ny))
            dgrl = d_y
            alpha = np.arctan(np.gradient(draft, axis=0)/dy)
            grl_adv = np.full_like(draft, -1000)
        elif case==3:
            mask[:,0] = 0
            mask[-1,:] = 0
            mask[0,:] = 0
            mask[:,-1] = 2
            xx, yy = np.meshgrid(np.linspace(1,0,nx), np.linspace(0,np.pi,ny))
            curv = 250
            draft = -500-((500-curv)+curv*np.sin(yy)**2)*xx
            dgrl = d_x
            alpha = np.arctan(np.gradient(draft, axis=1)/dx)
            grl_adv = grl_adv = np.tile(draft[:,0], (nx, 1)).T

        kwargs = {'dims':['y','x'], 'coords':{'x':x, 'y':y}}
        da0 = xr.DataArray(data=mask               , name='mask'   , **kwargs)
        da1 = xr.DataArray(data=draft              , name='draft'  , **kwargs)
        da2 = xr.DataArray(data=Ta*np.ones((ny,nx)), name='Ta'     , **kwargs)
        da3 = xr.DataArray(data=Sa*np.ones((ny,nx)), name='Sa'     , **kwargs)
        da4 = xr.DataArray(data=dgrl               , name='dgrl'   , **kwargs)
        da5 = xr.DataArray(data=alpha              , name='alpha'  , **kwargs)
        da6 = xr.DataArray(data=grl_adv            , name='grl_adv', **kwargs)
        ds = xr.merge([da0, da1, da2, da3, da4, da5, da6])
        return ds

    def plume_1D_geometry(self, x, draft, Ta, Sa):
        """ make 1D dataset that Plume model can work with
        
        input:
        x     .. (np.array)  dimensional position of points
        draft .. (np.array)  ice-ocean interface depth at points
        Ta/Sa .. (float)     ambient temperature/salinity
        """
        assert len(x)==len(draft)
        N = len(x)
        ds = xr.Dataset(coords={'x':x})
        ds['dgrl'] = ('x', x)
        ds['draft'] = ('x', draft)
        ds['Ta'] =  Ta
        ds['Sa'] = Sa
        ds['alpha'] = ('x', np.arctan(np.gradient(draft, x)))
        ds['grl_adv'] = ('x', np.full_like(draft, draft[0]))
        return ds

    def create(self):
        """ function to return geometry dataset """
        if self.name=='plumeref':
            N = 101
            ds = self.plume_1D_geometry(x=np.linspace(0,5e5,N), draft=np.linspace(-1000,0,N), Ta=-1.9, Sa=34.65)
        elif self.name=='plume1D':
            p = self.pdict
            for q in ['x', 'draft', 'Ta', 'Sa']:  assert q in p
            ds = self.plume_1D_geometry(x=p['x'], draft=p['draft'], Ta=p['Ta'], Sa=p['Sa'])
        elif self.name[:4]=='test':
            Ta, Sa = 0, 34
            if self.pdict is not None:
                if 'Ta' in self.pdict:   Ta = self.pdict['Ta']
                if 'Sa' in self.pdict:   Sa = self.pdict['Sa']
            case = int(self.name[4:])
            ds = self.test_geometry(case=case, Ta=Ta, Sa=Sa)

        # metadata
        ds.dgrl.attrs    = {'long_name':'distance to grounding line (`X` in publication)', 'units':'m'}
        ds.draft.attrs   = {'long_name':'ice shelf base depth', 'units':'m'}
        ds.Ta.attrs      = {'long_name':'ambient temperature', 'units':'degC'}
        ds.Sa.attrs      = {'long_name':'ambient salinity', 'units':'psu'}
        ds.alpha.attrs   = {'long_name':'local slope angle along stream lines', 'units':'rad'}
        ds.grl_adv.attrs = {'long_name':'advected grounding line depth / plume origin depth', 'units':'m'}
        return ds