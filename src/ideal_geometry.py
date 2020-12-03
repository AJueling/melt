import numpy as np
import xarray as xr

from constants import ModelConstants

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
        mask    ..  [0] ocean, [1] grounded ice, [3] ice shelf; like BedMachine dataset
        u/v     ..  
        Ta/Sa   ..  constant, ambient temperature/salinity
        angle   ..  local angle, needed for Plume and PICOP
        grl_adv ..  advected grounding line depth, needed for Plume and PICOP
        box     ..  box number [1,n] for PICO and PICOP models
        area_k  ..  area per box
        """

        assert type(case)==int

        nx, ny = 30,30
        x = np.linspace(0,1e5,nx)
        y = np.linspace(0,1e5,ny)
        dx, dy = x[1]-x[0], y[1]-y[0]
        d_x, d_y = np.meshgrid(x, y)
        area = dx*dy  # area per grid cell

        ds = xr.Dataset({'mask':(['y','x'], 3*np.ones((ny,nx)))}, coords={'x':x, 'y':y})
        ds.mask[:,0] = 2
        ds.mask[0,:] = 2

        def define_boxes(dim, n):
            """ create DataArray with equally spaced integer box numbers [1,n+1] """
            assert dim in ['x', 'y']
            if dim=='x': other_dim = 'y'
            elif dim=='y': other_dim = 'x'
            da = xr.DataArray(data=np.arange(1,n+1,n/len(ds[dim])).astype(int), dims=dim, coords={dim:ds[dim]})
            da = da.expand_dims({other_dim:ds[other_dim]})
            if dim=='y':
                da = da.T
            return da

        def area_per_box(n):
            mask_ = xr.where(ds.mask==3, 1, 0)  # binary mask where floating ice
            area = dx*dy*mask_
            A = np.zeros((n+1))
            A[0] = (mask_*area).sum(['x','y'])
            for k in np.arange(1,n+1):
                A[k] = area.where(ds.box==k).sum(['x','y'])  
            da = xr.DataArray(data=A, dims='boxnr', coords={'boxnr':np.arange(n+1)})
            da.attrs = {'long_name':'area per box', 'units':'m^2'}
            return da

        if case==1:
            draft, _ = np.meshgrid(np.linspace(-1000,-500,nx), np.ones((ny)))
            alpha = np.arctan(np.gradient(draft, axis=1)/dx)
            grl_adv = np.full_like(draft, -1000)
            box_dim = 'x'
        elif case==2:
            _, draft = np.meshgrid( np.ones((nx)), np.linspace(-1000,-500,ny))
            alpha = np.arctan(np.gradient(draft, axis=0)/dy)
            grl_adv = np.full_like(draft, -1000)
            box_dim = 'y'
        elif case==3:
            xx, yy = np.meshgrid(np.linspace(1,0,nx), np.linspace(0,np.pi,ny))
            curv = 250
            draft = -500-((500-curv)+curv*np.sin(yy)**2)*xx
            alpha = np.arctan(np.gradient(draft, axis=1)/dx)
            grl_adv = grl_adv = np.tile(draft[:,0], (nx, 1)).T
            box_dim = 'x'
        
        if box_dim=='x':
            ds.mask[-1,:] = 2
            ds.mask[:,-1] = 0
            dgrl = d_x
        elif box_dim=='y':
            ds.mask[:,-1] = 2
            ds.mask[-1,:] = 0
            dgrl = d_y

        ds['box']     = define_boxes(dim=box_dim, n=n)
        ds['area_k']  = area_per_box(n=3)
        ds['draft']   = (['y','x'], draft              )
        ds['Ta']      = (['y','x'], Ta*np.ones((ny,nx)))
        ds['Sa']      = (['y','x'], Sa*np.ones((ny,nx)))
        ds['dgrl']    = (['y','x'], dgrl               )
        ds['alpha']   = (['y','x'], alpha              )
        ds['grl_adv'] = (['y','x'], grl_adv            )
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
    
def FavierTest(iceshelf,forcing,dx=2e3):
    dy=dx
    if iceshelf == 'fris':
        lx,ly = 7e5,7e5
        zdeep,zshallow = -1000,0
    elif iceshelf == 'totten':
        lx,ly = 1.5e5,3e4
        zdeep,zshallow = -2000,-200
    elif iceshelf == 'thwaites':
        lx,ly = 4e4,4e4
        zdeep,zshallow = -1000,-200
    elif iceshelf == 'test':
        lx,ly = 1e5,1e5
        zdeep,zshallow = -1000,-500
    elif iceshelf == 'test2':
        lx,ly = 1e5,6e4
        
    x = np.arange(0,lx,dx)
    y = np.arange(0,ly,dy)
    nx, ny = len(x), len(y)
    mask = 3*np.ones((ny,nx))
    mask[:,0] = 2
    if iceshelf == 'thwaites':
        mask[0,:] = 0
        mask[-1,:] = 0
    else:
        mask[0,:] = 2
        mask[-1,:] = 2
    mask[:,-1] = 0
    
    if iceshelf == 'test2':
        xx, yy = np.meshgrid(np.linspace(1,0,nx), np.linspace(0,np.pi,ny))
        curv = 250
        draft = -500-((500-curv)+curv*np.sin(yy)**2)*xx
    else:
        draft, _ = np.meshgrid(np.linspace(zdeep,zshallow,nx), np.ones((ny)))
    
    if forcing == 'cold0':
        z = [-5000,-700,-300,0]
        Tz = [-1.5,-1.5,-1.5,-1.5]
        Sz = [34,34,34,34]
    elif forcing == 'cold1':
        z = [-5000,-700,-300,0]
        Tz = [1.2,1.2,-0.6,-0.6]
        Sz = [34.5,34.5,34,34]
    elif forcing == 'warm0':
        z = [-5000,-700,-300,0]
        Tz = [1.2,1.2,-1.0,-1.0]
        Sz = [34.5,34.5,34,34]
    elif forcing == 'warm1':
        z = [-5000,-700,-300,0]
        Tz = [2.2,2.2,0,0]
        Sz = [34.5,34.5,34,34]        
    elif forcing == 'warm2':
        z = [-5000,-700,-300,0]
        Tz = [2.2,2.2,-1,-1]
        Sz = [34.5,34.5,34,34]         
    elif forcing == 'warm3':
        z = [-5000,-500,-100,0]
        Tz = [1.2,1.2,-1.0,-1.0]
        Sz = [34.5,34.5,34,34]        
        
    Ta = np.interp(draft,z,Tz)
    Sa = np.interp(draft,z,Sz) - 0.0002*draft
    
    ds = xr.Dataset({'mask':(['y','x'], mask)}, coords={'x':x, 'y':y})    
    ds['draft']   = (['y','x'], draft)
    ds['Ta']      = (['y','x'], Ta)
    ds['Sa']      = (['y','x'], Sa)
    
    return ds


class Forcing(ModelConstants):

    def __init__(self):
    
        ModelConstants.__init__(self)

        self.S0 = 34 #Reference surface salinity
        self.T0 = self.l1*self.S0+self.l2 #Surface freezing temperature

        self.z1 = 100 #Thermocline sharpness in m
        self.z  = np.arange(-5000,0,1)
    
    def create(self,ztcl,Tdeep,drhodz):
        
        self.Tz = Tdeep + (self.T0-Tdeep) * (1+np.tanh((self.z-ztcl)/self.z1))/2
        
        self.Sz = self.S0 + self.alpha*(self.Tz-self.T0)/self.beta - drhodz*self.z/(self.beta*self.rho0)
        
        z = xr.DataArray(self.z,name='z')
        Tz = xr.DataArray(self.Tz,name='Tz')
        Sz = xr.DataArray(self.Sz,name='Sz')

        ds = xr.merge([z,Tz,Sz])
        
        return ds

    
    