import os
import numpy as np
import xarray as xr

from constants import ModelConstants

cases = ['plumeref',  # reference 1D case for Plume model
         'plume1D',   # any 1D domain for plume model
         'plume2D',   # a quasi-1D 2D domain for plume model
         'test1',     # constant slope, constant u, v=0
         'test2',     # same as test 1 but in y direction
         'test3',     # sinusoidal grounding line
         'Ocean1',    # ISOMIP+ initial steady state position
         'Ocean2',    # ISOMIP+ retreated position
         ]


class IdealGeometry(ModelConstants):
    """ creating idealized geometries for comparing melt models; illustrated in `Geometry.ipynb`
    
        output:
        ds  (xr.Dataset)             containing:
            draft    (x,y)    [m]    vertical position of ice shelf base  (<=0)
            p        (x,y)    [Pa]   hydrostatic pressure
            mask     (x,y)    [int]  [0] ocean, [1] grounded ice, [3] ice shelf
                                     (like BedMachine dataset)
            dgrl     (x,y)    [m]    distance to grounding line, needed for PICO/P boxes
            alpha    (x,y)    [rad]  local angle, needed for Plume and PICOP
            grl_adv  (x,y)    [m]    advected grounding line depth, needed for Plume and PICOP
            box      (x,y)    [int]  box number [1,n] for PICO and PICOP models
            area_k   (boxnr)  [m^2]  area per box
    """
    def __init__(self, name, pdict=None):
        """ 
        input:
        name  .. (str) 
        pdict .. (dict)  parameter dictionary
        """
        ModelConstants.__init__(self)
        assert name in cases
        self.name = name
        self.pdict = pdict
        self.n = 5
        return

    def test_geometry(self, case):
        """ creates standard test case geometries
        format designed to be compatible with Bedmachine data
        
        input:   case   (int)
        output:  (see class docstring)
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

        ds['box']     = define_boxes(dim=box_dim, n=self.n)
        ds['area_k']  = area_per_box(n=self.n)
        ds['draft']   = (['y','x'], draft              )
        ds['dgrl']    = (['y','x'], dgrl               )
        ds['alpha']   = (['y','x'], alpha              )
        ds['grl_adv'] = (['y','x'], grl_adv            )
        self.ds = ds
        return self.ds

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
        ds['Ta'] = Ta
        ds['Sa'] = Sa
        ds['alpha'] = ('x', np.arctan(np.gradient(draft, x)))
        ds['grl_adv'] = ('x', np.full_like(draft, draft[0]))
        self.ds = ds
        return self.ds

    def isomip_geometry(self, case):
        """ generate geometry file from ISOMIP netcdfs """
        fn = f'../data/isomip/Ocean{case}_input_geom_v1.01.nc'
        if os.path.exists(fn):
            dsi = xr.open_dataset(fn)
        elif os.path.exists('../'+fn):  # if called from `src/ipynb` folder
            dsi = xr.open_dataset('../'+fn)
        else:
            print('ISOMIP `Ocean{case}` file does not exist')
        ds = xr.Dataset(coords={'x':('x',np.arange(416_000,641_000,500.)),
                        'y':('y',np.arange(250,80_000,500.)),
                        'boxnr':('boxnr',np.arange(self.n+1))},
                       )
        ds['draft'] = dsi.lowerSurface.interp_like(ds)
        ds['mask'] = (dsi.groundedMask + 3*dsi.floatingMask).interp_like(ds,method='nearest').astype(int)
        ds.mask[[0,-1],:] = ds.mask[[1,-2],:].values  # otherwise the lateral boundaries have nonsensical values
        dist = xr.DataArray(dims=('y','x'), coords={'x':ds.x, 'y':ds.y}, data=np.meshgrid(ds.x,ds.y)[0])
        ds['dgrl'] = dist - xr.where((ds.mask-ds.mask.shift(x=-1))==-2,dist,0).sum('x')
        ds['disf'] = abs(dist - xr.where((ds.mask-ds.mask.shift(x=-1))== 3,dist,0).sum('x'))
        ds['alpha'] = (('y','x'), np.gradient(ds.draft.values,500, axis=1))
        a = xr.where((ds.mask-ds.mask.shift(x=-1))==-2,ds.draft,0).sum('x')
        ds['grl_adv'] = (('y','x'), np.tile(a,len(ds.x)).reshape((len(ds.x),len(ds.y))).T)
        rd = ds.dgrl/(ds.dgrl+ds.disf)  # dimensionless relative distance
        ds['box'] = (('y','x'), np.zeros_like(ds.mask))
        ds['area_k'] = (('boxnr'), np.zeros(self.n+1))
        for k in np.arange(1,self.n+1):
            lb = xr.where(rd>=1-np.sqrt((self.n-k+1)/self.n),1,0)
            ub = xr.where(rd<=1-np.sqrt((self.n-k)  /self.n),1,0)
            ds.box.values += xr.where(ub*lb==1, k, 0).values
            ds.area_k[k] = (ds.x[1]-ds.x[0])*(ds.y[1]-ds.y[0]).where(ds.box==k).sum()
        ds.area_k[0] = ds.area_k[1:].sum()
        return ds

    def create(self):
        """ function to return geometry dataset """

        # geometry
        if self.name=='plumeref':
            N = 101
            self.ds = self.plume_1D_geometry(x=np.linspace(0,5e5,N), draft=np.linspace(-1000,0,N), Ta=-1.9, Sa=34.65)
        elif self.name=='plume1D':
            p = self.pdict
            for q in ['x', 'draft', 'Ta', 'Sa']:  assert q in p
            self.ds = self.plume_1D_geometry(x=p['x'], draft=p['draft'], Ta=p['Ta'], Sa=p['Sa'])
        elif self.name[:4]=='test':
            case = int(self.name[4:])
            self.ds = self.test_geometry(case=case)
        elif self.name[:5]=='Ocean':
            case = int(self.name[5:])
            self.ds = self.isomip_geometry(case=case)

        self.ds['p'] = abs(self.ds.draft)*self.rho0*self.g  # assuming constant density
        self.ds['n'] = self.n
        self.ds['name_geo'] = f'{self.name}_{self.n}'
        # metadata
        self.ds.dgrl.attrs    = {'long_name':'distance to grounding line (`X` in publication)', 'units':'m'}
        self.ds.draft.attrs   = {'long_name':'ice shelf base depth', 'units':'m'}
        self.ds.alpha.attrs   = {'long_name':'local slope angle along stream lines', 'units':'rad'}
        self.ds.grl_adv.attrs = {'long_name':'advected grounding line depth / plume origin depth', 'units':'m'}
        self.ds.p.attrs       = {'long_name':'hydrostatic pressure', 'units':'Pa'}  
        self.ds.n.attrs       = {'long_name':'box number; 0 is ambient'}
        self.ds.area_k.attrs  = {'long_name':'area per box', 'units':'m^2'}

        return self.ds
    
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
    
    return ds

