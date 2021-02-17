import os
import sys
import numpy as np
import xesmf as xe
import pyproj
import xarray as xr
import pandas as pd
import warnings
import rioxarray
import geopandas
import matplotlib
import matplotlib.pyplot as plt

from advect import advect_grl
from constants import ModelConstants
from scipy.ndimage import gaussian_filter
from shapely.geometry import mapping
from tqdm.autonotebook import tqdm

# to suppress xarray's "Mean of empty slice" warning
warnings.filterwarnings("ignore", category=RuntimeWarning)

if sys.platform=='darwin':  # my macbook
    path = '/Users/Andre/git/melt'
elif sys.platform=='linux':  # cartesius
    path = '/home/ajueling/melt'

fn_BedMachine = f'{path}/data/BedMachine/BedMachineAntarctica_2020-07-15_v02.nc'
fn_IceVelocity = f'{path}/data/IceVelocity/antarctic_ice_vel_phase_map_v01.nc'

glaciers = ['Amery',
            'Totten',
            'MoscowUniversity',
            'Ross',
            'Dotson', 
            'Thwaites',
            'PineIsland',
            'FilchnerRonne',
           ]
coarse_resolution = ['Ross', 'FilchnerRonne']
# total advection time  [yrs]
T_adv = {'Totten'          : 400,
         'MoscowUniversity': 400,
         'Thwaites'        : 200, 
        }
# ice shelves not listes in PICO publication
# n from Fig. 3, Ta/Sa from Fig. 2;                             # drainage basin
noPICO = {'MoscowUniversity': {'n':2, 'Ta':-0.73, 'Sa':34.73},  #  8
          'Dotson'          : {'n':2, 'Ta':+0.47, 'Sa':34.73},  # 14
         }
# Table 2 of Reese et al. (2018)
table2 = pd.read_csv(f'{path}/doc/Reese2018/Table2.csv', index_col=0)


class RealGeometry(ModelConstants):
    """ create geometry files of realistic ice shelves based on BedMachine2 data
    
        output:
        ds  (xr.Dataset)             containing:
            draft    (x,y)    [m]    vertical position of ice shelf base  (<=0)
            p        (x,y)    [Pa]   hydrostatic pressure
            # mask     (x,y)    [int]  [0] ocean, [1] grounded ice, [3] ice shelf; like BedMachine dataset
            mask     (x,y) *  [bool] mask of ice shelf in domain 
            grl      (x,y) *  [bool] grounding line mask
            isf      (x,y) *  [bool] ice shelf front mask
            dgrl     (x,y)    [m]    distance to grounding line, needed for Plume and PICOP  !!! not yet implemented
            dglm     (x,y) *  [m]    distance to grounding line, needed for PICO/P boxes
            disf     (x,y) *  [m]    minimum distance to isf
            rd       (x,y) *  [0,1]  relative distance
            alpha    (x,y)    [rad]  local angle, needed for Plume and PICOP
            grl_adv  (x,y)    [m]    advected grounding line depth, needed for Plume and PICOP
            box      (x,y)    [int]  box number [1,n] for PICO and PICOP models
            area_k   (boxnr)  [m^2]  area per box

            (*) intermediate results only, not needed for any of the models
                        . mask   .. 
            . 
            . 
            . box    .. (int)    box number [1,...,n]
            . area   .. (float)  area of each box [m^2]
    """
    def __init__(self, name, n=None):
        """
        input:
        name .. name of ice shelf
        n    .. number of boxes in PICO model

        output:

        """
        ModelConstants.__init__(self)
        assert name in glaciers
        self.name = name
        if n is None:  self.n = RealGeometry.find(self.name, 'n')
        else:          self.n = n
        self.fn     = f'{path}/results/geometry/{name}_n{self.n}_geometry.nc'
        self.fn_evo = f'{path}/results/advection/{name}_evo.nc'
        self.fn_isf = f'{path}/data/mask_polygons/{name}_isf.geojson'
        self.fn_grl = f'{path}/data/mask_polygons/{name}_grl.geojson'
        self.fn_outline = f'{path}/data/mask_polygons/{name}_polygon.geojson'
        for fn in [self.fn_outline, self.fn_grl, self.fn_isf]:
            assert os.path.exists(fn), f'file does not exists:  {fn}'
        return
    
    def select_geometry(self):
        """ selects the appropriate domain of BedMachine dataset 
        the polygon stored in the .geojson was created in QGIS
        large ice shelves: grid sopacing is decreased to 2.5 km
        """
        ds = xr.open_dataset(fn_BedMachine)
        ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y')
        ds = ds.rio.write_crs('epsg:3031')
        poly = geopandas.read_file(self.fn_outline, crs='espg:3031')
        clipped = ds.mask.rio.clip(poly.geometry.apply(mapping), poly.crs, drop=True)
        self.ds = ds.where(clipped)
        #
        if self.name in coarse_resolution:
            self.ds = self.ds.coarsen(x=5,y=5).mean()
        return

    @staticmethod
    def find(name, q):
        """ find quantitity `q` either from PICO publication or dict above """
        if q=='n':     nPn, dfn = 'n', 'bn'
        elif q=='Ta':  nPn, dfn = 'Ta', 'T0'
        elif q=='Sa':  nPn, dfn = 'Sa', 'S0'
        else:          raise ValueError('argument `q` needs to be `n`, `Ta` or `Sa`')
        if name in noPICO:
            Q = noPICO[name][nPn]
        else:
            Q = table2[dfn].loc[name]
        if q=='n':  Q = int(Q)
        else:       Q = float(Q)
        return Q

    def calc_draft(self):
        """  add (draft=surface-thickness) to self.ds """
        self.ds['draft'] = (self.ds.surface-self.ds.thickness).where(self.ds.mask==3)
        self.ds.draft.attrs = {'long_name':'depth of ice shelf-ocean interface', 'units':'m'}
        return

    def calc_p(self):
        self.ds['p'] = abs(self.ds.draft)*self.rho0*self.g  # assuming constant density
        self.ds.p.attrs = {'long_name':'hydrostatic pressure', 'units':'Pa'}  
        return

    def determine_grl_isf(self):
        """ add ice shelf, grounding line, and ice shelf masks to self.ds """
        mask = self.ds.mask
        mask = mask.fillna(0)

        def find_grl_isf(line):
            """ finds mask boundaries and selects points within polygon in geojson files """
            assert line in ['grl', 'isf']
            if line=='grl':    diff, fn = 1, self.fn_grl
            elif line=='isf':  diff, fn = 3, self.fn_isf
            poly = geopandas.read_file(fn, crs='espg:3031')
            new = xr.where(mask-mask.shift(x= 1)==diff, mask, 0) + \
                  xr.where(mask-mask.shift(x=-1)==diff, mask, 0) + \
                  xr.where(mask-mask.shift(y= 1)==diff, mask, 0) + \
                  xr.where(mask-mask.shift(y=-1)==diff, mask, 0)
            new = new.where(mask==3)/new
            new.name = line
            new = new.rio.set_spatial_dims(x_dim='x', y_dim='y')
            new = new.rio.write_crs('epsg:3031')
            new = new.rio.clip(poly.geometry.apply(mapping), poly.crs, drop=False)
            new = new.fillna(0)
            return new

        grl  = find_grl_isf('grl')
        grl.attrs = {'long_name':'grounding line mask'}
        isf  = find_grl_isf('isf')
        isf.attrs = {'long_name':'ice shelf front mask'}
        # now new `mask`: ice shelf = 1, rest = 0
        mask = xr.where(self.ds['mask']==3, 1, 0)
        self.ds = self.ds.rename({'mask':'mask_orig'})
        # self.ds = self.ds.drop('mask')
        self.ds = xr.merge([self.ds, mask, grl, isf])
        return

    @staticmethod
    def distance_to_line(mask_a, mask_b):
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
    
    @staticmethod
    def remove_points(da, line, direction='below'):
        """ removing points in a direction from line
        input:
        da         .. (DataArray)       boolean values
        line       .. (lists of tuples) x, y coordinates of line
        direction  .. (str)             ['below', 'above']

        output:
        da_        .. (DataArray)       copy of da with points south of line removed
        """
        assert direction in ['below','above']
        x_, y_ = [], []
        for (x,y) in line:
            x_.append(x)
            y_.append(y)
        da_ = da.copy()
        for i in range(len(x_)-1):
            x1, y1 = x_[i]  , y_[i]
            x2, y2 = x_[i+1], y_[i+1]
            a = (y2-y1)/(x2-x1)
            if a!=0:    b = ((y1+y2)-(x1+x2)*a)/2
            elif a==0:  b = y1
            x_seg = da.x.where(da.x>x1).where(da.x<=x2)
            for j, x in enumerate(x_seg.values):
                if np.isnan(x)==False:
                    y = a*x+b
                    if direction=='below':
                        da_[:,j] = da_.where(da_.y>y).values[:,j]
                    elif direction=='above':
                        da_[:,j] = da_.where(da_.y<y).values[:,j]
        return da_
    
    def find_distances(self):
        """ calculate minimum distances to ice shelf front / grounding line
        
        output:
        dglm  .. min distance to grounding line
        disf  .. min distance to ice shelf front
        rd    .. relative distance from grounding line
        """
        self.ds['dglm'] = RealGeometry.distance_to_line(self.ds.mask, self.ds.grl)
        self.ds['dgrl'] = self.ds.dglm.copy()
        self.ds.dglm.attrs = {'long_name':'minimum distance to grounding line', 'units':'m'}
        self.ds.dgrl.attrs = {'long_name':'minimum distance to grounding line', 'units':'m'}
        self.ds['disf'] = RealGeometry.distance_to_line(self.ds.mask, self.ds.isf)
        self.ds.disf.attrs = {'long_name':'minimum distance to ice shelf front', 'units':'m'}
        self.ds['rd'] = self.ds.dglm/(self.ds.dglm+self.ds.disf)
        self.ds.rd.attrs = {'long_name':'dimensionless relative distance'}
        return

    def add_latlon(self):
        """ add `lat` and `lon` coordinate to `self.ds` """
        project = pyproj.Proj("epsg:3031")
        xx, yy = np.meshgrid(self.ds.x, self.ds.y)
        lons, lats = project(xx, yy, inverse=True)
        dims = ['y','x']
        self.ds = self.ds.assign_coords({'lat':(dims,lats), 'lon':(dims,lons)})
        return

    def define_boxes(self):
        """ boxes based on total and relative distance to grl and isf
        output:
        ds.box  .. [int]  mask number field (x,y)
        """
        self.ds['box'] = xr.DataArray(data=np.zeros(np.shape(self.ds.mask)),
                                      name='box',
                                      dims=['y','x'],
                                      coords=self.ds.coords)
        for k in np.arange(1,self.n+1):
            lb = self.ds.mask.where(self.ds.rd>=1-np.sqrt((self.n-k+1)/self.n))
            ub = self.ds.mask.where(self.ds.rd<=1-np.sqrt((self.n-k)/self.n))
            self.ds.box.values += xr.where(ub*lb==1, k*self.ds.mask, 0).values
        return
    
    def calc_area(self):
        """ calculate area of box k
        introduce boxnr coordinate
        assumes regularly space coordinates named x and y
        """
        dx = abs(self.ds.x[1]-self.ds.x[0])
        dy = abs(self.ds.y[1]-self.ds.y[0])
        self.ds['area'] = dx*dy*self.ds.mask
        A = np.zeros((self.n+1))
        A[0] = (self.ds.mask*self.ds.area).sum(['x','y'])
        for k in np.arange(1,self.n+1):
            A[k] = self.ds.area.where(self.ds.box==k).sum(['x','y'])  
        # assert A[0]==np.sum(A[1:]), f'{A[0]=} should be{np.sum(A[1:])=}'
        self.ds['area_k'] = xr.DataArray(data=A, dims='boxnr', coords={'boxnr':np.arange(self.n+1)})
        self.ds.area_k.attrs = {'long_name':'area per box', 'units':'m^2'}
        return

    def interpolate_velocity(self):
        """ interpolates 450 m spaced velocity onto geometry (500 m spaced) grid 
        add new velocity fields (`u` and `v`) to dataset `self.ds`
        interpolation
        """
        xlim = slice(self.ds.x[0],self.ds.x[-1])
        ylim = slice(self.ds.y[0],self.ds.y[-1])
        vel = xr.open_dataset(fn_IceVelocity)
        vel = vel.sel({'x':xlim, 'y':ylim})

        # create new lat/lon coords for dataset `ds` to enable regridding
        project = pyproj.Proj("epsg:3031")
        xx, yy = np.meshgrid(self.ds.x, self.ds.y)
        lons, lats = project(xx, yy, inverse=True)
        dims = ['y','x']
        self.ds = self.ds.assign_coords({'lat':(dims,lats), 'lon':(dims,lons)})

        regridder = xe.Regridder(vel, self.ds, 'bilinear')#, reuse_weights=True)
        u = regridder(vel.VX)
        v = regridder(vel.VY)
        u.name = 'u'
        u.attrs = {'long_name':'velocity in x-direction', 'units':'/yr'}
        v.name = 'v'
        u.attrs = {'long_name':'velocity in y-direction', 'units':'m/yr'}
        self.ds = xr.merge([self.ds, u, v])
        return

    def extend_velocity(self):
        """ extends velocity field by diffusion to areas where there are no observations """
        # exists = (xr.where(self.ds.u**2+self.ds.v**2>0,1,0)*xr.where(np.isnan(self.ds.draft),0,1))
        missing = (xr.where(np.isnan(self.ds.u**2+self.ds.v**2),0,1)*xr.where(np.isnan(self.ds.draft),0,1))
        if missing.sum().values>0:
            print('need to extend velocity')
            u_new = self.ds.u+missing
            v_new = self.ds.v+missing
            for t in tqdm(range(2000)):
                u_new = gaussian_filter(u_new, sigma=1, mode='reflect')
                v_new = gaussian_filter(v_new, sigma=1, mode='reflect')
                u_new = xr.where(missing==0, self.ds.u, u_new)
                v_new = xr.where(missing==0, self.ds.v, v_new)
            self.ds['u'] = u_new.where(np.isfinite(self.ds.draft))
            self.ds['v'] = v_new.where(np.isfinite(self.ds.draft))
        else:
            print('no need to expand velosity')
        return

    def adv_grl(self):
        """ solves advection-diffusion equation as described in Pelle et al. (2019)
        uses `advect_grl` frunction from `advect.py`
        
        """
        ds = self.ds
        ds['u'] = ds.u.fillna(0)
        ds['v'] = ds.v.fillna(0)
        if self.name in T_adv:  T = T_adv[self.name]
        else:                   T = 500
        kw_isel = dict(x=slice(1,-1), y=slice(1,-1))  # remove padding
        evo = advect_grl(ds=ds, eps=1/50, T=T, plots=False).isel(**kw_isel)
        # evo.to_netcdf(self.fn_evo)
        self.ds['grl_adv'] = evo.isel(time=-1).drop('time')
        self.ds.attrs = {'long_name':'advected groundling line depth', 'units':'m'}
        return

    @staticmethod
    def smoothen(da):
        """ smoothens any DataArray wtih Gaussian filter of sigma=2(dx)
        inspired by https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
        """
        U = xr.where(np.isnan(da),0.,da)
        V = xr.where(np.isnan(da),0.,1.)
        kw = dict(sigma=2, mode='reflect')
        Z = xr.DataArray(data=gaussian_filter(U, **kw)/gaussian_filter(V, **kw), dims=da.dims, coords=da.coords)
        return Z  # xr.where(np.isnan(da),np.nan,Z)

    def calc_angles(self):
        """ local slope angles (in radians)
        based on (smoothed) draft field `D` and velocity flowlines `F`

        n1    .. vector perpendicular to scalar draft field
        n2    .. horizontal flowline vector field `F`
        n3    .. vertical unit vector = [0,0,1]
        alpha .. slope angle with respect to flowlines, i.e. b/w n1 and n3
        beta  .. maximum slope angle, i.e. between n1 and n2,
                 is also the angle between the plane and the horizontal
        
        using the four points one grid point away in the x and y directions
        calculate the normal vector to the two x ad y-slopes `n1 = xslope x yslope`
        `n_1 = [-2*dy*(a_{i+1}-a_{i-1}), -2*dx*(a_{j+1}-a_{j-1}), 4*dx*dy]`
        where `a` is the draft, `dx` and `dy` are the grid spacings

        the flow vectors are perpendicular to the gradient of flowline field A
        `\nabla F = [dF/dx, dF/dy]` so that `n2 = [-dFdy, dFdx, 0]` 
        this garantuees orthogonality `\nabla A \cdot n2 = 0`

        the inner product of vectors `a` and `b` contains the angle `gamma`
        `a \cdot b = |a| |b| \cos \gamma`
        `\gamma = \arccos ( \frac{a \cdot b}{|a|*|b|} )`
        """
        if self.name in coarse_resolution:
            D = self.ds.draft
        else:  # smoothing `draft` with a rolling mean first
            # D = (self.ds.draft.rolling(x=5).mean(skipna=True)+self.ds.draft.rolling(y=5).mean(skipna=True))/2
            D = RealGeometry.smoothen(self.ds.draft)
        dx, dy = D.x[1]-D.x[0], D.y[1]-D.y[0]
        print(dx, dy)
        dxdy = abs((D.y-D.y.shift(y=1))*(D.x-D.x.shift(x=1)))
        ip, im = D.shift(x=-1), D.shift(x= 1)
        jp, jm = D.shift(y=-1), D.shift(y= 1)
        n1 = np.array([-2*dy*(ip-im), -2*dx*(jp-jm), 4*dxdy])
        n1_norm = np.linalg.norm(n1, axis=0)
        beta = np.arccos(4*dxdy/n1_norm)  # n3 already normalized
        self.ds['beta']  = xr.where(np.isnan(self.ds.draft), np.nan, beta)
        self.ds.attrs = {'long_name':'largest angle with horizontal', 'units':'rad'}

        F = self.ds['grl_adv']
        ip, im = F.shift(x=-1), F.shift(x= 1)
        jp, jm = F.shift(y=-1), F.shift(y= 1)
        dFdx = (ip-im)/2/dx
        dFdy = (jp-jm)/2/dy

        # selecting where centered difference results in NaN but one-sided difference does not
        mask = xr.where(np.isnan(F),0,1)
        mask_xp = xr.where(np.isnan(ip),1,0)*mask
        dFdx = xr.where(mask_xp, (F-im)/dx, dFdx)
        mask_xm = xr.where(np.isnan(im),1,0)*mask
        dFdx = xr.where(mask_xm, (ip-F)/dx, dFdx)

        mask_yp = xr.where(np.isnan(jp),1,0)*mask
        dFdy = xr.where(mask_yp, (F-jm)/dy, dFdy)
        mask_ym = xr.where(np.isnan(jm),1,0)*mask
        dFdy = xr.where(mask_ym, (jp-F)/dy, dFdy)

        n2 = np.array([-dFdy, dFdx, xr.zeros_like(dFdx)])
        n2_norm = np.linalg.norm(n2, axis=0)
        alpha = np.arcsin((-dFdy*n1[0]+dFdx*n1[1])/n1_norm/n2_norm)
        alpha = abs(alpha)
        alpha = xr.where(xr.where(np.isnan(alpha),1,0)*xr.where(np.isnan(self.ds.draft),0,1), 0, alpha)
        self.ds['alpha'] = xr.where(np.isnan(self.ds.draft), np.nan, alpha)
        self.ds.attrs = {'long_name':'angle along streamlines', 'units':'rad'}
        return

    def create(self, new=False):
        """ creates geometry Dataset for all models (see class docstring for output details) """
        if os.path.exists(self.fn) and new==False:
            print(f'\n--- load geometry file: {self.name} ---')
            self.ds = xr.open_dataset(self.fn)
        else:
            print(f'\n--- generate geometry {self.name} n={self.n} ---')
            self.select_geometry()  # creates self.ds from BedMachine data
            self.calc_draft()
            self.calc_p()
            self.determine_grl_isf()
            self.find_distances()
            self.add_latlon()
            self.define_boxes()
            self.calc_area()
            self.interpolate_velocity()
            self.extend_velocity()
            self.adv_grl()
            self.calc_angles()

            # needs dgrl for PICOP
            self.ds['n'] = self.n
            self.ds.n.attrs = {'long_name':'box number; 0 is ambient'}
            self.ds['name_geo'] = f'{self.name}_{self.n}'
            self.ds = self.ds.drop(['mapping', 'spatial_ref'])
            self.ds.to_netcdf(self.fn) # .drop(['mapping', 'spatial_ref'])
        return self.ds

    """ plotting """

    def plot_PICO(self):
        """ plots all PICOP fields """
        if os.path.exists(self.fn):
            ds = xr.open_dataset(self.fn)
        else:
            ds = self.create()
        kwargs = {'cbar_kwargs':{'orientation':'horizontal'}}
        f, ax = plt.subplots(1, 5, figsize=(15,5), constrained_layout=True, sharey=True, sharex=True)
        ds.draft.name = 'draft [meters]'
        ds.draft.plot(ax=ax[0], cmap='plasma', **kwargs, vmax=0)
        ds.grl.where(ds.grl>0).plot(ax=ax[0], cmap='Blues', add_colorbar=False)
        ds.isf.where(ds.isf>0).plot(ax=ax[0], cmap='Reds' , add_colorbar=False)
        ds.disf.name = 'min. to ice shelf front [km]'
        (ds.disf/1e3).plot(ax=ax[1], **kwargs)
        ds.dglm.name = 'min. to grounding line [km]'
        (ds.dglm/1e3).plot(ax=ax[2], **kwargs)
        ds.rd.name = 'relative distance'
        (ds.rd).plot(ax=ax[3], **kwargs)
        ds.box.name = 'box nr.'
        (ds.box).plot(ax=ax[4], **kwargs)
        f.suptitle(f'{self.name} Ice Shelf', fontsize=16)
        return

    def plot_PICOP(self):
        """ plots important PICO fields + interpolated velocity """
        if os.path.exists(self.fn):
            ds = xr.open_dataset(self.fn)
        else:
            ds = self.create()
        
        f, ax = plt.subplots(1, 4, figsize=(12,5), constrained_layout=True, sharey=True)
        kwargs = {'cbar_kwargs':{'orientation':'horizontal'}}

        # draft 
        ds.draft.name = 'draft [meters]'
        ds.draft.plot(ax=ax[0], cmap='plasma', **kwargs, vmax=0)
        ds.grl.where(ds.grl>0).plot(ax=ax[0], cmap='Blues', add_colorbar=False)
        ds.isf.where(ds.isf>0).plot(ax=ax[0], cmap='Reds' , add_colorbar=False)

        # velocity stream
        vel = np.sqrt(ds.u**2+ds.v**2)
        vel.name = 'velocity [m/yr]'
        xx, yy = np.meshgrid(ds.x, ds.y)
        vel.plot(ax=ax[1], **kwargs)
        ax[1].streamplot(xx, yy, ds.u, ds.v, color='w', linewidth=vel.fillna(0).values/5e2)

        # alpha
        ds.alpha.name = 'local angle [$^\circ$]'
        ds.alpha.plot(ax=ax[2], **kwargs)

        # advected grounding line
        ds.grl_adv.name = 'advected grl depth'
        ds.grl_adv.plot(ax=ax[3], **kwargs)

        f.suptitle(f'{self.name} Ice Shelf', fontsize=16)
        return


if __name__=='__main__':
    """ calculate geometries for individual or all glaciers
    called as `python geometry.py new {glacier_name}`
    """
    new = False  # skip calculations if files exist; this is the default
    if len(sys.argv)>1:
        if sys.argv[1]=='new':
            new = True   # overwrite existing files

    if len(sys.argv)>2:  # if glacier is named, only calculate geometry for this one
        glacier = sys.argv[2]
        assert glacier in glaciers, f'input {glacier} not recognized, must be in {glaciers}'
        RealGeometry(name=glacier).create(new=new)
    else:  # calculate geometry for all glaciers
        for i, glacier in enumerate(glaciers):
            if i in [0,3,7]:  continue
            RealGeometry(name=glacier).create(new=new)