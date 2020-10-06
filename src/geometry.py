import os
import dask
import numpy as np
import xesmf as xe
import pyproj
import xarray as xr
import pandas as pd
import rioxarray
import geopandas
import matplotlib
import matplotlib.pyplot as plt

from shapely.geometry import mapping

if sys.platform=='darwin':  # my macbook
    data = '/Users/Andre/git/melt/data'
    results = '/Users/Andre/git/melt/results'
elif sys.platform=='linux':  # cartesius
    data = '/home/ajueling/melt/data'
    results = '/home/ajueling/melt/results'

fn_BedMachine = f'{data}/BedMachine/BedMachineAntarctica_2020-07-15_v02.nc'
fn_IceVelocity = f'{data}/IceVelocity/antarctic_ice_vel_phase_map_v01.nc'
fn_IceVelocity_remapped = f'{data}/IceVelocity/IceVelocity_remapped_v01.nc'

glaciers = ['Lambert',
            'Totten',
            'MoscowUniversity',
            'Ross',
            'Dotson', 
            'Thwaites',
            'PineIsland',
            'FilchnerRonne',
           ]

class ModelGeometry(object):
    """ create geometry files for PICO and PICOP """
    def __init__(self, name, n=None):
        """
        input:
        name .. name of ice shelf
        n    .. number of boxes in PICO model
        """
        assert name in glaciers
        self.name = name
        if n is None:
            self.n = 3
        self.fn_PICO = f'{results}/PICO/{name}.nc'
        self.fn_PICOP = f'{results}/PICO/{name}.nc'
        self.fn_isf = f'{data}/mask_polygons/{name}_isf.geojson'
        self.fn_grl = f'{data}/mask_polygons/{name}_grl.geojson'
        self.fn_outline = f'{data}/mask_polygons/{name}_polygon.geojson'
        for fn in [self.fn_outline, self.fn_grl, self.fn_isf]:
            assert os.path.exists(fn), f'file does not exists:  {fn}'
        return
    
    ### PICO methods
    def select_geometry(self):
        """ selects the appropriate domain of BedMachine dataset 
        the polygon stored in the .geojson was created in QGIS
        """
        ds = xr.open_dataset(fn_BedMachine)
        ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y')
        ds = ds.rio.write_crs('epsg:3031')
        poly = geopandas.read_file(self.fn_outline, crs='espg:3031')
        self.clipped = ds.mask.rio.clip(poly.geometry.apply(mapping), poly.crs, drop=True)
        self.ds = ds.where(self.clipped)
        return

    def calc_draft(self):
        """  add draft ()=surface-thickness) to self.ds """
        draft = (self.ds.surface-self.ds.thickness).where(self.ds.mask==3)
        draft.name = 'draft'
        self.ds = xr.merge([self.ds, draft])
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
            new = new/new
            new.name = line
            new = new.rio.set_spatial_dims(x_dim='x', y_dim='y')
            new = new.rio.write_crs('epsg:3031')
            new = new.rio.clip(poly.geometry.apply(mapping), poly.crs, drop=False)
            new = new.fillna(0)
            return new

        grl  = find_grl_isf('grl')
        isf  = find_grl_isf('isf')
        # print(self.ds)
        # print(grl)
        # print(isf)
        # now new `mask`: ice shelf = 1, rest = 0
        mask = xr.where(self.ds['mask']==3, 1, 0)
        self.ds = self.ds.drop('mask')
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
        dgrl  .. min distance to grounding line
        disf  .. min distance to ice shelf front
        rd    .. relative disantce from grounding line
        """
        self.ds['dgrl'] = ModelGeometry.distance_to_line(self.ds.mask, self.ds.grl)
        self.ds['disf'] = ModelGeometry.distance_to_line(self.ds.mask, self.ds.isf)
        self.ds['rd'] = self.ds.dgrl/(self.ds.dgrl+self.ds.disf)
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
        self.ds['box'] = xr.DataArray(data=np.zeros(np.shape(self.ds.mask)), name='box',
                           dims=['y','x'], coords=self.ds.coords)
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
        dx = self.ds.x[1]-self.ds.x[0]
        dy = self.ds.y[1]-self.ds.y[0]
        self.ds['area'] = dx*dy*self.ds.mask
        A = np.zeros((self.n+1))
        A[0] = (self.ds.mask*self.ds.area).sum(['x','y'])
        for k in np.arange(1,self.n+1):
            A[k] = self.ds.area.where(self.ds.box==k).sum(['x','y'])  
        # assert A[0]==np.sum(A[1:]), f'{A[0]=} should be{np.sum(A[1:])=}'
        self.ds['area_k'] = xr.DataArray(data=A, dims='boxnr', coords={'boxnr':np.arange(self.n+1)})
        return

    def PICO(self):
        """ creates geometry Dataset for PICO model containing
        coordinates:
        x, y    .. from BedMachine dataset [m]
        box     .. box number [1,...,n]

        output:
        self.ds .. (xr.Dataset)
            . mask   .. (bool)   mask of ice shelf in domain
            . draft  .. (float)  depth of ice shelf [m]
            . grl    .. (bool)   grounding line mask
            . isf    .. (bool)   ice shelf front mask
            . dgrl   .. (float)  distance to grl [m]
            . disf   .. (float)  distance to isf [m]
            . rd     .. (float)  relative distance [0,1]
            . box    .. (int)    box number [1,...,n]
            . area   .. (float)  area of each box [m^2]
        """
        if os.path.exists(self.fn_PICO):
            self.ds = xr.open_dataset(self.fn_PICO)
        else:
            print(f'\n --- {self.name} ---\n')
            print('self.select_geometry()')
            self.select_geometry()

            print('self.calc_draft()')
            self.calc_draft()

            print('self.determine_grl_isf()')
            self.determine_grl_isf()

            print('self.find_distances()')
            self.find_distances()

            print('self.add_latlon()')
            self.add_latlon()

            print('self.define_boxes()')
            self.define_boxes()

            print('self.calc_area()')
            self.calc_area()

            # print('self.ds.to_netcdf(self.fn_PICO)')
            self.ds.to_netcdf(self.fn_PICO)
        return self.ds

    def plot_PICO(self):
        """ plots all PICOP fields """
        print(self.fn_PICO)
        if os.path.exists(self.fn_PICO):
            print(self.fn_PICO, ' exists')
            
            ds = xr.open_dataset(self.fn_PICO)
        else:
            ds = self.PICO()
        divnorm = matplotlib.colors.DivergingNorm(vmin=-3000., vcenter=0, vmax=500)
        kwargs = {'cbar_kwargs':{'orientation':'horizontal'}}
        f, ax = plt.subplots(1, 5, figsize=(15,5), constrained_layout=True, sharey=True, sharex=True)
        ds.draft.name = 'draft [meters]'
        ds.draft.plot(ax=ax[0], cmap='plasma', **kwargs, vmax=0)
        ds.grl.where(ds.grl>0).plot(ax=ax[0], cmap='Blues', add_colorbar=False)
        ds.isf.where(ds.isf>0).plot(ax=ax[0], cmap='Reds'   , add_colorbar=False)
        ds.disf.name = 'to ice shelf front [km]'
        (ds.disf/1e3).plot(ax=ax[1], **kwargs)
        ds.dgrl.name = 'to grounding line [km]'
        (ds.dgrl/1e3).plot(ax=ax[2], **kwargs)
        ds.rd.name = 'relative distance'
        (ds.rd).plot(ax=ax[3], **kwargs)
        ds.box.name = 'box nr.'
        (ds.box).plot(ax=ax[4], **kwargs)
        f.suptitle(f'{self.name} Ice Shelf', fontsize=16)
        return

    ### PICOP methods
    def select_velocity(self):
        """ selects the appropriate domain """
        self.vel = xr.open_dataset(fn_IceVelocity).where(self.ds)
        return

    def interpolate_velocity(self):
        """ interpolates 450 m spaced velocity onto geometry (500 m spaced) grid 
        add new velocity fields (`u` and `v`) to dataset `self.ds`
        interpolation
        """
        if os.path.exists(fn_IceVelocity_remapped):
            vel = xr.open_dataset(fn_IceVelocity_remapped)
        else:  
            ds  = xr.open_dataset(fn_BedMachine)
            vel = xr.open_dataset(fn_IceVelocity)

            # create new lat/lon coords for dataset `ds`
            project = pyproj.Proj("epsg:3031")
            xx, yy = np.meshgrid(ds.x, ds.y)
            lons, lats = project(xx, yy, inverse=True)
            dims = ['y','x']
            ds = ds.assign_coords({'lat':(dims,lats), 'lon':(dims,lons)})

            regridder = xe.Regridder(self.vel, self.ds, 'bilinear')
            u = regridder(self.vel.VX)
            v = regridder(self.vel.VY)
            u.name = 'u'
            v.name = 'v'
        self.ds = xr.merge([self.ds, vel.u, vel.v])
        return

    def calc_alpha(self):
        """ local slope angle alpha (x,y) based on draft """
        self.ds['alpha'] = 0
        return

    def PICOP(self):
        """ creates geometry Dataset for PICOP model containing
        all PICO DataArrays
        u  .. x-velocity
        v  .. y-velocity
        """
        if os.path.exists(self.fn_PICOP):
            self.ds = xr.open_dataset(self.fn_PICOP)
        else:
            self.PICO()  # create all fields for PICO
            self.interpolate_velocity()
            self.select_velocity()
            self.calc_alpha()
            self.ds.to_netcdf(self.fn_PICOP)
        return self.ds

    def plot_PICOP(self):
        """ plots important PICO fields + interpolated velocity """
        # xx, yy = np.meshgrid(vel.x, vel.y)
        # plt.figure(figsize=(10,5), constrained_layout=True)

        # velocity stream
        # np.sqrt(vel.VX**2+vel.VY**2).plot(cbar_kwargs={'label':'velocity  [m/yr]'})
        # plt.streamplot(xx, yy, vel.VX, vel.VY, color='w', linewidth=np.sqrt(vel.VX**2+vel.VY**2).fillna(0).values/5e2)
        pass


if __name__=='__main__':
    """ calculate the Totten IS example """
    # for glacier in glaciers:
    #     ModelGeometry(name=glacier).PICO()
    # ModelGeometry(name='Totten').PICO()
    ModelGeometry(name='MoscowUniversity').PICO()
    ModelGeometry(name='Dotson').PICO()
    ModelGeometry(name='Thwaites').PICO()
    ModelGeometry(name='PineIsland').PICO()