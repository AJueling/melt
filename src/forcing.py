import numpy as np
import xarray as xr

from plotfunctions import add_lonlat
from constants import ModelConstants

class Forcing(ModelConstants):
    """ """

    def __init__(self, ds):
        """ 
        input:
        ds       (xr.Dataset)  geometry dataset from one of the Geometry classes

        output:
        self.ds  (xr.Dataset)  original `ds` with additional fields:
            Tz   (z)   [degC]  ambient potential temperature
            Sz   (z)   [psu]   ambient salinity

            Plume/Sheet/Simple:  spatially extended ambient T/S fields
            Ta   (x,y) [degC]  ambient potential temperature
            Sa   (x,y) [psu]   ambient salinity

            PICO/PICOP:        single temperature/salinity values for  model
            TaD  ()    [degC]  temperature at deepest part of grounding line
            Ta5  ()    [degC]  temperature at 500 m
            Ta7  ()    [degC]  temperature at 700 m
            TaL  ()    [degC]  temperature at 
        """
        assert 'draft' in ds
        self.ds = ds
        self.ds = self.ds.assign_coords({'z':np.arange(-5000.,0,1)})
        ModelConstants.__init__(self)
        return

    def constant(self, Ta=0, Sa=34):
        """ constant in-situ temperature and salinity """
        self.ds['Tz'] = Ta*np.ones_like(self.ds.z.values)
        self.ds['Sz'] = Sa*np.ones_like(self.ds.z.values)
        self.ds = self.calc_fields()
        self.ds.attrs['name_forcing'] = f'const_Ta{Ta}_Sa{Sa}'
        return self.ds

    def tanh(self, ztcl, Tdeep, drhodz=.6/720):
        """ creates tanh thermocline forcing profile
        input:
        ztcl    ..  (float)  [m]       thermocline depth
        Tdeep   ..  (float)  [degC]    in situ temperature at depth
        drhodz  ..  (float)  [kg/m^4]  linear density stratification
        """
        if ztcl>0:
            print('z-coordinate is postive upwards; ztcl was {ztcl}, now set ztcl=-{ztcl}')
            ztcl = -ztcl
        self.S0 = 34                       # [psu]  reference surface salinity
        self.T0 = self.l1*self.S0+self.l2  # [degC] surface freezing temperature
        self.z1 = 100                      # [m]    thermocline sharpness
        
        self.ds['Tz'] = Tdeep + (self.T0-Tdeep) * (1+np.tanh((self.ds.z-ztcl)/self.z1))/2
        self.ds['Sz'] = self.S0 + self.alpha*(self.ds.Tz-self.T0)/self.beta - drhodz*self.ds.z/(self.beta*self.rho0)
        self.ds = self.calc_fields()
        self.ds.attrs['name_forcing'] = f'tanh_Tdeep{Tdeep:.1f}_ztcl{ztcl}'
        return self.ds
    
    def tanh2(self, ztcl, Tdeep, drhodz=.6/720, T0 = -1.7):
        """ creates tanh thermocline forcing profile
        input:
        ztcl    ..  (float)  [m]       thermocline depth
        Tdeep   ..  (float)  [degC]    in situ temperature at depth
        drhodz  ..  (float)  [kg/m^4]  linear density stratification
        """
        if ztcl>0:
            print('z-coordinate is postive upwards; ztcl was {ztcl}, now set ztcl=-{ztcl}')
            ztcl = -ztcl
        self.S0 = 34                       # [psu]  reference surface salinity
        self.z1 = 100                      # [m]    thermocline sharpness
        
        self.ds['Tz'] = Tdeep + (T0-Tdeep) * (1+np.tanh((self.ds.z-ztcl)/self.z1))/2
        self.ds['Sz'] = self.S0 + self.alpha*(self.ds.Tz-T0)/self.beta - drhodz*self.ds.z/(self.beta*self.rho0)
        self.ds = self.calc_fields()
        self.ds.attrs['name_forcing'] = f'tanh2_Tdeep{Tdeep:.1f}_ztcl{ztcl}'
        return self.ds
    
    def favier(self, profile):
        """ piecewise linear profiles suggested by Favier et al. (2019), using potential temperature """
        assert profile in ['cold0', 'cold1', 'warm0', 'warm1', 'warm2', 'warm3']

        if profile == 'cold0':
            z = [-5000,-700,-300,0]
            Tz = [-1.5,-1.5,-1.5,-1.5]
            Sz = [34,34,34,34]
        elif profile == 'cold1':
            z = [-5000,-700,-300,0]
            Tz = [1.2,1.2,-0.6,-0.6]
            Sz = [34.5,34.5,34,34]
        elif profile == 'warm0':
            z = [-5000,-700,-300,0]
            Tz = [1.2,1.2,-1.0,-1.0]
            Sz = [34.5,34.5,34,34]
        elif profile == 'warm1':
            z = [-5000,-700,-300,0]
            Tz = [2.2,2.2,0,0]
            Sz = [34.5,34.5,34,34]        
        elif profile == 'warm2':
            z = [-5000,-700,-300,0]
            Tz = [2.2,2.2,-1,-1]
            Sz = [34.5,34.5,34,34]         
        elif profile == 'warm3':
            z = [-5000,-500,-100,0]
            Tz = [1.2,1.2,-1.0,-1.0]
            Sz = [34.5,34.5,34,34]
            
        self.ds['Tz'] = (['z'],np.interp(self.ds.z,z,Tz))
        self.ds['Sz'] = (['z'],np.interp(self.ds.z,z,Sz))
        self.ds = self.calc_fields()
        self.ds.attrs['name_forcing'] = f'Favier19_{profile}'

        return self.ds

    def isomip(self,profile):
        """ linear ISOMIP profiles,
        temperatures are given as potential temperatures
        """
        assert profile in ['WARM','COLD']
        if profile == 'COLD':
            z = [-720,0]
            Tz = [-1.9,-1.9]
            Sz = [34.55,33.8]
        elif profile == 'WARM':
            z = [-720,0]
            Tz = [1.0,-1.9]
            Sz = [34.7,33.8]
        self.ds['Tz'] = (['z'],np.interp(self.ds.z,z,Tz))
        self.ds['Sz'] = (['z'],np.interp(self.ds.z,z,Sz))
        self.ds = self.calc_fields()
        self.ds.attrs['name_forcing'] = f'ISOMIP_{profile}'
        return self.ds

    def isomip_frac(self,frac):
        """ linear interpolation between ISOMIP COLD and WARM profiles """
        assert frac>=0 and frac<=1
        z = [-720,0]
        Tz = frac*np.array([-1.9,-1.9])  + (1-frac)*np.array([1.0,-1.9])
        Sz = frac*np.array([34.55,33.8]) + (1-frac)*np.array([34.7,33.8])
        self.ds['Tz'] = (['z'],np.interp(self.ds.z,z,Tz))
        self.ds['Sz'] = (['z'],np.interp(self.ds.z,z,Sz))
        self.ds = self.calc_fields()
        self.ds.attrs['name_forcing'] = f'ISOMIP_{frac:.1f}'
        return self.ds

    def calc_fields(self):
        """ adds Ta/Sa fields to geometry dataset: forcing  = frac*COLD + (1-frac)*WARM """
        assert 'Tz' in self.ds
        assert 'Sz' in self.ds
        Sa = np.interp(self.ds.draft.values, self.ds.z.values, self.ds.Sz.values)
        Ta = np.interp(self.ds.draft.values, self.ds.z.values, self.ds.Tz.values)
        self.ds['Ta'] = (['y', 'x'], Ta)
        self.ds['Sa'] = (['y', 'x'], Sa)
        self.ds['Tf'] = self.l1*self.ds.Sa + self.l2 + self.l3*self.ds.draft  # l3 -> potential temperature
        self.ds.Ta.attrs = {'long_name':'ambient potential temperature' , 'units':'degC'}
        self.ds.Sa.attrs = {'long_name':'ambient salinity'              , 'units':'psu' }
        self.ds.Tf.attrs = {'long_name':'local potential freezing point', 'units':'degC'}  # from:Eq. 3 of Favier19
        return self.ds

    def holland07(self):
        z = [-5000,-2000,0]
        Tz = [-2.3,-2.3,-1.9]
        Sz = [34.8,34.8,34.5]
        self.ds['Tz'] = (['z'],np.interp(self.ds.z,z,Tz))
        self.ds['Sz'] = (['z'],np.interp(self.ds.z,z,Sz))
        self.ds = self.calc_fields()
        self.ds.attrs['name_forcing'] = f'holland07'
        return self.ds        
    
    def holland(self,option='interp',kup =2,kdwn = 1,nsm = 1):
        
        self.ds = add_lonlat(self.ds)
        lon3 = self.ds.lon.values
        lat3 = self.ds.lat.values
        mask = self.ds.mask.values
        
        #Read Holland data
        timep= slice("1989-1-1","2018-12-31")
        ds = xr.open_dataset('../../data/paulholland/PAS_851/stateTheta.nc')
        ds = ds.sel(LONGITUDE=slice(360+np.min(lon3),360+np.max(lon3)),LATITUDE=slice(np.min(lat3),np.max(lat3)),TIME=timep)
        ds = ds.mean(dim='TIME')
        lon   = (ds.LONGITUDE-360.).values
        lat   = (ds.LATITUDE-.05).values
        dep   = ds.DEPTH.values
        theta = ds.THETA.values
        ds.close()
        ds = xr.open_dataset('../../data/paulholland/PAS_851/stateSalt.nc')
        ds = ds.sel(LONGITUDE=slice(360+np.min(lon3),360+np.max(lon3)),LATITUDE=slice(np.min(lat3),np.max(lat3)),TIME=timep)
        ds = ds.mean(dim='TIME')
        salt  = ds.SALT.values
        ds.close()
        
        #Extrapolate profiles to top and bottom
        llon,llat = np.meshgrid(lon,lat)
        Th = theta.copy()
        Sh = salt.copy()
        for j,jj in enumerate(lat):
            for i,ii in enumerate(lon):
                if Sh[0,j,i] == 0:
                    k0 = np.argmax(Sh[:,j,i]!=0)+kup
                    Th[:k0,j,i] = Th[k0,j,i]
                    Sh[:k0,j,i] = Sh[k0,j,i]
                if Sh[-1,j,i] == 0:
                    k1 = np.argmin(Sh[:,j,i]!=0)-kdwn
                    Th[k1:,j,i] = Th[k1-1,j,i]
                    Sh[k1:,j,i] = Sh[k1-1,j,i]
                if sum(Sh[:,j,i]) == 0:
                    llon[j,i] = 1e36
                    llat[j,i] = 1e36

        #Apply nearest neighbour onto model grid
        depth = np.arange(5000) #depth also used as index, so must be positive with steps of 1
        Tz = np.zeros((len(depth),mask.shape[0],mask.shape[1]))
        Sz = np.zeros((len(depth),mask.shape[0],mask.shape[1]))
        for j in range(mask.shape[0]):
            for i in range(mask.shape[1]):
                if mask[j,i] == 3:
                    #Get nearest indices at low end
                    i0 = np.argmax(lon>lon3[j,i])-1
                    j0 = np.argmax(lat>lat3[j,i])-1
                    if option=='interp':
                        #Distance squared
                        dist = np.cos(np.pi*lat3[j,i]/180.)*(np.pi*(lon3[j,i]-llon[j0-nsm:j0+2+nsm,i0-nsm:i0+2+nsm])/180.)**2+(np.pi*(lat3[j,i]-llat[j0-nsm:j0+2+nsm,i0-nsm:i0+2+nsm])/180.)**2
               
                        weight = 1./(dist+5e-8)
                        TT = np.sum(Th[:,j0-nsm:j0+2+nsm,i0-nsm:i0+2+nsm]*weight,axis=(1,2))/np.sum(weight)
                        SS = np.sum(Sh[:,j0-nsm:j0+2+nsm,i0-nsm:i0+2+nsm]*weight,axis=(1,2))/np.sum(weight)
                    elif option=='nn':
                        #Direct nearest neighbor:
                        TT = Th[:,j0,i0]
                        SS = Sh[:,j0,i0]
                    
                    Tz[:,j,i] = np.interp(depth,dep,TT)
                    Sz[:,j,i] = np.interp(depth,dep,SS)
        del TT,SS,Th,Sh,theta,salt
        self.ds = self.ds.assign_coords({'z':depth})
        self.ds['Tz'] = (['z','y','x'],Tz)
        self.ds['Sz'] = (['z','y','x'],Sz)
        self.ds.attrs['name_forcing'] = 'holland'
        return self.ds
        
        
    def potential_to_insitu(self):
        """ convert potential to in-situ temperatures """
        assert 'Tz' in self.ds
        # self.ds['Tz'] = self.
        return self.ds