import numpy as np
import xarray as xr

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
        self.ds['name_forcing'] = f'const_Ta{Ta}_Sa{Sa}'
        return self.ds

    def tanh(self, ztcl, Tdeep, drhodz=0.0001):
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
        self.ds['name_forcing'] = f'tanh_Tdeep{Tdeep}_ztcl{ztcl}'
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
        self.ds['name_forcing'] = f'favier_{profile}'

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
        self.ds['name_forcing'] = f'ISOMIP_{profile}'
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
        self.ds['name_forcing'] = f'ISOMIP_{frac}'
        return self.ds

    def calc_fields(self):
        """ adds Ta/Sa fields to geometry dataset """
        assert 'Tz' in self.ds
        assert 'Sz' in self.ds
        Sa = np.interp(self.ds.draft.values, self.ds.z.values, self.ds.Sz.values)
        Ta = np.interp(self.ds.draft.values, self.ds.z.values, self.ds.Tz.values)
        self.ds['Ta'] = (['y', 'x'], Ta)
        self.ds['Sa'] = (['y', 'x'], Sa)
        self.ds['Tf'] = self.l1*self.ds.Sa + self.l2 + self.l3*self.ds.draft
        self.ds.Ta.attrs = {'long_name':'ambient in-situ temperature', 'units':'degC'}
        self.ds.Sa.attrs = {'long_name':'ambient salinity', 'units':'psu'}
        self.ds.Tf.attrs = {'long_name':'local (in-situ) freezing point', 'units':'degC'}  # from:Eq. 3 of Favier19
        return self.ds


    def potential_to_insitu(self):
        """ convert potential to in-situ temperatures """
        assert 'Tz' in self.ds
        # self.ds['Tz'] = self.
        return self.ds