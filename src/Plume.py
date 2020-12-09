import numpy as np
import xarray as xr

from constants import ModelConstants


class PlumeModel(ModelConstants):
    """ plume melt model analytical solution by Lazeroms et al. (2018)
        melt plume driven by the ice pump circulation
        assumes: small angle
        equation and table numbers refer to publication (doi: 10.1175/JPO-D-18-0131.1)
        
        input:
        dp  ..  xr.Dataset for Plume model containing
                name      dims/coords    unit  quantity
            .   dgrl        (x)/(x,y)     [m]  distance to grounding line
            .   draft       (x)/(x,y)     [m]  depth of ice shelf
            .   alpha    ()/(x)/(x,y)   [rad]  local angle
            .   grl_adv     (x)/(x,y)     [m]  advected grounding line depth
            .   Ta [*]   ()/(x)/(x,y)  [degC]  ambient temperature
            .   Sa [*]   ()/(x)/(x,y)   [psu]  ambient salinity
            .   Tf          (x)/(x,y)  [degC]  local freezing point 
            [*] vary only in PICOP model, taken from PICO model
        
        output:  [calling `.compute_plume()`]
        ds  ..  xr.Dataset holding additional quantities with their coordinates
    """
    
    def __init__(self, dp):
        ModelConstants.__init__(self)
        assert type(dp)==xr.core.dataset.Dataset
        assert 'x' in dp.coords
        for q in ['dgrl', 'draft', 'alpha','grl_adv','Ta','Sa','Tf']:
            assert q in dp, f'missing {q}'
        self.dp = dp

        # freezing point at corresponding grounding line, eqn (7)
        self.dp['Tf0'] = self.l1*self.dp.Sa + self.l2 + self.l3*self.dp.grl_adv
        self.dp.Tf0.attrs = {'long_name':'pressure freezing point at corresponding grounding line'}
        
        # calculate dimensionless coordinate dgrl_=$\tilde{x}$ (28b)
        self.Ea = self.E0*np.sin(self.dp.alpha)
        self.dp['dgrl_'] = self.l3*(self.dp.draft-self.dp.grl_adv)/(self.dp.Ta-self.dp.Tf0)/\
                           (1+self.Ce*(self.Ea/(self.CG+self.ct+self.Ea))**(3/4))
        self.dp.dgrl_.attrs = {'long_name':'dimensionless coordinate tilde{x} in limited range [0,1); eqn. (28b)'}
            
        # reused parameter combinations
        self.f1 = self.bs*self.dp.Sa*self.g/(self.l3*(self.L/self.cp)**3)
        self.f2 = (1-self.cr1*self.CG)/(self.Cd+self.Ea)
        self.f3 = self.CG*self.Ea/(self.CG+self.ct*self.Ea)

        return
        
    def nondim_M(self, x):
        """ nondimensional melt rate, eqn. (26)
        input:
        x .. nondimensional locations, either x or x_
        """
        return (3*(1-x)**(4/3)-1)*np.sqrt(1-(1-x)**(4/3))/(2*np.sqrt(2))
    
    def dim_M(self):
        """ dimensional melt rate in [m/yr], eqn. (28a)
        needs nondimensional melt rate at dgrl_
        """
        assert 'M' in self.dp
        return np.sqrt(self.f1*self.f2*self.f3**3)*(self.dp.Ta-self.dp.Tf0)**2*self.dp.M
    
    def phi0(self, x):
        """ non-dimensional cavity circulation, eqn. (25) """
        return (1-(1-x)**(4/3))**(3/2)/(2*np.sqrt(2))
    
    def Phi(self):
        """ dimensional cavity circulation, eqn. (29) """
        return self.E0*np.sqrt(self.f1*self.f2*self.f3)*(self.dp.Ta-self.dp.Tf0)**2*self.dp.phi0

    def compute_plume(self, full_nondim=False):
        """ combines all output into single xarray dataset """
        def compute_nondimensional(x):
            """ both nondim melt rate and circulation """
            M = self.nondim_M(x)
            M.attrs = {'long_name':'dimensionless meltrate; eqn. (26)'}
            phi0 = self.phi0(x)
            phi0.attrs = {'long_name':'dimensionless circulation; eqn. (25)'}
            return M, phi0

        # calculations
        self.dp['M'], self.dp['phi0'] = compute_nondimensional(self.dp.dgrl_)

        self.dp['m']  = self.dim_M()*3600*24*365  # [m/s] -> [m/yr]
        self.dp.m.attrs = {'long_name':'dimensional meltrates; eqn. (28a)', 'units':'m/yr'}
        
        self.dp['Phi']   = self.Phi()
        self.dp.Phi.attrs = {'long_name':'dimensional circulation; eqn. (29)', 'units':'m^3/s'}

        if full_nondim:   # compute non-dimensional 1D melt curve for full [0,1] interval
            self.dp = self.dp.assign_coords({'x_':np.linspace(0,1,51)})
            self.dp.x_.attrs['long_name'] = 'non-dimensional coordinates in [0,1]'
            self.dp['M_full'], self.dp['phi0_full'] = compute_nondimensional(self.dp.coords['x_'])
        
        return self.dp