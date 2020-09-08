import numpy as np
import xarray as xr

class PlumeModel(object):
    """ plume melt model analytical solution
    based on Lazerom et al. (2018)
    melt plume driven by the ice pump circulation
    assumes: small angle
    equation and table numbers refer to publication (doi: 10.1175/JPO-D-18-0131.1)
    
    input:
    X   ..  [m]     distance from grounding line
    zb  ..  [m]     ice shelf depth
    Ta  ..  [degC]  ambient temperature
    Ta  ..  [psu]   ambient salinity
    
    output:  [calling `.compute()`]
    ds  ..  xarray Dataset holding all quantities with their coordinates
    """
    
    def __init__(self, X, zb, Ta, Sa):
        assert len(X)==len(zb)
        assert X[0]==0  # (starting at grounding line)
        self.X  = X
        self.Ta = Ta
        self.Sa = Sa
        
        N   = len(X)

        # parameters (from Table 1)
        self.l1  = -5.73e-2  # [degC]      freezing point salinity coefficient
        self.l2  = 8.32e-2   # [degC]      freezing point offset
        self.l3  = 7.61e-4   # [degC/m]    freezing point depth coefficient
        self.Ce  = 0.6       # [1]         slope correction factor
        self.E0  = 3.6e-2    # [1]         entrainment coefficient
        self.CG  = 5.9e-4    # [1]         effective thermal Stanton number
        self.ct  = 1.4e-5    # [1]         c_{tau}
        self.L   = 3.35e5    # [J/kg]      Latent heat of fusion for ice
        self.g   = 9.81      # [m/s^2]     gravitational acceleration
        self.c   = 3.974e3   # [J/kg/degC] specific heat capacity of ocean
        self.bs  = 7.86e-4   # [1/psu]     haline coontraction coefficient 
        self.Cd  = 2.5e-3    # [1]         drag coefficient
        self.cr1 = 2e2       # [1]         c_{\rho 1}
        
        # geometry and dependent variables
        zgl = zb[0]  # grounding line depth [m]
        self.Tf0 = self.l1*self.Sa + self.l2 + self.l3*zgl
                        # freezing point at grounding line, eqn (7)
        Tf = self.l1*self.Sa + self.l2 + self.l3*zb
                        # freezing point at ice sheet base,
                        # should be eqn (8), but adapted for non-uniform slope
        alpha = np.arctan(np.gradient(zb, X)) 
                        # angle at each position X

        
        # calculate dimensionless coordinate x_=$\tilde{x}$ (28b)
        self.Ea = self.E0*np.sin(alpha)
        x_ = self.l3*(zb-zgl)/(self.Ta-self.Tf0)/\
             (1+self.Ce*(self.Ea/(self.CG+self.ct+self.Ea))**(3/4))
            
        # reused parameter combinations
        self.f1 = self.bs*self.Sa*self.g/(self.l3*(self.L/self.c)**3)
        self.f2 = (1-self.cr1*self.CG)/(self.Cd+self.Ea)
        self.f3 = self.CG*self.Ea/(self.CG+self.ct*self.Ea)
        
        # create xarray Dataset to contain quantities
        data_vars = {'zb':('X',zb), 'Tf':('X',Tf)}
        coords = {'x':np.linspace(0,1,51),
                  'x_':x_,
                  'X':X
                 }
        self.ds = xr.Dataset(data_vars, coords=coords)
        return
        
        
    def nondim_M(self, x):
        """ nondimensional melt rate, eqn. (26)
        input:  x .. nondimensional locations, either x or x_ 
        """
        return (3*(1-x)**(4/3)-1)*np.sqrt(1-(1-x)**(4/3))/(2*np.sqrt(2))
    
    
    def dim_M(self):
        """ dimensional melt rate in [m/yr], eqn. (28a); needs nondimensional melt rate at x_ """
        assert 'M_' in self.ds
        return np.sqrt(self.f1*self.f2*self.f3**3)*(self.Ta-self.Tf0)**2*self.ds['M_']
    
    
    def phi0(self, x):
        """ non-dimensional cavity circulation, eqn. (25) """
        return (1-(1-x)**(4/3))**(3/2)/(2*np.sqrt(2))
    
    def Phi(self):
        """ dimensional cavity circulation, eqn. (29) """
        return self.E0*np.sqrt(self.f1*self.f2*self.f3)*(self.Ta-self.Tf0)**2*self.ds['phi0_']
    
    
    def compute(self):
        """ combines all output into single xarray dataset """
        # calcultions
        self.ds['M']  = ('x' , self.nondim_M(self.ds.x))
        self.ds['M_'] = ('x_', self.nondim_M(self.ds.x_))
        self.ds['m']  = ('X' , self.dim_M())
        
        self.ds['phi0']  = ('x' , self.phi0(self.ds.x))
        self.ds['phi0_'] = ('x_', self.phi0(self.ds.x_))
        self.ds['Phi']   = ('X' , self.Phi())
        
        # metadata
        attrs = {'X' :'dimensional coordinate $X$ [m]',
                 'x' :'dimensionless coordinate $x$ in full range [0,1)',
                 'x_':'dimensionless coordinate tilde{x} in limited range [0,1); eqn. (28b)',
                 'zb':'dimensional depths at X [m]',
                 'Tf':'pressure freezing points at X [degC]',
                 'M' :'dimensionless meltrate at x; eqn. (26)',
                 'M_':'dimensionless meltrate at x_; eqn. (26)',
                 'm' :'dimensional meltrates at X [m/s]; eqn. (28a)',
                 'phi0' :'dimensionless circulation at x; eqn. (25)',
                 'phi0_':'dimensionless circulation at x_; eqn. (25)',
                 'Phi'  :'dimensional circulation, eqn. (29)',
                
                }
        self.ds.attrs = attrs
        
        return self.ds