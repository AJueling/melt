class ModelConstants(object):
    """ shared functions for different melt models
    
    init: defines all constants used in 
    - PICO   Reese et al. (2018)
    - Plume  Lazeroms et al. (2019)
    - PICOP  Pelle et al. (2019)
    """
    def __init__(self):
        """ defining constants; created mainly to avoid naming conflicts """
        # shared
        self.g      = 9.81      # [m/s^2]     gravitational acceleration
        self.L      = 3.35e5    # [J/kg]      Latent heat of fusion for ice
        self.cp     = 3.974e3   # [J/kg/degC] specific heat capacity of ocean
        
        # PICO model
        self.a      = -0.0572   # [degC /psu]
        self.b      =  0.0788   # [degC]
        self.c      =  7.77e-8  # [degC/Pa]
        self.alpha  =  7.5e-5   # [1/degC]
        self.beta   =  7.7e-4   # [1/psu]
        self.rho0   =  1033     # [kg/m^3]
        self.rhoi   =   910     # [kg/m^3]
        self.rhow   =  1028     # [kg/m^3]
        self.gammaS =  2e-6     # [m/s]       salinity mixing coefficient
        self.gammaT =  5e-5     # [m/s]       temperature mixing coefficient $$
        self.gammae =  2e-5     # [m/s]       effective mixing coefficient $\gamma^\star_T$
        self.C      =  1e6      # [m^6/s/kg]
        
        # Plume model (from Table 1)
        self.l1     = -5.73e-2  # [degC]      freezing point salinity coefficient
        self.l2     = 8.32e-2   # [degC]      freezing point offset
        self.l3     = 7.61e-4   # [degC/m]    freezing point depth coefficient
        self.Ce     = 0.6       # [1]         slope correction factor
        self.E0     = 3.6e-2    # [1]         entrainment coefficient
        self.CG     = 5.9e-4    # [1]         effective thermal Stanton number
        self.ct     = 1.4e-5    # [1]         c_{tau}
        self.bs     = 7.86e-4   # [1/psu]     haline coontraction coefficient 
        self.Cd     = 2.5e-3    # [1]         drag coefficient
        self.cr1    = 2e2       # [1]         c_{\rho 1}
        return
    
    