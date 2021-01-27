import numpy as np
import xarray as xr

from constants import ModelConstants

class LayerModel(ModelConstants):
    """ Layer model based on Holland et al (2007)
    
        input:
        ds including:
            x      ..  [m]     x coordinate
            y      ..  [m]     y coordinate

            2D [y,x] fields:
            mask   ..  [bin]   mask identifying ocean (0), grounded ice (2), ice shelf (3)
            draft  ..  [m]     ice shelf draft
            
            1D [z] fields:
            Tz     ..  [degC]  ambient temperature
            Sz     ..  [psu]   ambient salinity

        output:  [calling `.compute()`]
        ds  ..  xarray Dataset holding all quantities with their coordinates
    """
    
    def __init__(self, ds):
        #Read input
        self.dx = ds.x[1]-ds.x[0]
        self.dy = ds.y[1]-ds.y[0]
        if self.dx<0:
            print('inverting x-coordinates')
            ds = ds.reindex(x=list(reversed(ds.x)))
            self.dx = -self.dx
        if self.dy<0:
            print('inverting y-coordinates')
            ds = ds.reindex(y=list(reversed(ds.y)))
            self.dy = -self.dy
            
        self.ds   = ds
        self.x    = ds.x
        self.y    = ds.y        
        self.mask = ds.mask
        self.zb   = ds.draft
        self.z    = ds.z.values
        self.Tz   = ds.Tz.values
        self.Sz   = ds.Sz.values

        #Physical parameters
        ModelConstants.__init__(self)
        self.f = -1.37e-4     # Coriolis parameter [1/s]
        
        #Some input params
        self.nu = .8          # Nondimensional factor for Robert Asselin time filter
        self.slip = 2         # Nondimensional factor Free slip: 0, no slip: 2, partial no slip: [0..2]  
        self.Ah = 100         # Laplacian viscosity [m^2/s]
        self.dt = 50          # Time step [s]
        self.Kh = 5           # Diffusivity [m^2/s]
        
        self.minD = 1.        # Cutoff thickness [m]
        self.vcut = 1.        # Velocity above which drag is increased [m/s]
        self.Cdfac = 1000      # Magnification factor of drag on velocity above vcut
        
        #Some parameters for displaying output
        self.diagint = 100    # Timestep at which to print diagnostics
        self.verbose = True

    def integrate(self):
        """Integration of 2 time steps, now-centered Leapfrog scheme"""
        intD(self,2*self.dt)
        intu(self,2*self.dt)
        intv(self,2*self.dt)
        intT(self,2*self.dt)
        intS(self,2*self.dt)        
        return
    
    def timefilter(self):
        """Time filter, Robert Asselin scheme"""
        self.D[1,:,:] += self.nu/2 * (self.D[0,:,:]+self.D[2,:,:]-2*self.D[1,:,:]) * self.tmask
        self.u[1,:,:] += self.nu/2 * (self.u[0,:,:]+self.u[2,:,:]-2*self.u[1,:,:]) * self.umask
        self.v[1,:,:] += self.nu/2 * (self.v[0,:,:]+self.v[2,:,:]-2*self.v[1,:,:]) * self.vmask
        self.T[1,:,:] += self.nu/2 * (self.T[0,:,:]+self.T[2,:,:]-2*self.T[1,:,:]) * self.tmask
        self.S[1,:,:] += self.nu/2 * (self.S[0,:,:]+self.S[2,:,:]-2*self.S[1,:,:]) * self.tmask
        return
                
    def updatevars(self):
        """Update temporary variables"""
        self.D = np.roll(self.D,-1,axis=0)
        self.u = np.roll(self.u,-1,axis=0)
        self.v = np.roll(self.v,-1,axis=0)
        self.T = np.roll(self.T,-1,axis=0)
        self.S = np.roll(self.S,-1,axis=0)
        updatesecondary(self)
        return
    
    def compute(self,days=12):
        create_mask(self)
        create_grid(self)
        initialize_vars(self)

        self.nt = int(days*24*3600/self.dt)+1    # Number of time steps
        self.time = np.linspace(0,days,self.nt)  # Time in days
        for self.t in range(self.nt):
            self.updatevars()
            self.integrate()
            self.timefilter()
            self.D = np.where(self.D<self.minD,self.minD,self.D)
            if self.t in np.arange(self.diagint,self.nt,self.diagint):
                printdiags(self)
        if self.verbose:
            print('-----------------------------')
            print(f'Run completed, final values:')
            printdiags(self)
        
        #Output

        self.ds['U'] = (['y','x'], im(self.u[1,:,:]))
        self.ds['V'] = (['y','x'], jm(self.v[1,:,:]))
        self.ds['D'] = (['y','x'], self.D[1,:,:])
        self.ds['T'] = (['y','x'], self.T[1,:,:])
        self.ds['S'] = (['y','x'], self.S[1,:,:])
        
        self.ds['tmask'] = (['y','x'], self.tmask)
        
        self.ds['melt'] = (['y','x'], 3600*24*365.25*self.melt)
        self.ds['mav'] = 3600*24*365.25*(self.melt*self.dx*self.dy).sum()/(self.tmask*self.dx*self.dy).sum()
        self.ds['mmax'] = 3600*24*365.25*self.melt.max()
        
        self.ds['name_model'] = 'Layer'
        self.ds['filename'] = f"../../results/{self.ds['name_model'].values}_{self.ds['name_geo'].values}_{self.ds['name_forcing'].values}"
        self.ds.to_netcdf(f"{self.ds['filename'].values}.nc")
        print('-----------------------------')
        print(f"Output saved as {self.ds['filename'].values}.nc")
        print('=============================')
    
        return self.ds

def create_mask(self):

    #Read mask input
    self.tmask = np.where(self.mask==3,1,0)
    self.grd   = np.where(self.mask==2,1,0)
    self.grd   = np.where(self.mask==1,1,self.grd)
    self.ocn   = np.where(self.mask==0,1,0)

    #Rolled masks
    self.tmaskym1    = np.roll(self.tmask,-1,axis=0)
    self.tmaskyp1    = np.roll(self.tmask, 1,axis=0)
    self.tmaskxm1    = np.roll(self.tmask,-1,axis=1)
    self.tmaskxp1    = np.roll(self.tmask, 1,axis=1)    
    self.tmaskxm1ym1 = np.roll(np.roll(self.tmask,-1,axis=0),-1,axis=1)
    self.tmaskxm1yp1 = np.roll(np.roll(self.tmask, 1,axis=0),-1,axis=1)
    self.tmaskxp1ym1 = np.roll(np.roll(self.tmask,-1,axis=0), 1,axis=1)

    self.ocnym1      = np.roll(self.ocn,-1,axis=0)
    self.ocnyp1      = np.roll(self.ocn, 1,axis=0)
    self.ocnxm1      = np.roll(self.ocn,-1,axis=1)
    self.ocnxp1      = np.roll(self.ocn, 1,axis=1)

    self.grdNu = 1-np.roll((1-self.grd)*(1-np.roll(self.grd,-1,axis=1)),-1,axis=0)
    self.grdSu = 1-np.roll((1-self.grd)*(1-np.roll(self.grd,-1,axis=1)), 1,axis=0)
    self.grdEv = 1-np.roll((1-self.grd)*(1-np.roll(self.grd,-1,axis=0)),-1,axis=1)
    self.grdWv = 1-np.roll((1-self.grd)*(1-np.roll(self.grd,-1,axis=0)), 1,axis=1)

    #Extract ice shelf front
    self.isfE = self.ocn*self.tmaskxp1
    self.isfN = self.ocn*self.tmaskyp1
    self.isfW = self.ocn*self.tmaskxm1
    self.isfS = self.ocn*self.tmaskym1
    self.isf  = self.isfE+self.isfN+self.isfW+self.isfS

    #Extract grounding line 
    self.grlE = self.grd*self.tmaskxp1
    self.grlN = self.grd*self.tmaskyp1
    self.grlW = self.grd*self.tmaskxm1
    self.grlS = self.grd*self.tmaskym1
    self.grl  = self.grlE+self.grlN+self.grlW+self.grlS

    #Create masks for U- and V- velocities at N/E faces of grid points
    self.umask = (self.tmask+self.isfW)*(1-np.roll(self.grlE,-1,axis=1))
    self.vmask = (self.tmask+self.isfS)*(1-np.roll(self.grlN,-1,axis=0))

    self.umaskxp1 = np.roll(self.umask,1,axis=1)
    self.vmaskyp1 = np.roll(self.vmask,1,axis=0)

    return
    
def create_grid(self):   
    #Spatial parameters
    self.nx = len(self.x)
    self.ny = len(self.y)
    self.dx = (self.x[1]-self.x[0]).values
    self.dy = (self.y[1]-self.y[0]).values
    self.xu = self.x+self.dx
    self.yv = self.y+self.dy

    # Assure free-slip is used in 1D simulation
    if (len(self.y)==3 or len(self.x)==3):
        print('1D run, using free slip')
        self.slip = 0                             

    return

def initialize_vars(self):
    #Major variables. Three arrays for storage of previous timestep, current timestep, and next timestep
    self.u = np.zeros((3,self.ny,self.nx)).astype('float64')
    self.v = np.zeros((3,self.ny,self.nx)).astype('float64')
    self.D = np.zeros((3,self.ny,self.nx)).astype('float64')
    self.T = np.zeros((3,self.ny,self.nx)).astype('float64')
    self.S = np.zeros((3,self.ny,self.nx)).astype('float64')

    #Draft dz/dx and dz/dy on t-grid
    self.dzdx = np.gradient(self.zb,self.dx,axis=1)
    self.dzdy = np.gradient(self.zb,self.dy,axis=0)

    #Initial values
    self.Ta = np.interp(self.zb,self.z,self.Tz)
    self.Sa = np.interp(self.zb,self.z,self.Sz)   

    self.D += self.minD+10.#.1
    for n in range(3):
        self.T[n,:,:] = self.Ta
        self.S[n,:,:] = self.Sa-.1

    #Perform first integration step with 1 dt
    updatesecondary(self)
    intD(self,self.dt)
    intu(self,self.dt)
    intv(self,self.dt)
    intT(self,self.dt)
    intS(self,self.dt)
    return    

    
def div0(a,b):
    return np.divide(a,b,out=np.zeros_like(a), where=b!=0)

def im(var):
    """Value at i-1/2 """
    return .5*(var+np.roll(var,1,axis=1))

def ip(var):
    """Value at i+1/2"""
    return .5*(var+np.roll(var,-1,axis=1))

def jm(var):
    """Value at j-1/2"""
    return .5*(var+np.roll(var,1,axis=0))

def jp(var):
    """Value at j+1/2"""
    return .5*(var+np.roll(var,-1,axis=0))

def im_t(self,var):
    """Value at i-1/2, no gradient across boundary"""
    return div0((var*self.tmask + np.roll(var*self.tmask, 1,axis=1)),self.tmask+self.tmaskxp1)

def ip_t(self,var):
    """Value at i+1/2, no gradient across boundary"""   
    return div0((var*self.tmask + np.roll(var*self.tmask,-1,axis=1)),self.tmask+self.tmaskxm1)

def jm_t(self,var):
    """Value at j-1/2, no gradient across boundary"""
    return div0((var*self.tmask + np.roll(var*self.tmask, 1,axis=0)),self.tmask+self.tmaskyp1)

def jp_t(self,var):
    """Value at j+1/2, no gradient across boundary"""
    return div0((var*self.tmask + np.roll(var*self.tmask,-1,axis=0)),self.tmask+self.tmaskym1)

def im_(var,mask):
    """Value at i-1/2, no gradient across boundary"""
    return div0((var*mask + np.roll(var*mask, 1,axis=1)),(mask+np.roll(mask, 1,axis=1)))

def ip_(var,mask):
    """Value at i+1/2, no gradient across boundary"""
    return div0((var*mask + np.roll(var*mask,-1,axis=1)),(mask+np.roll(mask,-1,axis=1)))

def jm_(var,mask):
    """Value at j-1/2, no gradient across boundary"""
    return div0((var*mask + np.roll(var*mask, 1,axis=0)),(mask+np.roll(mask, 1,axis=0)))

def jp_(var,mask):
    """Value at j+1/2, no gradient across boundary"""
    return div0((var*mask + np.roll(var*mask,-1,axis=0)),(mask+np.roll(mask,-1,axis=0)))    

def lapT(self,var):
    """Laplacian operator for DT and DS"""
    
    tN = jp_t(self,self.D[0,:,:])*(np.roll(var,-1,axis=0)-var)*self.tmaskym1/self.dy**2
    tS = jm_t(self,self.D[0,:,:])*(np.roll(var, 1,axis=0)-var)*self.tmaskyp1/self.dy**2
    tE = ip_t(self,self.D[0,:,:])*(np.roll(var,-1,axis=1)-var)*self.tmaskxm1/self.dx**2
    tW = im_t(self,self.D[0,:,:])*(np.roll(var, 1,axis=1)-var)*self.tmaskxp1/self.dx**2    
    
    return tN+tS+tE+tW

def lapu(self):
    """Laplacian operator for Du"""
    Dcent = ip_t(self,self.D[0,:,:])
    var = self.u[0,:,:]

    tN = jp_t(self,Dcent)                            * (np.roll(var,-1,axis=0)-var)/self.dy**2 * (1-self.ocnym1) - self.slip*Dcent*var*self.grdNu/self.dy**2
    tS = jm_t(self,Dcent)                            * (np.roll(var, 1,axis=0)-var)/self.dy**2 * (1-self.ocnyp1) - self.slip*Dcent*var*self.grdSu/self.dy**2  
    tE = np.roll(self.D[0,:,:]*self.tmask,-1,axis=1) * (np.roll(var,-1,axis=1)-var)/self.dx**2 * (1-self.ocnxm1)
    tW = self.D[0,:,:]                               * (np.roll(var, 1,axis=1)-var)/self.dx**2 * (1-self.ocn   )
    
    return (tN+tS+tE+tW) * self.umask

def lapv(self):
    """Laplacian operator for Dv"""
    Dcent = jp_t(self,self.D[0,:,:])
    var = self.v[0,:,:]
    
    tN = np.roll(self.D[1,:,:]*self.tmask,-1,axis=0) * (np.roll(var,-1,axis=0)-var)/self.dy**2 * (1-self.ocnym1) 
    tS = self.D[0,:,:]                               * (np.roll(var, 1,axis=0)-var)/self.dy**2 * (1-self.ocn   )
    tE = ip_t(self,Dcent)                            * (np.roll(var,-1,axis=1)-var)/self.dx**2 * (1-self.ocnxm1) - self.slip*Dcent*var*self.grdEv/self.dx**2
    tW = im_t(self,Dcent)                            * (np.roll(var, 1,axis=1)-var)/self.dx**2 * (1-self.ocnxp1) - self.slip*Dcent*var*self.grdWv/self.dx**2  
    
    return (tN+tS+tE+tW) * self.vmask

def convT(self,var):
    """Upstream convergence scheme for D, DT, DS"""
    tN = - (np.maximum(self.v[1,:,:],0)*var                   + np.minimum(self.v[1,:,:],0)*np.roll(var,-1,axis=0)) / self.dy * self.vmask
    tS =   (np.maximum(self.vyp1    ,0)*np.roll(var,1,axis=0) + np.minimum(self.vyp1    ,0)*var                   ) / self.dy * self.vmaskyp1
    tE = - (np.maximum(self.u[1,:,:],0)*var                   + np.minimum(self.u[1,:,:],0)*np.roll(var,-1,axis=1)) / self.dx * self.umask
    tW =   (np.maximum(self.uxp1    ,0)*np.roll(var,1,axis=1) + np.minimum(self.uxp1    ,0)*var                   ) / self.dx * self.umaskxp1
    return (tN+tS+tE+tW) * self.tmask

def convu(self):
    """Convergence for Du"""
   
    #Get D at north, south, east, west points
    DN = div0((self.D[1,:,:]*self.tmask + self.Dxm1 + self.Dym1 + self.Dxm1ym1),(self.tmask + self.tmaskxm1 + self.tmaskym1 + self.tmaskxm1ym1))
    DS = div0((self.D[1,:,:]*self.tmask + self.Dxm1 + self.Dyp1 + self.Dxm1yp1),(self.tmask + self.tmaskxm1 + self.tmaskyp1 + self.tmaskxm1yp1))
    DE = self.Dxm1                 + self.ocnxm1 * self.D[1,:,:]*self.tmask
    DW = self.D[1,:,:]*self.tmask  + self.ocn    * self.Dxm1
    
    tN = -DN *        ip_(self.v[1,:,:],self.vmask)           *(jp_(self.u[1,:,:],self.umask)-self.slip*self.u[1,:,:]*self.grdNu) /self.dy
    tS =  DS *np.roll(ip_(self.v[1,:,:],self.vmask),1,axis=0) *(jm_(self.u[1,:,:],self.umask)-self.slip*self.u[1,:,:]*self.grdSu) /self.dy
    tE = -DE *        ip_(self.u[1,:,:],self.umask)           * ip_(self.u[1,:,:],self.umask)                                     /self.dx
    tW =  DW *        im_(self.u[1,:,:],self.umask)           * im_(self.u[1,:,:],self.umask)                                     /self.dx
    
    return (tN+tS+tE+tW) * self.umask

def convv(self):
    """Covnergence for Dv"""
    
    #Get D at north, south, east, west points
    DE = div0((self.D[1,:,:]*self.tmask + self.Dym1 + self.Dxm1 + self.Dxm1ym1),(self.tmask + self.tmaskym1 + self.tmaskxm1 + self.tmaskxm1ym1))
    DW = div0((self.D[1,:,:]*self.tmask + self.Dym1 + self.Dxp1 + self.Dxp1ym1),(self.tmask + self.tmaskym1 + self.tmaskxp1 + self.tmaskxp1ym1))
    DN = self.Dym1                 + self.ocnym1 * self.D[1,:,:]*self.tmask
    DS = self.D[1,:,:]*self.tmask  + self.ocn    * self.Dym1 
    
    tN = -DN *        jp_(self.v[1,:,:],self.vmask)           * jp_(self.v[1,:,:],self.vmask)                                     /self.dy
    tS =  DS *        jm_(self.v[1,:,:],self.vmask)           * jm_(self.v[1,:,:],self.vmask)                                     /self.dy
    tE = -DE *        jp_(self.u[1,:,:],self.umask)           *(ip_(self.v[1,:,:],self.vmask)-self.slip*self.v[1,:,:]*self.grdEv) /self.dx
    tW =  DW *np.roll(jp_(self.u[1,:,:],self.umask),1,axis=1) *(im_(self.v[1,:,:],self.vmask)-self.slip*self.v[1,:,:]*self.grdWv) /self.dx
    
    return (tN+tS+tE+tW) * self.vmask     

def updatesecondary(self):
    self.Ta   = np.interp(self.zb-self.D[1,:,:],self.z,self.Tz)
    self.Sa   = np.interp(self.zb-self.D[1,:,:],self.z,self.Sz)
    self.Tf   = (self.l1*self.S[1,:,:]+self.l2+self.l3*self.zb).values
    
    self.drho = (self.beta*(self.Sa-self.S[1,:,:]) - self.alpha*(self.Ta-self.T[1,:,:])) * self.tmask
    self.entr = self.E0*(np.abs(im(self.u[1,:,:])*self.dzdx + jm(self.v[1,:,:])*self.dzdy)) * self.tmask
    self.melt = self.cp/self.L*self.CG*(im(self.u[1,:,:])**2+jm(self.v[1,:,:])**2)**.5*(self.T[1,:,:]-self.Tf) * self.tmask
    
    self.Dym1    = np.roll(        self.D[1,:,:]*self.tmask,-1,axis=0)
    self.Dyp1    = np.roll(        self.D[1,:,:]*self.tmask, 1,axis=0)
    self.Dxm1    = np.roll(        self.D[1,:,:]*self.tmask,-1,axis=1)
    self.Dxp1    = np.roll(        self.D[1,:,:]*self.tmask, 1,axis=1)
    self.Dxm1ym1 = np.roll(np.roll(self.D[1,:,:]*self.tmask,-1,axis=1),-1,axis=0)
    self.Dxp1ym1 = np.roll(np.roll(self.D[1,:,:]*self.tmask, 1,axis=1),-1,axis=0)
    self.Dxm1yp1 = np.roll(np.roll(self.D[1,:,:]*self.tmask,-1,axis=1), 1,axis=0)
    
    self.vyp1    = np.roll(self.v[1,:,:],1,axis=0)  
    self.uxp1    = np.roll(self.u[1,:,:],1,axis=1)      

def intD(self,delt):
    """Integrate D"""
    self.D[2,:,:] = self.D[0,:,:] \
                    + (convT(self,self.D[1,:,:]) \
                    +  self.melt \
                    +  self.entr \
                    ) * self.tmask * delt


    
    
def intu(self,delt):
    """Integrate u"""
    self.u[2,:,:] = self.u[0,:,:] \
                    +div0((-self.u[1,:,:] * ip_t(self,(self.D[2,:,:]-self.D[0,:,:]))/(2*self.dt) \
                    +  convu(self) \
                    +  -self.g*ip_t(self,self.drho*self.D[1,:,:])*(self.Dxm1-self.D[1,:,:])/self.dx * self.tmask*self.tmaskxm1 \
                    +  self.g*ip_t(self,self.drho*self.D[1,:,:]*self.dzdx) \
                    +  -.5*self.g*ip_t(self,self.D[1,:,:])**2*(np.roll(self.drho,-1,axis=1)-self.drho)/self.dx * self.tmask * self.tmaskxm1 \
                    +  self.f*ip_t(self,self.D[1,:,:]*jm_(self.v[1,:,:],self.vmask)) \
                    +  -self.Cd* self.u[1,:,:] *(self.u[1,:,:]**2 + ip(jm(self.v[1,:,:]))**2)**.5 \
                    +  -self.Cd*self.Cdfac* np.maximum(0,np.abs(self.u[1,:,:])-self.vcut)**2*np.sign(self.u[1,:,:]) *(self.u[1,:,:]**2 + ip(jm(self.v[1,:,:]))**2)**.5 \
                    +  self.Ah*lapu(self)
                    ),ip_t(self,self.D[1,:,:])) * self.umask * delt

def intv(self,delt):
    """Integrate v"""
    self.v[2,:,:] = self.v[0,:,:] \
                    +div0((-self.v[1,:,:] * jp_t(self,(self.D[2,:,:]-self.D[0,:,:]))/(2*self.dt) \
                    + convv(self) \
                    + -self.g*jp_t(self,self.drho*self.D[1,:,:])*(self.Dym1-self.D[1,:,:])/self.dy * self.tmask*self.tmaskym1 \
                    + self.g*jp_t(self,self.drho*self.D[1,:,:]*self.dzdy) \
                    + -.5*self.g*jp_t(self,self.D[1,:,:])**2*(np.roll(self.drho,-1,axis=0)-self.drho)/self.dy * self.tmask * self.tmaskym1 \
                    + -self.f*jp_t(self,self.D[1,:,:]*im_(self.u[1,:,:],self.umask)) \
                    + -self.Cd* self.v[1,:,:] *(self.v[1,:,:]**2 + jp(im(self.u[1,:,:]))**2)**.5 \
                    +  -self.Cd*self.Cdfac* np.maximum(0,np.abs(self.v[1,:,:])-self.vcut)**2*np.sign(self.v[1,:,:]) *(self.u[1,:,:]**2 + ip(jm(self.v[1,:,:]))**2)**.5 \
                    + self.Ah*lapv(self)
                    ),jp_t(self,self.D[1,:,:])) * self.vmask * delt
    
def intT(self,delt):
    """Integrate T"""
    self.T[2,:,:] = self.T[0,:,:] \
                    +div0((-self.T[1,:,:] * (self.D[2,:,:]-self.D[0,:,:])/(2*self.dt) \
                    +  convT(self,self.D[1,:,:]*self.T[1,:,:]) \
                    +  self.entr*self.Ta \
                    +  self.melt*(self.Tf - self.L/self.cp) \
                    +  self.Kh*lapT(self,self.T[0,:,:]) \
                    ),self.D[1,:,:]) * self.tmask * delt

def intS(self,delt):
    """Integrate S"""
    self.S[2,:,:] = self.S[0,:,:] \
                    +div0((-self.S[1,:,:] * (self.D[2,:,:]-self.D[0,:,:])/(2*self.dt) \
                    +  convT(self,self.D[1,:,:]*self.S[1,:,:]) \
                    +  self.entr*self.Sa \
                    +  self.Kh*lapT(self,self.S[0,:,:]) \
                    ),self.D[1,:,:]) * self.tmask * delt

def printdiags(self):
    """Print diagnostics at given intervals as defined below"""
    #Maximum thickness
    diag0 = (self.D[1,:,:]*self.tmask).max()
    #Average thickness [m]
    diag1 = div0((self.D[1,:,:]*self.tmask*self.dx*self.dy).sum(),(self.tmask*self.dx*self.dy).sum())
    #Maximum melt rate [m/yr]
    diag2 = 3600*24*365.25*self.melt.max()
    #Average melt rate [m/yr]
    diag3 = 3600*24*365.25*div0((self.melt*self.dx*self.dy).sum(),(self.tmask*self.dx*self.dy).sum())
    #Minimum thickness
    diag4 = np.where(self.tmask==0,100,self.D[1,:,:]).min()
    #Integrated volume thickness convergence == net in/outflow [Sv]
    diag5 = 1e-6*(convT(self,self.D[1,:,:])*self.tmask*self.dx*self.dy).sum()
    #Maximum velocity [m/s]
    diag6 = ((im(self.u[1,:,:])**2 + jm(self.v[1,:,:])**2)**.5).max()
    diag7 = np.abs(self.u[1,:,:]).max()
    diag8 = np.abs(self.v[1,:,:]).max()

    print(f'{self.time[self.t]:.03f} days | D_av: {diag1:.03f}m | D_max: {diag0:.03f}m | D_min: {diag4:.03f}m | M_av: {diag3:.03f} m/yr | M_max: {diag2:.03f} m/yr | In/out: {diag5:.03f} Sv | Max. vel: {diag6:.03f} | {diag7:.03f} | {diag8:.03f} m/s')
              
    return