import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean as cmo
from numba import jit
from mpl_toolkits.axes_grid1 import make_axes_locatable

@jit(nopython=True)
def roll_ym1(X):
    return np.reshape(np.roll(np.ravel(X),-X.shape[1]),X.shape)

@jit(nopython=True)
def roll_yp1(X):
    return np.reshape(np.roll(np.ravel(X),X.shape[1]),X.shape)

@jit(nopython=True)
def roll_xm1(X):
    return np.transpose(np.reshape(np.roll(np.ravel(np.transpose(X)),-X.shape[0]),(X.shape[1],X.shape[0])))

@jit(nopython=True)
def roll_xp1(X):
    return np.transpose(np.reshape(np.roll(np.ravel(np.transpose(X)),X.shape[0]),(X.shape[1],X.shape[0])))

def create_mask(self):
        
    #Read mask input
    self.tmask = np.where(self.mask==3,1,0)
    self.grd   = np.where(self.mask==2,1,0)
    self.grd   = np.where(self.mask==1,1,self.grd)
    self.ocn   = np.where(self.mask==0,1,0)

    #Rolled masks
    self.tmaskym1    = roll_ym1(self.tmask)
    self.tmaskyp1    = roll_yp1(self.tmask)
    self.tmaskxm1    = roll_xm1(self.tmask)
    self.tmaskxp1    = roll_xp1(self.tmask)
    self.tmaskxm1ym1 = roll_xm1(roll_ym1(self.tmask))
    self.tmaskxm1yp1 = roll_xm1(roll_yp1(self.tmask))
    self.tmaskxp1ym1 = roll_xp1(roll_ym1(self.tmask))

    self.ocnym1      = roll_ym1(self.ocn)
    self.ocnyp1      = roll_yp1(self.ocn)
    self.ocnxm1      = roll_xm1(self.ocn)
    self.ocnxp1      = roll_xp1(self.ocn)
    
    self.grdNu = 1-roll_ym1((1-self.grd)*(1-roll_xm1(self.grd)))
    self.grdSu = 1-roll_yp1((1-self.grd)*(1-roll_xm1(self.grd)))
    self.grdEv = 1-roll_xm1((1-self.grd)*(1-roll_ym1(self.grd)))
    self.grdWv = 1-roll_xp1((1-self.grd)*(1-roll_ym1(self.grd)))
    
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
    self.umask = (self.tmask+self.isfW)*(1-roll_xm1(self.grlE))
    self.vmask = (self.tmask+self.isfS)*(1-roll_ym1(self.grlN))
    
    self.umaskxp1 = roll_xp1(self.umask)
    self.vmaskyp1 = roll_yp1(self.vmask)

    return

def create_grid(self):   
    #Spatial parameters
    self.nx = len(self.x)
    self.ny = len(self.y)
    self.dx = (self.x[1]-self.x[0]).values
    self.dy = (self.y[1]-self.y[0]).values
    self.xu = self.x+self.dx
    self.yv = self.y+self.dy
        
    #Temporal parameters
    self.nt = int(self.days*24*3600/self.dt)+1    # Number of time steps
    self.time = np.linspace(0,self.days,self.nt)  # Time in days 

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
    
    self.D += self.minD+.1
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

@jit(nopython=True)
def div0(a,b):
    return np.where(b>0,a/b,0)

@jit(nopython=True)
def im(var):
    """Value at i-1/2 """
    return .5*(var+roll_xp1(var))
    
@jit(nopython=True)
def ip(var):
    """Value at i+1/2"""
    return .5*(var+roll_xm1(var))

@jit(nopython=True)    
def jm(var):
    """Value at j-1/2"""
    return .5*(var+roll_yp1(var))

@jit(nopython=True)    
def jp(var):
    """Value at j+1/2"""
    return .5*(var+roll_ym1(var))

@jit(nopython=True)
def im_(var,mask):
    """Value at i-1/2, no gradient across boundary"""
    return div0((var*mask + roll_xp1(var*mask)),(mask+roll_xp1(mask)))

@jit(nopython=True)
def ip_(var,mask):
    """Value at i+1/2, no gradient across boundary"""
    return div0((var*mask + roll_xm1(var*mask)),(mask+roll_xm1(mask)))

@jit(nopython=True)
def jm_(var,mask):
    """Value at j-1/2, no gradient across boundary"""
    return div0((var*mask + roll_yp1(var*mask)),(mask+roll_yp1(mask)))

@jit(nopython=True)
def jp_(var,mask):
    """Value at j+1/2, no gradient across boundary"""
    return div0((var*mask + roll_ym1(var*mask)),(mask+roll_ym1(mask)))

def lapT(self,var):
    """Laplacian operator for DT and DS"""
    
    tN = jp_(self.D[0,:,:],self.tmask)*(roll_ym1(var)-var)*self.tmaskym1/self.dy**2
    tS = jm_(self.D[0,:,:],self.tmask)*(roll_yp1(var)-var)*self.tmaskyp1/self.dy**2
    tE = ip_(self.D[0,:,:],self.tmask)*(roll_xm1(var)-var)*self.tmaskxm1/self.dx**2
    tW = im_(self.D[0,:,:],self.tmask)*(roll_xp1(var)-var)*self.tmaskxp1/self.dx**2    
    
    return tN+tS+tE+tW

def lapu(self):
    """Laplacian operator for Du"""
    Dcent = ip_(self.D[0,:,:],self.tmask)
    var = self.u[0,:,:]

    tN = jp_(Dcent,self.tmask) * (roll_ym1(var)-var)/self.dy**2 * (1-self.ocnym1) - self.slip*Dcent*var*self.grdNu/self.dy**2
    tS = jm_(Dcent,self.tmask) * (roll_yp1(var)-var)/self.dy**2 * (1-self.ocnyp1) - self.slip*Dcent*var*self.grdSu/self.dy**2  
    tE = self.Dxm1             * (roll_xm1(var)-var)/self.dx**2 * (1-self.ocnxm1)
    tW = self.D[0,:,:]         * (roll_xp1(var)-var)/self.dx**2 * (1-self.ocn   )
    
    return (tN+tS+tE+tW) * self.umask

def lapv(self):
    """Laplacian operator for Dv"""
    Dcent = jp_(self.D[0,:,:],self.tmask)
    var = self.v[0,:,:]
    
    tN = self.Dym1             * (roll_ym1(var)-var)/self.dy**2 * (1-self.ocnym1) 
    tS = self.D[0,:,:]         * (roll_yp1(var)-var)/self.dy**2 * (1-self.ocn   )
    tE = ip_(Dcent,self.tmask) * (roll_xm1(var)-var)/self.dx**2 * (1-self.ocnxm1) - self.slip*Dcent*var*self.grdEv/self.dx**2
    tW = im_(Dcent,self.tmask) * (roll_xp1(var)-var)/self.dx**2 * (1-self.ocnxp1) - self.slip*Dcent*var*self.grdWv/self.dx**2  
    
    return (tN+tS+tE+tW) * self.vmask

def convT(self,var):
    """Upstream convergence scheme for D, DT, DS"""
    tN = - (np.maximum(self.v[1,:,:],0)*var           + np.minimum(self.v[1,:,:],0)*roll_ym1(var)) / self.dy * self.vmask
    tS =   (np.maximum(self.vyp1    ,0)*roll_yp1(var) + np.minimum(self.vyp1    ,0)*var          ) / self.dy * self.vmaskyp1
    tE = - (np.maximum(self.u[1,:,:],0)*var           + np.minimum(self.u[1,:,:],0)*roll_xm1(var)) / self.dx * self.umask
    tW =   (np.maximum(self.uxp1    ,0)*roll_xp1(var) + np.minimum(self.uxp1    ,0)*var          ) / self.dx * self.umaskxp1
    return (tN+tS+tE+tW) * self.tmask

def convTcen(self,var):
    """Centered convergence scheme for D, T, and S"""
    tN = -jp_(var,self.tmask)*self.v[1,:,:] /self.dy * self.vmask                             
    tS =  jm_(var,self.tmask)*self.vyp1     /self.dy * self.vmaskyp1
    tE = -ip_(var,self.tmask)*self.u[1,:,:] /self.dx * self.umask                             
    tW =  im_(var,self.tmask)*self.uxp1     /self.dx * self.umaskxp1
    return (tN+tS+tE+tW) * self.tmask

def convu(self):
    """Convergence for Du"""
   
    #Get D at north, south, east, west points
    DN = div0((self.D[1,:,:]*self.tmask + self.Dxm1 + self.Dym1 + self.Dxm1ym1),(self.tmask + self.tmaskxm1 + self.tmaskym1 + self.tmaskxm1ym1))
    DS = div0((self.D[1,:,:]*self.tmask + self.Dxm1 + self.Dyp1 + self.Dxm1yp1),(self.tmask + self.tmaskxm1 + self.tmaskyp1 + self.tmaskxm1yp1))
    DE = self.Dxm1                 + self.ocnxm1 * self.D[1,:,:]*self.tmask
    DW = self.D[1,:,:]*self.tmask  + self.ocn    * self.Dxm1
    
    tN = -DN *         ip_(self.v[1,:,:],self.vmask)  *(jp_(self.u[1,:,:],self.umask)-self.slip*self.u[1,:,:]*self.grdNu) /self.dy
    tS =  DS *roll_yp1(ip_(self.v[1,:,:],self.vmask)) *(jm_(self.u[1,:,:],self.umask)-self.slip*self.u[1,:,:]*self.grdSu) /self.dy
    tE = -DE *         ip_(self.u[1,:,:],self.umask)  * ip_(self.u[1,:,:],self.umask)                                     /self.dx
    tW =  DW *         im_(self.u[1,:,:],self.umask)  * im_(self.u[1,:,:],self.umask)                                     /self.dx
    
    return (tN+tS+tE+tW) * self.umask

def convv(self):
    """Covnergence for Dv"""
    
    #Get D at north, south, east, west points
    DE = div0((self.D[1,:,:]*self.tmask + self.Dym1 + self.Dxm1 + self.Dxm1ym1),(self.tmask + self.tmaskym1 + self.tmaskxm1 + self.tmaskxm1ym1))
    DW = div0((self.D[1,:,:]*self.tmask + self.Dym1 + self.Dxp1 + self.Dxp1ym1),(self.tmask + self.tmaskym1 + self.tmaskxp1 + self.tmaskxp1ym1))
    DN = self.Dym1                 + self.ocnym1 * self.D[1,:,:]*self.tmask
    DS = self.D[1,:,:]*self.tmask  + self.ocn    * self.Dym1 
    
    tN = -DN *         jp_(self.v[1,:,:],self.vmask)  * jp_(self.v[1,:,:],self.vmask)                                     /self.dy
    tS =  DS *         jm_(self.v[1,:,:],self.vmask)  * jm_(self.v[1,:,:],self.vmask)                                     /self.dy
    tE = -DE *         jp_(self.u[1,:,:],self.umask)  *(ip_(self.v[1,:,:],self.vmask)-self.slip*self.v[1,:,:]*self.grdEv) /self.dx
    tW =  DW *roll_xp1(jp_(self.u[1,:,:],self.umask)) *(im_(self.v[1,:,:],self.vmask)-self.slip*self.v[1,:,:]*self.grdWv) /self.dx
    
    return (tN+tS+tE+tW) * self.vmask     

"""Physical part below"""

def updatesecondary(self):
    self.Ta   = np.interp(self.zb-self.D[1,:,:],self.z,self.Tz)
    self.Sa   = np.interp(self.zb-self.D[1,:,:],self.z,self.Sz)
    self.Tf   = (self.l1*self.S[1,:,:]+self.l2+self.l3*self.zb).values
    
    self.drho = (self.beta*(self.Sa-self.S[1,:,:]) - self.alpha*(self.Ta-self.T[1,:,:])) * self.tmask
    self.entr = self.E0*(np.abs(im(self.u[1,:,:])*self.dzdx + jm(self.v[1,:,:])*self.dzdy)) * self.tmask
    self.melt = self.cp/self.L*self.CG*(im(self.u[1,:,:])**2+jm(self.v[1,:,:])**2)**.5*(self.T[1,:,:]-self.Tf) * self.tmask
    
    self.Dym1    = roll_ym1(         self.D[1,:,:]*self.tmask)
    self.Dyp1    = roll_yp1(         self.D[1,:,:]*self.tmask)
    self.Dxm1    = roll_xm1(         self.D[1,:,:]*self.tmask)
    self.Dxp1    = roll_xp1(         self.D[1,:,:]*self.tmask)
    self.Dxm1ym1 = roll_ym1(roll_xm1(self.D[1,:,:]*self.tmask))
    self.Dxp1ym1 = roll_ym1(roll_xp1(self.D[1,:,:]*self.tmask))
    self.Dxm1yp1 = roll_yp1(roll_xm1(self.D[1,:,:]*self.tmask))
    
    self.vyp1    = roll_yp1(self.v[1,:,:])  
    self.uxp1    = roll_xp1(self.u[1,:,:])      

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
                    +div0((-self.u[1,:,:] * ip_(self.D[2,:,:]-self.D[0,:,:],self.tmask)/(2*self.dt) \
                    +  convu(self) \
                    +  -self.g*ip_(self.drho*self.D[1,:,:],self.tmask)*(self.Dxm1-self.D[1,:,:])/self.dx * self.tmask*self.tmaskxm1 \
                    +  self.g*ip_(self.drho*self.D[1,:,:]*self.dzdx,self.tmask) \
                    +  -.5*self.g*ip_(self.D[1,:,:],self.tmask)**2*(roll_xm1(self.drho)-self.drho)/self.dx * self.tmask * self.tmaskxm1 \
                    +  self.f*ip_(self.D[1,:,:]*jm_(self.v[1,:,:],self.vmask),self.tmask) \
                    +  -self.Cd*self.u[1,:,:]*(self.u[1,:,:]**2 + ip(jm(self.v[1,:,:]))**2)**.5 \
                    +  self.Ah*lapu(self)
                    ),ip_(self.D[1,:,:],self.tmask)) * self.umask * delt

def intv(self,delt):
    """Integrate v"""
    self.v[2,:,:] = self.v[0,:,:] \
                    +div0((-self.v[1,:,:] * jp_(self.D[2,:,:]-self.D[0,:,:],self.tmask)/(2*self.dt) \
                    + convv(self) \
                    + -self.g*jp_(self.drho*self.D[1,:,:],self.tmask)*(self.Dym1-self.D[1,:,:])/self.dy * self.tmask*self.tmaskym1 \
                    + self.g*jp_(self.drho*self.D[1,:,:]*self.dzdy,self.tmask) \
                    + -.5*self.g*jp_(self.D[1,:,:],self.tmask)**2*(roll_ym1(self.drho)-self.drho)/self.dy * self.tmask * self.tmaskym1 \
                    + -self.f*jp_(self.D[1,:,:]*im_(self.u[1,:,:],self.umask),self.tmask) \
                    + -self.Cd*self.v[1,:,:]*(self.v[1,:,:]**2 + jp(im(self.u[1,:,:]))**2)**.5 \
                    + self.Ah*lapv(self)
                    ),jp_(self.D[1,:,:],self.tmask)) * self.vmask * delt
    
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

"""--------------------------"""
"""Functions for output below"""
"""--------------------------"""

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
    #Volume tendency [Sv]
    #diag5 = 1e-6*((self.D[2,:,:]-self.D[0,:,:])*self.tmask*self.dx*self.dy).sum()/2/self.dt.values
    #Integrated melt flux [Sv]
    #diag6 = 1e-6*(self.melt*self.dx*self.dy).sum().values
    #Integrated entrainment [Sv]
    #diag7 = 1e-6*(self.entr*self.dx*self.dy).sum().values
    #Integrated volume thickness convergence == net in/outflow [Sv]
    diag8 = 1e-6*(convT(self,self.D[1,:,:])*self.tmask*self.dx*self.dy).sum()
    #Maximum velocity [m/s]
    diag9 = ((im(self.u[1,:,:])**2 + jm(self.v[1,:,:])**2)**.5).max()


    print(f'{self.time[self.t]:.03f} days | D_av: {diag1:.03f}m | D_max: {diag0:.03f}m | D_min: {diag4:.03f}m | M_av: {diag3:.03f} m/yr | M_max: {diag2:.03f} m/yr | In/out: {diag8:.03f} Sv | Max. vel: {diag9:.03f} m/s')
              
    return

"""Functions for plotting below"""

def addpanel(self,dax,var,cmap,title,symm=True,stream=None,density=1,log=False):
    x = np.append(self.x.values,self.x[-1].values+self.dx)-self.dx/2
    y = np.append(self.y.values,self.y[-1].values+self.dy)-self.dy/2
    dax.pcolormesh(x,y,self.mask,cmap=plt.get_cmap('cmo.diff'),vmin=-1,vmax=3) 

    if log:
        vvar = np.ma.masked_where(var<=0,var)
        IM = dax.pcolormesh(x,y,xr.where(self.tmask,vvar,np.nan),cmap=plt.get_cmap(cmap),norm=mpl.colors.LogNorm())
    elif symm:
        IM = dax.pcolormesh(x,y,xr.where(self.tmask,var,np.nan),cmap=plt.get_cmap(cmap),vmax=np.max(np.abs(var)),vmin=-np.max(np.abs(var)))
    else:
        IM = dax.pcolormesh(x,y,xr.where(self.tmask,var,np.nan),cmap=plt.get_cmap(cmap))
          
    plt.colorbar(IM,ax=dax,orientation='horizontal')
    if stream is not None:
        spd = ((im(self.u[1,:,:]*self.umask)**2 + jm(self.v[1,:,:]*self.vmask)**2)**.5)
        lw = 3*spd/spd.max()
        strm = dax.streamplot(self.x.values,self.y.values,im(self.u[1,:,:]*self.umask),jm(self.v[1,:,:]*self.vmask),linewidth=lw,color=stream,density=density,arrowsize=.5)
                              
    dax.set_title(title)
    dax.set_aspect('equal', adjustable='box')
    return

def plotpanels(self):
    fig,ax = plt.subplots(2,4,figsize=self.figsize,sharex=True,sharey=True)            
    
    addpanel(self,ax[0,0],(self.u[1,:,:]**2+self.v[1,:,:]**2)**.5,'cmo.speed','Speed',symm=False)
    addpanel(self,ax[1,0],self.drho,'cmo.delta_r','Buoyancy')
    #addpanel(self,ax[0,0],self.Ta,'cmo.thermal','Ambient temp',symm=False)
    #addpanel(self,ax[1,0],self.Sa,'cmo.haline','Ambient saln',symm=False)
    
    addpanel(self,ax[0,1],self.D[1,:,:],'cmo.rain','Plume thickness',symm=False,log=True)
    addpanel(self,ax[1,1],self.zb,'cmo.deep_r','Ice draft',symm=False,stream='orangered')
    #addpanel(self,ax[1,1],self.dzdx,'cmo.deep_r','Draft slope',symm=False,stream=False)
        
    addpanel(self,ax[0,2],self.T[1,:,:],'cmo.thermal','Plume temperature',symm=False)          
    addpanel(self,ax[1,2],self.S[1,:,:],'cmo.haline','Plume salinity',symm=False)   
    #addpanel(self,ax[1,2],self.drho,'cmo.dense','Buoyancy',symm=False)
            
    addpanel(self,ax[0,3],3600*24*365.25*self.melt,'cmo.curl','Melt',symm=True)
    #addpanel(self,ax[0,3],3600*24*365.25*self.melt,'cmo.matter','Melt',symm=False)
    addpanel(self,ax[1,3],3600*24*365.25*self.entr,'cmo.turbid','Entraiment',symm=False)                

    plt.tight_layout()
    plt.show()

"""Functions for plotting derivaties for debugging purposes"""
    
def plotdudt(self):
    fig,ax = plt.subplots(2,5,figsize=self.figsize,sharex=True,sharey=True)            

    t1 =   -self.u[1,:,:] * ip_t(self,(self.D[2,:,:]-self.D[0,:,:]))/(2*self.dt) 
    t2 =   convu(self) 
    t3 =   -self.g*ip_t(self,self.drho*self.D[1,:,:])*(self.Dxm1-self.D[1,:,:])/self.dx * self.tmask*self.tmaskxm1 
    t4 =   self.g*ip_t(self,self.drho*self.D[1,:,:]*self.dzdx) 
    t5 =   -.5*self.g*ip_t(self,self.D[1,:,:])**2*(np.roll(self.drho,-1,axis=1)-self.drho)/self.dx * self.tmask * self.tmaskxm1 
    t6 =   self.f*ip_t(self,self.D[1,:,:]*jm_(self.v[1,:,:],self.vmask)) 
    t7 =   -self.Cd*self.u[1,:,:]*(self.u[1,:,:]**2 + ip(jm(self.v[1,:,:]))**2)**.5 
    t8 =   self.Ah*lapu(self)
    
    tt = t1+t2+t3+t4+t5+t6+t7+t8
    
    addpanel(self,ax[0,0],1e6*(self.u[1,:,:]) * self.umask,'cmo.curl','U')                                                                    
    addpanel(self,ax[1,0],1e6*div0(tt,ip_t(self,self.D[1,:,:])) * self.umask,'RdBu_r','dU/dt')
             
    addpanel(self,ax[0,1],1e6*div0(t1,ip_t(self,self.D[1,:,:])) * self.umask,'RdBu_r','dD/dt')
    addpanel(self,ax[1,1],1e6*div0(t2,ip_t(self,self.D[1,:,:])) * self.umask,'RdBu_r','conv')
             
    addpanel(self,ax[0,2],1e6*div0(t3,ip_t(self,self.D[1,:,:])) * self.umask,'RdBu_r','dD/dx')
    addpanel(self,ax[1,2],1e6*div0(t4,ip_t(self,self.D[1,:,:])) * self.umask,'RdBu_r','d z/dx')
             
    addpanel(self,ax[0,3],1e6*div0(t5,ip_t(self,self.D[1,:,:])) * self.umask,'RdBu_r','d rho/dx')       
    addpanel(self,ax[1,3],1e6*div0(t6,ip_t(self,self.D[1,:,:])) * self.umask,'RdBu_r','fV')
             
    addpanel(self,ax[0,4],1e6*div0(t7,ip_t(self,self.D[1,:,:])) * self.umask,'RdBu_r','drag')
    addpanel(self,ax[1,4],1e6*div0(t8,ip_t(self,self.D[1,:,:])) * self.umask,'RdBu_r','lap')         

    plt.tight_layout()
    plt.show()


def plotdTdt(self):
    fig,ax = plt.subplots(2,4,figsize=self.figsize,sharex=True,sharey=True)            
    
    t1 = -self.T[1,:,:] * (self.D[2,:,:]-self.D[0,:,:])/(2*self.dt)
    t2 = convT(self,self.D[1,:,:]*self.T[1,:,:]) 
    t3 = self.entr*self.Ta 
    t4 = self.melt*(self.Tf - self.L/self.cp) 
    t5 = self.Kh*lapT(self,self.T[0,:,:])
    
    tt = t1+t2+t3+t4+t5
    
    addpanel(self,ax[0,0],1e6*div0(tt,ip_t(self,self.D[1,:,:])) * self.tmask,'RdBu_r','dT/dt')
    addpanel(self,ax[1,0],1e6*div0(t1,ip_t(self,self.D[1,:,:])) * self.tmask,'RdBu_r','dD/dt')
            
    addpanel(self,ax[0,1],1e6*div0(t2,ip_t(self,self.D[1,:,:])) * self.tmask,'RdBu_r','conv')
    addpanel(self,ax[1,1],1e6*div0(t3,ip_t(self,self.D[1,:,:])) * self.tmask,'RdBu_r','entr')

    addpanel(self,ax[0,2],1e6*div0(t4,ip_t(self,self.D[1,:,:])) * self.tmask,'RdBu_r','melt')
    addpanel(self,ax[1,2],1e6*div0(t5,ip_t(self,self.D[1,:,:])) * self.tmask,'RdBu_r','lap')

    plt.tight_layout()
    plt.show()
    
def plotdSdt(self):
    fig,ax = plt.subplots(2,4,figsize=self.figsize,sharex=True,sharey=True)            
    
    t1 = -self.S[1,:,:] * (self.D[2,:,:]-self.D[0,:,:])/(2*self.dt) 
    t2 =   convT(self,self.D[1,:,:]*self.S[1,:,:]) 
    t3 =   self.entr*self.Sa 
    t4 =   self.Kh*lapT(self,self.S[0,:,:]) 
    
    tt = t1+t2+t3+t4
    
    addpanel(self,ax[0,0],1e6*div0(tt,ip_t(self,self.D[1,:,:])) * self.tmask,'RdBu_r','dS/dt')
    addpanel(self,ax[1,0],1e6*div0(t1,ip_t(self,self.D[1,:,:])) * self.tmask,'RdBu_r','dD/dt')
            
    addpanel(self,ax[0,1],1e6*div0(t2,ip_t(self,self.D[1,:,:])) * self.tmask,'RdBu_r','conv')
    addpanel(self,ax[1,1],1e6*div0(t3,ip_t(self,self.D[1,:,:])) * self.tmask,'RdBu_r','entr')

    addpanel(self,ax[0,2],1e6*div0(t4,ip_t(self,self.D[1,:,:])) * self.tmask,'RdBu_r','lap')

    plt.tight_layout()
    plt.show()
    
def plotdDdt(self):
    fig,ax = plt.subplots(2,4,figsize=self.figsize,sharex=True,sharey=True)     
    
    t1 = convT(self,self.D[1,:,:])
    t2 = self.melt 
    t3 = self.entr 
    
    tt = t1+t2+t3
    
    addpanel(self,ax[0,0],1e6*tt * self.tmask,'RdBu_r','dD/dt')
    addpanel(self,ax[1,0],1e6*t1 * self.tmask,'RdBu_r','conv')
            
    addpanel(self,ax[0,1],1e6*t2 * self.tmask,'RdBu_r','melt')
    addpanel(self,ax[1,1],1e6*t3 * self.tmask,'RdBu_r','entr')


    plt.tight_layout()
    plt.show()
    
def plotextra(self):
    fig,ax = plt.subplots(2,4,figsize=self.figsize,sharex=True,sharey=True)
    
    DN = div0((self.D[1,:,:]*self.tmask + self.Dxm1 + self.Dym1 + self.Dxm1ym1),(self.tmask + self.tmaskxm1 + self.tmaskym1 + self.tmaskxm1ym1))
    DS = div0((self.D[1,:,:]*self.tmask + self.Dxm1 + self.Dyp1 + self.Dxm1yp1),(self.tmask + self.tmaskxm1 + self.tmaskyp1 + self.tmaskxm1yp1))
    DE = self.Dxm1                 + self.ocnxm1 * self.D[1,:,:]*self.tmask
    DW = self.D[1,:,:]*self.tmask  + self.ocn    * self.Dxm1
    
    tN = -DN *        ip_(self.v[1,:,:],self.vmask)           *(jp_(self.u[1,:,:],self.umask)-self.slip*self.u[1,:,:]*self.grdNu) /self.dy
    tS =  DS *np.roll(ip_(self.v[1,:,:],self.vmask),1,axis=0) *(jm_(self.u[1,:,:],self.umask)-self.slip*self.u[1,:,:]*self.grdSu) /self.dy
    tE = -DE *        ip_(self.u[1,:,:],self.umask)           * ip_(self.u[1,:,:],self.umask)                                     /self.dx
    tW =  DW *        im_(self.u[1,:,:],self.umask)           * im_(self.u[1,:,:],self.umask)                                     /self.dx
    

    addpanel(self,ax[0,0],1e6*tN,'RdBu_r','N')
    addpanel(self,ax[1,0],1e6*tS,'RdBu_r','S')
            
    addpanel(self,ax[0,1],1e6*tE,'RdBu_r','E')
    addpanel(self,ax[1,1],1e6*tW,'RdBu_r','W')
    
def plotmelt(self,filename,figsize,density):
    fig,ax = plt.subplots(1,1,figsize=figsize)            

    ax.set_aspect('equal', adjustable='box')  
    x = np.append(self.x.values,self.x[-1].values+self.dx)-self.dx/2
    y = np.append(self.y.values,self.y[-1].values+self.dy)-self.dy/2
    xx,yy = np.meshgrid(self.x.values,self.y.values)

    ax.pcolormesh(x,y,self.mask,cmap=plt.get_cmap('cmo.diff'),vmin=-1,vmax=3)

    #cmap = plt.get_cmap('jet')
    cmap = plt.get_cmap('inferno')
    
    var = 3600*24*365.25*self.melt
    var = np.where(var>.1,var,.1)
    levs = np.power(10, np.arange(-1,2.51,.01))
    IM = ax.contourf(xx,yy,xr.where(self.tmask,var,np.nan),levs,cmap=cmap,norm=mpl.colors.LogNorm())      

    the_divider = make_axes_locatable(ax)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(IM, cax=color_axis)
    cbar.set_ticks([.1,1,10,100])
    cbar.set_ticklabels([.1,1,10,100])
    cbar.ax.tick_params(labelsize=21)
    cbar.set_label('Melt [m/yr]', fontsize=21, labelpad=-2)
    
    spd = ((im(self.u[1,:,:]*self.umask)**2 + jm(self.v[1,:,:]*self.vmask)**2)**.5)
    lw = 4*spd/spd.max()
    strm = ax.streamplot(self.x.values,self.y.values,im(self.u[1,:,:]*self.umask),jm(self.v[1,:,:]*self.vmask),linewidth=lw,color='w',density=density,arrowsize=.5)


    
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'../../results/{filename}.png')
    plt.show()
