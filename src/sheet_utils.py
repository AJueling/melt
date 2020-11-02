import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean as cmo

def create_mask(self):
        
    #Read mask input
    self.tmask = xr.where(self.mask==3,1,0)
    self.grd   = xr.where(self.mask==2,1,0)
    self.grd   = xr.where(self.mask==1,1,self.grd)
    self.ocn   = xr.where(self.mask==0,1,0)
    #assert zeros at all edges x 0,-1 and y 0,-1
    
    #Extract ice shelf front
    self.isfE = self.ocn*self.tmask.roll(x= 1,roll_coords=False)
    self.isfN = self.ocn*self.tmask.roll(y= 1,roll_coords=False)
    self.isfW = self.ocn*self.tmask.roll(x=-1,roll_coords=False)
    self.isfS = self.ocn*self.tmask.roll(y=-1,roll_coords=False)
    self.isf  = self.isfE+self.isfN+self.isfW+self.isfS
    
    #Extract grounding line 
    self.grlE = self.grd*self.tmask.roll(x= 1,roll_coords=False)
    self.grlN = self.grd*self.tmask.roll(y= 1,roll_coords=False)
    self.grlW = self.grd*self.tmask.roll(x=-1,roll_coords=False)
    self.grlS = self.grd*self.tmask.roll(y=-1,roll_coords=False)
    self.grl  = self.grlE+self.grlN+self.grlW+self.grlS
    
    #Create masks for U- and V- velocities at N/E faces of grid points
    self.umask = (self.tmask+self.isfW)*(1-self.grlE.roll(x=-1,roll_coords=False))
    self.vmask = (self.tmask+self.isfS)*(1-self.grlN.roll(y=-1,roll_coords=False))
    
    return

def create_grid(self):   
    #Spatial parameters
    self.nx = len(self.x)
    self.ny = len(self.y)
    self.dx = self.x[1]-self.x[0]
    self.dy = self.y[1]-self.y[0]
    self.xu = self.x+self.dx
    self.yv = self.y+self.dy
        
    #Temporal parameters
    self.Ah = .5*self.maxvel*self.dx.values                          # Laplacian diffusivity, equal for U,V,T,S
    self.dt = min(self.dx/2/self.maxvel,self.dx**2/self.Ah/8).values # Time step in seconds
    self.nt = int(self.days*24*3600/self.dt)+1                       # Number of time steps
    self.time = np.linspace(0,self.days,self.nt)                     # Time in days 
    
    #Other parameters
    self.minD = .01                                                  # Minimum value for thickness
    
    if (len(self.y)==3 or len(self.x)==3):
        print('1D run, using free slip')
        self.slip = 0                                                # Assure free-slip is used in 1D simulation
    
    print(f'Ah: {self.Ah:.0f} m2/s  | dt: {self.dt:.0f} sec | nt: {self.nt} steps')
    return

def initialize_vars(self):
    #Major variables. Three arrays for storage of previous timestep, current timestep, and next timestep
    self.u = xr.DataArray(np.zeros((3,self.ny,self.nx)),dims=['n','y','x'],coords={'y':self.y,'x':self.x},name='U')
    self.v = xr.DataArray(np.zeros((3,self.ny,self.nx)),dims=['n','y','x'],coords={'y':self.y,'x':self.x},name='V')
    self.D = xr.DataArray(np.zeros((3,self.ny,self.nx)),dims=['n','y','x'],coords={'y':self.y,'x':self.x},name='D')
    self.T = xr.DataArray(np.zeros((3,self.ny,self.nx)),dims=['n','y','x'],coords={'y':self.y,'x':self.x},name='T')
    self.S = xr.DataArray(np.zeros((3,self.ny,self.nx)),dims=['n','y','x'],coords={'y':self.y,'x':self.x},name='S')

    #Draft dz/dx and dz/dy on t-grid
    self.dzdx = ddxT_e(self,self.zb)
    self.dzdy = ddyT_e(self,self.zb)
    
    #Local freezing point [degC]
    self.Tf = self.l1*self.Sa+self.l2+self.l3*self.zb
        
    #Initial values
    self.D += 1
    self.T += self.Tf 
    self.S += 30 
    
    #Perform first integration step with 1 dt
    self.D[2,:,:] = self.D[0,:,:] + self.dt * self.rhsD()
    self.u[2,:,:] = self.u[0,:,:] + self.dt * self.rhsu()
    self.v[2,:,:] = self.v[0,:,:] + self.dt * self.rhsv()
    self.T[2,:,:] = self.T[0,:,:] + self.dt * self.rhsT()
    self.S[2,:,:] = self.S[0,:,:] + self.dt * self.rhsS()       
    
    return

def ddxT_e(self,var):
    """Computes d/dx at tgrid, extrapolating gradients outside valid area. Specific for gradient in draft at boundaries """
    t1 = (var.roll(x=-1,roll_coords=False) - var)*self.tmask.roll(x=-1,roll_coords=False)
    t2 = (var - var.roll(x= 1,roll_coords=False))*self.tmask.roll(x= 1,roll_coords=False)
    return ((t1+t2)/((self.tmask.roll(x=-1,roll_coords=False)+self.tmask.roll(x=1,roll_coords=False))*self.dx)).fillna(0)# * self.tmask

def ddyT_e(self,var):
    """Computes d/dy at tgrid, extrapolating gradients outside valid area. Specific for gradient in draft at boundaries """
    t1 = (var.roll(y=-1,roll_coords=False) - var)*self.tmask.roll(y=-1,roll_coords=False)
    t2 = (var - var.roll(y= 1,roll_coords=False))*self.tmask.roll(y= 1,roll_coords=False)
    return ((t1+t2)/((self.tmask.roll(y=-1,roll_coords=False)+self.tmask.roll(y=1,roll_coords=False))*self.dy)).fillna(0)# * self.tmask

def im(var):
    """Value at i-1/2 """
    return .5*(var+var.roll(x=1,roll_coords=False))
    
def ip(var):
    """Value at i+1/2"""
    return .5*(var+var.roll(x=-1,roll_coords=False))
    
def jm(var):
    """Value at j-1/2"""
    return .5*(var+var.roll(y=1,roll_coords=False))
    
def jp(var):
    """Value at j+1/2"""
    return .5*(var+var.roll(y=-1,roll_coords=False))

def im_(var,mask):
    """Value at i-1/2, no gradient across boundary"""
    return ((var*mask + (var*mask).roll(x= 1,roll_coords=False))/(mask+mask.roll(x= 1,roll_coords=False))).fillna(0)

def ip_(var,mask):
    """Value at i+1/2, no gradient across boundary"""
    return ((var*mask + (var*mask).roll(x=-1,roll_coords=False))/(mask+mask.roll(x=-1,roll_coords=False))).fillna(0)

def jm_(var,mask):
    """Value at j-1/2, no gradient across boundary"""
    return ((var*mask + (var*mask).roll(y= 1,roll_coords=False))/(mask+mask.roll(y= 1,roll_coords=False))).fillna(0)

def jp_(var,mask):
    """Value at j+1/2, no gradient across boundary"""
    return ((var*mask + (var*mask).roll(y=-1,roll_coords=False))/(mask+mask.roll(y=-1,roll_coords=False))).fillna(0)

def lap(self):
    """Laplacian operator for D"""
    var = self.D[0,:,:]
    
    tN = (var.roll(y=-1,roll_coords=False)-var)*self.tmask.roll(y=-1,roll_coords=False)/self.dy**2
    tS = (var.roll(y= 1,roll_coords=False)-var)*self.tmask.roll(y= 1,roll_coords=False)/self.dy**2
    tE = (var.roll(x=-1,roll_coords=False)-var)*self.tmask.roll(x=-1,roll_coords=False)/self.dx**2
    tW = (var.roll(x= 1,roll_coords=False)-var)*self.tmask.roll(x= 1,roll_coords=False)/self.dx**2
    
    return tN+tS+tE+tW

def lapT(self,var):
    """Laplacian operator for DT and DS"""
    Dcent = self.D[0,:,:]
    
    tN = jp_(Dcent,self.tmask)*(var.roll(y=-1,roll_coords=False)-var)*self.tmask.roll(y=-1,roll_coords=False)/self.dy**2
    tS = jm_(Dcent,self.tmask)*(var.roll(y= 1,roll_coords=False)-var)*self.tmask.roll(y= 1,roll_coords=False)/self.dy**2
    tE = ip_(Dcent,self.tmask)*(var.roll(x=-1,roll_coords=False)-var)*self.tmask.roll(x=-1,roll_coords=False)/self.dx**2
    tW = im_(Dcent,self.tmask)*(var.roll(x= 1,roll_coords=False)-var)*self.tmask.roll(x= 1,roll_coords=False)/self.dx**2    
    
    return tN+tS+tE+tW

def lapu(self):
    """Laplacian operator for Du"""
    Dcent = ip_(self.D[0,:,:],self.tmask)
    var = self.u[0,:,:]
    
    tN = jp_(Dcent,self.tmask)*(var.roll(y=-1,roll_coords=False)-var)/self.dy**2 * (1-self.ocn).roll(y=-1,roll_coords=False) - self.slip*Dcent*var*ip(self.grd.roll(y=-1,roll_coords=False))/self.dy**2
    tS = jm_(Dcent,self.tmask)*(var.roll(y= 1,roll_coords=False)-var)/self.dy**2 * (1-self.ocn).roll(y= 1,roll_coords=False) - self.slip*Dcent*var*ip(self.grd.roll(y= 1,roll_coords=False))/self.dy**2  
    tE = self.D[0,:,:].roll(x=-1,roll_coords=False)   *(var.roll(x=-1,roll_coords=False)-var)/self.dx**2 * (1-self.ocn).roll(x=-1,roll_coords=False)
    tW = self.D[0,:,:]                                *(var.roll(x= 1,roll_coords=False)-var)/self.dx**2 * (1-self.ocn)
    return (tN+tS+tE+tW) * self.umask

def lapv(self):
    """Laplacian operator for Dv"""
    Dcent = jp_(self.D[0,:,:],self.tmask)
    var = self.v[0,:,:]
    
    tN = self.D[0,:,:].roll(y=-1,roll_coords=False)   *(var.roll(y=-1,roll_coords=False)-var)/self.dy**2 * (1-self.ocn).roll(y=-1,roll_coords=False) 
    tS = self.D[0,:,:]                                *(var.roll(y= 1,roll_coords=False)-var)/self.dy**2 * (1-self.ocn)
    tE = ip_(Dcent,self.tmask)*(var.roll(x=-1,roll_coords=False)-var)/self.dx**2 * (1-self.ocn).roll(x=-1,roll_coords=False) - self.slip*Dcent*var*jp(self.grd.roll(x=-1,roll_coords=False))/self.dx**2
    tW = im_(Dcent,self.tmask)*(var.roll(x= 1,roll_coords=False)-var)/self.dx**2 * (1-self.ocn).roll(x= 1,roll_coords=False) - self.slip*Dcent*var*jp(self.grd.roll(x= 1,roll_coords=False))/self.dx**2  
    return (tN+tS+tE+tW) * self.vmask

def convT(self,var):
    """Convergence for D, T, and S"""
    tN = -jp_(var,self.tmask)*self.v[1,:,:]                            /self.dy * self.vmask                             
    tS =  jm_(var,self.tmask)*self.v[1,:,:].roll(y=1,roll_coords=False)/self.dy * self.vmask.roll(y=1,roll_coords=False) 
    tE = -ip_(var,self.tmask)*self.u[1,:,:]                            /self.dx * self.umask                             
    tW =  im_(var,self.tmask)*self.u[1,:,:].roll(x=1,roll_coords=False)/self.dx * self.umask.roll(x=1,roll_coords=False)
    return (tN+tS+tE+tW) * self.tmask

def convu(self):
    """Convergence for Du"""
    DD = self.D[1,:,:]*self.tmask
    mm = self.tmask
    #Get D at north and south points (average of 4 values, weighted by mask, so assuming zero gradient across boundaries)
    DN = ((DD + DD.roll(x=-1,roll_coords=False) + DD.roll(y=-1,roll_coords=False) + DD.roll(x=-1,roll_coords=False).roll(y=-1,roll_coords=False))\
          /(mm + mm.roll(x=-1,roll_coords=False) + mm.roll(y=-1,roll_coords=False) + mm.roll(x=-1,roll_coords=False).roll(y=-1,roll_coords=False))).fillna(0)
    DS = ((DD + DD.roll(x=-1,roll_coords=False) + DD.roll(y= 1,roll_coords=False) + DD.roll(x=-1,roll_coords=False).roll(y= 1,roll_coords=False))\
          /(mm + mm.roll(x=-1,roll_coords=False) + mm.roll(y= 1,roll_coords=False) + mm.roll(x=-1,roll_coords=False).roll(y= 1,roll_coords=False))).fillna(0)
    
    tN = -DN                              *jp_(self.u[1,:,:],self.umask) *ip(self.v[1,:,:])                             /self.dy #* self.umask.roll(y=-1,roll_coords=False)
    tS =  DS                              *jm_(self.u[1,:,:],self.umask) *ip(self.v[1,:,:]).roll(y=1,roll_coords=False) /self.dy #* self.umask.roll(y= 1,roll_coords=False)
    tE = -DD.roll(x=-1,roll_coords=False) *ip_(self.u[1,:,:],self.umask) *ip(self.u[1,:,:])                             /self.dx #* self.umask.roll(x=-1,roll_coords=False)
    tW =  DD                              *im_(self.u[1,:,:],self.umask) *im(self.u[1,:,:])                             /self.dx #* self.umask.roll(x= 1,roll_coords=False)
    return (tN+tS+tE+tW) * self.umask

def convv(self):
    """Covnergence for Dv"""
    DD = self.D[1,:,:]*self.tmask
    mm = self.tmask
    #Similar D at east and west points
    DE = ((DD + DD.roll(x=-1,roll_coords=False) + DD.roll(y=-1,roll_coords=False) + DD.roll(x=-1,roll_coords=False).roll(y=-1,roll_coords=False))\
          /(mm + mm.roll(x=-1,roll_coords=False) + mm.roll(y=-1,roll_coords=False) + mm.roll(x=-1,roll_coords=False).roll(y=-1,roll_coords=False))).fillna(0)
    DW = ((DD + DD.roll(x= 1,roll_coords=False) + DD.roll(y=-1,roll_coords=False) + DD.roll(x= 1,roll_coords=False).roll(y= 1,roll_coords=False))\
          /(mm + mm.roll(x= 1,roll_coords=False) + mm.roll(y=-1,roll_coords=False) + mm.roll(x= 1,roll_coords=False).roll(y=-1,roll_coords=False))).fillna(0)
    
    tN = -DD.roll(y=-1,roll_coords=False)  *jp_(self.v[1,:,:],self.vmask) *jp(self.v[1,:,:])                             /self.dy #* self.vmask.roll(y=-1,roll_coords=False)
    tS =  DD                               *jm_(self.v[1,:,:],self.vmask) *jm(self.v[1,:,:])                             /self.dy #* self.vmask.roll(y= 1,roll_coords=False)
    tE = -DE                               *ip_(self.v[1,:,:],self.vmask) *jp(self.u[1,:,:])                             /self.dx #* self.vmask.roll(x=-1,roll_coords=False)
    tW =  DW                               *im_(self.v[1,:,:],self.vmask) *jp(self.u[1,:,:]).roll(x=1,roll_coords=False) /self.dx #* self.vmask.roll(x= 1,roll_coords=False)
    return (tN+tS+tE+tW) * self.vmask     

def updatevar(self,var):
    """Rearrange variable arrays at the start of each time step"""
    var[0,:,:] = var[1,:,:]
    var[1,:,:] = var[2,:,:]
    var[2,:,:] *= 0
    return

"""Functions for plotting below"""

def addpanel(self,dax,var,cmap,title,symm=True,stream=False):
    x = np.append(self.x.values,self.x[-1].values+self.dx.values)-self.dx.values/2
    y = np.append(self.y.values,self.y[-1].values+self.dy.values)-self.dy.values/2
    dax.pcolormesh(x,y,self.mask,cmap=plt.get_cmap('cmo.diff'),vmin=-1,vmax=3.5) 

    if symm:
        IM = dax.pcolormesh(x,y,xr.where(self.tmask,var,np.nan).values,cmap=plt.get_cmap(cmap),vmax=np.max(np.abs(var)),vmin=-np.max(np.abs(var)))
        #IM = dax.pcolormesh(x,y,var.values,cmap=plt.get_cmap(cmap),vmax=np.max(np.abs(var)),vmin=-np.max(np.abs(var)))
    else:
        IM = dax.pcolormesh(x,y,xr.where(self.tmask,var,np.nan).values,cmap=plt.get_cmap(cmap))
        #IM = dax.pcolormesh(x,y,var.values,cmap=plt.get_cmap(cmap))
          
    plt.colorbar(IM,ax=dax,orientation='horizontal')
    if stream:
        spd = ((im(self.u[1,:,:]*self.umask)**2 + jm(self.v[1,:,:]*self.vmask)**2)**.5).values
        lw = 3*spd/spd.max()
        strm = dax.streamplot(self.x.values,self.y.values,im(self.u[1,:,:]*self.umask).values,jm(self.v[1,:,:]*self.vmask).values,linewidth=lw,color='w')
                              
    dax.set_title(title)
    dax.set_aspect('equal', adjustable='box')
    return

def plotpanels(self):
    fig,ax = plt.subplots(2,4,figsize=(15,8),sharex=True,sharey=True)            
    
    #pltvar = self.rhsu()
    #pltvar2 = -.5*self.g*((self.drho()*self.D[1,:,:]**2).roll(x=-1,roll_coords=False) - self.drho()*self.D[1,:,:]**2)/self.dx * self.tmask
    #addpanel(self,ax[0,0],pltvar ,'cmo.curl','rhsu')
    #addpanel(self,ax[1,0],pltvar2,'cmo.curl','Diff u')
    addpanel(self,ax[0,0],self.u[1,:,:],'cmo.curl','U velocity')
    addpanel(self,ax[1,0],self.v[1,:,:],'cmo.curl','V velocity')
            
    addpanel(self,ax[0,1],self.D[1,:,:],'cmo.rain','Plume thickness',symm=False)
    addpanel(self,ax[1,1],self.zb,'cmo.deep_r','Ice draft',symm=False,stream=True)
            
    addpanel(self,ax[0,2],self.T[1,:,:],'cmo.thermal','Plume temperature',symm=False)          
    addpanel(self,ax[1,2],self.S[1,:,:],'cmo.haline','Plume salinity',symm=False)              
            
    addpanel(self,ax[0,3],3600*24*365.25*self.melt(),'cmo.matter','Melt',symm=False)
    addpanel(self,ax[1,3],3600*24*365.25*self.entr(),'cmo.turbid','Entraiment',symm=False)                

    plt.tight_layout()
    plt.show()