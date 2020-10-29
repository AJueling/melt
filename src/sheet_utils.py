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
        self.nfs  = 1
    
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

def lap(self,var):
    """Laplacian operator for D"""
    tN = (var.roll(y=-1,roll_coords=False)-var)*self.tmask.roll(y=-1,roll_coords=False)/self.dy**2
    tS = (var.roll(y= 1,roll_coords=False)-var)*self.tmask.roll(y= 1,roll_coords=False)/self.dy**2
    tE = (var.roll(x=-1,roll_coords=False)-var)*self.tmask.roll(x=-1,roll_coords=False)/self.dx**2
    tW = (var.roll(x= 1,roll_coords=False)-var)*self.tmask.roll(x= 1,roll_coords=False)/self.dx**2
    
    return tN+tS+tE+tW

def lapT(self,var):
    """Laplacian operator for DT and DS"""
    Dcent = self.D[0,:,:]
    
    tN = jp(Dcent)*(var.roll(y=-1,roll_coords=False)-var)*self.tmask.roll(y=-1,roll_coords=False)/self.dy**2
    tS = jm(Dcent)*(var.roll(y= 1,roll_coords=False)-var)*self.tmask.roll(y= 1,roll_coords=False)/self.dy**2
    tE = ip(Dcent)*(var.roll(x=-1,roll_coords=False)-var)*self.tmask.roll(x=-1,roll_coords=False)/self.dx**2
    tW = im(Dcent)*(var.roll(x= 1,roll_coords=False)-var)*self.tmask.roll(x= 1,roll_coords=False)/self.dx**2    
    
    return tN+tS+tE+tW

def lapu(self,var):
    """Laplacian operator for Du"""
    Dcent = ip(self.D[0,:,:])
    
    tN = jp(Dcent)*(var.roll(y=-1,roll_coords=False)-var)*self.tmask.roll(y=-1,roll_coords=False)/self.dy**2 - self.slip*Dcent*var*ip(self.grd.roll(y=-1,roll_coords=False))/self.dy**2
    tS = jm(Dcent)*(var.roll(y= 1,roll_coords=False)-var)*self.tmask.roll(y= 1,roll_coords=False)/self.dy**2 - self.slip*Dcent*var*ip(self.grd.roll(y= 1,roll_coords=False))/self.dy**2  
    tE = self.D[0,:,:].roll(x=-1,roll_coords=False)*(var.roll(x=-1,roll_coords=False)-var)*self.tmask.roll(x=-1,roll_coords=False)/self.dx**2
    tW = self.D[0,:,:]*(var.roll(x= 1,roll_coords=False)-var)*self.tmask.roll(x= 1,roll_coords=False)/self.dx**2  
    
    return tN+tS+tE+tW

def lapv(self,var):
    """Laplacian operator for Dv"""
    Dcent = jp(self.D[0,:,:])
    
    tN = self.D[0,:,:].roll(y=-1,roll_coords=False)*(var.roll(y=-1,roll_coords=False)-var)*self.tmask.roll(y=-1,roll_coords=False)/self.dy**2 
    tS = self.D[0,:,:]*(var.roll(y= 1,roll_coords=False)-var)*self.tmask.roll(y= 1,roll_coords=False)/self.dy**2
    tE = ip(Dcent)*(var.roll(x=-1,roll_coords=False)-var)*self.tmask.roll(x=-1,roll_coords=False)/self.dx**2 - self.slip*Dcent*var*jp(self.grd.roll(x=-1,roll_coords=False))/self.dx**2
    tW = im(Dcent)*(var.roll(x= 1,roll_coords=False)-var)*self.tmask.roll(x= 1,roll_coords=False)/self.dx**2 - self.slip*Dcent*var*jp(self.grd.roll(x= 1,roll_coords=False))/self.dx**2  
    
    return tN+tS+tE+tW

def convT(self,var):
    """Convergence for D, T, and S"""
    tN = -jp(var)*self.v[1,:,:]/self.dy * self.vmask * self.tmask
    tS =  jm(var)*self.v[1,:,:].roll(y=1,roll_coords=False)/self.dy * self.vmask.roll(y=1,roll_coords=False) * self.tmask
    tE = -ip(var)*self.u[1,:,:]/self.dx * self.umask * self.tmask
    tW =  im(var)*self.u[1,:,:].roll(x=1,roll_coords=False)/self.dx * self.umask.roll(x=1,roll_coords=False) * self.tmask
    
    return tN+tS+tE+tW


def convu(self):
    """Convergence for Du"""
    tN = 0
    tS = 0
    tE = self.D[1,:,:].roll(x=-1,roll_coords=False) * ip(self.u[1,:,:])**2 / self.dx * self.tmask.roll(x=-1,roll_coords=False)
    tW = self.D[1,:,:]*im(self.u[1,:,:])**2/self.dx * self.tmask
    
    return tN+tS+tE+tW

def convv(self):
    tN = 0
    tS = 0
    tE = 0
    tW = 0
    return tN+tS+tE+tW    

def convu_g(self):
    t1 =  (self.D[1,:,:]*im(self.u[1,:,:])**2 - self.D[1,:,:].roll(x=-1,roll_coords=False)*ip(self.u[1,:,:])**2)/self.dx
    t2 =  (jm(ip(self.D[1,:,:]))*jm(self.u[1,:,:])*ip(self.v[1,:,:].roll(y=-1,roll_coords=False)) - jp(ip(self.D[1,:,:]))*jp(self.u[1,:,:])*ip(self.v[1,:,:]))/self.dy * self.vmask
    return t1+t2

def convv_g(self):
    t1 =  (im(jp(self.D[1,:,:]))*jp(self.u[1,:,:].roll(x=-1,roll_coords=False))*im(self.v[1,:,:]) - jp(ip(self.D[1,:,:]))*jp(self.u[1,:,:])*ip(self.v[1,:,:]))/self.dx * self.umask
    t2 =  (self.D[1,:,:]*jm(self.v[1,:,:])**2 - self.D[1,:,:].roll(y=-1,roll_coords=False)*jp(self.v[1,:,:])**2)/self.dy * self.vmask
    return t1+t2    


def updatevar(self,var):
    """Rearrange variable arrays at the start of each time step"""
    var[0,:,:] = var[1,:,:]
    var[1,:,:] = var[2,:,:]
    var[2,:,:] *= 0
    return

def boundaries(self):          
    """Update ghost values at boundaries to ensure zero derivatives"""
    
    self.D[1,:,:] = self.tmask*self.D[1,:,:] \
        +((  (self.grlN+self.isfN)*self.D[1,:,:].roll(y= 1,roll_coords=False) \
         +   (self.grlS+self.isfS)*self.D[1,:,:].roll(y=-1,roll_coords=False) \
         +   (self.grlE+self.isfE)*self.D[1,:,:].roll(x= 1,roll_coords=False) \
         +   (self.grlW+self.isfW)*self.D[1,:,:].roll(x=-1,roll_coords=False)) / \
        (self.grl+self.isf)).fillna(0)
    
    self.D[1,:,:] = xr.where(self.D[1,:,:]<self.minD,self.minD,self.D[1,:,:])
    
    self.T[1,:,:] = self.tmask*self.T[1,:,:] \
        +((  (self.grlN+self.isfN)*self.T[1,:,:].roll(y= 1,roll_coords=False) \
         +   (self.grlS+self.isfS)*self.T[1,:,:].roll(y=-1,roll_coords=False) \
         +   (self.grlE+self.isfE)*self.T[1,:,:].roll(x= 1,roll_coords=False) \
         +   (self.grlW+self.isfW)*self.T[1,:,:].roll(x=-1,roll_coords=False)) / \
        (self.grl+self.isf)).fillna(0)
    self.S[1,:,:] = self.tmask*self.S[1,:,:] \
        +((  (self.grlN+self.isfN)*self.S[1,:,:].roll(y= 1,roll_coords=False) \
         +   (self.grlS+self.isfS)*self.S[1,:,:].roll(y=-1,roll_coords=False) \
         +   (self.grlE+self.isfE)*self.S[1,:,:].roll(x= 1,roll_coords=False) \
         +   (self.grlW+self.isfW)*self.S[1,:,:].roll(x=-1,roll_coords=False)) / \
        (self.grl+self.isf)).fillna(0)
    
    self.u[1,:,:] *= self.umask
    self.v[1,:,:] *= self.vmask
    
    self.u[1,:,:] = xr.where(self.grlN,self.nfs*self.u[1,:,:].roll(y= 1,roll_coords=False),self.u[1,:,:])
    self.u[1,:,:] = xr.where(self.grlS,self.nfs*self.u[1,:,:].roll(y=-1,roll_coords=False),self.u[1,:,:])    
    
    self.v[1,:,:] = xr.where(self.grlE,self.nfs*self.v[1,:,:].roll(x= 1,roll_coords=False),self.v[1,:,:])
    self.v[1,:,:] = xr.where(self.grlW,self.nfs*self.v[1,:,:].roll(x=-1,roll_coords=False),self.v[1,:,:])
    
    self.u[1,:,:] = xr.where(self.isfN,self.u[1,:,:].roll(y= 1,roll_coords=False),self.u[1,:,:])
    self.u[1,:,:] = xr.where(self.isfS,self.u[1,:,:].roll(y=-1,roll_coords=False),self.u[1,:,:])        
    self.u[1,:,:] = xr.where(self.isfE,self.u[1,:,:].roll(x= 1,roll_coords=False),self.u[1,:,:])
    self.u[1,:,:] = xr.where(self.isfW,self.u[1,:,:].roll(x=-1,roll_coords=False),self.u[1,:,:])
    
    self.v[1,:,:] = xr.where(self.isfN,self.v[1,:,:].roll(y= 1,roll_coords=False),self.v[1,:,:])
    self.v[1,:,:] = xr.where(self.isfS,self.v[1,:,:].roll(y=-1,roll_coords=False),self.v[1,:,:])         
    self.v[1,:,:] = xr.where(self.isfE,self.v[1,:,:].roll(x= 1,roll_coords=False),self.v[1,:,:])
    self.v[1,:,:] = xr.where(self.isfW,self.v[1,:,:].roll(x=-1,roll_coords=False),self.v[1,:,:]) 
        
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
    
    
"""Old and unused functions below"""
    
def lapT_g(self,var):
    """Laplacian operator for DT and DS based on ghost values"""
    Dcent = self.D[0,:,:]
    
    t1 = Dcent*(var.roll(x=-1,roll_coords=False)+var.roll(x=1,roll_coords=False)-2*var)/self.dx**2
    t2 = (ip(var)-im(var))*(ip(Dcent)-im(Dcent))/self.dx**2
    t3 = Dcent*(var.roll(y=-1,roll_coords=False)+var.roll(y=1,roll_coords=False)-2*var)/self.dy**2
    t4 = (jp(var)-jm(var))*(jp(Dcent)-jm(Dcent))/self.dy**2
    
    return t1+t2+t3+t4
    
    
def lapu_g(self,var):
    """Laplacian operator for Du based on ghost values"""
    Dcent = ip(self.D[0,:,:])
    
    t1 = Dcent*(var.roll(x=-1,roll_coords=False)+var.roll(x=1,roll_coords=False)-2*var)/self.dx**2
    t2 = (ip(var)-im(var))*(ip(Dcent)-im(Dcent))/self.dx**2
    t3 = Dcent*(var.roll(y=-1,roll_coords=False)+var.roll(y=1,roll_coords=False)-2*var)/self.dy**2
    t4 = (jp(var)-jm(var))*(jp(Dcent)-jm(Dcent))/self.dy**2
    
    return t1+t2+t3+t4
    
def lapv_g(self,var):
    """Laplacian operator for Dv based on ghost values"""
    Dcent = jp(self.D[0,:,:])
    
    t1 = Dcent*(var.roll(x=-1,roll_coords=False)+var.roll(x=1,roll_coords=False)-2*var)/self.dx**2
    t2 = (ip(var)-im(var))*(ip(Dcent)-im(Dcent))/self.dx**2
    t3 = Dcent*(var.roll(y=-1,roll_coords=False)+var.roll(y=1,roll_coords=False)-2*var)/self.dy**2
    t4 = (jp(var)-jm(var))*(jp(Dcent)-jm(Dcent))/self.dy**2

    return t1+t2+t3+t4