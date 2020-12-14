import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean as cmo

"""All functions are directly copied from sheetmodel and must be rewritten to work with general input"""

def plotmelt(self,filename,figsize,density):
    fig,ax = plt.subplots(1,1,figsize=figsize)            

    ax.set_aspect('equal', adjustable='box')  
    x = np.append(self.x.values,self.x[-1].values+self.dx)-self.dx/2
    y = np.append(self.y.values,self.y[-1].values+self.dy)-self.dy/2
    xx,yy = np.meshgrid(self.x.values,self.y.values)

    ax.pcolormesh(x,y,self.mask,cmap=plt.get_cmap('cmo.diff'),vmin=-1,vmax=3)

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
