import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from geometry import glaciers
from PICO import PicoModel, table2

def compare_PICO():
    """ compares out PICO results to the ones in original publication and observations
        - melt rates
        - model overurning
        - model ambient and box n temperatute
        - model ambient and box n salinity
    """
    renames = {'FilchnerRonne':'Filchner\nRonne', 'MoscowUniversity':'Moscow\nUniversity'}
    gnames = [renames[x] if x in renames.keys() else x for x in glaciers]

    fig, ax = plt.subplots(2, 2, figsize=(10,8), constrained_layout=True)
    ax[0,0].set_ylabel('melt rates  [m/yr]')
    ax[0,1].set_ylabel('overturning  [Sv]')
    ax[1,0].set_ylabel(r'box temperature  [$^\circ$C]')
    ax[1,1].set_ylabel('box salinity  [psu]')
    for i, glacier in enumerate(glaciers):
        c = f'C{i}'
        
        if glacier in table2.index:
            l1 = ax[0,0].scatter(i-.1, table2['m'].loc[glacier], c=c, label='original PICO', marker='s')
            l2 = ax[0,0].errorbar(x=i, y=table2['m_obs'].loc[glacier], yerr=table2['m_unc'].loc[glacier], c=c, marker='o', label='observations')
            ax[0,1].scatter(i-.1, table2['q'] .loc[glacier], c=c, marker='s')
            ax[1,0].scatter(i-.1, table2['T0'].loc[glacier], c=c, marker='s')
            ax[1,0].scatter(i-.1, table2['Tn'].loc[glacier], c=c, marker='s')
            ax[1,1].scatter(i-.1, table2['S0'].loc[glacier], c=c, marker='s')
            ax[1,1].scatter(i-.1, table2['Sn'].loc[glacier], c=c, marker='s')
        if glacier in ['Ross', 'FilchnerRonne']:  continue
        fn = PicoModel(name=glacier).fn_PICO_output
        ds = xr.open_dataset(fn)
        l3 = ax[0,0].scatter(i+.1, ds.mk[0] , c=c, marker='D', label='our PICO')
        ax[0,1].scatter(i+.1, ds.q/1e6 , c=c, marker='D')
        ax[1,0].scatter(i+.1, ds.Tk[0] , c=c, marker='D')
        ax[1,0].scatter(i+.1, ds.Tk[-1], c=c, marker='D')
        ax[1,1].scatter(i+.1, ds.Sk[0] , c=c, marker='D')
        ax[1,1].scatter(i+.1, ds.Sk[-1], c=c, marker='D')
        if i==1:
            ax[0,0].legend(handles=[l1,l2,l3])
    for i in range(2):
        for j in range(2):
            ax[j,i].set_xticks(np.arange(len(glaciers)))
            ax[j,i].set_xticklabels(gnames, rotation=45)
    return