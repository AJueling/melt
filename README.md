# `melt` project: comparing ice shelf melt models

## Intercomparison of different models of basal melt
### Geometries and Forcing

### Example Ice Shelf: Totten Glacier
data from the Bedmachine dataset; figure created in `src/notebooks/Bedmachine.ipynb`

<img src="results/Bedmachine/TottenIS_geometry.png">

__Figure 1:__  Geometry of the Totten Ice Shelf. The data in the top row is provided by the Bedmachine dataset. Left to right: surface elevation, bed topography/bathymetry, and ice thickness. Bottom left, derived draft of the ice shelf (elevation-thickness) together with the grounding line (blue) and the ice shelf front (red). Bottom center and right: minimum distance to the grounding line/ice shelf front.

### Models of basal ice shelf melt

1. Simple: simple parametrizations based on thermal driving, i.e. difference between pressure freezing temperature and in-situ ocean temperature (e.g. Favier _et al._ ())
1. Plume: analytical approximtation of the 1D plume model (Lazeroms _et al._ (2019))
1. PICO: box model of cavity circulation (Reese _et al._ (2018))
1. PICOP: combination the two abovementioned models (Pelle _et al._ (2019))
1. 2D Sheet model

## Code structure
```
melt
│   README.md
│   LICENSE
│
└───data
│   └───BedMachine    (Antarctic topography data)
│   │   │   BedMachineAntarctica_2020-07-15_v02.nc
│   │
│   └───IceVelocity   (Radar derived surface velocities)
│   │   │   antarctic_ice_vel_phase_map_v01.nc
│   │
│   └───mask_polygons (grounding lines and ice shelf front masks)
│       │   Amery_grl.geojson, ...
│
└───doc               (documentation)
│
└───results           (images & netcdf files, mostly not committed)
│   
└───src
    └───ipynb         (notebooks with output, not committed)
    │   │   BedMachineAntarctica_2020-07-15_v02.nc
    │
    └───notebooks     (notebooks without output, version controlled)
    │   │   Geometry.ipynb
    │   │   ...
    │
    │   forcing.py        │
    │   ideal_geometry.py │  (set up model domains)
    │   real_geometry.py  │
    │   PICO.py         │
    │   PICOP.py        │
    │   Plume.py        │    (models)
    │   sheet.py        │
    │   Simple.py       │
    │   advect.py         │
    │   constants.py      │  (auxiliary functions)
    │   sheet_utils.py    │
```

### How to run models

1. choose geometry
    ```
    from ideal_geometry import IdealGeometry
    ds = IdealGeometry('test1')
    ```
2. choose forcing (temperature and salinity profiles)
    ```
    from forcing import Forcing
    pdict = dict(Tdeep=-2, ztcl=500)
    ds = Forcing(ds).tanh(pdict)
    ```
3. pass those to model of choice
    ```
    from Simple import SimpleModels
    ds = SimpleModels(ds).compute()
    ```