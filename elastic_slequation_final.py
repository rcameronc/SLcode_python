import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import spharm as ps
from scipy import interpolate as interp
import cartopy.crs as ccrs
import time
import argparse


parser = argparse.ArgumentParser(description='import vars via c-line')
parser.add_argument("--maxdeg", default='256')

args = parser.parse_args()
maxdeg = args.maxdeg

## ------ Load topo, GR_mask, WAIS mask, lat and lon on GL grids ---- ##

topo_ice = pd.read_csv('topography/etopo_ice_15.txt', sep=" ", header=None, names=['lon', 'lat', 'topo'])
GR_mask = pd.read_csv('ice_masks/Greenland_mask.txt', sep=" ", header=None, names=['lon', 'lat', 'topo'])
WAIS_mask = pd.read_csv('ice_masks/WAIS_mask.txt', sep=" ", header=None, names=['lon', 'lat', 'topo'])
lat_GL = pd.read_csv('GL_grid/lat_GL256.txt', sep=" ", header=None, names=['lat'], usecols=range(0,1))
lon_GL = pd.read_csv('GL_grid/lon_GL256.txt', sep=" ", header=None, names=['lon'], usecols=range(0,1))
love_numbs = pd.read_csv('lovenumbers_ed.txt', sep='\s+', header=None, usecols=[0,1,2],
                         names=['degree', 'el_h', 'el_k', 'el_k_mult'])
# parameters
g = 9.8062      #gravity
rho_I = 934     #density of ice
rho_W = 1025    #density of water
a = 6.371e6     #earth radius (meters)
M_e = 5.976e24  #earth mass
maxdeg = 256
nlons = maxdeg*2
nlats = maxdeg

### -----  Arrange data into arrays & interpolate ------##

#import data into python
gb = topo_ice.groupby('lat')
row=[]
for key, group in gb:
    row.append(group.topo)
TOPO_ice = np.asarray(row)

gb = GR_mask.groupby('lat')
row=[]
for key, group in gb:
    row.append(group.topo)
GR_MASK = np.asarray(row)

gb = WAIS_mask.groupby('lat')
row=[]
for key, group in gb:
    row.append(group.topo)
WAIS_MASK = np.asarray(row)

# Not interpolated
lon_in = topo_ice.lon.unique() # shape (1440,)
lat_in = topo_ice.lat.unique() # shape (720,)
LON, LAT = np.meshgrid(lon_in, lat_in) # shape (720, 1440)

#already on GL grid
lon_out = lon_GL.values.squeeze() # shape (512,)
lon_out[0] = 0; lon_out[-1] = 360; # avoid white gap at edges
lat_out = lat_GL.values.squeeze() # shape (256,)
LON_out, LAT_out = np.meshgrid(lon_out, lat_out) #shapes (256, 512)
WAIS_MASK; # shape (256, 512)
GR_MASK; # shape (256, 512)

#Interpolate topo onto GL grid
topo_interp = interp.interp2d(lon_in, lat_in[::-1], TOPO_ice)
TOPO_interp = topo_interp(lon_out, lat_out)

# -------- Define Ocean Function ----- #

def Oc_func(topo):
    ocean = topo<0
    topo[ocean] = 1
    topo[~ocean] = 0
    return topo

#Filter topography with ocean function
TOPO_interp_new = TOPO_interp.copy()
topo_oc = Oc_func(TOPO_interp_new)

#slice WAIS out of topo
del_ice = TOPO_interp * GR_MASK[::-1]

#----------- Define T_l and E_l -------------#

def make_tl(maxdeg):

    #constant
    C = 4*np.pi*a**3/M_e

    mdlist = np.arange(0,maxdeg,1).tolist()
    reps = np.arange(1,maxdeg+1,1).tolist()

    l_list = np.repeat(mdlist, reps, axis=0)
    T_l = C/(2*l_list + 1)
    return T_l

def make_el(maxdeg):

    reps = np.arange(1,maxdeg+1,1).tolist() # maxdeg +2 to get Jacky's

    k_list = love_numbs.el_k[:(maxdeg)].tolist() #max deg +1 to get Jacky's
    el_k = np.repeat(k_list, reps, axis=0)

    h_list = love_numbs.el_h[:(maxdeg)].tolist() #max deg +1 to get Jacky's
    el_h = np.repeat(h_list, reps, axis=0)

    E_l = 1 + el_k - el_h
    return E_l


# reorder a list of shape {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
# to {1, 2, 4, 7, 3, 5, 8, 6, 9, 10}, the order of pyspharm's spherical harmonics.
def m_to_l_order(inpt, maxdeg):
    """
    takes as input:
    1.  inpt: a list with shape [1,2,3,...n]
    2. maximum degree (e.g. length of input list)
    """
    tri_ord = []
    for j in range(1, maxdeg+1):
        for i in range((j-1), maxdeg):
            tri_ord = np.append(tri_ord, (j + (i ** 2 + i)//2))
    tri_ord[:] = [x - 1 for x in tri_ord]

    return np.asarray([inpt[x] for x in tri_ord.astype(int)])

T_l = make_tl(maxdeg)
T_l = m_to_l_order(T_l, maxdeg)

E_l=make_el(maxdeg)
E_l = m_to_l_order(E_l, maxdeg)

### ------------- Initial guess : ice melt distributed evenly -----##

sph = ps.Spharmt(nlons, nlats, rsphere=a, gridtype='gaussian', legfunc='stored')

# Topo_gl into Spherical harmonics
del_ice_sh = sph.grdtospec(del_ice)

# Oc function of topo into Spherical harmonics
topo_oc_sh = sph.grdtospec(topo_oc)

#initial guess
del_S = (rho_I/rho_W) * np.real(del_ice_sh[0]/topo_oc_sh[0]) * topo_oc
del_S_sh = sph.grdtospec(del_S)

# ------------ Iterate through SL equation -----------#

#define max iterations
i_max = 10
i=0

#define convergence criterion chi
delta = 10**-5
chi = delta * 3

start = time.time()
while (chi >= delta) and (i<i_max):


    #non-uniform part of SL change
    del_SLcurl_sh = T_l*E_l*(rho_I*del_ice_sh + rho_W*del_S_sh)
    del_SLcurl = sph.spectogrd(del_SLcurl_sh)

    #change in radial position of surface
    del_RO = del_SLcurl * topo_oc
    del_RO_sh = sph.grdtospec(del_RO)

    #uniform perturbation to grav equipotential surface
    delPhi_g = np.real(- rho_I/rho_W * del_ice_sh[0]/topo_oc_sh[0]
                       - del_RO_sh[0]/topo_oc_sh[0])

    #Change in sea surface height
    del_S_new = (del_SLcurl + delPhi_g) * topo_oc
    del_S_sh_new = sph.grdtospec(del_S_new)

    #define convergence criterion
    chi = np.abs((np.sum(np.abs(del_S_sh_new))
                  - np.sum(np.abs(del_S_sh)))/np.sum(np.abs(del_S_sh)))

    del_S_sh = del_S_sh_new

    print('chi is', chi)
    print(i)
    i += 1

end = time.time()
print ("Time elapsed", end - start)

del_SL = del_SLcurl + delPhi_g

#----------- normalize fingerprint & plot -----------#

#normalize fingerprint
SLchange = del_SL/np.real(del_S_sh_new[0])

#plot SL fingerprint
fig = plt.figure(figsize=(20, 16))
ax1 = plt.subplot(projection=ccrs.Robinson())
ax1.coastlines()

levs = np.linspace(-4.8,2.4, 28)

ax1 = plt.contourf(lon_out, lat_out, SLchange, levels=levs,
                   transform=ccrs.PlateCarree(central_longitude=0), cmap='coolwarm_r',
                   extend='both')

cbar = plt.colorbar(ax1, shrink=.4)
cbar.ax.set_ylabel('RSL (m)', rotation=90, fontsize=17);
cbar.ax.tick_params(labelsize=15)

plt.title('SL fingerprint of GRIS collapse', fontsize=25);

plt.show()
