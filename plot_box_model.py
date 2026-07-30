import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import cmocean as cmo


n = 18

print("running script ...")
ds = xr.open_dataset("box_model%d.nc" % n, decode_times=False)
print("opened dataset...")
print(ds)
date = pd.to_datetime("2020-07-16 20:10:01")
ds = ds.dropna(dim="N", how="any")
mtime = [date + pd.Timedelta(t, unit='s') for t in ds.time.values]
nt = len(ds.time) 
Nens = len(ds.N)
Ns = len(ds.Ds)
# [1] 1002943

date = pd.to_datetime("2020-07-16 20:10:01")
# ds = ds.dropna(dim="N", how="any")
mtime = [date + pd.Timedelta(t, unit='s') for t in ds.time.values]

g = 9.81            # gravitational constant [m/s^2]
nu = 1.5e-6          # visosity of water [m^2/s] 
rho_w = 1000          # density of water [kg/m^3]
rho_s = 2650          # density of sediment kg/m^3
Dp = 1e-6

def calculate_mass(nf, D, N):
  
    # calculate the mass of a floc in size class i
    # parameters:
    #   floc_density: particle density (kg/m^3)
    #   D: array of particle diameters (m)
    #   i: index of the particle size class
    #   Dp: reference diameter (m)
    #   nf: fractal dimension exponent
    # returns: mass of a particle in size class i (kg)

    density_ = np.zeros(N)
    for i in range(N):
        density_[i] = rho_w + (rho_s - rho_w) * (Dp/D[i])**(nf - 3)
     
    mass_ = np.zeros(N)
    for i in range(N): 
        mass_[i] = rho_s * np.pi/6 * Dp**3 * (D[i]/Dp)**nf

    # mass_s = rho_s 
    mass_s = np.zeros(N)
    for i in range(N): 
        mass_s[i] = rho_s * np.pi/6 * D[i]**(3-nf) * (Dp/D[i])**nf

    ws_ = np.zeros(N)
    for i in range(N):
        ws_[i] = g/(18*nu) * (density_[i] - rho_w)/(rho_w) * D[i]**2 
     
    return mass_, mass_s, ws_, density_

D = ds.Ds

nf0 = 2.1
# ssc = dsm.ssc1.isel(time=0)

print(ds)
fig, axs = plt.subplots(nrows=2, ncols=1)

for i in range(0, Nens):
    nf = ds.nf.isel(N=i)
    nf = nf0 * np.exp(0.1*nf) 
    color = cmo.cm.haline(i/Nens)

    mass, mass_s, ws, density = calculate_mass(nf, D, Ns)

    s1 = ds.ssc1.isel(N=i) 
    s1 = np.nansum(s1 * mass, axis=1) * 1e6
    s1[s1>50] = np.nan
    print("shoal:", s1)

    axs[0].plot(mtime, s1, '-o', color=color)

    s1 = ds.ssc2.isel(N=i) 
    s1 = np.nansum(s1 * mass, axis=1) * 1e6
    s1[s1>50] = np.nan
    print("channel:", s1)
    axs[1].set_title("Channel")
    axs[0].set_title("Shoal")

    label = "Nf = %2.1f, beta=%2.1f , alpha=%2.1f" % (ds.nf.isel(N=i), ds.beta.isel(N=i), ds.alpha.isel(N=i))
    axs[1].plot(mtime, s1, '-', color=color, label=label)

for ax in axs:
    ax.legend()
    ax.set_ylim(0, 140)
    ax.grid(alpha=0.2)
    ax.set_ylabel("SSC [mg/L]")


plt.tight_layout() 
fig.savefig("TEST%d.png" % n)