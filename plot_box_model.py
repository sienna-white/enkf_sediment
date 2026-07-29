import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

print("TEST?")
ds = xr.open_dataset("box_model4.nc", decode_times=False)
print("opened dataset...")
print(ds)
date = pd.to_datetime("2020-07-16 20:10:01")
# ds = ds.dropna(dim="N", how="any")
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

dsm = ds.mean(dim="N")
D = ds.Ds
nf = dsm.nf.isel(time=0) 
nf = 0
nf0 = 2.1
nf = nf0 * np.exp(0.1*nf) 
ssc = dsm.ssc1.isel(time=0)

mass, mass_s, ws, density = calculate_mass(nf, D, 36)

print(mass)


print(ds)
fig = plt.figure()
ax = plt.gca() 

for i in range(0, Nens):
    s1 = ds.ssc1.isel(N=i) 
    s1 = np.nansum(s1 * mass, axis=1) * 1e6
    print(s1)
    ax.plot(ds.time, s1, '-o')

    s1 = ds.ssc2.isel(N=i) 
    s1 = np.nansum(s1 * mass, axis=1) * 1e6
    ax.plot(ds.time, s1, '-.')

ax.set_ylabel("SSC [mg/L]")
fig.savefig("TEST.png")