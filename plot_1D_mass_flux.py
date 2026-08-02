import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmocean as cmo


# /global/scratch/users/siennaw/scripts/enkf_sediment/1DModel_sediment_NOFLOCMOD.nc
dsn = xr.open_dataset("sediment_1D_NOFLOCMOD.nc", decode_times=False)
dsf = xr.open_dataset("sediment_1D_model_24.nc", decode_times=False)



print(dsn.Ds)

print("LOGRANGE:")
print(dsf.Ds.values[5], dsf.Ds.values[15])
models = {"Fixed size classes" : dsn,  "FLOCMOD" : dsf}
colors = {"Fixed size classes" : "#A59837",  "FLOCMOD" : "#7A3D5D"}
keys = list(models.keys())


# print('no floc')
# print(dsn.ssc.isel(time=0).mean(dim="Nens")) 


# print('floc:')
# print(dsf.ssc.isel(time=0).mean(dim="Nens")) 

# z = -ds.z 
# nd = len(ds.Ds)
# nt = len(ds.time) 
# nt = nt  #// 2
# Nens = len(ds.Nens)


# N = 0 

# volume = 4 / 3 * np.pi * (ds.Ds.values / 2) ** 3


# MASS 
def get_mass(ds):
    D = ds.Ds #.values
    Dp = 1e-6 
    nf = 2.2 
    rho_s = 2650
    rho_w = 1000
    mass = rho_s * (np.pi/6) * Dp**(3-nf) * D**nf
    return mass 
######

def get_mass_flux(ds):
    mass = get_mass(ds)
    mass_conc = ds.ssc * mass #* 1e3
    mass_conc = mass_conc.sum(dim="Ds")
    # print("mass_conc", mass_conc)
    mass_flux = mass_conc * ds.U
    mass_flux = mass_flux.integrate(coord="z") #.sum(dim="z")
    return mass_flux




# ######################### PLOT MASS FLUX #########################
fig = plt.figure(figsize=(8, 3))
ax = plt.gca()
plt.suptitle("Suspended sediment flux")

for key in keys:
    # print(key)
    ds = models[key]
    mass_flux = get_mass_flux(ds)
    time = ds.time.values / 3600 
    Nens = len(ds.Nens)
    color = colors[key]
    for N in range(Nens):
        var = mass_flux.isel(Nens=N).values 
        ax.plot(time, var, color=color, linewidth=3, alpha=0.1)


    var = mass_flux.mean(dim="Nens")
    ax.plot(time, var, color=color, linewidth=1, alpha=0.9, label=key)
    # print("average flux: ", var[0:6])
ax.set_ylim(dsf.time.values[0]/3600 , dsf.time.values[-1]/3600)
fig.savefig('1DModel_MASSFLUX.png')






print("OFF")
# mass = get_mass(dsn)
print(dsn.ssc.isel(time=0).mean(dim="Nens")) 

# print(mass)

print("ON")
mass = get_mass(dsf)
print(dsf.ssc.isel(time=1).mean(dim="Nens")) 
# print(mass[5], mass[15])


# ######################### PLOT MASS  #########################
fig = plt.figure(figsize=(8, 3))
ax = plt.gca()
plt.suptitle("Suspended sediment flux")

for key in keys:
    # print(key)
    ds = models[key]
    mass = get_mass(ds)
    print("")
    mass_conc = ds.ssc * mass #* 1e3

    mass_conc = mass_conc.sum(dim="Ds")
    total_mass = mass_conc.integrate(coord="z") #.sum(dim="z")


    time = ds.time.values / 3600 
    Nens = len(ds.Nens)
    color = colors[key]
    for N in range(Nens):
        var = total_mass.isel(Nens=N).values 
        ax.plot(time, var, color=color, linewidth=3, alpha=0.1)

    var = total_mass.mean(dim="Nens")
    print(key, " total mass :")
    print(var.values[0:10])
    ax.plot(time, var, color=color, linewidth=1, alpha=0.9, label=key)
    # print("average flux: ", var[0:6])
ax.set_ylim(dsf.time.values[0]/3600 , dsf.time.values[-1]/3600)
fig.savefig('1DModel_TOTAL_MASS.png')
assert(False)



fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(12, 5), sharex=False, sharey=True)
axs = axs.flatten()

times2plot = np.arange(0, 175, 40)
for key in keys:

    ds = models[key]
    mass = get_mass(ds)
    time = ds.time.values / 3600 
    Nens = len(ds.Nens)
    z = -ds.z 
    for it, t in enumerate(times2plot):
        if it>5:
            continue 
        color = colors[key] 
        axs[it].set_title("%d min" % (t/60))
        
        for N in range(Nens):
            var = ds.ssc.isel(Nens=N).sel(time=t, method='nearest')
            var = var * mass * 1e3
            var= var.sum(dim="Ds")
            h= axs[it].plot(var, z, color=color, linewidth=3, alpha=0.01)

        var = ds.ssc.sel(time=t, method='nearest')
        var = var.mean(dim='Nens')
        var = var * mass * 1e3
        var= var.sum(dim="Ds")
        h= axs[it].plot(var, z, color=color, linewidth=3, alpha=1, label=key)

    axs[0].legend()

#     Nens = len(ds.Nens)
#     color = colors[key]
#     for N in range(Nens):
#         var = mass_flux.isel(Nens=N).values 
#         ax.plot(time, var, color=color, linewidth=3, alpha=0.1)

#     var = mass_flux.mean(dim="Nens")
#     ax.plot(time, var, color=color, linewidth=1, alpha=0.9, label=key)
#     print("average flux: ", var[0:6])
# ax.set_ylim(ds.time.values[0]/3600 , ds.time.values[-1]/3600)
fig.savefig('1DModel_COMPARE_VOL.png')

   
    
    

#     for N in range(Nens):
#         var = ds.ssc.isel(time=t, Nens=N)
#         var = var * mass[i] * 1e3
#         h= axs[it].plot(np.sum(var, axis=0), z, color=color, linewidth=3, alpha=0.1)

#     var = ds.ssc.isel(time=t).mean(dim="Nens")
#     var = var * mass[i] * 1e3
#     total_volume = np.sum(var, axis=0)

#     h= axs[it].plot(total_volume, z, color=color, label="T=%d min" % (ds.time[t]/60), linewidth=2, alpha=1)

#     axs[it].grid(alpha=0.3)
#     axs[it].set_xlabel("SSC (mg/L)")



#     axs[it].set_title("t=%d min" % (ds.time.values[t]/60))
#     # axs[it].set_ylim(-4, 0)
#         # axs2[i].set_title("Ds = %.2f $\mu$m" % (ds.Ds.isel(Ds=d).values*1e6))

# axs[0].set_ylabel("Depth (m)")
# plt.tight_layout()

# fig.savefig('1DModel_VOLUME_time.png')
# # fig3.savefig('test1d2.png')

assert(False)
######################### PHYSICAL FORCINGS #########################

ds= dsf 
z = ds.z 
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7, 7), sharex=False, sharey=True)
axs = axs.flatten()

h= axs[0].pcolormesh(time, z, ds.G.values.T, vmin=0, vmax=10, cmap=cmo.cm.speed)
axs[0].set_title("Shear (1/s)")
plt.colorbar(h, label='Shear (1/s)', shrink=0.7)


h= axs[1].pcolormesh(time, z, np.log(ds.Kz.values.T), vmin=-10, vmax=-1, cmap=cmo.cm.rain)
axs[1].set_title("Diffusivity (m2/s)")
plt.colorbar(h, label=r'$\kappa$ (m$^2$/s)', shrink=0.7)

h= axs[2].pcolormesh(time, z, ds.U.values.T, vmin=-0.5, vmax=0.5, cmap=cmo.cm.balance)
axs[2].set_title("U (m/s)")
plt.colorbar(h, label='U (m/s)', shrink=0.7)
print(ds.U.values.T)

for ax in axs:
    ax.grid(alpha=0.2)
    ax.set_xlim(2.6, 12)

plt.tight_layout() 

fig.savefig("1DModel_Kappa_SHEAR_U.png")    # 
#################################################################################


# SHOW ROUSE PROFILES? 