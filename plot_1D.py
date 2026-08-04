import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmocean as cmo


# /global/scratch/users/siennaw/scripts/enkf_sediment/1DModel_sediment_NOFLOCMOD.nc
ds = xr.open_dataset("sediment_1D_model_35_slow_NEWMC.nc", decode_times=False)


z = ds.z 

nd = len(ds.Ds)
nt = len(ds.time) 
nt = nt  #// 2
Nens = len(ds.Nens)



fig3 = plt.figure(figsize=(2, 5))
ax3 = plt.gca() 

N = 0 

volume = 4 / 3 * np.pi * (ds.Ds.values / 2) ** 3
D = ds.Ds #.values

# MASS 
Dp = 1e-6 
nf = 2.2 
rho_s = 2650
rho_w = 1000
mass = rho_s * (np.pi/6) * Dp**(3-nf) * D**nf
######



# FLUX 
fig = plt.figure(figsize=(8, 3))
ax = plt.gca()
plt.suptitle("Suspended sediment flux")

mass_conc = ds.ssc * mass  #* 1e3
mass_conc = mass_conc.sum(dim="Ds")
mass_flux = mass_conc * ds.U
mass_flux = mass_flux.sum(dim="z")

time = ds.time.values / 3600 

color = 'skyblue'
for N in range(Nens):
    var = mass_flux.isel(Nens=N).values 
    ax.plot(time, var, color=color, linewidth=3, alpha=0.1)

var = mass_flux.mean(dim="Nens")
ax.plot(time, var, color='k', linewidth=1, alpha=0.9)
ax.grid(alpha=0.2)
fig.savefig('1DModel_MASSFLUX.png')



    # h= axs[i].plot(var, z, color=color, label="T=%d min" % (ds.time[t].values/60), linewidth=2, alpha=1)

#     axs[i].grid(alpha=0.3)
#     axs[i].set_xlabel("SSC (uL/L)")

#     axs[i].set_title(r"Ds = %.2f $\mu$m" % (ds.Ds.isel(Ds=d).values*1e6))
#     # axs[i].set_ylim(-4, 0)
#         # axs2[i].set_title("Ds = %.2f $\mu$m" % (ds.Ds.isel(Ds=d).values*1e6))


# axs[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0. )
# # axs2[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0. )
# axs[0].set_ylabel("Depth (m)")


# plt.tight_layout()
# fig.savefig('1DModel_Volume_Ds.png')
# # fig3.savefig('test1d2.png')




#####
print(mass)


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(7, 5), sharex=False, sharey=True)
axs = axs.flatten()

for t in range(0, nt, nt//5):
    color = cmo.cm.thermal(t/nt) 
    i = 0 
    h= axs[i].plot(ds.G.isel(time=t), z, color=color, label="T=%d min" % (t*10/60), linewidth=2, alpha=1)
    axs[i].grid(alpha=0.3)
    axs[i].set_xlabel("Shear (1/s)")
    # axs[i].set_ylim(-4, 0)

    i = 1
    h= axs[i].plot(ds.Kz.isel(time=t), z, color=color, label="T=%d min" % (t*10/60), linewidth=2, alpha=1)
    axs[i].grid(alpha=0.3)
    axs[i].set_xlabel(r"$\kappa$ (m$^2$/s)")


    i = 2
    h= axs[i].plot(ds.U.isel(time=t), z, color=color, label="T=%d min" % (t*10/60), linewidth=2, alpha=1)
    axs[i].grid(alpha=0.3)
    axs[i].set_xlabel("U (m/s)")

    axs[i].set_ylim(0, 4)

axs[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0. )
axs[0].set_ylabel("Depth (m)")
plt.tight_layout()
fig.savefig('1DModel_diff+shear.png')





fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(12, 5), sharex=False, sharey=True)
axs = axs.flatten()
plt.suptitle("FlocMod on <-> advection/diff off")
for i, d in enumerate(range(0, nd, 6)):


    for t in range(0, nt, nt//5):
        color = cmo.cm.thermal(t/nt) 
        print("Time = min", ds.time[t].values/60)

        for N in range(Nens):
            var = ds.ssc.isel(time=t, Nens=N, Ds=d)
            var = var * mass[i] * 1e3
            h= axs[i].plot(var, z, color=color, linewidth=3, alpha=0.1)
        var = ds.ssc.isel(time=t, Ds=d).mean(dim="Nens")
        var = var * mass[i] * 1e3
        h= axs[i].plot(var, z, color=color, label="T=%d min" % (ds.time[t].values/60), linewidth=2, alpha=1)

    axs[i].grid(alpha=0.3)
    axs[i].set_xlabel("SSC (uL/L)")

    axs[i].set_title(r"Ds = %.2f $\mu$m" % (ds.Ds.isel(Ds=d).values*1e6))
    # axs[i].set_ylim(-4, 0)
        # axs2[i].set_title("Ds = %.2f $\mu$m" % (ds.Ds.isel(Ds=d).values*1e6))


axs[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0. )
# axs2[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0. )
axs[0].set_ylabel("Depth (m)")


plt.tight_layout()
fig.savefig('1DModel_Volume_Ds.png')
# fig3.savefig('test1d2.png')







fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(12, 5), sharex=False, sharey=True)
axs = axs.flatten()
# plt.suptitle("FlocMod on <-> advection/diff off")
    
for it, t in enumerate(range(0, nt, nt//5)):

    if it>5:
        continue
    
    color = cmo.cm.thermal(t/nt) 

    for N in range(Nens):
        var = ds.ssc.isel(time=t, Nens=N)
        var = var * mass[i] * 1e3
        h= axs[it].plot(var.sum(dim="Ds"), z, color=color, linewidth=3, alpha=0.1)

    var = ds.ssc.isel(time=t).mean(dim="Nens")
    var = var * mass[i] * 1e3
    total_volume = var.sum(dim="Ds") # np.sum(var, axis=0)

    h= axs[it].plot(total_volume, z, color=color, label="T=%d min" % (ds.time[t]/60), linewidth=2, alpha=1)
    h= axs[0].plot(total_volume, z, color=color, label="T=%d min" % (ds.time[t]/60), linewidth=2, alpha=1)

    axs[it].grid(alpha=0.3)
    axs[it].set_xlabel("SSC (mg/L)")

    axs[it].set_title("t=%d min" % (ds.time.values[t]/60))
    axs[it].set_ylim(0, 4)
        # axs2[i].set_title("Ds = %.2f $\mu$m" % (ds.Ds.isel(Ds=d).values*1e6))

axs[0].set_ylabel("Depth (m)")
plt.tight_layout()

fig.savefig('1DModel_SSC(z).png')
# fig3.savefig('test1d2.png')




#*#*#*#*#*#*##*#*#*#*#*#*##*#*#*#*#*#*##*#*#*#*#*#*#

fig = plt.figure(figsize=(8,3))
ax = plt.gca() 


for i,t in enumerate(range(0, nt, nt//5)):
    color = cmo.cm.thermal(t/nt)

    for N in range(Nens):
        var = ds.ssc.isel(Nens=N, time=t) * mass * 1e3
        var = var.mean(dim='z') 
        axs[i].plot(ds.Ds.values*1e6, var, '-', color=color, alpha=0.2, linewidth=3) 
    
    var = ds.ssc.isel(time=t)
    var = var.mean(dim='z') 
    var = var.mean(dim='Nens') * mass * 1e3
    ax.plot(ds.Ds.values*1e6, var, '-', color=color, linewidth=1, label="t=%d min" % (ds.time.values[t]/60)) 
    
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3)
ax.set_title("T=%d min" % (ds.time.values[t]/60))
ax.set_xlabel("Sediment grain size (µm)")
ax.legend()

plt.tight_layout()
fig.savefig("1DModel_SedDist(t).png" )



#*#*#*#*#*#*##*#*#*#*#*#*##*#*#*#*#*#*##*#*#*#*#*#*#

for dep in range(0, 50, 15):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), sharex=True, sharey=False)
    axs = axs.flatten()

    # z = z.values

    for i,t in enumerate(range(0, nt, nt//5)):
        # var = ds.Kz.isel(time=12)

        if i>5:
            continue 


        for N in range(Nens):
            var = ds.ssc.isel(Nens=N, z=dep, time=t) * mass * 1e3
            axs[i].plot(ds.Ds.values*1e6, var, '-', color=cmo.cm.haline(z[dep]/10), alpha=0.6, linewidth=3) 
        
        var = ds.ssc.isel(z=dep, time=t)
        var = var.mean(dim='Nens') * mass * 1e3
        axs[i].plot(ds.Ds.values*1e6, var, 'k-') 
        
        axs[i].set_xscale('log')
        axs[i].grid(alpha=0.3)
        axs[i].set_title("T=%d min" % (ds.time.values[t]/60))
        # axs[i].set_yscale('log')

        axs[i].set_xlabel("Sediment grain size (µm)")
        # axs[i].set_ylabel("10$^6$ particles/m$^3$")
    # axs[0].legend()
    plt.suptitle("Depth = %2.1f m" % z[dep])
    plt.tight_layout()
    fig.savefig("1DModel_dep%d.png" % dep)


