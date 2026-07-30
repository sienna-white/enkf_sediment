import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmocean as cmo


# /global/scratch/users/siennaw/scripts/enkf_sediment/test_sediment_NOFLOCMOD.nc
ds = xr.open_dataset("test_sediment_NOFLOCMOD.nc", decode_times=False)
print(ds)

nd = len(ds.Ds)
nt = len(ds.time) 
Nens = len(ds.Nens)



fig3 = plt.figure(figsize=(2, 5))
ax3 = plt.gca() 

N = 0 

volume = 4 / 3 * np.pi * (ds.Ds.values / 2) ** 3



fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(12, 5), sharex=False, sharey=True)
axs = axs.flatten()
plt.suptitle("FlocMod on <-> advection/diff off")
for i, d in enumerate(range(0, nd, 6)):


    for t in range(0, nt, nt//4):
        color = cmo.cm.thermal(t/nt) 

        for N in range(Nens):
            var = ds.ssc.isel(time=t, Nens=N, Ds=d).values 
            var = var * volume[i] * 1e6
            h= axs[i].plot(var, -ds.z, color=color, linewidth=3, alpha=0.1)
        var = ds.ssc.isel(time=t, Ds=d).mean(dim="Nens")
        var = var * volume[i] * 1e6
        h= axs[i].plot(var, -ds.z, color=color, label="T=%d min" % (t*10/60), linewidth=2, alpha=1)

    axs[i].grid(alpha=0.3)
    axs[i].set_xlabel("SSC (g/L)")
    # axs2[i].grid(alpha=0.3)

    axs[i].set_title(r"Ds = %.2f $\mu$m" % (ds.Ds.isel(Ds=d).values*1e6))
    axs[i].set_ylim(-4, 0)
        # axs2[i].set_title("Ds = %.2f $\mu$m" % (ds.Ds.isel(Ds=d).values*1e6))


axs[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0. )
# axs2[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0. )
axs[0].set_ylabel("Depth (m)")



fig.savefig('test1d.png')
# fig3.savefig('test1d2.png')







fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(12, 5), sharex=True, sharey=True)
axs = axs.flatten()
plt.suptitle("FlocMod on <-> advection/diff off")
    
for it, t in enumerate(range(0, nt, nt//6)):

    if it>5:
        continue
    
    color = cmo.cm.thermal(t/nt) 

    for N in range(Nens):
        var = ds.ssc.isel(time=t, Nens=N).values 
        var = var * volume[i] * 1e6
        h= axs[it].plot(np.sum(var, axis=0), -ds.z, color=color, linewidth=3, alpha=0.1)

    var = ds.ssc.isel(time=t).mean(dim="Nens")
    var = var * volume[i] * 1e6
    total_volume = np.sum(var, axis=0)

    h= axs[it].plot(total_volume, -ds.z, color=color, label="T=%d min" % (t*10/60), linewidth=2, alpha=1)

    axs[it].grid(alpha=0.3)
    axs[it].set_xlabel("SSC (g/L)")

    axs[it].set_title("t=%d min" % (ds.time.values[it]/60))
    axs[it].set_ylim(-4, 0)
        # axs2[i].set_title("Ds = %.2f $\mu$m" % (ds.Ds.isel(Ds=d).values*1e6))

axs[0].set_ylabel("Depth (m)")
fig.savefig('test_VOLUME_time.png')
# fig3.savefig('test1d2.png')

assert(False)

#*#*#*#*#*#*##*#*#*#*#*#*##*#*#*#*#*#*##*#*#*#*#*#*#


for dep in range(0, 50, 15):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), sharex=True, sharey=False)
    axs = axs.flatten()

    z = ds.z.values

    for i,t in enumerate(range(0, nt, nt//5)):
        # var = ds.Kz.isel(time=12)

        if i>5:
            continue 


        for N in range(Nens):
            var = ds.ssc.isel(Nens=N, z=dep, time=t) * volume
            axs[i].plot(ds.Ds.values*1e6, var, '-', color=cmo.cm.haline(z[dep]/10), alpha=0.6, linewidth=3) 
        
        var = ds.ssc.isel(z=dep, time=t)
        var = var.mean(dim='Nens') * volume
        axs[i].plot(ds.Ds.values*1e6, var, 'k-') 

        
        axs[i].set_xscale('log')
        axs[i].grid(alpha=0.3)
        axs[i].set_title("T=%d min" % (t*10/60))
        # axs[i].set_yscale('log')

        axs[i].set_xlabel("Sediment grain size (µm)")
        # axs[i].set_ylabel("10$^6$ particles/m$^3$")
    # axs[0].legend()
    plt.suptitle("Depth = %2.1f m" % z[dep])
    plt.tight_layout()
    fig.savefig("test1d_dep%d.png" % dep)