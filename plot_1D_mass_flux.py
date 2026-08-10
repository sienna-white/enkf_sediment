import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmocean as cmo


# /global/scratch/users/siennaw/scripts/enkf_sediment/1DModel_sediment_NOFLOCMOD.nc
dsn = xr.open_dataset("sediment_1D_NOFLOCMOD.nc", decode_times=False)
dsf = xr.open_dataset("sediment_1D_model_91.nc", decode_times=False)
ds3 = xr.open_dataset("sediment_1D_model_82.nc", decode_times=False)
# ds4 = xr.open_dataset("sediment_1D_model_67.nc", decode_times=False)
print(dsn)
print(dsf)

max_time = max(dsf.time.values)

models = {"Fixed size classes" : dsn, "FLOCMOD on" : dsf}
colors = {"Fixed size classes" : "#A59837",  "FLOCMOD on" : "#7A3D5D", "FLOCMOD (90)" : "#D73220", "FLOCMOD (78)" : 'b'}

# colors = {"Fixed size classes" : "#A59837",  "FLOCMOD" : "#7A3D5D", }
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


# # ######################### PLOT MASS FLUX #########################
# fig = plt.figure(figsize=(8, 3))
# ax = plt.gca()
# plt.suptitle("Suspended sediment flux")

# for key in keys:
#     # print(key)
#     ds = models[key]
#     mass_flux = get_mass_flux(ds)
#     time = ds.time.values / 3600 
#     Nens = len(ds.Nens)
#     color = colors[key]
#     for N in range(Nens):
#         var = mass_flux.isel(Nens=N).values 
#         ax.plot(time, var, color=color, linewidth=3, alpha=0.03)


#     var = mass_flux.mean(dim="Nens")
#     ax.plot(time, var, color=color, linewidth=1, alpha=0.9, label=key)
#     # print("average flux: ", var[0:6])
# ax.set_ylabel(r"Sed. flux (kg/m$^2$)")
# ax.grid(alpha=0.2)
# ax.set_xlim(dsf.time.values[0]/3600 , dsf.time.values[-1]/3600)
# ax.legend()
# fig.savefig('1DModel_MASSFLUX(t).png')



### 
fig = plt.figure(figsize=(8, 3))
ax = plt.gca()
plt.suptitle("Suspended sediment flux")

for key in keys:
    ds = models[key]
    mass_flux = get_mass_flux(ds)
    time = ds.time.values / 3600
    color = colors[key]

    var_mean = mass_flux.mean(dim="Nens")
    var_std = mass_flux.std(dim="Nens")

    # ax.fill_between(time, var_mean - var_std, var_mean + var_std,
    #                  color=color, alpha=0.2, linewidth=0)

    # NEW PLOT 
    q_lo, q_med, q_hi = mass_flux.quantile([0.05, 0.5, 0.95], dim="Nens")
    ax.plot(time, q_med, color=color, linewidth=2, alpha=1, label=key)
    ax.fill_between(time, q_lo, q_hi, color=color, alpha=0.2, linewidth=0)


    # ax.plot(time, var_mean, color=color, linewidth=1.5, alpha=0.9, label=key)

ax.set_ylabel(r"Sed. flux (kg m$^{-2}$ s$^{-1}$)")
ax.grid(alpha=0.2)
ax.set_xlim(dsf.time.values[0] / 3600, dsf.time.values[-1] / 3600)
ax.legend()
ax.set_xlabel("Time [hrs]")
plt.tight_layout()
fig.savefig('1DModel_MASSFLUX(t).png', dpi=250)

# ######################### PLOT MASS  #########################
fig = plt.figure(figsize=(8, 3))
ax = plt.gca()
plt.suptitle("Suspended sediment flux")

for key in keys:
    # print(key)
    ds = models[key]
    mass = get_mass(ds)
    mass_conc = ds.ssc * mass #* 1e3

    mass_conc = mass_conc.sum(dim="Ds")
    total_mass = mass_conc.integrate(coord="z") #.sum(dim="z")

    time = ds.time.values / 3600 
    Nens = len(ds.Nens)
    color = colors[key]

    # var_mean = total_mass.mean(dim="Nens")
    # var_std = total_mass.std(dim="Nens")

    q_lo, q_med, q_hi = total_mass.quantile([0.05, 0.5, 0.95], dim="Nens")
    ax.plot(time, q_med, color=color, linewidth=2, alpha=1, label=key)
    ax.fill_between(time, q_lo, q_hi, color=color, alpha=0.2, linewidth=0)
    
    # ax.fill_between(time, var_mean - var_std, var_mean + var_std,
    #                  color=color, alpha=0.2, linewidth=0)
    # ax.plot(time, var_mean, color=color, linewidth=1.5, alpha=0.9, label=key)

    # print("average flux: ", var[0:6])
ax.legend()
ax.set_xlabel("Time [hrs]")
ax.set_ylabel("Suspended sediment [kg]")
ax.set_xlim(dsf.time.values[0]/3600 , dsf.time.values[-1]/3600)
plt.tight_layout()
fig.savefig('1DModel_TOTAL_MASS(t).png')



fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(14, 5), sharex=True, sharey=True)
axs = axs.flatten()

times2plot = np.arange(0, max_time, max_time//5)
# print("plotting time at ", times2plot)
for key in keys:

    ds = models[key]
    mass = get_mass(ds)
    time = ds.time.values / 3600 
    Nens = len(ds.Nens)
    z = ds.z 
    for it, t in enumerate(times2plot):
        if it>5:
            continue 
        color = colors[key] 
        axs[it].set_title("%d hrs" % (t/3600))
        axs[it].set_ylim(0, 4)
        axs[it].grid(alpha=0.2) 
        
        # for N in range(Nens):
        var = ds.ssc.sel(time=t, method='nearest')

        var = var * mass * 1e3
        total_mass = var.sum(dim="Ds")

        q_lo, q_med, q_hi = total_mass.quantile([0.05, 0.5, 0.95], dim="Nens")

        h = axs[it].plot(q_med, z, color=color, linewidth=2, alpha=1, label=key)
        axs[it].fill_betweenx(z, q_lo, q_hi, color=color, alpha=0.2, linewidth=0)
        axs[it].set_xlabel("SSC (mg/L)")

    axs[0].legend()
axs[0].set_ylabel("Depth (m)")
plt.tight_layout()
fig.savefig('1DModel_COMPARE_Mass(z).png')



#### 



   
    
    

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

######################### PHYSICAL FORCINGS #########################

ds= dsf 
print(ds)
z = ds.z 

time = ds.time.values/3600
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7, 7), sharex=False, sharey=True)
axs = axs.flatten()

h= axs[0].pcolormesh(time, z, ds.G.values.T, vmin=0, vmax=1, cmap=cmo.cm.speed)
# axs[0].set_title("Shear (1/s)")
plt.colorbar(h, label='Shear (1/s)', shrink=0.7)


h= axs[1].pcolormesh(time, z, np.log(ds.Kz.values.T), vmin=-10, vmax=-1, cmap=cmo.cm.rain)
# axs[1].set_title("log(Diffusivity) (m$^2$/s)")
plt.colorbar(h, label=r'log($\kappa$)', shrink=0.7)

h= axs[2].pcolormesh(time, z, ds.U.values.T, vmin=-0.5, vmax=0.5, cmap=cmo.cm.balance)
# axs[2].set_title("U (m/s)")
plt.colorbar(h, label='U (m/s)', shrink=0.7)

for ax in axs:
    ax.grid(alpha=0.2)
    ax.set_ylabel("Depth (m)")
axs[-1].set_xlabel("Time [hrs]")
plt.tight_layout() 

fig.savefig("1DModel_Kappa_SHEAR_U.png")    # 
#################################################################################




# Define diameter bin edges in meters (Ds is assumed to be in meters)
bin_edges_um = [1, 10, 50, 100, np.inf]
bin_edges = np.array(bin_edges_um) * 1e-6
bin_labels = [r"1-5 $\mu$m", r"5-10 $\mu$m", r"10-50 $\mu$m", r"50-100 $\mu$m", r"100+ $\mu$m"]
n_bins = len(bin_labels)
bin_colors = cmo.cm.haline(np.linspace(0, 1, n_bins))

fig, axs = plt.subplots(nrows=len(keys), ncols=1, figsize=(12, len(keys) * 2),
                         sharex=False, sharey=True, constrained_layout=True)
axs = axs.flatten()

for ik, key in enumerate(keys):
    print(key)
    ds = models[key]
    mass = get_mass(ds)                     # mass per particle, per size class [kg]
    time = ds.time.values / 3600
    z = ds.z

    # Convert mass concentration -> number concentration (particles per volume, per size class)
    num_conc = ds.ssc  * mass * 1e3               # [# / m^3] per Ds bin
    num_conc = num_conc.mean(dim="Nens")

    # Depth-integrate to get particles per unit area, per size class
    num_conc_zint = num_conc.integrate(coord="z")

    axs[ik].grid(alpha=0.2)

    # Assign each Ds value to a diameter bin, then sum within each bin
    Ds_vals = ds.Ds.values
    bin_idx = np.digitize(Ds_vals, bin_edges) - 1  # index 0..n_bins-1

    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        binned = num_conc_zint.isel(Ds=mask).sum(dim="Ds")
        axs[ik].plot(time, binned, linewidth=2, color=bin_colors[b], label=bin_labels[b])
        # print(binned)

    axs[ik].set_yscale("log")
    axs[ik].set_ylim(1e-1,)
    axs[ik].set_ylabel(key)
    axs[ik].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncols=1)

axs[-1].set_xlabel("Time [hrs]")
fig.suptitle("Particle number concentration by size bin")
fig.savefig("1DModel_Compare_SizeBins(t).png", dpi=150, bbox_inches="tight")


fig, axs = plt.subplots(nrows=len(keys), ncols=1, figsize=(12, len(keys)*2), sharex=False, sharey=True)
axs = axs.flatten()

for ik, key in enumerate(keys):
    ds = models[key]
    mass = get_mass(ds)
    time = ds.time.values / 3600 
    Nens = len(ds.Nens)
    z = ds.z 
    var = ds.ssc.mean(dim='Nens')
    # var = var * mass * 1e3
    var = var.integrate(coord='z')
    axs[ik].grid(alpha=0.2) 
    axs[ik].set_title(key)

    for di, ds_ in enumerate(ds.Ds.values):
        color = cmo.cm.haline(di/len(ds.Ds))
        if di%3==0:
            axs[ik].plot(time, var.isel(Ds=di), color=color, linewidth=2)
        else:
            axs[ik].plot(time, var.isel(Ds=di), linewidth=2, label="%2.1f $\mu$m" % (ds_*1e6), color=color)

    axs[ik].legend(bbox_to_anchor=(1.05, 1), ncols=3)
    axs[ik].set_yscale('log')
    axs[ik].set_ylim(1e2,)

plt.tight_layout()
fig.savefig("1DModel_Compare_DS(t)")   




fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(16, 4))
axs= axs.flatten()
j=0 


for ik, key in enumerate(keys):

    ds = models[key]
    mass = get_mass(ds)
    time = ds.time.values / 3600 
    Nens = len(ds.Nens)

    color = colors[key] 
    ws = ds.ws.mean(dim="Nens")
    for it, t in enumerate(times2plot):
        if it>5:
            continue 
       
        axs[it].set_title("%d hrs" % (t/3600))
        # axs[it].set_ylim(0, 4)
        axs[it].grid(alpha=0.2) 

        var = ds.ssc.mean(dim="Nens").sel(time=t, method='nearest')
        ssc = var * mass
        ssc = ssc.integrate(coord="z")

        axs[it].plot(ws*100, ssc, 'o', color=color, label=key)
        axs[it].grid(alpha=0.2)
        axs[it].set_xscale('log')

for ax in axs:
    ax.set_xlabel("$w_s$ (cm/s)" )
    ax.set_ylabel("SSC [kg]")
    
axs[-1].legend()
plt.tight_layout()
fig.savefig("1DModel_Compare_ws.png")
# axs[-5].set_xlabel("$w_s$ (cm/s)" )
# axs[-4].set_xlabel("$w_s$ (cm/s)" )
# axs[-3].set_xlabel("$w_s$ (cm/s)" )
# axs[-2].set_xlabel("$w_s$ (cm/s)" )
# axs[-1].set_xlabel("$w_s$ (cm/s)" )
    # ax.set_ylabel("$w_s$ (cm/s)")


# SHOW ROUSE PROFILES? 


from scipy.integrate import cumulative_trapezoid

# ######################### PLOT MASS + CUMULATIVE MASS #########################
fig = plt.figure(figsize=(8, 3))
ax = plt.gca()

for key in keys:
    ds = models[key]
    mass_flux = get_mass_flux(ds)


    hours = ds.time.values / 3600 
    Nens = len(ds.Nens)
    color = colors[key]

    time_sec = ds.time.values


    time_axis = mass_flux.get_axis_num("time")
    cum_flux = cumulative_trapezoid(mass_flux.values, time_sec, axis=time_axis, initial=0)
    cum_flux = xr.DataArray(cum_flux, dims=mass_flux.dims, coords=mass_flux.coords)

    cum_mean = cum_flux.mean(dim="Nens")
    cum_lo, cum_hi = cum_flux.quantile([0.05, 0.95], dim="Nens")

    ax.plot(hours, cum_mean, '-', color=color, linewidth=0.9, alpha=0.9, label=key)
    ax.fill_between(hours, cum_lo, cum_hi, color=color, alpha=0.2, linewidth=0)


    # mass_flux = mass_flux.mean(dim="Nens")
    # cum_mean = cumulative_trapezoid(mass_flux, time_sec, initial=0)
    # ax.plot(hours, cum_mean, '-', color=color, linewidth=0.9, alpha=0.9, label=key)

ax.set_ylabel("Cumulative mass\n(time-integrated)")
ax.set_xlabel("Time [hrs]")
ax.grid(alpha=0.2)
ax.legend(fontsize=8)

ax.set_xlim(dsf.time.values[0] / 3600, dsf.time.values[-1] / 3600)
plt.tight_layout()
fig.savefig('1DModel_CUMULATIVE(t).png', dpi=150)