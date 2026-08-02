#!/usr/bin/env python3
"""
plot_floc_mod_ens.py

Converted from plot_floc_mod_ens.ipynb.

Generates the full set of FLOCMOD ensemble / LISST comparison plots and
saves each one as a PNG, with the run identifier (derived from the model
output filename, e.g. "flocmod_56") appended to every plot's filename.

Usage:
    python plot_floc_mod_ens.py
    python plot_floc_mod_ens.py --nc flocmod_61.nc --outdir plot

Since this runs headless on the HPC (no notebook display), it uses the
non-interactive "Agg" matplotlib backend, so nothing is shown on screen --
everything is written straight to disk in --outdir.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend -- no display needed on the HPC

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmocean as cmo
import matplotlib.dates as mdates
from scipy.integrate import cumulative_trapezoid


# --------------------------------------------------------------------------- #
#  CONFIG
# --------------------------------------------------------------------------- #
# NOTE: the ADCP and LISST paths below are copied verbatim from the original
# notebook (hardcoded to your scratch directory). Update them here, or pass
# --adcp / --lisst on the command line, if this is run from somewhere else.
DEFAULT_NC = "flocmod_108_NOASSIM.nc"
DEFAULT_ADCP = "/global/scratch/users/siennaw/data/usgs/resampled_adcp_usgs.nc"
DEFAULT_LISST = "/global/scratch/users/siennaw/data/usgs/CSF20_Shallows_Time_Series/CSF20SC104ls-b.nc"
DEFAULT_LISST_CSV = "lisst_data.csv"
DEFAULT_START_DATE = "2020-07-16 20:10:01"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nc", default=DEFAULT_NC, help="Model output NetCDF file (e.g. flocmod_56.nc)")
    p.add_argument("--adcp", default=DEFAULT_ADCP, help="ADCP NetCDF file")
    p.add_argument("--lisst", default=DEFAULT_LISST, help="LISST NetCDF file")
    p.add_argument("--lisst-csv", default=DEFAULT_LISST_CSV, help="LISST data CSV (as fed to the EnKF)")
    p.add_argument("--start-date", default=DEFAULT_START_DATE, help="Model t=0 real-world timestamp")
    p.add_argument("--outdir", default="plots", help="Directory to save all figures into")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI for saved PNGs")
    return p.parse_args()


def save_and_close(fig, name, outdir, run_tag, dpi):
    """Save a figure as <name>_<run_tag>.png in outdir, then close it."""
    fname = outdir / f"{name}_{run_tag}.png"
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# --------------------------------------------------------------------------- #
#  SUPPORTING FUNCTIONS  (from notebook cell 0 / cell 7)
# --------------------------------------------------------------------------- #
def julian_to_datetime(julian_days, seconds_, base_julian=2440000, base_date_str="1968-05-23 00:00:00"):
    julian_days = np.asarray(julian_days)
    seconds_ = np.asarray(seconds_)
    base_date = np.datetime64(base_date_str)
    days_diff = (julian_days - base_julian) * np.timedelta64(1, "D")
    seconds = seconds_ * np.timedelta64(1, "ms")
    dates = base_date + days_diff + seconds
    return dates


def format_date_ax(ax, interval=1):
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
    date_format = mdates.DateFormatter("%m/%d %H:%M")
    ax.xaxis.set_major_formatter(date_format)


def weighted_median_axis1(data, weights):
    sort_idx = np.argsort(data, axis=1)
    sorted_data = np.take_along_axis(data, sort_idx, axis=1)
    sorted_weights = np.take_along_axis(weights, sort_idx, axis=1)
    cum_weights = np.cumsum(sorted_weights, axis=1)
    cutoff = np.sum(sorted_weights, axis=1, keepdims=True) / 2.0
    idx = np.argmax(cum_weights >= cutoff, axis=1, keepdims=True)
    return np.take_along_axis(sorted_data, idx, axis=1).squeeze()


def date2color(date, mtime, cmap=cmo.cm.haline):
    d0 = mtime[0]
    d1 = (d0 - mtime[-1]).total_seconds()
    normalize = (d0 - date).total_seconds() / d1
    return cmap(normalize)


# --------------------------------------------------------------------------- #
#  MAIN
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Run tag comes straight from the model output filename, e.g.
    # "flocmod_56.nc" -> "flocmod_56". This gets appended to every saved plot.
    run_tag = Path(args.nc).stem
    dpi = args.dpi
    print(f"Run tag: {run_tag}")
    print("Opening file: ", args.nc)
    print(f"Saving figures to: {outdir.resolve()}")

    # ----------------------------------------------------------------- #
    #  OPEN MODEL DATA
    # ----------------------------------------------------------------- #
    fn = args.nc
    ds = xr.open_dataset(fn, decode_times=False)
    volume = 4 / 3 * np.pi * (ds.Ds / 2) ** 3

    date = pd.to_datetime(args.start_date)
    ds = ds.dropna(dim="N", how="any")
    mtime = [date + pd.Timedelta(t, unit="s") for t in ds.time.values]
    nt = len(ds.time)
    Nens = len(ds.N)
    Ns = len(ds.Ds)

    # ----------------------------------------------------------------- #
    #  OPEN ADCP DATA
    # ----------------------------------------------------------------- #
    adcp = xr.open_dataset(args.adcp)
    rtime = adcp.time.values
    offset = 3.537306921274909
    levels = [-100, -80, -50, -20, -1, 0, 1, 20, 50, 80, 100]
    adcp_u = np.squeeze(adcp.u_1205)
    adcp_z = adcp.depth.values
    print(adcp.lon.values, adcp.lat.values)

    # ----------------------------------------------------------------- #
    #  OPEN LISST DATA
    # ----------------------------------------------------------------- #
    lds = xr.open_dataset(args.lisst, decode_times=False)
    ldates = julian_to_datetime(lds["time"].values, lds["time2"].values)
    lds = lds.assign_coords(time=ldates)

    # =================================================================== #
    #  PLOT 1: ensemble SSC time series, per size class
    # =================================================================== #
    print("Plotting: ensemble SSC time series...")
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10, 11), sharex=True, sharey=False)
    axs = axs.flatten()

    for j, Dsi in enumerate(range(0, Ns, 4)):
        if j > 9:
            continue
        axs[j].grid(alpha=0.3)
        axs[j].set_title(r"Ds = %.2f $\mu$m" % (ds.Ds.isel(Ds=Dsi).values * 1e6))
        time = ds.time.values / 3600
        axs[j].set_ylim(1, 2)
        axs[j].set_ylabel("SSC [uL/L]")
        for i in range(0, Nens):
            s0 = ds.ssc.isel(N=i).isel(Ds=Dsi) * volume[Dsi] * 1e6
            axs[j].plot(time, s0, color=cmo.cm.thermal(i / Nens), linewidth=2)

    axs[-1].set_xlabel("Time [hr]")
    axs[-2].set_xlabel("Time [hr]")
    save_and_close(fig, "ensemble_ssc_timeseries", outdir, run_tag, dpi)

    # =================================================================== #
    #  PLOT 2: per-ensemble-member Hovmoller of number density + params
    # =================================================================== #
    print("Plotting: per-ensemble number density Hovmoller...")
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(14, 7), sharex=True, sharey=True)
    axs[0, 0].set_ylabel(r"D [$\mu$m]")
    axs[1, 0].set_ylabel(r"D [$\mu$m]")
    axs[2, 0].set_ylabel(r"D [$\mu$m]")
    for j in range(5):
        axs[2, j].set_xlabel("Time [hr]")
    axs = axs.flatten()

    h = None
    ax2 = None
    for i in range(0, 15):
        axs[i].grid(alpha=0.3)
        axs[i].set_title("Ensemble #%d" % i)
        ssc = ds.ssc.isel(N=i)
        alpha = ds.alpha.isel(N=i)
        beta = ds.beta.isel(N=i)
        beta2 = ds.beta2.isel(N=i)
        ax2 = axs[i].twinx()
        ax2.plot(mtime, alpha, color="r", label="alpha", linewidth=2)
        ax2.plot(mtime, beta, color="g", label="beta", linewidth=2)
        h = axs[i].pcolormesh(mtime, ds.Ds.values * 1e6, ssc.T, cmap=cmo.cm.rain, shading="auto", vmin=0, vmax=1e10)
        axs[i].set_yscale("log")
        format_date_ax(axs[i], 72)

    plt.colorbar(h, ax=axs, orientation="vertical", label="particle density", shrink=0.5)
    if ax2 is not None:
        ax2.legend()
    save_and_close(fig, "ensemble_number_density_hovmoller", outdir, run_tag, dpi)

    # =================================================================== #
    #  PLOT 3: per-ensemble-member Hovmoller of volume density + params
    # =================================================================== #
    print("Plotting: per-ensemble volume density Hovmoller...")
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(14, 7), sharex=True, sharey=True)
    axs[0, 0].set_ylabel(r"D [$\mu$m]")
    axs[1, 0].set_ylabel(r"D [$\mu$m]")
    axs[2, 0].set_ylabel(r"D [$\mu$m]")
    for j in range(5):
        axs[2, j].set_xlabel("Time [hr]")
    axs = axs.flatten()

    h = None
    for i in range(0, 15):
        axs[i].grid(alpha=0.3)
        axs[i].set_title("Ensemble #%d" % i)
        time = ds.time.values / 3600
        ssc = ds.ssc.isel(N=i) * volume * 1e6

        h = axs[i].pcolormesh(time, ds.Ds.values * 1e6, ssc.T, cmap=cmo.cm.rain, shading="auto", vmin=0, vmax=5)
        axs[i].set_yscale("log")

        alpha = ds.alpha.isel(N=i)
        beta = ds.beta.isel(N=i)
        beta2 = ds.beta2.isel(N=i)
        ax2 = axs[i].twinx()
        ax2.plot(time, alpha, color="r", label="alpha", linewidth=2)
        ax2.plot(time, beta, color="g", label="beta", linewidth=2)

    plt.colorbar(h, ax=axs, orientation="vertical", label="volume density", shrink=0.5)
    save_and_close(fig, "ensemble_volume_density_hovmoller", outdir, run_tag, dpi)

    # =================================================================== #
    #  PLOT 4 (disabled in notebook -- flux coefficient plot, no flux
    #  variable currently saved by the model). Left out here as it was
    #  entirely commented out in the source notebook.
    # =================================================================== #

    # =================================================================== #
    #  PLOT 5 (disabled in notebook via `if 0:`) -- ADCP flux comparison.
    #  Kept disabled here for parity with the notebook; flip to `if 1:`
    #  if you want these two figures generated.
    # =================================================================== #
    if 0:
        print("Plotting: ADCP flux comparison (disabled)...")
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(14, 7), sharex=True, sharey=False)

        depths = np.squeeze(adcp.P_1.values)
        ax2 = axs[0].twinx()
        ax2.plot(rtime, depths, color="k", label="Depth")

        U = adcp_u.mean(dim="depth")
        temp = pd.DataFrame({"u": np.squeeze(U.values)})
        temp.index = rtime
        output = pd.DataFrame(index=mtime)

        combined_index = temp.index.union(output.index)
        adcp_aligned = temp.reindex(combined_index).interpolate(method="time")
        output["u"] = adcp_aligned

        keys = []
        colors = []
        for i in range(0, 2):
            color = cmo.cm.haline(i / 2)
            f0 = ds.flux.isel(Nf=i).mean(dim="N")
            f0 = f0.rolling(time=2).mean()
            f0 = np.sin(f0)
            keys.append(i)
            colors.append(color)
            output[i] = f0.values
            axs[1].plot(output[i], "-", color=color, alpha=0.5, linewidth=3)
            axs[2].plot(output.u / 100, "--k")

        axs[0].set_xlim(mtime[0], mtime[-1])
        save_and_close(fig, "adcp_flux_timeseries", outdir, run_tag, dpi)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), sharex=True, sharey=False)
        axs = axs.flatten()
        for i, (key, c) in enumerate(zip(keys, colors)):
            if i > 11:
                continue
            axs[i].set_title(r"Ds = %s $\mu$m" % key)
            axs[i].hist(output[key].loc[output.u > 0], bins=5, alpha=0.5, label="Flood", color="skyblue", density=True)
            axs[i].hist(output[key].loc[output.u < 0], bins=5, alpha=0.5, label="Ebb", color="red", density=True)
            axs[i].grid(alpha=0.3)
            axs[i].set_xlim(-0.2, 0.2)
            axs[i].vlines(0, 0, 20, color="k", alpha=0.5, linewidth=1)
        axs[0].legend()
        save_and_close(fig, "flux_flood_ebb_histogram", outdir, run_tag, dpi)

    # =================================================================== #
    #  PLOT 6: mean size-distribution snapshots at 4 times
    # =================================================================== #
    print("Plotting: size-distribution snapshots...")
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(6, 6), sharex=True, sharey=False)

    Ds_um = ds.Ds.values * 1e6
    for it, t in enumerate(range(0, nt, max(nt // 4, 1))):
        if it > 3:
            continue
        time_ = mtime[it]
        mean = ds.ssc.isel(time=t).mean(dim="N") * volume * 1e6
        axs[it].plot(Ds_um, mean, "-", color="k", linewidth=2)
        axs[it].set_title(time_.strftime("%b %d %H:%M"))
        axs[it].set_xscale("log")
        axs[it].grid(alpha=0.3)
        axs[it].set_ylabel(r"$\mu$L/L")
        axs[it].set_ylim(0, )
        axs[it].set_xlim(Ds_um[0], Ds_um[-1])

    axs[-1].set_xlabel("Sediment grain size (\u00b5m)")
    axs[-1].legend()
    plt.tight_layout()
    save_and_close(fig, "size_distribution_snapshots", outdir, run_tag, dpi)

    # =================================================================== #
    #  PLOT 8: LISST vs. model ensemble-mean Hovmoller comparison
    # =================================================================== #
    print("Plotting: LISST vs. model Hovmoller comparison...")
    l_ds = lds.size_class.values
    vconc = lds.vconc.mean(dim="sample").isel(depth=0).values
    lds_time = lds.time.values

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(14, 7), sharex=True, sharey=True)
    axs = axs.flatten()

    axs[0].pcolormesh(lds_time, l_ds, vconc.T, cmap=cmo.cm.rain, shading="auto", vmin=0, vmax=5)

    ssc = ds.ssc
    ssc0 = ssc.mean(dim="N") * volume * 1e6
    axs[0].set_title("LISST data")
    axs[1].set_title("Model ensemble mean")

    i = 1
    axs[i].grid(alpha=0.3)
    axs[i].pcolormesh(mtime, ds.Ds.values * 1e6, ssc0.T, cmap=cmo.cm.rain, shading="auto", vmin=0, vmax=5)
    axs[i].set_yscale("log")
    axs[i].set_xlim(mtime[0], mtime[-1])
    axs[i].set_ylim(ds.Ds.values[0] * 1e6, ds.Ds.values[-1] * 1e6)
    format_date_ax(axs[i], 24)

    for ax in axs:
        ax.set_ylabel("D (\u00b5m)")
    save_and_close(fig, "lisst_vs_model_hovmoller", outdir, run_tag, dpi)

    # =================================================================== #
    #  PLOT 10: D50 timeseries comparison (model vs. LISST) + ADCP shear
    # =================================================================== #
    print("Plotting: D50 timeseries comparison...")
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 5), sharex=True, sharey=False, constrained_layout=True)

    shear = ds.G.isel(N=0)
    ax = axs[0]
    ax2 = axs[1].twinx()
    ax2.plot(mtime, shear, "-", color="#B21235", linewidth=2, label="Shear stress")
    ax2.set_ylim(0, 10)
    ax2.tick_params(axis="y", colors="#B21235")
    ax2.set_ylabel(r"ADV shear stress (s$^{-1}$)", color="#B21235")

    var = ds.ssc.rolling(time=5).mean()
    Ds_aligned = ds.Ds.values[None, :, None]

    var = xr.where(var < 0, 0, var)
    weights = var * volume

    mean_d = np.sum(Ds_aligned * weights, axis=1) / np.sum(weights, axis=1)
    std_ = mean_d.std(axis=1)

    mean_d = np.median(mean_d, axis=1)
    ax.plot(mtime, mean_d * 1e6, color="#CD7A3E", linewidth=3, label="Model ensemble mean")
    ax.fill_between(mtime, np.squeeze(mean_d - std_) * 1e6, np.squeeze(mean_d + std_) * 1e6,
                     color="#CD7A3E", alpha=0.2, label="Model std. deviation")

    ax.grid(alpha=0.3)
    ax.set_xlabel("Time [hrs]")
    ax.set_ylabel(r"D$_{50}$ (\u00b5m)")
    ax.set_ylim(10, 200)

    d50 = lds["D50"].isel(depth=0)
    stds = lds["D50"].std(axis=1)
    d50 = lds["D50"].mean(axis=1)

    ax.fill_between(ldates, np.squeeze(d50 - stds), np.squeeze(d50 + stds), color="#CBE2FE", alpha=0.55,
                     label="LISST std. deviation")
    ax.plot(ldates, d50, "o", color="#10288C", label=r"LISST D$_{50}$", markersize=3)
    format_date_ax(ax, 24)
    ax.set_ylim(0, 200)

    h = axs[1].contourf(rtime, -adcp_z + offset, adcp_u.T, cmap=cmo.cm.balance, levels=levels)
    plt.colorbar(h, label="U Velocity (cm/s)", shrink=0.7)
    axs[1].grid(alpha=0.3)
    axs[0].set_xlim(mtime[0], mtime[-1])
    axs[1].set_xlim(mtime[0], mtime[-1])
    save_and_close(fig, "d50_timeseries_comparison", outdir, run_tag, dpi)

    # =================================================================== #
    #  PLOT 11 (disabled in notebook via `if 0:`) -- PDF comparison
    # =================================================================== #
    if 1:
        print("Plotting: PDF comparison (disabled)...")
        x = lds.size_class.values
        x = np.append(x, x[-1])

        fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharey=True)
        Ds_ = ds.Ds.values * 1e6
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(5, nt, max(nt // 6, 1)):
                conc = ds.ssc.isel(time=i)
                conc = conc * volume
                conc = conc.mean(dim="N").values

                cdf_integrated = cumulative_trapezoid(conc, Ds_, initial=0)
                cdf = cdf_integrated / cdf_integrated[-1]
                pdf = np.gradient(cdf, Ds_)

                color = date2color(mtime[i], mtime)
                axs[1].plot(Ds_, pdf, "-o", color=color, markersize=3, label=mtime[i].strftime("%b %d %H:%M"))

                cdf = lds.CDF.sel(time=mtime[i], method="nearest").isel(depth=0)
                cdf = cdf.mean(dim="sample")
                pdf = np.gradient(cdf, x)
                d0 = pd.to_datetime(cdf.time.values)
                color = date2color(d0, mtime)
                axs[0].plot(x, pdf, "-", color=color, linewidth=3, label=d0.strftime("%b %d %H:%M"))

        axs[1].set_title("Floc Model PDF")
        axs[0].set_title("LISST PDF")
        for ax in axs:
            ax.set_xlim(1, 1000)
            ax.set_xlabel(r"D$_{50}$ ($\mu$m)")
            ax.grid(alpha=0.2)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax.set_xscale("log")
        save_and_close(fig, "pdf_comparison", outdir, run_tag, dpi)

    # =================================================================== #
    #  PLOT 12 (disabled in notebook via `if 0:`) -- CDF comparison
    # =================================================================== #
    if 0:
        print("Plotting: CDF comparison (disabled)...")
        x = lds.size_class.values
        x = np.append(x, x[-1])
        Ds_ = ds.Ds.values * 1e6

        fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharey=False)
        for i in range(5, nt, 5):
            conc = ds.ssc.isel(time=i)
            conc = conc * volume / np.sum(volume * conc)
            conc = conc.mean(dim="N").values

            cdf_integrated = cumulative_trapezoid(conc, Ds_, initial=0)
            cdf = cdf_integrated / cdf_integrated[-1]

            color = date2color(mtime[i], mtime)
            axs[1].plot(Ds_, cdf, "-o", color=color, markersize=3, label=mtime[i].strftime("%b %d %H:%M"))

            cdf = lds.CDF.sel(time=mtime[i], method="nearest").isel(depth=0)
            cdf = cdf.mean(dim="sample")
            d0 = pd.to_datetime(cdf.time.values)
            color = date2color(d0, mtime)
            axs[0].plot(x, cdf, "-", color=color, linewidth=3, label=d0.strftime("%b %d %H:%M"))

        axs[1].set_title("Floc Model CDF")
        axs[0].set_title("LISST CDF")
        for ax in axs:
            ax.set_xlim(1, 1000)
            ax.set_xlabel(r"D$_{50}$ ($\mu$m)")
            ax.grid(alpha=0.2)
            ax.set_xscale("log")
        save_and_close(fig, "cdf_comparison", outdir, run_tag, dpi)

    # =================================================================== #
    #  PLOT 13: volume concentration comparison (model vs. LISST)
    # =================================================================== #
    print("Plotting: volume concentration comparison...")
    x = lds.size_class.values
    x = np.append(x, x[-1])
    Ds_um = ds.Ds.values * 1e6

    fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharey=True)
    for i in range(5, nt, 20):
        conc = ds.ssc.isel(time=i)
        conc = conc.mean(dim="N").values
        interpolated = conc * volume * 1e6
        color = date2color(mtime[i], mtime)

        axs[1].plot(Ds_um, interpolated, "-.", color=color, markersize=3, label=mtime[i].strftime("%b %d %H:%M"))
        interpolated = conc @ ds.H.values
        axs[1].plot(x[0:-1], interpolated, "-o", color=color, markersize=3, label=mtime[i].strftime("%b %d %H:%M"))

        cdf = lds.vconc.sel(time=mtime[i], method="nearest").isel(depth=0)
        cdf = cdf.mean(dim="sample")
        d0 = pd.to_datetime(cdf.time.values)
        color = date2color(d0, mtime)
        axs[0].plot(x[0:-1], cdf, "-o", color=color, linewidth=3, label=d0.strftime("%b %d %H:%M"))

    axs[1].set_title("Floc Model Vconc")
    axs[0].set_title("LISST Vconc")
    for ax in axs:
        ax.set_ylim(0, 4)
        ax.set_xlabel(r"D$_{50}$ ($\mu$m)")
        ax.set_xlim(x[0], x[-1])
        ax.grid(alpha=0.2)
        ax.set_ylabel(r"$\mu$L/L")
    save_and_close(fig, "volume_concentration_comparison", outdir, run_tag, dpi)

    # =================================================================== #
    #  PLOT 14: model vs. LISST scatter + error time series
    # =================================================================== #
    print("Plotting: model vs. LISST scatter / error...")
    lstdata = pd.read_csv(args.lisst_csv)
    seconds = lstdata["seconds"].values

    fig2 = plt.figure(figsize=(8, 4))
    ax2 = fig2.gca()
    fig = plt.figure(figsize=(4, 4))
    ax = fig.gca()

    for i in range(0, nt, 5):
        color = cmo.cm.phase(i / 10)
        t = seconds[i]
        conc = ds.ssc.sel(time=t, method="nearest")
        conc = conc @ ds.H
        conc = conc.mean(dim="N")
        l = lstdata.iloc[i, 1:].values
        ax.plot(conc, l, "o", color=color, markersize=5, label="t=%d s" % t)
        error = np.mean(np.sqrt((conc - l) ** 2))
        ax2.plot(t, error, "o", color=color, markersize=5, label="t=%d s" % t)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.plot([0, 10], [0, 10], "--k", alpha=0.5)
    ax.set_xlabel("Floc Model")
    ax.set_ylabel("LISST")
    save_and_close(fig, "model_vs_lisst_scatter", outdir, run_tag, dpi)

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("RMS error")
    ax2.grid(alpha=0.3)
    save_and_close(fig2, "model_vs_lisst_error_timeseries", outdir, run_tag, dpi)

    # =================================================================== #
    #  PLOT 16: ensemble parameter histograms at 4 times
    # =================================================================== #
    print("Plotting: parameter histograms...")
    fig, axs = plt.subplots(figsize=(10, 6), nrows=4, ncols=4, sharex=False)
    shears = ds.G.values.flatten()
    shear = np.mean(shears)
    plt.suptitle(r"Avg. shear rate = %2.1f s$^{-1}$" % shear)

    total = nt
    for i, t in enumerate(range(0, total, max(total // 4, 1))):
        if i > 3:
            continue
        alphas = ds.alpha.isel(time=t).values
        betas = ds.beta.isel(time=t).values
        nfs = ds.nf.values
        beta2s = ds.beta2.values

        axs[0, i].hist(alphas, density=True, bins=30)
        print("alpha:", np.mean(alphas), np.std(alphas))
        print(np.mean(betas), np.std(alphas))
        print(np.mean(nfs), np.std(nfs))

        axs[0, i].set_title(r"$\alpha$ (t=%.2f hr)" % (ds.time.isel(time=t).values / 3600))
        axs[1, i].hist(betas, density=True, bins=30)
        axs[1, i].plot(np.mean(betas), 0, "ro", label="mean")
        axs[1, i].set_title(r"$\beta$  (t=%.2f hr)" % (ds.time.isel(time=t).values / 3600))

        axs[2, i].hist(nfs, density=True, bins=30)
        axs[2, i].set_title(r"$n_f$ (t=%.2f hr)" % (ds.time.isel(time=t).values / 3600))
        axs[2, i].plot(np.mean(nfs), 0, "ro", label="mean")

        axs[3, i].hist(beta2s, density=True, bins=30)
        axs[3, i].set_title(r"$\beta_2$ (t=%.2f hr)" % (ds.time.isel(time=t).values / 3600))
        axs[3, i].plot(np.mean(beta2s), 0, "ro", label="mean")

    for ax in axs.flatten():
        ax.grid(alpha=0.3)
    plt.tight_layout()
    save_and_close(fig, "parameter_histograms", outdir, run_tag, dpi)

    # =================================================================== #
    #  PLOT 17: ensemble parameter time series (mean +/- std)
    # =================================================================== #
    print("Plotting: parameter time series...")
    fig, axs = plt.subplots(figsize=(10, 6), nrows=2, ncols=1, sharex=False)

    alpha0 = 0.15
    beta0 = 0.15
    nf0 = 2.5
    beta20 = 1.5

    alpha = alpha0 * np.exp(0.1 * ds["alpha"])
    beta = beta0 * np.exp(0.1 * ds["beta"])


    dictionary = {"alpha": alpha, "beta": beta}
    for iv, variable in enumerate(["alpha", "beta"]):
        var = dictionary[variable]
        mean_var = var.mean(dim="N").values
        std_var = var.std(dim="N").values
        axs[iv].plot(mtime, mean_var, color="#F95454", linewidth=2, label="Mean %s" % variable)
        axs[iv].fill_between(mtime, np.squeeze(mean_var - std_var), np.squeeze(mean_var + std_var),
                              color="#F9547F", alpha=0.2, label="Std. deviation")
        axs[iv].set_title(variable)
        axs[iv].grid(alpha=0.3)
        axs[iv].set_ylim(1e-6, )
        axs[iv].set_yscale('log')

    format_date_ax(axs[-1], 24)
    plt.tight_layout()
    save_and_close(fig, "parameter_timeseries", outdir, run_tag, dpi)

    print(f"\nDone. All figures saved to {outdir.resolve()} with run tag '{run_tag}'.")


if __name__ == "__main__":
    main()