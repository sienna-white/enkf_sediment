#!/usr/bin/env julia

#  module load  julia/1.11.7 
using Printf
using DataStructures: OrderedDict
using NCDatasets
using CSV, DataFrames
using Printf
using Profile
using Statistics 
# using Arrow, DataFrames
# using LaTeXStrings
using LinearAlgebra

include("hydro/calculate_physical_variables.jl") 
include("hydro/advance_variables.jl")
include("hydro/forcings.jl") 
include("hydro/output.jl")
include("floc_mod.jl")
using .floc_mod

# ---------------------------------------------------------------------------
# Linear interpolation in time for a hydrodynamic field.
#
# `data` is (Nz x Nt) -- one column per timestamp in `t_data` (seconds).
# `t_query` is the time (seconds) to interpolate to. Values outside the
# range of `t_data` are clamped to the nearest available timestamp rather
# than extrapolated.
# ---------------------------------------------------------------------------
function interp_time(data::AbstractMatrix{<:Real}, t_data::AbstractVector{<:Real}, t_query::Real)
    nt = length(t_data)
    if t_query <= t_data[1]
        return data[:, 1]
    elseif t_query >= t_data[end]
        return data[:, end]
    end
    j = clamp(searchsortedlast(t_data, t_query), 1, nt - 1)
    t0, t1 = t_data[j], t_data[j + 1]
    w = (t_query - t0) / (t1 - t0)
    return @. (1 - w) * data[:, j] + w * data[:, j + 1]
end

function run_my_model(file_out_name::String, floc_on::Bool=true)

    #***********************************************************************
    # Wind time series 
    # wind_fn = "/global/homes/s/siennaw/scratch/siennaw/stockton_field_data/forcing_for_model/wind_on_august_10-16.csv"
    # df = CSV.read(wind_fn, DataFrame)
    # wind = df[!,"WindSpeed"]
    # real_time = df[!,"time"]
    # println("Read in wind data ...")
    
    
    # function get_wind_speed(index::Int, wind=wind)
    #     return wind[index]
    # end
    #***********************************************************************

    
    @info "Floc on = $floc_on"

    #********************** SPATIAL DOMAIN  ***************************
    N = 50    # number of grid points
    H = 10    # depth (meters)
    dz = H/N  # grid spacing - may need to adjust to reduce oscillations
    dt = 0.05 # (seconds) size of time step -- must be <= hydro.nc's 1 s sampling interval
    total_duration_s = 3600*5  # total simulated time (s); same 5-hour window as before
    M  = Int(round(total_duration_s / dt))

    # Increments for saving profiles are now expressed as a fixed number of
    # *seconds* between saves (rather than a fixed step count), so the saved
    # cadence stays the same regardless of dt. Was `isave = 30` steps at
    # dt=1s (i.e. a save every 30 s); this reproduces that same 30 s cadence.
    save_every_s = 30
    isave = max(Int(round(save_every_s / dt)), 1)
    var2save = ["U", "Kq", "Nu", "C", "Kz", "L", "Q2", "Q2L", "N_BV2"]

    create_output_dict(M, isave, var2save, N)

    # Create depth vector 
    z = collect(H:-dz:dz) .- dz/2 # depth vector

    #********************** FIXED CONSTANTS  ***************************
    rhoA = 1.23                     # Density of air, kg/m^3
    rhoW = 1000                     # Density of water, kg/m^3
    specific_heat_water = 4181      # J/kg-degC
    specific_heat_air = 1007        # J/kg-degC x RH
    c_d = 0.05                      # Drag coefficient [-]
    cm2m = 0.01
    hr2s = 1/3600
    base_temp = 22.0                   # Base temperature for density calculation
    #********************** DEFINE HYDRODYNAMIC FORCINGS ***************************
    # NOTE: no longer used now that hydrodynamics are read from hydro.nc
    # instead of solved internally. Left here in case you want to switch back
    # to the internal solver later.
    # (1) PRESSURE 
    Px0 = 2e-4          # Pressure gradient forcing
    T_Px = 6           # Period [hours] on pressure gradient forcing. Set to 0 for steady

    # (2) Wind
    W0 = 2.5
    # Wind = 1                       # u_star =m/s >> 0.05 is  drag coefficient, 10 is my wind speed 
    # WIND = (c_d * Wind)^2 * rhoA   # this is rho * u*^2

    # ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
    Ns = 40                  # Number of sediment size classes
    ssc0 = zeros(N, Ns)  #.+ 1   # Matrix for sediment concentration (Nz x Ns)
    ssc0[:, 1:30] .= 5e2    # Matrix for sediment concentration (Nz x Ns)
    D = logrange(10e-6, 1500e-6, Ns)      # Sediment grain sizes (\mu m )
    D = collect(D)



    init_params!(D, Ns, ssc0[1,:])

    # Calculate settling velocities for each size class
    ws = floc_mod.ws[]  #.* 10
    for i in 1:Ns
        CFL = ws[i] * dt / dz
        println("Size class $i: ws = $(ws[i]*100) cm/s, CFL = $CFL")
    end

    add_sediment_to_output(Ns, isave, M, N)

    #***************************************************************************
    #   Initialize variables
    #***************************************************************************

    # Create dictionary to hold important discretization parameters
    discretization = Dict("beta" => (dt/dz^2), "dz" => dz, "dt" => dt, "N" => N, "z"=> z, "H" => H)

    #********************** LOAD HYDRODYNAMIC FORCING FROM FILE ***************************
    # Hydrodynamics are no longer solved internally -- they're read from
    # hydro.nc (nominally 1 s resolution) and linearly interpolated onto this
    # model's (finer) dt time grid every substep.
    hydro_fn = "hydro.nc"
    hydro_vars = ["U", "Kq", "Nu", "C", "Kz", "L", "Q2", "Q2L", "N_BV2"]

    local t_hydro::Vector{Float64}
    local hydro_data::Dict{String, Matrix{Float64}}
    NCDataset(hydro_fn, "r") do hds
        t_hydro = Float64.(Array(hds["time"]))  # seconds
        if haskey(hds, "z")
            z_hydro = Array(hds["z"])
            if length(z_hydro) != N
                error("hydro.nc has $(length(z_hydro)) depth levels but this run uses N=$N. " *
                      "Vertical grids must match -- this script only interpolates in time, not space.")
            end
        end
        missing_vars = filter(v -> !haskey(hds, v), hydro_vars)
        if !isempty(missing_vars)
            error("hydro.nc is missing expected hydrodynamic variable(s): $(join(missing_vars, ", "))")
        end
        hydro_data = Dict(v => Float64.(Array(hds[v])) for v in hydro_vars)  # each (z, time)
    end

    @info "Loaded hydrodynamic forcing from $hydro_fn: $(length(t_hydro)) timestamps " *
          "spanning $(t_hydro[1])s to $(t_hydro[end])s"
    if total_duration_s > (t_hydro[end] - t_hydro[1])
        @warn "Requested run duration ($(total_duration_s)s) exceeds the time span available " *
              "in hydro.nc ($(t_hydro[end]-t_hydro[1])s). Timestamps beyond the file's range " *
              "will be held fixed at the last available value."
    end

    function get_hydro_at(time::Real)
        return Dict(v => interp_time(hydro_data[v], t_hydro, time) for v in hydro_vars)
    end
    #****************************************************************************************

    # Initial dictionary to store variables
    variables = Dict() 
    Times = collect(0:dt:(M*dt))

    hydro0 = get_hydro_at(Times[1])
    variables["U"]     = hydro0["U"]
    variables["C"]     = hydro0["C"]
    variables["N_BV2"] = hydro0["N_BV2"]
    variables["Q2"]    = hydro0["Q2"]
    variables["Q2L"]   = hydro0["Q2L"]
    variables["Q"]     = sqrt.(max.(hydro0["Q2"], 0))
    variables["L"]     = hydro0["L"]
    variables["Nu"]    = hydro0["Nu"]
    variables["Kq"]    = hydro0["Kq"]
    variables["Kz"]    = hydro0["Kz"]
    variables["SSC"]   = ssc0

    #***************************************************************************
    # Save initial condition
    save2output(1, 1, "U", variables["U"])
    save2output(1, 1, "Kz", variables["Kz"])
    save2output(1, 1, "C", variables["C"])
    save2output(1, 1, "L", variables["L"])
    save2output(1, 1, "Q2", variables["Q2"])
    save2output(1, 1, "Q2L", variables["Q2L"])
    save2output(1, 1, "N_BV2", variables["N_BV2"])
    save2output(1, 1, "Kq", variables["Kq"])
    save2output(1, 1, "Nu", variables["Nu"])
    save_sediment2output(1, 1, ssc0)
    real_times_saved = [Times[1]]
    #***************************************************************************


    #*************************** TIME LOOP  *********************************
    for i in 2:(M-1)
        time = Times[i];

        # [1]-[8] Hydrodynamic state is no longer solved internally -- every
        # substep it's looked up from hydro.nc (1 s resolution) and linearly
        # interpolated onto this model's dt time grid.
        hydro_t = get_hydro_at(time)
        U     = hydro_t["U"]
        C     = hydro_t["C"]
        N_BV2 = hydro_t["N_BV2"]
        Q2    = hydro_t["Q2"]
        Q2L   = hydro_t["Q2L"]
        Q     = sqrt.(max.(Q2, 0))
        L     = hydro_t["L"]
        nu_t  = hydro_t["Nu"]
        Kq    = hydro_t["Kq"]
        Kz    = hydro_t["Kz"]

        # Turbulent shear driving the floc model -- computed exactly the same
        # way as the original internal hydro solver did, so floc behavior is
        # unaffected by this change.
        turbulent_shear = (Q2 ./ Q2L) .* Q .* 100
        turbulent_shear[end] = 0.1

        # ***************************************************************
        # [8] Advance sediment concentrations for each size class
        ssc = variables["SSC"]

        gamma = similar(ssc) 
        gamma .= 0
        if floc_on  
            # At each depth 'N' evaluate and update the sediment distribution 
            for ix in 1:N           # sed distribution ~  Ns ~ Shear ~ dt 
                ssc[ix,:] .= run_floc_mod(ssc[ix, :], Ns, turbulent_shear[ix], dt) 
            end
        end

        # # println("gamma = ", gamma)
        # for i_sed_class in 1:Ns                          # ssc ~ (Nz x Ns)
        #     s0 = advance_sediment3(variables, ssc[:,i_sed_class], -ws[i_sed_class], gamma[:,i_sed_class], discretization)
        #     ssc[:, i_sed_class] = s0 

        
        # [9] Pack variables for next timestep 
        variables["U"] = U
        variables["C"] = C
        variables["N_BV2"] = N_BV2
        variables["Nu"] = nu_t
        variables["Q2"] = Q2
        variables["Q2L"] = Q2L
        variables["Kq"] = Kq
        variables["Kz"] = Kz
        variables["L"] = L
        variables["SSC"] = ssc

        if i % isave == 0
           
            index = div(i, isave) + 1 
            save2output(time, index, "U", variables["U"])
            save2output(time, index, "Kz", variables["Kz"])
            save2output(time, index, "C", variables["C"])
            save2output(time, index, "L", variables["L"])
            save2output(time, index, "Q2", variables["Q2"])
            save2output(time, index, "Q2L", variables["Q2L"])
            save2output(time, index, "N_BV2", variables["N_BV2"])
            save2output(time, index, "Nu", variables["Nu"])
            save2output(time, index, "Kq", variables["Kq"])
            save_sediment2output(time, index, ssc)
            push!(real_times_saved, time)
        end
    end
     
    # ********************** save data ****************************
    units_dict = Dict("U" => "m/s", 
        "C" => "deg C", 
        "Kz" => "m\$^2\$ s\$^{-1}\$", 
        "L" => "Turbulent length scale", 
        "Q2" => "TKE", "Q2L" => "TKE*L",
        "N_BV2" => "Brunt-Vaisala frequency", "Kq" => "Kq", "Nu" => "Nu_t")

    var2name = Dict("U" => "Velocity", 
                "C" => "Temperature", 
                "Kz" => "Turbulent diffusivity", 
                "L" => "Turbulent length scale", 
                "Q2" => "TKE","Q2L" => "TKE*L",
                "N_BV2" => "Brunt-Vaisala frequency", "Kq" => "Kq", "Nu" => "Nu_t")

    # times_unique = unique(times) 
    nt = div(M,isave) #+ 1
    
    ds = NCDataset(file_out_name,"c")
    ds.attrib["title"] = "testing"

    # model_time = collect(1:M)
    defDim(ds, "z", length(z)) 
    defDim(ds, "time", nt)

    # println("Length of times_unique is ", size(t2))
    println("Length of SSC is ", size(output["ssc"]))

    defDim(ds, "Ds", length(D))
    v = defVar(ds, "Ds", Float32, ("Ds",))
    v[:] = D



    # N, Ns, n_saved_steps
    v = defVar(ds, "ssc", Float64,("z", "Ds", "time"), attrib = OrderedDict(
        "units" =>  "parts/m3", "long_name" => "suspended sediment concentration"))
    v[:,:,:] = output["ssc"]


    D1, M1 = get_particle_density() 
    v = defVar(ds, "np", Float64, ("Ds",), attrib = OrderedDict(
        "units" =>  "particle/floc", "long_name" => "number primary particle per floc"))
    v[:] = D1

    v = defVar(ds, "mass", Float64, ("Ds",), attrib = OrderedDict(
        "units" =>  "kg/particle", "long_name" => "mass of each floc (fractal!)"))
    v[:] = M1

    v = defVar(ds, "z", Float32, ("z",))
    v[:] = z

    v = defVar(ds, "time", Float32, ("time",), attrib = OrderedDict("units" => "seconds"))
    v[:] = real_times_saved #real_times_saved #collect(1:nt) #model_time

    for var in var2save
        v = defVar(ds, var, Float64,("z","time"), attrib = OrderedDict(
        "units" =>  units_dict[var], "long_name" => var2name[var]))
        v[:,:] = output[var];
    end

    print("Saved $file_out_name \n")
    close(ds)

end 

# file_out_name = @sprintf("hydro.nc") 
run_my_model("hydro_FlocOn_4.nc", true)
# run_my_model("hydro_FlocOff.nc", false)



# using StatProfilerHTML 
# # using ProfileView   
# using Profile 




# @profilehtml run_my_model(ws1, ws2, pmax1, pmax2, file_out_name)

# StatProfilerHTML.view()
# Profile.print()