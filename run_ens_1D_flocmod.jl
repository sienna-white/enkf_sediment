#!/usr/bin/env julia
using Printf
using DataStructures: OrderedDict
using NCDatasets
# using Arrow, DataFrames
using CSV, DataFrames
using Printf
using Profile
using Statistics 
using LinearAlgebra
using Random
using Compat # for older julia versoins 


fp="/global/homes/s/siennaw/scratch/siennaw/two_species/adjoint_phytoplankton/model_code"
include("model1D/calculate_physical_variables.jl") 
include("model1D/advance_variables.jl")
include("model1D/phytoplankton.jl")
include("model1D/forcings.jl") 
include("model1D/output.jl")
include("floc_mod_enkf.jl")
using .floc_mod

#********************** SPATIAL DOMAIN  ***************************
N = 50   # number of grid points
Nens = 5 
H = 4   # depth (meters)
dz = H/N # grid spacing - may need to adjust to reduce oscillations

 #********************** SPATIAL DOMAIN  ***************************
 dt = 0.05 #0.5 #0.5    # (seconds) size of time step 
 M = 200 #12000*20  #000 *5#72000*100 #*24*12 # 15 hours @ dt = 0.05
 # Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
 isave = 10 #1200# 00 #6000 
 var2save = ["G", "Kz"]
# Create vector to hold the time steps 

Times = collect(0:dt:(M*dt))
println("Initialized time vector of length $M")
real_times_saved = [Times[1]]

create_output_dict(M, isave, var2save, N)

 # ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
 println("Initializing sediment properties...")
 Ns = 36 #40                 # Number of sediment size classes

 ssc0 = zeros(N, Ns)     # Matrix for sediment concentration (Nz x Ns)
 ssc0[:, 1:20] .= abs.(randn(N, 20))*1e10  # Matrix for sediment concentration (N x Ns)
 ssc0 = ssc0 .+ abs.(randn(N, Ns)) 

 D = collect(logrange(1e-6, 1000e-6, Ns))      # Sediment grain sizes (\mu m )

#  D  = [  1,   1.6 ,   1.89,  2.23 ,  2.63,  3.11,  3.67,   4.33,   5.11 ,  6.03,
#                7.11,   8.39,   9.9,   11.7,   13.8,  16.3,  19.2,   22.7,   26.7,   31.6,
#                37.2,   43.9,   51.9,  61.2,   72.2,  85.2,  101.,   119.,   140.,   165.,
#                195.,   230.,   273.,  324.,   386.,  459.]
#  D = D .* 1e-6 # convert to meters

 # Volume of each size class (m^3)
 Volumes = (4/3)*pi*(D./2).^3

 # Only need to calculate this once, can pass to all sediment parameter sets 
 collision_matrix = calculate_collision_matrix(D, Ns)

alpha0 = 0.3 
beta0  = 0.1
nf0 = 2.2  
beta20 = 1.5

Alphas = randn(N) 
Betas = randn(N)
Nfs = randn(N)
Beta2s = randn(N)

floc_params_list = Vector{Any}(undef, Nens)
ws = Vector{Any}(undef, Nens)

for EID in 1:Nens
    Alphas[EID] = alpha0 * exp(0.1*Alphas[EID]) 
    Betas[EID] = beta0 * exp(0.1*Betas[EID]) 
    Beta2s[EID] = beta20 * exp(0.1*Beta2s[EID]) 
    Nfs[EID] = nf0 * exp(0.1* Nfs[EID])  
    floc_params = init_params(D, Ns, ssc0[EID,:], Alphas[EID], Betas[EID], Beta2s[EID], Nfs[EID], collision_matrix)
    floc_params_list[EID] = floc_params
    ws[EID] = floc_params.ws
end 

 #***************************************************************************
add_sediment_to_output(Ns, isave, M, Nens, N)
file_out_name = "floc_1D_model.nc"
 #***************************************************************************



 #********************** LOAD HYDRODYNAMIC FORCING FROM FILE ***************************
# Hydrodynamics are no longer solved internally -- they're read from
# hydro.nc (nominally 1 s resolution) and linearly interpolated onto this
# model's (finer) dt time grid every substep.

println("Setting up interpolator for hydrodynamic data...")

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


hydro_fn = "hydro.nc"
hydro_vars = ["U", "Kq", "Nu", "C", "Kz", "L", "Q2", "Q2L", "N_BV2"]
hydro_vars = ["Kq", "Nu",  "Kz", "Q2", "Q2L"]

local t_hydro::Vector{Float64}
local hydro_data::Dict{String, Matrix{Float64}}


function load_hydro_forcing(hydro_fn::String, hydro_vars::Vector{String}, N::Int)
    ds = NCDataset(hydro_fn, "r")
    local t_hydro, hydro_data
    try
        if !haskey(ds, "time")
            error("hydro.nc has no \"time\" variable.")
        end
        t_hydro = Float64.(Array(ds["time"]))  # seconds
 
        if haskey(ds, "z")
            z_hydro = Array(ds["z"])
            if length(z_hydro) != N
                error("hydro.nc has $(length(z_hydro)) depth levels but this run uses N=$N. " *
                      "Vertical grids must match -- this script only interpolates in time, not space.")
            end
        end
 
        missing_vars = filter(v -> !haskey(ds, v), hydro_vars)
        if !isempty(missing_vars)
            error("hydro.nc is missing expected hydrodynamic variable(s): $(join(missing_vars, ", "))")
        end
 
        hydro_data = Dict(v => Float64.(Array(ds[v])) for v in hydro_vars)  # each (z, time)
    finally
        close(ds)
    end
    return t_hydro, hydro_data
end

t_hydro, hydro_data = load_hydro_forcing(hydro_fn, hydro_vars, N)


function get_hydro_at(time::Real)
    return Dict(v => interp_time(hydro_data[v], t_hydro, time) for v in hydro_vars)
end
#****************************************************************************************


######################################################################################################

# Calculate settling velocities for each size class
# ws = floc_params.ws
# for i in 1:Ns
#     println("\t Size class $i: ws = $(ws[i]*100) cm/s")
#     @printf("\t Size class %d, ws =  %2.2f cm/s\n", i, (ws[i_ens]*100))
# end


# Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
# isave = 1 

# Create depth vector 
z = collect(H:-dz:dz) .- dz/2 

#********************** FIXED CONSTANTS  ***************************
rhoA = 1.23                     # Density of air, kg/m^3
rhoW = 1000                     # Density of water, kg/m^3
specific_heat_water = 4181      # J/kg-degC
specific_heat_air = 1007        # J/kg-degC x RH
c_d = 0.05                      # Drag coefficient [-]
cm2m = 0.01
hr2s = 1/3600

#********************** DEFINE SEDIMENT PARAMETERS ***************************

# Create dictionary to hold important discretization parameters
discretization = Dict("beta" => (dt/dz^2), "dz" => dz, "dt" => dt, "N" => N, "z"=> z, "H" => H)


ssc = zeros(Nens, N, Ns)
ssc[:, :, 1:20] .= 1e10  # Matrix for sediment concentration (N x Ns)


# Nens, N, Ns, n_saved_steps

# make variables a global variable

function run_forward_model(EID, ssc_past, diffusivity, shear)

    ssc_ = ssc_past
    ssc_next = similar(ssc_)
    
    Gammas = zeros(N, Ns) 
    for depth in 1:N # Calculate floc mod ROC over each depth
        Gammas[depth,:] = run_floc_mod_gamma(floc_params_list[EID], ssc_[depth, :], Ns,  shear[depth], dt) 
        # ssc1[ssc1 .* vec(Volumes) .> 15e-5] .= 100

    end 
    Gammas[.!isfinite.(Gammas)] .= 0 

    # Now loop through sediment classes 
    for sed in 1:Ns
        ssc_next[:,sed] = advance_sediment3(ssc_[:,sed], diffusivity, ws[EID][sed], Gammas[:,sed], discretization)
    end 
    clamp!(ssc_next, 0, Inf)
    return ssc_next

end


for i in 2:(M-1)
    time = Times[i];

    # [1]-[8] Hydrodynamic state is no longer solved internally -- every
    # substep it's looked up from hydro.nc (1 s resolution) and linearly
    # interpolated onto this model's dt time grid.
    hydro_t = get_hydro_at(time)

    Q2    = hydro_t["Q2"]
    Q     = sqrt.(max.(Q2, 0))
    Q2L   = hydro_t["Q2L"]
    Kq    = hydro_t["Kq"]
    Kz    = hydro_t["Kz"]

    # Turbulent shear driving the floc model 
    turbulent_shear = (Q2 ./ Q2L) .* Q .* 100
    turbulent_shear[end] = 0.1

    # println("On time $i")

    for EID in 1:Nens
        # println("Running ensemble member $EID")

        ssc[EID, :, :] = run_forward_model(EID, ssc[EID, :, :], Kz, turbulent_shear)

        if i % isave == 0
            index = div(i, isave)
            # save_sediment2output(index, ssc, "ssc")
            save_sediment2output_ens(EID, index, ssc[EID, :, :], "ssc")
            if EID==1
                println("Saving @ $i/$M $(i/M)")
                save2output(index, "G", turbulent_shear)
                save2output(index, "Kz", Kz)


                push!(real_times_saved, time)
            end 
        end
    end
end 



fout = "test_sedimentDEL.nc"


# ********************** save data ****************************
units_dict = Dict(
    "ssc" => "ssc")

var2name = Dict("ssc" => "HAB concentration",
            "floc2" => "Diatom concentration",
            "floc3" => "Cyanobacteria concentration")


ds = NCDataset(fout,"c")
nt = div(M,isave) + 1 

# Define Dimensions 
defDim(ds, "Nens", Nens)
defDim(ds, "Ds", length(D))
defDim(ds, "z", length(z)) 
defDim(ds, "time", length(real_times_saved))


v = defVar(ds, "Ds", Float32, ("Ds",))
v[:] = D

v = defVar(ds, "z", Float32, ("z",))
v[:] = z

v = defVar(ds, "time", Int, ("time",), attrib = OrderedDict("units" => "s"))
v[:] = real_times_saved #collect(1:nt)


v = defVar(ds, "Nens", Int, ("Nens",))
v[:] = collect(1:Nens)


# Nens, N, Ns, n_saved_steps
v = defVar(ds, "ssc", Float64,("Nens", "z", "Ds", "time"), attrib = OrderedDict(
    "units" =>  "parts/m3", "long_name" => "suspended sediment concentration"))
v[:,:,:,:] = output["ssc"]



for var in var2save
    println("Saving ... $var")
    v2 = defVar(ds, var, Float64,("z", "time"))
    v2[:,:] = output[var];
end

print("Saved $file_out_name \n")
close(ds)



    # detrended = total_sediment .- ensemble_mean
    # ensemble_covariance = (detrended * transpose(detrended)) *  1/(N_ensemble - 1)