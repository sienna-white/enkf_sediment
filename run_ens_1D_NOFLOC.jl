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
Nens = 50 #30 
H = 4   # depth (meters)
dz = H/N # grid spacing - may need to adjust to reduce oscillations

 #********************** SPATIAL DOMAIN  ***************************
 dt = 1 #0.5 #0.5    # (seconds) size of time step 
 M = 3600*12  #000 *5#72000*100 #*24*12 # 15 hours @ dt = 0.05
 # Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
 isave = 60# 00 #6000 
#  var2save = ["G", "alpha", "beta"]
# Create vector to hold the time steps 

Times = collect(0:dt:(M*dt))
println("Initialized time vector of length $M")
real_times_saved = [Times[1]]
var2save = ["G", "Kz", "U"]

create_output_dict(M, isave, var2save, N)

# ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
println("Initializing sediment properties...")
Ns = 2               # Number of sediment size classes
# Set settling speeds
ws = zeros(N, Ns)     # Matrix for sediment concentration (Nz x Ns)
ws[:, 1] = 1.922e-6 .+ randn(N)*1e-6
ws[:, 2] = 2.375e-5 .+ randn(N)*1e-5

# ssc0 = zeros(N, Ns)     # Matrix for sediment concentration (Nz x Ns)
# ssc0[:, 1:2] .= 5e12 # Matrix for sediment concentration (N x Ns)
# ssc0 = ssc0 .+ abs.(randn(N, Ns)) 

# 
ssc = zeros(Nens, N, Ns)
ssc[:,:,1] .= 5e12 
ssc[:,:,2] .= 5e10  # Matrix for sediment concentration (N x Ns)

# D = [10e-6, 50e-6]

D = [2.6826958e-06, 1.9306977e-05]
 # Volume of each size class (m^3)
#  Volumes = (4/3)*pi*(D./2).^3

 #***************************************************************************
add_sediment_to_output(Ns, isave, M, Nens, N)
 #***************************************************************************

 #********************** LOAD HYDRODYNAMIC FORCING FROM FILE ***************************
# Hydrodynamics are no longer solved internally -- they're read from
# hydro.nc (nominally 1 s resolution) and linearly interpolated onto this
# model's (finer) dt time grid every substep.

println("Setting up interpolator for hydrodynamic data...")

function nearest_time(data::AbstractMatrix{<:Real}, t_data::AbstractVector{<:Real}, t_query::Real)
    j = searchsortedfirst(t_data, t_query)
    if j <= 1
        return data[:, 1]
    elseif j > length(t_data)
        return data[:, end]
    else
        # j is the first index with t_data[j] >= t_query; compare it against j-1
        # to find whichever timestamp is actually closest
        return abs(t_data[j] - t_query) <= abs(t_query - t_data[j-1]) ? data[:, j] : data[:, j-1]
    end
end


hydro_fn = "hydro_newZ.nc"
hydro_vars = ["U", "Kq", "Nu", "C", "Kz", "L", "Q2", "Q2L", "N_BV2"]
hydro_vars = ["Kq", "Nu",  "Kz", "Q2", "Q2L", "U"]

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
    t_query = time + 9600 
    return Dict(v => nearest_time(hydro_data[v], t_hydro, t_query) for v in hydro_vars)
end


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
# z = collect(H:-dz:dz) .- dz/2 
z = collect(dz:dz:H) .- dz/2 
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



function run_forward_model(EID, ssc_past, diffusivity, shear)
    ssc_ = ssc_past
    ssc_next = similar(ssc_)
    Gammas = zeros(N, Ns) 

    # Now loop through sediment classes 
    for sed in 1:Ns
        ssc_next[:,sed] = advance_sediment3(ssc_[:,sed], diffusivity, ws[EID,sed], Gammas[:,sed], discretization)
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
    # Kq    = hydro_t["Kq"]
    Kz    = hydro_t["Kz"]
    U    = hydro_t["U"]

    # Turbulent shear driving the floc model 
    turbulent_shear = (Q2 ./ Q2L) .* Q .* 100
    turbulent_shear[end] = 0.1

    # println("On time $i")

    for EID in 1:Nens
        # println("Running ensemble member $EID")

        ssc[EID, :, :] = run_forward_model(EID, ssc[EID, :, :], Kz, turbulent_shear)

        if i % isave == 0
            index = div(i, isave) + 1
            # save2output(index, "G", get_shear(time))
            # save_sediment2output(index, ssc, "ssc")
            save_sediment2output_ens(EID, index, ssc[EID, :, :], "ssc")
            if EID==1
                println("Saving @ $i/$M $(i/M)")
                save2output(index, "G", turbulent_shear)
                save2output(index, "Kz", Kz)
                save2output(index, "U", U)

                push!(real_times_saved, time)
                flush(stdout)

            end 
        end
    end
end 



fout = "sediment_1D_NOFLOCMOD.nc"


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
    v2 = defVar(ds, var, Float64,("z", "time"))
    v2[:,:] = output[var];
end

print("Saved $fout \n")
close(ds)



    # detrended = total_sediment .- ensemble_mean
    # ensemble_covariance = (detrended * transpose(detrended)) *  1/(N_ensemble - 1)