#!/usr/bin/env julia

# using Plots
using Printf
using DataStructures: OrderedDict
using NCDatasets
# using Arrow, DataFrames
using CSV, DataFrames
# using Colors
# using ColorSchemes
# using Plots
using Printf
# using LaTeXStrings
using Profile
using Statistics 
using LinearAlgebra
using Random

fp="/global/homes/s/siennaw/scratch/siennaw/two_species/adjoint_phytoplankton/model_code"
include("old/model_code/calculate_physical_variables.jl") 
include("old/model_code/advance_variables.jl")
include("old/model_code/phytoplankton.jl")
include("old/model_code/forcings.jl") 
include("old/model_code/output.jl")
include("old/model_code/define_params.jl")

#********************** SPATIAL DOMAIN  ***************************
N = 40   # number of grid points
Nens = 40 
H = 4   # depth (meters)
dz = H/N # grid spacing - may need to adjust to reduce oscillations
dt = 10 # (seconds) size of time step
M = 360*3  #360*2 # number of time steps


########
 #********************** SPATIAL DOMAIN  ***************************
 N = 100 #35 #50 #250    # number of ensembles points
 dt = 0.05 #0.5 #0.5    # (seconds) size of time step 
 M = 8640 #000 *5#72000*100 #*24*12 # 15 hours @ dt = 0.05
 # Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
 isave = 300# 00 #6000 
#  var2save = ["G", "alpha", "beta"]

#  create_output_dict(M, isave, var2save, N)
 #********************** FIXED CONSTANTS  ***************************
 rhoW = 1000                     # Density of water, kg/m^3
 specific_heat_water = 4181      # J/kg-degC
 specific_heat_air = 1007        # J/kg-degC x RH

 # ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
 Ns = 36 #40                 # Number of sediment size classes

 ssc0 = zeros(N, Ns)     # Matrix for sediment concentration (Nz x Ns)
 ssc0[:, 1:20] .= abs.(randn(N, 20))*1e10  # Matrix for sediment concentration (N x Ns)
 ssc0 = ssc0 .+ abs.(randn(N, Ns)) 

 # D = collect(logrange(1e-6, 1000e-6, Ns))      # Sediment grain sizes (\mu m )

 D  = [  1,   1.6 ,   1.89,  2.23 ,  2.63,  3.11,  3.67,   4.33,   5.11 ,  6.03,
               7.11,   8.39,   9.9,   11.7,   13.8,  16.3,  19.2,   22.7,   26.7,   31.6,
               37.2,   43.9,   51.9,  61.2,   72.2,  85.2,  101.,   119.,   140.,   165.,
               195.,   230.,   273.,  324.,   386.,  459.]
 D = D .* 1e-6 # convert to meters

 # Volume of each size class (m^3)
 Volumes = (4/3)*pi*(D./2).^3

 # Only need to calculate this once, can pass to all sediment parameter sets 
 collision_matrix = calculate_collision_matrix(D, Ns)

 #***************************************************************************
#***************************************************************************
alpha0 = 0.3 #35
beta0  = 0.1
nf0 = 2.1 # change?  
beta20 = 1.5

Alphas = randn(N) 
Betas = randn(N)
Nfs = randn(N)
Beta2s = randn(N)

for EID in 1:N
    Alphas[EID] = alpha0 * exp(0.1*Alphas[EID]) 
    Betas[EID] = beta0 * exp(0.1*Betas[EID]) 
    Beta2s[EID] = beta20 * exp(0.1*Beta2s[EID]) 
    Nfs[EID] = nf0 * exp(0.1* Nfs[EID])  
end 

variables = Dict() 
variables["SSC1"] = ssc0
variables["SSC2"] = ssc0

floc_params_list = Vector{Any}(undef, Nens)
ws_list = Vector{Any}(undef, Nens)
variables = Dict() 
variables["SSC1"] = ssc0
variables["SSC2"] = ssc0

for EID in 1:N
    floc_params = init_params(D, Ns, ssc0[EID,:], Alphas[EID], Betas[EID], Beta2s[EID], Nfs[EID], collision_matrix)
    floc_params_list[EID] = floc_params
    ws_list[EID] = floc_params.ws
    println("Ws is,", ws*100)
end 


 #***************************************************************************
 add_sediment_to_output(Ns, isave, M, Nens)

if M <= 0
    error("M must be greater than 0. Check istart and iend values.")
end

# time_index_vec = collect(istart:(iend+1))
time_index_vec = collect(1:M)

file_out_name = "floc_1D_model.nc"

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



# Hydrodynamic dataset 
# ds = NCDataset("hydro.nc")

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


######################################################################################################

# Calculate settling velocities for each size class
ws = floc_params.ws
for i in 1:Ns
    println("\t Size class $i: ws = $(ws[i]*100) cm/s")
    @printf("\t Size class %d, ws =  %2.2f cm/s\n", i, (ws[i_ens]*100))
end


# Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
isave = 1 

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

floc1 = Dict("d50" => 20e-6,             # mean diameter [m]
        "rho_sed" => 1001,          # specific density [kg/m^3]
        "ws" => 0.01,                # vertical velocity [m/s]
        "name" => "floc1")                #  name 

floc2 = Dict("d50" => 20e-6,             # mean diameter [m]
        "rho_sed" => 1001,          # specific density [kg/m^3]
        "ws" => 0.03,                # vertical velocity [m/s]
        "name" => "floc2")   
                
floc3 = Dict("d50" => 20e-6,             # mean diameter [m]
        "rho_sed" => 1001,          # specific density [kg/m^3]
        "ws" => 0.05,                # vertical velocity [m/s]
        "name" => "floc3")   
                



# Create dictionary to hold important discretization parameters
discretization = Dict("beta" => (dt/dz^2), "dz" => dz, "dt" => dt, "N" => N, "z"=> z, "H" => H)

N_ensemble = 500

output = Dict()

var2save = ["floc1","floc2", "floc3"]     

for var in var2save
    output[var] = zeros(Float64, N, M,  N_ensemble)
end 

init_conc = 115 

for j in 1:N_ensemble
   output["floc1"][:,1,j] .=  init_conc  .+ (abs.(randn(Float64, (N)))).*20
   output["floc2"][:,1,j] .=  init_conc  .+ (abs.(randn(Float64, (N))*20))
   output["floc3"][:,1,j] .=  init_conc  .+ (abs.(randn(Float64, (N))*20))
end

# Create vector to hold the time steps 
Times = collect(1:dt:(M*dt))
println("Initialized time vector of length $M")

# Generate fake observation 
observations = @. sin(Times/3600) 

                                          # N x M x  N_ensemble
variables = Dict("floc1" => output["floc1"][:, 1, :],
                 "floc2" => output["floc2"][:, 1, :], 
                 "floc3" => output["floc3"][:, 1, :])

# make variables a global variable
global variables

function run_forward_model(EID, it, time, Diffusivity, variables)



        ws = floc1["ws"]  * abs(rand(0:4)) 
        variables["floc1"][:, EID] = advance_sediment(variables["floc1"][:, EID], Diffusivity, floc1, gamma1, discretization, ws)
        
        ws = floc2["ws"]  * abs(rand(0:4))
        variables["floc2"][:, EID] = advance_sediment(variables["floc2"][:, EID], Diffusivity, floc2, gamma2, discretization, ws) 

        ws = floc3["ws"]  * abs(rand(0:4))
        variables["floc3"][:, EID] = advance_sediment(variables["floc3"][:, EID], Diffusivity, floc3, gamma2, discretization, ws)
                                # N  x  N_ensemble

        # Inject randomness 
        variables["floc1"][:, EID] = variables["floc1"][:, EID] .+ randn(Float64, (N))
        variables["floc2"][:, EID] = variables["floc2"][:, EID] .+ randn(Float64, (N))
        variables["floc3"][:, EID] = variables["floc3"][:, EID]  .+ randn(Float64, (N))
        return variables
end


for i in 2:(M-1)
    time = Times[i];

    index = time_index_vec[i]
    time = index 

    # [1]-[8] Hydrodynamic state is no longer solved internally -- every
    # substep it's looked up from hydro.nc (1 s resolution) and linearly
    # interpolated onto this model's dt time grid.
    hydro_t = get_hydro_at(time)
    U     = hydro_t["U"]
    C     = hydro_t["C"]
    N_BV2 = hydro_t["N_BV2"]
    Q2    = hydro_t["Q2"]
    Q2L   = hydro_t["Q2L"]
    L     = hydro_t["L"]
    nu_t  = hydro_t["Nu"]
    Kq    = hydro_t["Kq"]
    Kz    = hydro_t["Kz"]

    # Turbulent shear driving the floc model -- computed exactly the same
    # way as the original internal hydro solver did, so floc behavior is
    # unaffected by this change.
    turbulent_shear = (Q2 ./ Q2L) .* Q .* 100
    turbulent_shear[end] = 0.1

    # println("On time $i")

    for EID in 1:N_ensemble
        # println("Running ensemble member $EID")
        run_forward_model(EID, index, time, Diffusivity, variables)

        output["floc1"][:,i, EID] = variables["floc1"][:, EID]
        output["floc2"][:,i, EID] = variables["floc2"][:, EID]
        output["floc3"][:,i, EID] = variables["floc3"][:, EID]
        # println("EID = $EID completed")
        # println("floc1 = $(variables["floc1"][1:5, EID])")
    end

    if i % isave == 0
        index = div(i, isave)
        # save2output(index, "G", get_shear(time))
        # save_sediment2output(index, ssc, "ssc")
        save_sediment2output_ens(EID, index, ssc, "ssc")
        push!(real_times_saved, time)
    end


end 



# ********************** save data ****************************
units_dict = Dict(
    "floc1" => "conc1",
    "floc2" => "conc2",
    "floc3" => "conc3")

var2name = Dict("floc1" => "HAB concentration",
            "floc2" => "Diatom concentration",
            "floc3" => "Cyanobacteria concentration")

fout = "test_sediment.nc"

ds = NCDataset(fout,"c")
nt = div(M,isave) + 1 
defDim(ds, "z", length(z)) 
defDim(ds, "t", length(time_index_vec))
defDim(ds, "eid", N_ensemble)

v = defVar(ds, "z", Float32, ("z",))
v[:] = z

v = defVar(ds, "t", Int, ("t",), attrib = OrderedDict("units" => "s"))
v[:] = time_index_vec #collect(1:nt)

v = defVar(ds, "eid", Int, ("eid",))
v[:] = collect(1:N_ensemble)

observed_var = defVar(ds, "observations", Float64, ("t",), attrib = OrderedDict(
    "units" => "n/a", "long_name" => "Observed concentration"))
observed_var[:] =  observed_values

for var in var2save
    v2 = defVar(ds, var, Float64,("z","t", "eid" ), attrib = OrderedDict(
    "units" =>  units_dict[var], "long_name" => var2name[var]))
    v2[:,:,:] = output[var];
end

print("Saved $file_out_name \n")
close(ds)



    # detrended = total_sediment .- ensemble_mean
    # ensemble_covariance = (detrended * transpose(detrended)) *  1/(N_ensemble - 1)