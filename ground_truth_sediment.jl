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
include("/global/homes/s/siennaw/scratch/siennaw/scripts/enkf_sediment/model_code/calculate_physical_variables.jl") 
include("/global/homes/s/siennaw/scratch/siennaw/scripts/enkf_sediment/model_code/advance_variables.jl")
include("/global/homes/s/siennaw/scratch/siennaw/scripts/enkf_sediment/model_code/phytoplankton.jl")
include("/global/homes/s/siennaw/scratch/siennaw/scripts/enkf_sediment/model_code/forcings.jl") 
include("/global/homes/s/siennaw/scratch/siennaw/scripts/enkf_sediment/model_code/output.jl")
include("/global/homes/s/siennaw/scratch/siennaw/scripts/enkf_sediment/model_code/define_params.jl")

#********************** SPATIAL DOMAIN  ***************************
N = 40   # number of grid points
H = 4   # depth (meters)
dz = H/N # grid spacing - may need to adjust to reduce oscillations
dt = 10 # (seconds) size of time step

println("Courant condition = $(0.01*dt/dz)")

# dz = 0.03 
# dt = 10 * 0.001 = 0.01  1e-2 * 10 = 0.01 / 
istart = global_params["istart"] # start time step
iend = global_params["iend"] # end time step
M = 360*12  #360*2 # number of time steps

if M <= 0
    error("M must be greater than 0. Check istart and iend values.")
end

# time_index_vec = collect(istart:(iend+1))
time_index_vec = collect(1:M)

file_out_name = "ground_truth_sediment.nc"

forcing_folder = "/pscratch/sd/s/siennaw/stockton_field_data/forcing_for_model/2024/august6-28/"


# Hydrodynamic dataset 
ds = NCDataset("/pscratch/sd/s/siennaw/two_species/adjoint_phytoplankton/run_hydro/HYDRO_AUGUST6-28.nc")


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
        "name" => "sand")                #  name 

floc2 = Dict("d50" => 20e-6,             # mean diameter [m]
        "rho_sed" => 1001,          # specific density [kg/m^3]
        "ws" => 5e-4,                # vertical velocity [m/s]
        "name" => "floc2")   
                
floc3 = Dict("d50" => 20e-6,             # mean diameter [m]
        "rho_sed" => 1001,          # specific density [kg/m^3]
        "ws" => 8e-5,                # vertical velocity [m/s]
        "name" => "floc3")   
                



# Create dictionary to hold important discretization parameters
discretization = Dict("beta" => (dt/dz^2), "dz" => dz, "dt" => dt, "N" => N, "z"=> z, "H" => H)


N_ensemble = 1

output = Dict()

var2save = ["floc1","floc2", "floc3"]      # Only save growth + algae 

for var in var2save
    output[var] = zeros(Float64, N, M)
end 

init_conc = 120 
output["floc1"][:,1] .=  init_conc  
output["floc2"][:,1] .=  init_conc  
output["floc3"][:,1] .=  init_conc  


# Create vector to hold the time steps 
Times = collect(1:dt:(M*dt))
println("Initialized time vector of length $M")

# Generate fake observation 
ws_time_varying = @. 1 + sin(Times/3600)^2 



                  # N x M x  N_ensemble
variables = Dict("floc1" => output["floc1"][:, 1],
                 "floc2" => output["floc2"][:, 1], 
                 "floc3" => output["floc3"][:, 1])

# make variables a global variable
global variables

function run_forward_model(EID, it, time, Diffusivity, variables)

        gamma1 = zeros(Float64, N)
        gamma2 = zeros(Float64, N)

        ws_multiplier = ws_time_varying[it]

        ws = 1e-4 #floc1["ws"]  # * ws_multiplier

        variables["floc1"][:] = advance_sediment(variables["floc1"][:], Diffusivity, floc1, gamma1, discretization, ws)
        println( variables["floc1"][end-10:end])

        ws = floc2["ws"]  * ws_multiplier
        variables["floc2"][:, EID] = advance_sediment(variables["floc2"][:, EID], Diffusivity, floc2, gamma2, discretization, ws)
        
        ws = floc3["ws"]  * ws_multiplier
        variables["floc3"][:, EID] = advance_sediment(variables["floc3"][:, EID], Diffusivity, floc3, gamma2, discretization, ws)
        #                         # N  x  N_ensemble
        return variables

end


for i in 2:M
    index = time_index_vec[i]
    time = index #Times[i];

    # Hydrodynamics
    Diffusivity = zeros(N) .+ 1e-4 # ds["Kz"][:,index]

    # println("On time $i")

    run_forward_model(1, index, time, Diffusivity, variables)
    output["floc1"][:,i] = variables["floc1"][:]
    output["floc2"][:,i] = variables["floc2"][:]
    output["floc3"][:,i] = variables["floc3"][:]
end 


# Create synthetic observation
OBS_LOC = N-10 
observation =  0.2 .* output["floc1"][OBS_LOC, :] 
observation += 0.5 .* output["floc2"][OBS_LOC, :] 
observation += 0.3 .* output["floc3"][OBS_LOC, :]

println("Created synthetic observations at location $OBS_LOC")
# println(observation)


# ********************** save data ****************************
units_dict = Dict(
    "floc1" => "conc1",
    "floc2" => "conc2",
    "floc3" => "conc3")

var2name = Dict("floc1" => "HAB concentration",
            "floc2" => "Diatom concentration",
            "floc3" => "Cyanobacteria concentration")

fout = "ground_truth_sediment.nc"

ds = NCDataset(fout,"c")
nt = div(M,isave) + 1 
defDim(ds, "z", length(z)) 
defDim(ds, "t", length(time_index_vec))
defDim(ds, "eid", N_ensemble)

v = defVar(ds, "z", Float32, ("z",))
v[:] = z

v = defVar(ds, "t", Int, ("t",), attrib = OrderedDict("units" => "s"))
v[:] = time_index_vec #collect(1:nt)



observed_var = defVar(ds, "observation", Float64, ("t",), attrib = OrderedDict(
    "units" => "n/a", "long_name" => "Observed concentration"))
observed_var[:] =  observation

for var in var2save
    v2 = defVar(ds, var, Float64,("z","t" ), attrib = OrderedDict(
    "units" =>  units_dict[var], "long_name" => var2name[var]))
    v2[:,:] = output[var];
end

print("Saved $file_out_name \n")
close(ds)

