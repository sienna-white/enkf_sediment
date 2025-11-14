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
N = global_params["N"]   # number of grid points
H = global_params["H"]   # depth (meters)
dz = global_params["dz"] # grid spacing - may need to adjust to reduce oscillations
dt = global_params["dt"] # (seconds) size of time step
time_range = global_params["time_range"] # number of time steps

istart = global_params["istart"] # start time step
iend = global_params["iend"] # end time step
M = 360 #360*2 # number of time steps

if M <= 0
    error("M must be greater than 0. Check istart and iend values.")
end

# time_index_vec = collect(istart:(iend+1))
time_index_vec = collect(1:M)


file_out_name = "test1_sediment.nc"

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
            "ws" => 1.38e-5,                # vertical velocity [m/s]
            "Li" => 1.38e-5,             # specific loss rate [1/hour]
            "name" => "floc1")                #  name 

floc2 = Dict("d50" => 20e-6,             # mean diameter [m]
            "rho_sed" => 1001,          # specific density [kg/m^3]
            "ws" => 1.e-4,                # vertical velocity [m/s]
            "Li" => 1.38e-6,             # specific loss rate [1/hour]
            "name" => "floc2")   
                
                



# Create dictionary to hold important discretization parameters
discretization = Dict("beta" => (dt/dz^2), "dz" => dz, "dt" => dt, "N" => N, "z"=> z, "H" => H)


N_ensemble = 500

output = Dict()

var2save = ["floc1","floc2"]      # Only save growth + algae 

for var in var2save
    output[var] = zeros(Float64, N, M,  N_ensemble)
end 

init_conc = 100 

for j in 1:N_ensemble
   output["floc1"][:,1,j] .=  init_conc  .+ (abs.(randn(Float64, (N)))).*100
   output["floc2"][:,1,j] .=  init_conc  .+ log.(abs.(randn(Float64, (N))*400))
end


# Create vector to hold the time steps 
Times = collect(1:dt:(M*dt))
println("Initialized time vector of length $M")

# Generate fake observation 
observations = @. sin(Times/3600) 

                                          # N x M x  N_ensemble
variables = Dict("floc1" => output["floc1"][:, 1, :],
                 "floc2" => output["floc2"][:, 1, :])

# make variables a global variable
global variables

function run_forward_model(EID, it, time, Diffusivity, variables)

        gamma1 = zeros(Float64, N)
        gamma2 = zeros(Float64, N)

        # Estimate source term 
        # for j in 1:N 
        #     growth1[j] = 1e-3
        #     growth2[j] = 1e-3
        # end

        # # Split up loss + growth
        # gamma1 = growth1 .- floc1["Li"]  # subtract the loss rate
        # gamma2 = growth2 .- floc2["Li"]  # subtract the loss rate
        ws = floc1["ws"]  * abs(rand()) 

        # Algae 
        variables["floc1"][:, EID] = advance_sediment(variables["floc1"][:, EID], Diffusivity, floc1, gamma1, discretization, ws)  
        # variables["floc2"][:, EID] = advance_sediment(variables["floc2"][:, EID], Diffusivity, floc2, gamma2, discretization, ws) 
                            # N  x  N_ensemble
        return variables

end


observations = 200  #zeros(Float64, N)
# observations[4] = 10 
println("observation inserted at z= $(z[4]) ")

# (10x10) (60x1)
# H X 

H0 =  zeros(1, N)  # zeros(Float64, N, N)
H0[50] = 1
println("H0 = $H0")


# R = 1 * 
# Iterate through time 

observed_values = @. time_index_vec/300 # sin(time_index_vec/100) * 100
for i in 2:M
    index = time_index_vec[i]
    time = index #Times[i];

    # Hydrodynamics
    Diffusivity = zeros(N) .+ 1e-5 # ds["Kz"][:,index]

    println("On time $i")
    for EID in 1:N_ensemble
        # println("Running ensemble member $EID")
        run_forward_model(EID, index, time, Diffusivity, variables)
        output["floc1"][:,i, EID] = variables["floc1"][:, EID]
        output["floc2"][:,i, EID] = variables["floc2"][:, EID]
        # println("EID = $EID completed")
        # println("floc1 = $(variables["floc1"][1:5, EID])")
    end


    if i%50 == 0
        println("Performing EnKF update at time step $i")
        # EnKF step  
        
        # Size : Nz x N_ensemble
        total_sediment = variables["floc1"] #.+ variables["floc2"]
        ensemble_mean = mean(total_sediment, dims=2)    # Size : Nz x 1
        R = 1e-3 #* Matrix(I, N, N)                         #c Size : Nz x Nz
    
        ensemble_covariance = cov(total_sediment, dims=2)  # Nz x Nz    
        kalman_gain = ensemble_covariance * transpose(H0) * inv(H0 * ensemble_covariance * transpose(H0) .+ R)

        for EID in 1:N_ensemble
            shift = kalman_gain * (observations .- H0 * variables["floc1"][:, EID]) 
            variables["floc1"][:, EID] = variables["floc1"][:, EID] + shift
        end
    end
end 



# ********************** save data ****************************
units_dict = Dict(
    "floc1" => "conc1",
    "floc2" => "conc2")

var2name = Dict("floc1" => "HAB concentration",
            "floc2" => "Diatom concentration")


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