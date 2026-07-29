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
H = 4   # depth (meters)
dz = H/N # grid spacing - may need to adjust to reduce oscillations
dt = 10 # (seconds) size of time step
M = 360*3  #360*2 # number of time steps

if M <= 0
    error("M must be greater than 0. Check istart and iend values.")
end

# time_index_vec = collect(istart:(iend+1))
time_index_vec = collect(1:M)

file_out_name = "test1_sediment.nc"

# truth_dataset = NCDataset("ground_truth_sediment.nc")

forcing_folder = "/pscratch/sd/s/siennaw/stockton_field_data/forcing_for_model/2024/august6-28/"

# Hydrodynamic dataset 
# ds = NCDataset("/pscratch/sd/s/siennaw/two_species/adjoint_phytoplankton/run_hydro/HYDRO_AUGUST6-28.nc")


######################################################################################################

Calculate settling velocities for each size class
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

        gamma1 = zeros(Float64, N)
        gamma2 = zeros(Float64, N)

        # Estimate source term 
        # for j in 1:N 
        #     growth1[j] = 1e-3
        #     growth2[j] = 1e-3
        # end


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


observations = 120  #zeros(Float64, N)
# observations[4] = 10 

# (10x10) (60x1)
# H X 

# H0 =  zeros(1, N)  # zeros(Float64, N, N)
# H0[50] = 1
# println("H0 = $H0")


# For both flocs 
H0 = zeros(1, N*3)  # zeros(Float64, N, N)
x_index = N-10 
H0[x_index] = 0.2   # floc1
H0[N + x_index] = 0.5  # floc2
H0[N*2 + x_index] = 0.3  # floc3
println("observation inserted at z= $(z[x_index]) ")

# R = 1 * 
# Iterate through time 

observed_values = @. time_index_vec/300 # sin(time_index_vec/100) * 100
for i in 2:M
    index = time_index_vec[i]
    time = index #Times[i];

    # Hydrodynamics
    Diffusivity = zeros(N) .+ 1e-4 # ds["Kz"][:,index]

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

    # Update step : BOTH FLOCS 
    if i%60 == 0
        println("Performing EnKF update at time step $i")
        # EnKF step  

        # Size : Nz x N_ensemble
        total_sediment = zeros(Float64, N*3, N_ensemble)
        total_sediment[1:N, :] = variables["floc1"]
        total_sediment[N+1:2N, :] = variables["floc2"]
        total_sediment[2N+1:end, :] = variables["floc3"]
        println("total_sediment size = $(size(total_sediment))")    
        # total_sediment = variables["floc1"] #.+ variables["floc2"]
        ensemble_mean = mean(total_sediment, dims=2)    # Size : Nz x 1
        R = 1e-3 #* Matrix(I, N, N)                         #c Size : Nz x Nz
    
        ensemble_covariance = cov(total_sediment, dims=2)  # Nz x Nz    
        kalman_gain = ensemble_covariance * transpose(H0) * inv(H0 * ensemble_covariance * transpose(H0) .+ R)

        for EID in 1:N_ensemble
            current_state = zeros(Float64, N*3)
            current_state[1:N] = variables["floc1"][:, EID]
            current_state[N+1:2N] = variables["floc2"][:, EID]
            current_state[2N+1:end] = variables["floc3"][:, EID]
            shift = kalman_gain * (truth_dataset["observation"][i] .- H0 * current_state) 
            variables["floc1"][:, EID] = variables["floc1"][:, EID] + shift[1:N]
            variables["floc2"][:, EID] = variables["floc2"][:, EID] + shift[N+1:2N]
            variables["floc3"][:, EID] = variables["floc3"][:, EID] + shift[2N+1:end]
        end
    end

    # if i%50 == 0
    #     println("Performing EnKF update at time step $i")
    #     # EnKF step  
        
    #     # Size : Nz x N_ensemble
    #     total_sediment = variables["floc1"] #.+ variables["floc2"]
    #     ensemble_mean = mean(total_sediment, dims=2)    # Size : Nz x 1
    #     R = 1e-3 #* Matrix(I, N, N)                         #c Size : Nz x Nz
    
    #     ensemble_covariance = cov(total_sediment, dims=2)  # Nz x Nz    
    #     kalman_gain = ensemble_covariance * transpose(H0) * inv(H0 * ensemble_covariance * transpose(H0) .+ R)

    #     for EID in 1:N_ensemble
    #         shift = kalman_gain * (observations .- H0 * variables["floc1"][:, EID]) 
    #         variables["floc1"][:, EID] = variables["floc1"][:, EID] + shift
    #     end
    # end
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