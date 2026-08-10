#!/usr/bin/env julia
using Printf
using DataStructures: OrderedDict
using NCDatasets
# using Arrow, DataFrames
using CSV, DataFrames
using Printf
using Base.Threads
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
Nens = 30 #50 # 50 #30 
H = 4   # depth (meters)
dz = H/N # grid spacing - may need to adjust to reduce oscillations

 #********************** SPATIAL DOMAIN  ***************************
 dt = 1 #0.02 #0.5 #0.5    # (seconds) size of time step 
#  M = 360000 * 6360036 #8 #0*5 #*8 #0 #* 3 # 5 #12 #0 * 12 #100*60*3 #0 # 864000 
    #72000 #0 #*2
#  M = 3600*24 #2 #0/2 #4320000 # 360000 # for 0.03
 M = div(3600*24*7, dt)
 M = Int(M)
 println("There are $M time steps")
 println("dt= $dt")

 # Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
 isave = div(180, dt) #100 #30 #6000 #*2 #1200 #0 #2400  
 isave = Int(isave)
 var2save = ["G", "Kz", "U"]
 # Create vector to hold the time steps 

Times = collect(0:dt:(M*dt))
println("Initialized time vector of length $M")
real_times_saved = [Times[1]]

create_output_dict(M, isave, var2save, N)

 # ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
 println("Initializing sediment properties...")
 Ns = 36 #36 #40                 # Number of sediment size classes

ssc = zeros(Nens, N, Ns)
ssc[:,:,5]  .= 5e12 
ssc[:,:,15] .= 5e10  # Matrix for sediment concentration (N x Ns)




# ssc = zeros(Nens, N, Ns)
# ssc[:,:,5]  .= 5e12 
# ssc[:,:,15] .= 5e10  # Matrix for sediment concentration (N x Ns)



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

# alpha0 = 0.03 #2e-5 #0.003 
# beta0  = 0.004 #2e-6 #0.001

alpha0 = 2e-3 #5 #2e-5 #0.003 
beta0  = 0.5e-4 #6 #2e-6 #0.001
nf0 = 2.2  
beta20 = 1.5

Alphas = randn(Nens) 
Betas = randn(Nens)
Nfs = zeros(Nens) .+ 2.2 #randn(Nens)
Beta2s = randn(Nens)

floc_params_list = Vector{Any}(undef, Nens)
ws = Vector{Any}(undef, Nens)

for EID in 1:Nens
    Alphas[EID] = alpha0 * exp(0.5*Alphas[EID]) 
    Betas[EID] = beta0 * exp(0.5*Betas[EID]) 
    Beta2s[EID] = beta20 * exp(0.5*Beta2s[EID]) 
    Nfs[EID] = nf0 * exp(0.3* Nfs[EID])  
    floc_params = init_params(D, Ns, ssc[EID, 1, :], Alphas[EID], Betas[EID], Beta2s[EID], Nfs[EID], collision_matrix)
    floc_params_list[EID] = floc_params
    ws[EID] = floc_params.ws
end 

 #***************************************************************************
 add_sediments2_to_output(Ns, isave, M, Nens, N)
# add_sediment_to_output(Ns, isave, M, Nens, N)

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
hydro_vars = ["Kq", "Nu",  "Kz", "Q2", "Q2L", "U",]

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


# Nens, N, Ns, n_saved_steps

# make variables a global variable

function run_forward_model(EID, ssc_past, diffusivity, shear)

    ssc_ = ssc_past
    ssc_next = similar(ssc_)
  
    # TEST: loop 10 tens here with small values 
    
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
    return ssc_next, Gammas

end



hydro_t = get_hydro_at(0)
Kz    = hydro_t["Kz"]
U    = hydro_t["U"]

# save2output(1, "G", turbulent_shear)
save2output(1, "Kz", Kz)
save2output(1, "U", U)
for EID in 1:Nens
    save_sediment2output_ens(EID, 1, ssc[EID, :, :], "ssc")
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
    turbulent_shear = (Q2 ./ Q2L) .* Q # .* 100
    turbulent_shear[end] = 0.1

    @threads for EID in 1:Nens
    
        ssc[EID, :, :], Gammas = run_forward_model(EID, ssc[EID, :, :], Kz, turbulent_shear)

        if i % isave == 0
            index = div(i, isave) + 1
            # save_sediment2output(index, ssc, "ssc")
            save_sediment2output_ens(EID, index, ssc[EID, :, :], "ssc")
            save_sediment2output_ens(EID, index, Gammas, "gamma")
            if EID==1
                println("Saving @ index $index")

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



fout = "sediment_1D_model_91.nc"

# 60 
# alpha0 = 2e-4 #2e-5 #0.003 
# beta0  = 0.5e-6 #2e-6 #0.001
#61 
# alpha0 = 2e-3 #2e-5 #0.003 
# beta0  = 0.5e-5 #2e-6 #0.001

# 63: 
# alpha0 = 2e-2 #2e-5 #0.003 
# beta0  = 0.5e-4 #2e-6 #0.001


# 64: 
# alpha0 = 2e-2 
# beta0  = 0.5e-3 

# 65
# alpha0 = 2e-5 #2e-5 #0.003 
# beta0  = 0.5e-6 #2e-6 #0.001

# 66 (dt = 0.1)
# alpha0 = 2e-5 #2e-5 #0.003 
# beta0  = 0.5e-6 #2e-6 #0.001
 
# 67 > amping up the params a lot 


# all of these have new mass Conserv
# 50's: turbulent shear no longer*100 
# 50: dt = 0.02
# 51: dt = 0.01 
# 52: dt = 0.01 [shorter]
# 53: dt = 0.02, [shorter]
# alpha0 = 2e-4 #2e-5 #0.003 
# beta0  = 0.5e-6 #2e-6 #0.001

# 40: 
# 2e-4, 0.5e-6 parameters
# 40 is 2 hrs 
# 41 is 8 hours
# 42 is 8 hours (0.02 ts)
# 43 is 8 hours (0.03 ts)
# 44 is 2 hrs (0.02 ts)

# 34: 
# alpha0 = 0.03 #2e-5 #0.003 
# beta0  = 0.002 #2e-6 #0.001


# 35: 
# alpha0 = 0.03 
# beta0  = 0.004 

## 35: 

# 24 no floc mod --> test for IC 
# 28 > alpha, beta back to 0.03, 0.01
# 30 >> 0.05 time step 
# 31 >> parameters /10 again 


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

v = defVar(ds, "gamma", Float64,("Nens", "z", "Ds", "time"))
v[:,:,:,:] = output["gamma"]



# convert ws to a 2D matrix for saving 
matrix_2d = Float64.(stack(ws)')
v2 = defVar(ds, "ws", Float64,("Nens", "Ds"))
v2[:,:] = matrix_2d;

for var in var2save
    println("Saving ... $var")
    v2 = defVar(ds, var, Float64,("z", "time"))
    v2[:,:] = output[var];
end

print("Saved $fout \n")
close(ds)



    # detrended = total_sediment .- ensemble_mean
    # ensemble_covariance = (detrended * transpose(detrended)) *  1/(N_ensemble - 1)