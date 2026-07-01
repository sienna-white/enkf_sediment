#!/usr/bin/env julia

#  module load  julia/1.11.7 
using Printf
using DataStructures: OrderedDict
using NCDatasets
using CSV, DataFrames
using Printf
using Profile
using Statistics 
using LinearAlgebra
using Distributions
using Base.Threads

include("hydro/calculate_physical_variables.jl") 
include("hydro/advance_variables.jl")
include("hydro/forcings.jl") 
include("hydro/output.jl")
include("floc_mod_enkf.jl")
using .floc_mod


function create_lognormal_distribution(mean::Float64, var::Float64, N::Int)

    # 2. Convert to the underlying Normal distribution parameters (μ and σ)
    # Formula for underlying variance
    σ² = log(1.0 + (var/mean^2))
    σ = sqrt(σ²)

    # Formula for underlying mean
    μ = log(mean) - 0.5 * σ²

    # 3. Create the Log-Normal distribution object
    dist = LogNormal(μ, σ)

    # 4. Sample from it
    # Generate an array of 10,000 samples:
    samples = rand(dist, N)
    
    
    # # Create a log-normal distribution object
    # dist = LogNormal(mean, std_dev)
    
    # # Generate N random samples from the distribution
    # samples = rand(dist, N)
    
    return samples
end

# const α = 0.35 #0.35 #4.5 #1.5  #0.55
# const nf = 1.9 #2.1 #1.9 #2.0                # fractal dimension exponent
# const β = 0.045 #12 #12 

function run_my_model(file_out_name::String, floc_on::Bool=true)


    #********************** SPATIAL DOMAIN  ***************************
    N = 40    # number of ensembles points
    dt = 1    # (seconds) size of time step 
    M  = 3600*5 #3600

    # Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
    isave = 30 # 6 #1000
    var2save = ["G", "nf", "alpha", "beta"]

    create_output_dict(M, isave, var2save, N)

    #********************** FIXED CONSTANTS  ***************************
    rhoA = 1.23                     # Density of air, kg/m^3
    rhoW = 1000                     # Density of water, kg/m^3
    specific_heat_water = 4181      # J/kg-degC
    specific_heat_air = 1007        # J/kg-degC x RH

    # ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
    Ns = 50                  # Number of sediment size classes

    ssc0 = zeros(N, Ns)  #.+ 1   # Matrix for sediment concentration (Nz x Ns)
    ssc0[:, 1:30] .= 200    # Matrix for sediment concentration (N x Ns)
    ssc_init = ssc0[1, :]  # Initial sediment concentration for each size class
    D = logrange(10e-6, 1400e-6, Ns)      # Sediment grain sizes (\mu m )
    D = collect(D)

    Alphas = create_lognormal_distribution(0.35, 0.15^2, N)
    Betas = create_lognormal_distribution(0.055, 0.03^2, N)
    Nfs = create_lognormal_distribution(1.9, 0.3^2, N)

    Betas = clamp.(Betas, 0.01, 0.2) # Ensure Betas values are within the range [0.01, 0.1]
    Alphas = clamp.(Alphas, 0.01, 1.0) # Ensure Alphas values are within the range [0.1, 0.5]
    Nfs = clamp.(Nfs, 1.1, 2.9) # Ensure Nfs values are within the range [1.5, 2.5]

    add_sediment_to_output(Ns, isave, M, N)

    #***************************************************************************
    #   Initialize variables
    #***************************************************************************
    Times = collect(0:dt:(M*dt))

    save_sediment2output(1, 1, ssc0)
    real_times_saved = [Times[1]]
    #***************************************************************************

    turbulent_shears = zeros(N) .+ 30  


    # Create fake observation # 

    H0 = zeros(1, Ns+3)  # zeros(Float64, N, N)
    x_index = 5
    H0[x_index] = 1 #e4      # floc1
    H0[1 + x_index] = 1 #e4  # floc2
    H0[2 + x_index] = 1 # e4  # floc3


    Y = zeros(1, Ns+3)
    Y[x_index] = 1e4      # floc1
    Y[1 + x_index] = 1e4  # floc2
    Y[2 + x_index] = 1e4  # floc3


    # Initialize parameter set for each ensemble member
    floc_params_list = [] 
    for i_ens in 1:N
        alpha = Alphas[i_ens]
        beta = Betas[i_ens]
        nf = Nfs[i_ens]
        floc_params = init_params(D, Ns, ssc_init, alpha, beta, nf)
        push!(floc_params_list, floc_params)
        @printf("\tParameters for ensemble [%d]: alpha=%2.2f, beta=%2.2f, nf=%2.2f\n", i_ens, alpha, beta, nf)

    end

    variables = Dict() 
    variables["SSC"] = ssc0
    
    
        
        # Calculate settling velocities for each size class
        # ws = floc_params.ws
        # for i in 1:Ns
        #     # println("\t Size class $i: ws = $(ws[i]*100) cm/s")
        #     @printf("\t Size class %d, ws =  %2.2f cm/s\n", i, (ws[i_ens]*100))
        # end
     
        

    #*************************** TIME LOOP  *********************************
    @info "Starting time loop for $M steps with dt = $dt seconds"
    for i in 2:(M-1)
        time = Times[i];

        # ***************************************************************
        # [1] Advance sediment concentrations for each size class
        ssc = variables["SSC"]

        # [2] Ensemble loop
        for i_ens in 1:N 
            # println("Running ensemble point $i_ens")
                       # sed distribution ~  Ns ~ Shear ~ dt 
            ssc[i_ens,:] .= run_floc_mod(floc_params_list[i_ens], ssc[i_ens, :], Ns, turbulent_shears[i_ens], dt) 
        
        # [3] Pack variables for next timestep 
        variables["G"] = turbulent_shears
        variables["SSC"] = ssc

        # ***************************************************************
        # [4] Perform EnKF update  
        if i%15 ==0 
            println("Performing EnKF update at time = $time")
            augmented_matrix = zeros(Ns + 3, N) 
            for EID in 1:N
                augmented_matrix[1:Ns, EID] = ssc[EID, :]
                augmented_matrix[Ns + 1, EID] = Alphas[EID]
                augmented_matrix[Ns + 2, EID] = Betas[EID]
                augmented_matrix[Ns + 3, EID] = Nfs[EID]
            end

            ensemble_covariance = cov(augmented_matrix, dims=2) 
            ensemble_mean = mean(augmented_matrix, dims=2)
            R = 1e-3 

            println("Size of ensemble_covariance: ", size(ensemble_covariance))
            println("Size of H0: ", size(H0))
            println("Size of augmented_matrix: ", size(augmented_matrix))

            kalman_gain = (ensemble_covariance * transpose(H0)) / (H0 * ensemble_covariance * transpose(H0) .+ R)
            println("size of kalman gain", size(kalman_gain))
 
            for EID in 1:N
                current_state = augmented_matrix[:, EID]
                println("size of current state", size(current_state))
                println("size of Y", size(Y))
                println("size of H0", size(H0))
                innovation = (Y .- H0 * current_state)
                shift = kalman_gain .* innovation'
                println("size of shift", size(shift))
                println("size of innovation", size(innovation))
                println(innovation)

            end

            # shift = kalman_gain * (Y .- H0 * augmented_matrix[:,1]) 



            exit()
        end 

            if i % isave == 0
                index = div(i, isave) + 1 
                save2output_ens(i_ens, index, "G", variables["G"][i_ens])
                # save2output_ens(i_ens, index, "alpha", alpha)
                # save2output_ens(i_ens, index, "nf", nf)
                # save2output_ens(i_ens, index, "beta", beta)
                # save2output_ens(i_ens, index, "sed_mass", floc_params.mass)
                save_sediment2output_ens(i_ens, index, ssc[i_ens,:])
                if i_ens == 1
                    push!(real_times_saved, time)
                end
            end
        end
    end 
    # ********************** save data ****************************
    units_dict = Dict("G" => "1/s", "alpha" => "m^3/s", "beta" => "1/s", "nf" => "-", "sed_mass" => "kg/m^3")
    var2name = Dict("G" => "Turbulent shear", "alpha" => "Aggregation coefficient", 
                    "beta" => "Fragmentation coefficient", "nf" => "Fractal dimension exponent", "sed_mass" => "Sediment mass concentration")

    nt = div(M,isave)
    
    ds = NCDataset(file_out_name,"c")
    ds.attrib["title"] = "floc mod test"

    # model_time = collect(1:M)
    defDim(ds, "N", N) 
    defDim(ds, "time", nt)

    # println("Length of times_unique is ", size(t2))
    println("Length of SSC is ", size(output["ssc"]))

    defDim(ds, "Ds", length(D))
    v = defVar(ds, "Ds", Float32, ("Ds",))
    v[:] = D


    # N, Ns, n_saved_steps
    v = defVar(ds, "ssc", Float64,("N", "Ds", "time"), attrib = OrderedDict(
        "units" =>  "parts/m3", "long_name" => "suspended sediment concentration"))
    v[:,:,:] = output["ssc"]

    # D1, M1 = get_particle_density() 
    # v = defVar(ds, "np", Float64, ("Ds",), attrib = OrderedDict(
    #     "units" =>  "particle/floc", "long_name" => "number primary particle per floc"))
    # v[:] = D1

    # v = defVar(ds, "mass", Float64, ("Ds",), attrib = OrderedDict(
    #     "units" =>  "kg/particle", "long_name" => "mass of each floc (fractal!)"))
    # v[:] = M1

    v = defVar(ds, "N", Float32, ("N",))
    v[:] = collect(1:N)

    v = defVar(ds, "time", Float32, ("time",), attrib = OrderedDict("units" => "seconds"))
    v[:] = real_times_saved #real_times_saved #collect(1:nt) #model_time

    # println()
    for var in var2save
        v = defVar(ds, var, Float64,("N","time"), attrib = OrderedDict(
        "units" =>  units_dict[var], "long_name" => var2name[var]))
        v[:,:] = output[var]; 
    end
# end 
    print("Saved $file_out_name \n")
    close(ds)

end 

run_my_model("FlocMod_TEST.nc", true)
