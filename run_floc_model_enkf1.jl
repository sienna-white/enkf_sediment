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

    df = CSV.read("/global/homes/s/siennaw/scratch/siennaw/scripts/enkf_sediment/adv_shear_4_model.csv", DataFrame)
    time_steps = df[!, "seconds"]
    turbulent_shears = df[!, "smoothed_shear"] #./10

    function get_shear(tstep::Int)
        # divide by 10 
        index = div(tstep, 2) + 1
        return turbulent_shears[index]
    end 

    #********************** SPATIAL DOMAIN  ***************************
    N = 50 #250    # number of ensembles points
    dt = 0.5    # (seconds) size of time step 
    M  = 3600*2*25*5  #24*5 #3600*5 #3600

    interval = 2000 # M*2 #60*5//dt 

    # Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
    isave = 60*5*10
    var2save = ["G", "nf", "alpha", "beta", "beta2"]

    create_output_dict(M, isave, var2save, N)

    #********************** FIXED CONSTANTS  ***************************
    rhoA = 1.23                     # Density of air, kg/m^3
    rhoW = 1000                     # Density of water, kg/m^3
    specific_heat_water = 4181      # J/kg-degC
    specific_heat_air = 1007        # J/kg-degC x RH

    # ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
    Ns = 40                  # Number of sediment size classes

    ssc0 = zeros(N, Ns)     # Matrix for sediment concentration (Nz x Ns)
    ssc0[:, 1:30] .= 2000    # Matrix for sediment concentration (N x Ns)
    ssc_init = ssc0[1, :]   # Initial sediment concentration for each size class
    D = logrange(1e-6, 1500e-6, Ns)      # Sediment grain sizes (\mu m )
    D = collect(D)

    Alphas = create_lognormal_distribution(0.35, 0.3^2, N)
    Betas = create_lognormal_distribution(0.045, 0.05^2, N) #(0.055, 0.05^2, N)
    Beta2s = create_lognormal_distribution(1.0, 0.5^2, N) #(0.055, 0.05^2, N)
    Nfs = create_lognormal_distribution(1.9, 0.3^2, N)

    Betas = clamp.(Betas, 0.001, 0.5) # Ensure Betas values are within the range [0.01, 0.1]
    Alphas = clamp.(Alphas, 0.001, 100.0) # Ensure Alphas values are within the range [0.1, 1.0]
    Nfs = clamp.(Nfs, 1.5, 2.8) # Ensure Nfs values are within the range [1.5, 2.5]

    add_sediment_to_output(Ns, isave, M, N)

    #***************************************************************************
    #   Initialize variables
    #***************************************************************************
    Times = collect(0:dt:(M*dt))

    save_sediment2output(1, ssc0)
    real_times_saved = [Times[1]]
    #***************************************************************************

    # turbulent_shears = zeros(N) .+ 30  

    #***************************************************************************
    #   Observations
    #***************************************************************************
    Ny = 2
    Y = zeros(Ny) 

    
    x_index =10
    Y[1] = 2000     # floc1
    Y[2] = 3000     # floc2

   
    H = zeros(Ny, Ns+4)  # + 3 for augmented matrix 
    H[1, x_index] = 1
    H[2, x_index+1] = 1
    println("Inserting observations at size classes $x_index and $(x_index+1)--> Ds=$(D[x_index]*1e6) and Ds=$(D[x_index+1]*1e6) microns")


    # Initialize parameter set for each ensemble member
    floc_params_list = [] 
    for i_ens in 1:N
        alpha = Alphas[i_ens]
        beta = Betas[i_ens]
        beta2 = Beta2s[i_ens]
        nf = Nfs[i_ens]
        floc_params = init_params(D, Ns, ssc_init, alpha, beta, beta2, nf)
        push!(floc_params_list, floc_params)
        @printf("\tParameters for ensemble [%d]: alpha=%2.2f, beta=%2.2f, beta2=%2.2f, nf=%2.2f\n", i_ens, alpha, beta, beta2, nf)

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
    for i in 2:(M)
        time = Times[i];

        # ***************************************************************
        # [1] Advance sediment concentrations for each size class
        ssc = variables["SSC"]

        # [2] Ensemble loop @ time step 
        @threads for i_ens in 1:N 
                       # sed distribution ~  Ns ~ Shear ~ dt                           #  turbulent_shears[i]
            ssc_ = run_floc_mod(floc_params_list[i_ens], ssc[i_ens, :], Ns,  get_shear(i), dt) 
            ssc_[.!isfinite.(ssc_)] .= 0
            ssc[i_ens,:] .= ssc_ 
        end 
        
        # [3] Pack variables for next timestep 
        variables["G"] = turbulent_shears
        variables["SSC"] = ssc

        # ***************************************************************
        # [4] Perform EnKF update  
        if i%interval ==0 
            println("Performing EnKF update at time = $time")
            augmented_matrix = zeros(Ns+4, N) 
            for EID in 1:N
                augmented_matrix[1:Ns, EID] = ssc[EID, :]
                augmented_matrix[Ns + 1, EID] = Alphas[EID] #+ abs(randn()*0.01)
                augmented_matrix[Ns + 2, EID] = Betas[EID] #+ abs(randn()*0.01)
                augmented_matrix[Ns + 3, EID] = Nfs[EID]  #+ abs(randn()*0.01)
                augmented_matrix[Ns + 4, EID] = Beta2s[EID] #+ abs(randn()*0.01)
            end

            ensemble_covariance = cov(augmented_matrix, dims=2) 
            ensemble_mean = mean(augmented_matrix, dims=2)
            R = 4e2 #1e-3  # Observation error covariance matrix
            # R = 4e3 #1e-3 


            kalman_gain = (ensemble_covariance * transpose(H)) / (H * ensemble_covariance * transpose(H) + R * I)
 
            @threads for EID in 1:N
                # Analysis step 
                current_state = augmented_matrix[:, EID]
                innovation = (Y .- H * current_state)
                shift = kalman_gain * innovation
                augmented_matrix[:, EID] = augmented_matrix[:, EID] + shift

                # Clamp @ zero for no negative concentrations and parameters 
                augmented_matrix[:, EID] = clamp.(augmented_matrix[:, EID], 1e-5, Inf)
                
                # Update sediment concentration for the ensemble 
                ssc[EID, :] = augmented_matrix[1:Ns, EID]

                # Update parameters for the ensemble
                Alphas[EID] = clamp(augmented_matrix[Ns + 1, EID], 0.0001, 10.0) 
                Betas[EID]  = clamp(augmented_matrix[Ns + 2, EID], 0.0001, 10.0)
                Nfs[EID]    = clamp(augmented_matrix[Ns + 3, EID], 1.1, 2.9)
                Beta2s[EID] = clamp(augmented_matrix[Ns + 4, EID], 0.0, 3)
                # println("New alpha = ", Alphas[EID], ", New beta = ", Betas[EID], ", New nf = ", Nfs[EID])
                # println("ssc[EID, :] = ", ssc[EID, :])

                # Re-calculate sediment parameters
                floc_params_list[EID] = init_params(D, Ns, ssc[EID, :], Alphas[EID], Betas[EID], Beta2s[EID], Nfs[EID])
                
            end
        end 

        if i % isave == 0
            index = div(i, isave)
            save2output(index, "alpha", Alphas) 
            save2output(index, "beta", Betas)
            save2output(index, "beta2", Beta2s)
            save2output(index, "nf", Nfs)
            save2output(index, "G", get_shear(i)) #turbulent_shears[i])
            # save2output_ens(i_ens, index, "G", variables["G"][i_ens])
            # save2output_ens(i_ens, index, "sed_mass", floc_params.mass)
            save_sediment2output(index, ssc)
            push!(real_times_saved, time)
        end
    end 
    # ********************** save data ****************************
    units_dict = Dict("G" => "1/s", "alpha" => "m^3/s", "beta" => "1/s", "beta2" => "-", "nf" => "-", "sed_mass" => "kg/m^3")
    var2name = Dict("G" => "Turbulent shear", "alpha" => "Aggregation coefficient", 
                    "beta" => "Fragmentation coefficient","beta2" => "Fragmentation exponent", 
                    "nf" => "Fractal dimension exponent", "sed_mass" => "Sediment mass concentration")

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
    v[:] = real_times_saved[1:end-1] #real_times_saved #collect(1:nt) #model_time

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

run_my_model("FlocMod_ADV_G_16.nc", true)

