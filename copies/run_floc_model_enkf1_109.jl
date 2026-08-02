#!/usr/bin/env julia

#  module load  julia/1.11.7 
using Printf
using DataStructures: OrderedDict
using NCDatasets
using CSV, DataFrames, DataInterpolations
using Printf
# using Profile
using Random
using Statistics 
using LinearAlgebra
using Distributions
using Base.Threads
using Compat # for older julia versoins 

include("hydro/calculate_physical_variables.jl") 
include("hydro/advance_variables.jl")
include("hydro/forcings.jl") 
include("hydro/output.jl")
include("floc_mod_enkf.jl")
using .floc_mod

println("Using $(Threads.nthreads()) threads for parallelization.\n\n")

function run_my_model(file_out_name::String, floc_on::Bool=true)
    #********************** SPATIAL DOMAIN  ***************************
    N = 100 #35 #50 #250    # number of ensembles points
    dt = 0.01 #0.5 #0.5    # (seconds) size of time step 
    # M  = 3600*9 #*300 #0 #8*5 #*15 #*20 #*10 #10 #25 #*5 #*5*2  #24*5 #3600*5 #3600
    M = 864000*5 #8640000 #*2#72000*100 #*24*12 # 15 hours @ dt = 0.05
    # Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
    isave = 30000 #6000 #1200*2 #600 #2400 # every 2 min  #60*10 #*2 #10* #60*10*5 #*10
    var2save = ["G", "alpha", "beta"]

    create_output_dict(M, isave, var2save, N)
    #********************** FIXED CONSTANTS  ***************************
    rhoA = 1.23                     # Density of air, kg/m^3
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
    alpha0 = 0.3 #35
    beta0  = 0.1
    nf0 = 2.1 # change?  
    beta20 = 1.5

    Alphas = randn(N) 
    Betas = randn(N)
    Nfs = randn(N)
    Beta2s = randn(N)
    #***************************************************************************
    Nflux =  0 
    # Flux_ind = sum(D.<50e-6)
    # Flux   = rand(Normal(0, 0.3),  (N, 2))  # Random coefficients for our Flux term
    add_sediment_to_output(Ns, isave, M, N)
    # add_flux_to_output(Ns, isave, M, N, Nflux)

    #***************************************************************************
    #   OBSERVATIONAL DATA 
    #*************************************************************************** 
    @info "Reading observational data..."

    # Velocity data 
    # df = CSV.read("/global/homes/s/siennaw/scratch/siennaw/scripts/enkf_sediment/adcp_lateral_velocity.csv", DataFrame)
    # get_velocity = LinearInterpolation(df.lateral_velocity, df.seconds, extrapolation = ExtrapolationType.Linear)

    # Shear data
    df = CSV.read("adv_shear_4_model.csv", DataFrame)
    df.smoothed_shear = df.smoothed_shear 
    get_shear = LinearInterpolation(df.smoothed_shear, df.seconds) #, extrapolation = ExtrapolationType.Linear)

    time_steps = df[!, "seconds"]
    turbulent_shears = df[!, "smoothed_shear"] 
    @assert dt<=1 "Head's up: Time step dt must be less than 1 second for this shear data to work properly."

    # LISST data
    Ds_LISST = [  1.21,   1.6 ,   1.89,  2.23 ,  2.63,  3.11,  3.67,   4.33,   5.11 ,  6.03,
                  7.11,   8.39,   9.9,   11.7,   13.8,  16.3,  19.2,   22.7,   26.7,   31.6,
                  37.2,   43.9,   51.9,  61.2,   72.2,  85.2,  101.,   119.,   140.,   165.,
                  195.,   230.,   273.,  324.,   386.,  459.]
    N_lisst = length(Ds_LISST)

    # Initial observation operator matrix H (N_lisst x Ns)
    Ns_aug = Ns + Nflux + 2 
    H = zeros((N_lisst, Ns_aug))
    
    # Create volume-weighted observation operator matrix H
    for j in 1:Ns
        @info "Sorting floc size class: ", D[j]*1e6
        for i in 1:N_lisst
            if ((D[j]*1e6) <= Ds_LISST[i])
                @info "-> placing in LISST size class: ", Ds_LISST[i]
                H[i, j] = Volumes[j]*1e6
                break
            end 
            if i==N_lisst
                @info "Larger than max(LISST) -> placing in LISST size class: ", Ds_LISST[i]
                H[i, j] = Volumes[j]*1e6
                break
            end 
        end
    end 

    # Volume-weighted observations 
    dfL = CSV.read("lisst_data.csv", DataFrame)
    dfR = CSV.read("lisst_variance.csv", DataFrame)

    function get_observation_row(time_stamp::Real, df=dfL)
        # Thank you chatgpt for this function
        row_idx = findfirst(df.seconds .== time_stamp)
        # If a match is found, return the row data as a Vector, dropping the seconds column
        if !isnothing(row_idx)
            # println("Found observation at time = $time_stamp seconds (L_seconds=$(observation_times[row_idx]))")
            return true, Vector(df[row_idx, Not(:seconds)])
        else
            return false, nothing
        end
    end

    function get_R(time_stamp::Real, df=dfR)
        # Thank you chatgpt for this function
        row_idx = findfirst(df.seconds .== time_stamp)
        return Vector(df[row_idx, Not(:seconds)]) #.1e-6 #.*(1e-12)
       
    end
    #***************************************************************************
    #   Save initial variables
    #***************************************************************************
    Times = collect(0:dt:(M*dt))
    save_sediment2output(1, ssc0)
    save2output(1, "alpha", Alphas) 
    save2output(1, "beta", Betas)
    # save2output(1, "beta2", Beta2s)
    # save2output(1, "nf", Nfs)
    save2output(1, "G", get_shear(1))
    save_sediment2output(1, ssc0, "ssc")
    # save_sediment2output(1, Flux,"flux")
    real_times_saved = [Times[1]]
    #***************************************************************************

    #***************************************************************************
    #   Observations
    #***************************************************************************
    # Initialize parameter set for each ensemble member

    # Initialize list of length N for each ensemble member
    floc_params_list = Vector{Any}(undef, N)
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
    augmented_matrix = zeros(Ns_aug, N)  # Pre-allocate array 
    # noise_Nflux = zeros(N, Nflux) 
    noise_N     = zeros(N)
    noise_ssc   = zeros(N, Ns)
    # for EID in 1:N
    #     alpha = alpha0 * exp(0.1*Alphas[EID]) 
    #     beta = beta0 * exp(0.1*Betas[EID]) 
    #     beta2 = beta20 * exp(0.1*Beta2s[EID]) 
    #     nf = nf0 * exp(0.1* Nfs[EID])  #* sin(Nfs[EID])^2 
    #     floc_params = init_params(D, Ns, ssc0[EID,:], alpha, beta, beta2, nf, collision_matrix)
    #     # alpha = Alphas[EID]
    #     # beta = Betas[EID]
    #     # beta2 = Beta2s[EID]
    #     # nf = Nfs[EID]
    #     # floc_params = init_params(D, Ns, ssc0[EID,:], alpha, beta, beta2, nf)
    #     floc_params_list[EID] = floc_params
    #     # push!(floc_params_list, floc_params)
    #     @printf("\tParameters for ensemble [%d]: alpha=%2.2f, beta=%2.2f, beta2=%2.2f, nf=%2.2f\n", EID, alpha, beta, beta2, nf)
    # end
    VOLUME_NOISE = (1e-8./Volumes')

    for i in 2:(M)
        #***************************************************************************
        #   Random walk for parameters and flux term
        #***************************************************************************
        # Flux  .+=  randn(N, Nflux).* 0.0001 # Add some random noise to the flux term
        noise = 0.02
        Alphas.+=  randn!(noise_N) .* noise
        Betas .+=  randn!(noise_N) .* noise 
        # Beta2s.+=  randn!(noise_N).* noise
        # Nfs   .+=  randn!(noise_N) .* noise 

        # Flux  .=  (1 .+ randn(N, Nflux).* 0.01) # Add some random noise to the flux term
        # Alphas.=   randn!(noise_N) 
        # Betas .=   randn!(noise_N) 
        # Beta2s.=  randn!(noise_N) 
        # Nfs   .=   randn!(noise_N) 

        time = Times[i];
        # ***************************************************************
        # [1] Advance sediment concentrations for each size class
        ssc = variables["SSC"]

        ssc .+= randn!(noise_ssc) .*  VOLUME_NOISE
        clamp!(ssc, 0.0, Inf) # Ensure no negative concentrations

        # [2] Ensemble loop @ time step 
        shear = get_shear(time)
        # velocity = get_velocity(time)
        
        # ***************************************************************
        @threads for EID in 1:N 
            alpha = alpha0 * exp(0.1*Alphas[EID]) 
            beta = beta0 * exp(0.1*Betas[EID]) 
            beta2 = beta20 * exp(0.1*Beta2s[EID]) 
            nf = nf0 * exp(0.1*Nfs[EID]) 
            floc_params = init_params(D, Ns, ssc[EID,:], alpha, beta, beta2, nf, collision_matrix)
            ssc_ = run_floc_mod(floc_params, ssc[EID, :], Ns,  shear, dt) 
                       # sed distribution ~  Ns ~ Shear ~ dt                           #  turbulent_shears[i]
            # ssc_ = run_floc_mod(floc_params_list[EID], ssc[EID, :], Ns,  shear, dt) 
            # Flux_ = zeros(Ns)
            # Flux_[1:Flux_ind] .= Flux[EID, 1]
            # Flux_[Flux_ind+1:end] .= Flux[EID, 2]
            # ssc_ = ssc_ .+ dt.* 0.01*(sin.(Flux_).*ssc_) # velocity.*1e-3.* Add lateral Flux term to sediment concentration
            # clamp!(ssc_, 1e-6, 1e10) # Ensure no negative concentrations
            ssc_[ssc_ .* vec(Volumes) .> 15e-6] .= 100
            ssc_[.!isfinite.(ssc_)] .= 0 
            ssc[EID,:] .= ssc_ 
        end 
        
        # [3] Pack variables for next timestep 
        variables["G"] = turbulent_shears
        variables["SSC"] = ssc

        # ***************************************************************
        # [4] Perform EnKF update  
        run_analysis, observations = get_observation_row(time)
        # run_analysis = false 
        if run_analysis
            text = @sprintf("\t Observations available at time = %2.1f hours (timestep %d/%d)", time/3600, i, M)            
            println(text)
            R = get_R(time) #.* 1e-4 
            IND = Ns + Nflux 
                        # augmented_matrix[.!isfinite.(augmented_matrix)] .= 1e-8 
            for EID in 1:N
                augmented_matrix[1:Ns, EID] = ssc[EID, :]
                # augmented_matrix[Ns+1:IND, EID] = Flux[EID,:]
                augmented_matrix[IND + 1, EID] = Alphas[EID]
                augmented_matrix[IND + 2, EID] = Betas[EID] 
                # augmented_matrix[IND + 3, EID] = Nfs[EID] 
                # augmented_matrix[IND + 4, EID] = Beta2s[EID]
                # augmented_matrix[:, EID] = augmented_matrix[:, EID] # .+ randn(Ns_aug).*1e-7 # Add some random noise to the augmented matrix
            end
     
            augmented_matrix[.!isfinite.(augmented_matrix)] .= randn().*1e-6
            ensemble_covariance = cov(augmented_matrix, dims=2) 
            kalman_gain = (ensemble_covariance * transpose(H)) / (H * ensemble_covariance * transpose(H) + Diagonal(R))
            # HPHt = H * ensemble_covariance * transpose(H)
            # @info "‖HPHᵀ‖ = $(norm(HPHt)), ‖R‖ = $(norm(R)), mean(diag(HPHt))/mean(R) = $(mean(diag(HPHt))/mean(R))"
            flush(stdout)
            # ***************************************************************
            @threads for EID in 1:N
                # Analysis step 
                # current_state = augmented_matrix[:, EID]
                innovation = (observations .- H * augmented_matrix[:, EID])
                # obs2state = inv(H) * observations
                # augmented_matrix[:, EID] = obs2state # TEMP TEMP TEMP
                shift = kalman_gain * innovation
                augmented_matrix[:, EID] = augmented_matrix[:, EID] + shift

                # Update sediment concentration for the ensemble 
                ssc[EID, :] = augmented_matrix[1:Ns, EID]
                # if EID==1
                # cstate = H * current_state
                # astate = H * augmented_matrix[:, EID]

                #     println("[1-OBS]: ", observations[30:end])
                #     println("[1-MOD]: ", cstate[30:end])
                #     println("[2-MOD]: ", astate[30:end])
                #     println("\n\n")
                # end 
                # exit()

                # Update flux coefficents 
                # Flux[EID, :] = augmented_matrix[Ns+1:IND, EID]
                
                # Update parameters for the ensemble
                Alphas[EID] = augmented_matrix[IND + 1, EID] 
                Betas[EID]  = augmented_matrix[IND + 2, EID] 
                # Nfs[EID]    = augmented_matrix[IND + 3, EID] 
                # Beta2s[EID] = augmented_matrix[IND + 4, EID]
            end
            variables["SSC"] = ssc
        end 
        # ***************************************************************
        # ***************************************************************
        if i % isave == 0
            index = div(i, isave)
            save2output(index, "alpha", Alphas) 
            save2output(index, "beta", Betas)
            # save2output(index, "beta2", Beta2s)
            # save2output(index, "nf", Nfs)
            save2output(index, "G", get_shear(time))
            save_sediment2output(index, ssc, "ssc")
            # save_sediment2output(index, Flux,"flux")
            push!(real_times_saved, time)
        end
    end 
    # ********************** save data ****************************
    units_dict = Dict("G" => "1/s", "alpha" => "m^3/s", "flux" => "-", "beta" => "1/s", "beta2" => "-", "nf" => "-", "sed_mass" => "kg/m^3")
    var2name = Dict("G" => "Turbulent shear", "alpha" => "Aggregation coefficient",  "flux" => "sediment influx",
                    "beta" => "Fragmentation coefficient","beta2" => "Fragmentation exponent", 
                    "nf" => "Fractal dimension exponent", "sed_mass" => "Sediment mass concentration")

    nt = div(M,isave)
    ds = NCDataset(file_out_name,"c")
    ds.attrib["title"] = "floc mod test"

    defDim(ds, "N", N) 
    defDim(ds, "time", nt)
    defDim(ds, "Nf", Nflux)
    defDim(ds, "Ny", N_lisst)
    # println("Length of SSC is ", size(output["ssc"]))

    defDim(ds, "Ds", length(D))
    v = defVar(ds, "Ds", Float32, ("Ds",))
    v[:] = D

    v = defVar(ds, "nf", Float32, ("N",))
    v[:] = Nfs

    v = defVar(ds, "beta2", Float32, ("N",))
    v[:] = Beta2s

    v = defVar(ds, "ssc", Float64,("N", "Ds", "time"), attrib = OrderedDict(
        "units" =>  "parts/m3", "long_name" => "suspended sediment concentration"))
    v[:,:,:] = output["ssc"]

    # v = defVar(ds, "flux", Float64,("N", "Nf", "time"), attrib = OrderedDict(
    #     "units" =>  "parts/m3", "long_name" => "suspended sediment influx"))
    # v[:,:,:] = output["flux"]

    # D1, M1 = get_particle_density() 
    # v = defVar(ds, "np", Float64, ("Ds",), attrib = OrderedDict(
    #     "units" =>  "particle/floc", "long_name" => "number primary particle per floc"))
    # v[:] = D1

    # v = defVar(ds, "mass", Float64, ("Ds",), attrib = OrderedDict(
    #     "units" =>  "kg/particle", "long_name" => "mass of each floc (fractal!)"))
    # v[:] = M1

    v = defVar(ds, "H", Float64, ("Ny", "Ds",) , attrib = OrderedDict("obs operator" => "particle to vol-lisst"))
    v[:] = H[:, 1:Ns]

    v = defVar(ds, "time", Float32, ("time",), attrib = OrderedDict("units" => "seconds"))
    v[:] = real_times_saved[1:end-1] 

    for var in var2save
        v = defVar(ds, var, Float64,("N","time"), attrib = OrderedDict(
        "units" =>  units_dict[var], "long_name" => var2name[var]))
        v[:,:] = output[var]; 
    end
    print("Saved $file_out_name \n")
    close(ds)
end 

run_my_model("flocmod_109.nc", true)
# 71 but longer

# 101 >> trying out perturbed obs  [ voloume nosie to 1e-7 ]
# 102 >> trying out perturbed obs  [ voloume nosie to 1e-10 ]
# 103 >> trying out perturbed obs  [ voloume nosie to 1e-14 ]



# 70  >> more ens, dt=0.01, voloume nosie to 1e-7
# 71 >> more ens, dt=0.01, voloume nosie to 1e-8
# 72 >> more ens, dt=0.05, voloume nosie to 1e-8
# 73  >>  dt=0.05, voloume nosie to 1e-7
# 74  >>  dt=0.05, voloume nosie to 1e-6
# 75    dt=0.05,      noise = 0.03 /  voloume nosie to 1e-6
# 76    dt=0.05,      noise = 0.03 /  voloume nosie to 1e-7
# 77     dt=0.01,      noise = 0.03 /  voloume nosie to 1e-7
# 78     dt=0.01,      noise = 0.02 /  voloume nosie to 1e-7
# 79     dt=0.01,      noise = 0.02 /  voloume nosie to 1e-6 [delete]
# 80 -- longer     dt=0.01,      noise = 0.02 /  voloume nosie to 1e-6
# 81 -- longer     dt=0.01,      noise = 0.02 /  voloume nosie to 1e-8
# 82 -- longer     dt=0.01,      noise = 0.02 /  voloume nosie to 1e-7
# 83 -- 82 but w/o flocmod 




# 58 > hopefully the same as 57, testing 
# 59 > more noise 

# SAME AS 50, WITH RESET ON AGG TERM! 

# 52 >> increasing parameter noise! 
# 54 changed IC
# 55 


# 16 > long run
# 17 > updated the initial condition 
# 19 > chnaged noise floor
# 20 adding noise to state vector 
# 21 shear no longer divided by 10 
# 22 shear no longer divided by 10 + less noise for state 
# 23 more ensembles >> best so far 
# 25 >> N = 80 
# 26 >> N = 80 // less nosie ? 
# 27 is 26 but longer
# 28 >> 25 w/ more nosie (1e-12)
# 29 >> 1e-13 >> p good




