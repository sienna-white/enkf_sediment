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


# include("hydro/calculate_physical_variables.jl") 
# include("hydro/advance_variables.jl")
# include("hydro/forcings.jl") 
include("hydro/output.jl")
include("floc_mod_enkf.jl")
using .floc_mod

println("Using $(Threads.nthreads()) threads for parallelization.\n\n")

function run_my_model(file_out_name::String, floc_on::Bool=true)
    #********************** SPATIAL DOMAIN  ***************************
    N = 100 #65 #35 #50 #250    # number of ensembles points
    dt = 1 #0.5 #0.5    # (seconds) size of time step 
    M = 3600   #24*5 #3600*5 #3600

    # Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
    isave = 30 #*2 #10* #60*10*5 #*10
    var2save = ["G", "nf", "alpha", "beta", "beta2"]

    create_output_dict(M, isave, var2save, N)
    #********************** FIXED CONSTANTS  ***************************
    rhoA = 1.23                     # Density of air, kg/m^3
    rhoW = 1000                     # Density of water, kg/m^3
    specific_heat_water = 4181      # J/kg-degC
    specific_heat_air = 1007        # J/kg-degC x RH

    # ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
    Ns = 50                 # Number of sediment size classes
    ssc0 = zeros(N, Ns)     # Matrix for sediment concentration (Nz x Ns)
    ssc0[:, 1:4] .= abs.(randn(N, 4))*1e3  # Matrix for sediment concentration (N x Ns)
    ssc0 = ssc0 .+ abs.(randn(N, Ns)) #.*2
    D = logrange(1e-6, 500e-6, Ns)      # Sediment grain sizes (\mu m )
    D = collect(D)
    Volumes = (4/3)*pi*(D./2).^3

    #***************************************************************************
    alpha0 = 0.01
    beta0  = 0.1
    nf0 = 2.1 # change?  
    beta20 = 1.5

    Alphas = randn(N) * 1e-1
    Betas = randn(N) * 1e-1
    Nfs = randn(N) * 1e-1
    Beta2s = randn(N) * 1e-1
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

    # Shear data
    df = CSV.read("adv_shear_4_model.csv", DataFrame)
    get_shear = LinearInterpolation(df.smoothed_shear, df.seconds) #, extrapolation = DataInterpolations.ExtrapolationType.Linear)
    # get_shear = LinearInterpolation(df.smoothed_shear, df.seconds; extrapolation = ExtrapolationType.Linear)
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
    Ns_aug = Ns 
    H = zeros((N_lisst, Ns_aug))
    
    # Create volume-weighted observation operator matrix H
    for j in 1:Ns
        @debug "Sorting floc size class: ", D[j]*1e6
        for i in 1:N_lisst
            if ((D[j]*1e6) <= Ds_LISST[i])
                @debug "-> placing in LISST size class: ", Ds_LISST[i]
                H[i, j] = 1 #  Volumes[j]*1e6
                break
            end 
            if i==N_lisst
                @debug "Larger than max(LISST) -> placing in LISST size class: ", Ds_LISST[i]
                H[i, j] = 1 # Volumes[j]*1e6
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
    save2output(1, "beta2", Beta2s)
    save2output(1, "nf", Nfs)
    save2output(1, "G", get_shear(1))
    save_sediment2output(1, ssc0, "ssc")
    # save_sediment2output(1, Flux,"flux")
    real_times_saved = [Times[1]]
    #***************************************************************************
    variables = Dict() 
    variables["SSC"] = ssc0

    #*************************** TIME LOOP  *********************************
    @info "Starting time loop for $M steps with dt = $dt seconds"
    state_matrix = zeros(Ns_aug, N)  # Pre-allocate array 
    # noise_Nflux = zeros(N, Nflux) 
    noise_N     = zeros(N)
    noise_ssc   = zeros(N, 11)

    floc_params_list = Vector{Any}(undef, N)
    for EID in 1:N
        alpha = alpha0* exp(Alphas[EID]) 
        beta = beta0* exp(Betas[EID]) 
        beta2 = beta20* exp(Beta2s[EID]) 
        nf = nf0* exp(Nfs[EID])  # * sin(Nfs[EID])^2 
        floc_params = init_params(D, Ns, ssc0[EID,:], alpha, beta, beta2, nf)
        floc_params_list[EID] = floc_params
        @printf("\tParameters for ensemble [%d]: alpha=%2.2f, beta=%2.2f, beta2=%2.2f, nf=%2.2f\n", EID, alpha, beta, beta2, nf)
    end

    for i in 2:(M)
        time = Times[i];
        # ***************************************************************
        # [1] Advance sediment concentrations for each size class
        ssc = variables["SSC"]
        # svol = similar(ssc)
        # ssc += abs.(randn(N, Ns)) 
        # ssc .+= randn(N, Ns).* (1e-7./Volumes')  # ) #.* 0.1
        clamp!(ssc, 0.0, 1e20) # Ensure no negative concentrations

        # [2] Ensemble loop @ time step 
        # shear = get_shear(time)        
        # # ***************************************************************
        # @threads for EID in 1:N 
        #     # ssc_ = run_floc_mod(floc_params_list[EID], ssc[EID, :], Ns,  shear, dt) 
        #     # ssc_[.!isfinite.(ssc_)] .= 0 
        #     ssc[EID,:] .= abs.(randn(Ns)*1000) #ssc_ 
        # end 
        ssc .+= randn(N, Ns).* (1e-8./Volumes')
        clamp!(ssc, 0.0, 1e20) # Ensure no negative concentrations

        # [3] Pack variables for next timestep 
        variables["G"] = turbulent_shears
        variables["SSC"] = ssc
        # println("$(variables["SSC"][10, 1:2])")
        # ***************************************************************
        # [4] Perform EnKF update  
        run_analysis, observations = get_observation_row(time)
        # run_analysis = false 
        if run_analysis
            text = @sprintf("\t Observations available at time = %2.1f hours (timestep %d/%d)", time/3600, i, M)            
            R = get_R(time) #.* 1e-4 
            IND = Ns + Nflux 


            
            # exit()

            ensemble_covariance = cov(ssc, dims=1) 
            kalman_gain = (ensemble_covariance * transpose(H)) / (H * ensemble_covariance * transpose(H) + Diagonal(R))
            HPHt = H * ensemble_covariance * transpose(H)
            @info "‖HPHᵀ‖ = $(norm(HPHt)), ‖R‖ = $(norm(R)), mean(diag(HPHt))/mean(R) = $(mean(diag(HPHt))/mean(R))"
            diag_vars = diag(ensemble_covariance)
            @info "state variance range: min=$(minimum(diag_vars)), max=$(maximum(diag_vars)), ratio=$(maximum(diag_vars)/minimum(diag_vars))"
            @info "kalman_gain range: min=$(minimum(abs, kalman_gain)), max=$(maximum(abs, kalman_gain))"

            # ***************************************************************
             for EID in 1:N
                innovation = (observations .- H * ssc[EID, :])
                shift = kalman_gain * innovation
                ssc[EID, :] .+=  shift
            end
            variables["SSC"] = ssc
            # println("$(variables["SSC"][10, 1:2])")
            # println("Variables updated. \n\n")
        end 
        # ***************************************************************
        # ***************************************************************
        if i % isave == 0
            index = div(i, isave)
            save2output(index, "alpha", Alphas) 
            save2output(index, "beta", Betas)
            save2output(index, "beta2", Beta2s)
            save2output(index, "nf", Nfs)
            save2output(index, "G", get_shear(time))
            save_sediment2output(index, variables["SSC"], "ssc")
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

    defDim(ds, "Ds", length(D))
    v = defVar(ds, "Ds", Float32, ("Ds",))
    v[:] = D

    v = defVar(ds, "ssc", Float64,("N", "Ds", "time"), attrib = OrderedDict(
        "units" =>  "parts/m3", "long_name" => "suspended sediment concentration"))
    v[:,:,:] = output["ssc"]

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

run_my_model("FlocMod_NORMAL_1.nc", true)






