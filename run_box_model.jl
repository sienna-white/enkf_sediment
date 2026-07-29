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
    N = 5 #35 #50 #250    # number of ensembles points
    dt = 0.01 #0.5 #0.5    # (seconds) size of time step 
    M = 36000*50 #50 # 72000*24*12 # 15 hours @ dt = 0.05
    # Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
    isave = 6000 
    var2save = ["G"] #, "nf", "alpha", "beta", "beta2"]

    create_output_dict(M, isave, var2save, N)
    #********************** FIXED CONSTANTS  ***************************
    rhoA = 1.23                     # Density of air, kg/m^3
    rhoW = 1000                     # Density of water, kg/m^3
    specific_heat_water = 4181      # J/kg-degC
    specific_heat_air = 1007        # J/kg-degC x RH

    # ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
    Ns = 36 #40                 # Number of sediment size classes
    ssc0 = zeros(N, Ns)     # Matrix for sediment concentration (Nz x Ns)
    ssc0[:, 1:10] .= abs.(randn(N, 10))*1e9  # Matrix for sediment concentration (N x Ns)
    # ic =  [0, 0, 0, 0, 0, 1569393788, 13948421398, 34186735163, 18930699648, 9810032483, 6355491536, 7542553597, 8049608268, 4527016578, 2481863754, 1452287975, 1039818408, 765481304, 495921900, 376853152, 255783335, 185058484, 112515188, 73132356, 47481745, 34863237, 22586368, 18581423, 15560227, 10787057, 7192500, 4581169, 2739416, 1774748, 984630, 513701, 260245, 129624, 61627, 25981, 9148, 5926, 1635, 1059, 686, 444, 288, 186, 120, 0]
    ssc0 = ssc0 .+ abs.(randn(N, Ns)) #.*2
    D = collect(logrange(1e-6, 1000e-6, Ns))      # Sediment grain sizes (\mu m )

    # D  = [  1,   1.6 ,   1.89,  2.23 ,  2.63,  3.11,  3.67,   4.33,   5.11 ,  6.03,
    #               7.11,   8.39,   9.9,   11.7,   13.8,  16.3,  19.2,   22.7,   26.7,   31.6,
    #               37.2,   43.9,   51.9,  61.2,   72.2,  85.2,  101.,   119.,   140.,   165.,
    #               195.,   230.,   273.,  324.,   386.,  459.]
    # D = D .* 1e-6 # convert to meters

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

    add_sediments2_to_output(Ns, isave, M, N)

    #***************************************************************************
    #   OBSERVATIONAL DATA 
    #*************************************************************************** 
    @info "Reading observational data..."

    # Velocity data 
    df = CSV.read("adcp_lateral_velocity.csv", DataFrame)
    get_velocity = LinearInterpolation(df.u, df.seconds, extrapolate = true) #, extrapolation = ExtrapolationType.Linear)

    # Shear data
    df = CSV.read("adv_shear_4_model.csv", DataFrame)
    df.smoothed_shear = df.smoothed_shear 
    get_shear = LinearInterpolation(df.smoothed_shear, df.seconds, extrapolate = true) #, extrapolation = ExtrapolationType.Linear)

    time_steps = df[!, "seconds"]
    turbulent_shears = df[!, "smoothed_shear"] 
    @assert dt<=1 "Head's up: Time step dt must be less than 1 second for this shear data to work properly."

    #***************************************************************************
    #   Save initial variables
    #***************************************************************************
    Times = collect(0:dt:(M*dt))
    # save_sediment2output(1, ssc0)
    # save2output(1, "alpha", Alphas) 
    # save2output(1, "beta", Betas)
    # save2output(1, "beta2", Beta2s)
    # save2output(1, "nf", Nfs)
    save2output(1, "G", get_shear(1))
    save_sediment2output(1, ssc0, "ssc1")
    save_sediment2output(1, ssc0, "ssc2")

    real_times_saved = [Times[1]]
    #***************************************************************************

    #***************************************************************************
    #   Observations
    #***************************************************************************
    # Initialize parameter set for each ensemble member

    # Initialize list of length N for each ensemble member
    floc_params_list = Vector{Any}(undef, N)
    variables = Dict() 
    variables["SSC1"] = ssc0
    variables["SSC2"] = ssc0

    # Calculate settling velocities for each size class
    # ws = floc_params.ws
    # for i in 1:Ns
    #     # println("\t Size class $i: ws = $(ws[i]*100) cm/s")
    #     @printf("\t Size class %d, ws =  %2.2f cm/s\n", i, (ws[i_ens]*100))
    # end
    #*************************** TIME LOOP  *********************************
    @info "Starting time loop for $M steps with dt = $dt seconds"
    noise_N     = zeros(N)
    noise_ssc   = zeros(N, 11)

    H1 = 4
    H2 = 10 

    for EID in 1:N
        alpha = alpha0 * exp(0.1*Alphas[EID]) 
        beta = beta0 * exp(0.1*Betas[EID]) 
        beta2 = beta20 * exp(0.1*Beta2s[EID]) 
        nf = nf0 * exp(0.1* Nfs[EID])  #* sin(Nfs[EID])^2 
        floc_params = init_params(D, Ns, ssc0[EID,:], alpha, beta, beta2, nf, collision_matrix)
        floc_params_list[EID] = floc_params
        @printf("\tParameters for ensemble [%d]: alpha=%2.2f, beta=%2.2f, beta2=%2.2f, nf=%2.2f\n", EID, alpha, beta, beta2, nf)
    end 

    @threads for EID in 1:N

        ssc1 = copy(ssc0[EID, :])
        ssc2 = copy(ssc0[EID, :])
        ws = floc_params_list[EID].ws

        C_n1 = similar(ssc1)
        C_n2 = similar(ssc2)

        for i in 2:(M)
            copyto!(C_n1, ssc1)
            copyto!(C_n2, ssc2)

            time = Times[i];
            # ***************************************************************
            # # [1] Advance sediment concentrations for each size class
            # ssc1 = variables["SSC1"]
            # ssc2 = variables["SSC2"]
            clamp!(ssc1, 0.0, Inf) # Ensure no negative concentrations
            clamp!(ssc2, 0.0, Inf) # Ensure no negative concentrations

            # [2] Ensemble loop @ time step 
            shear = get_shear(time)
            u = get_velocity(time)

            ssc1_ = run_floc_mod(floc_params_list[EID], C_n1, Ns,  shear, dt) 
            ssc2_ = run_floc_mod(floc_params_list[EID], C_n2, Ns,  shear/10, dt) 

            @inbounds for k in 1:Ns
                ssc1[k] = ssc1_[k] +  dt*(u/H1 * (C_n2[k] - C_n1[k]))  - dt*(ws[k]/H1 * C_n1[k])  
                ssc2[k] = ssc2_[k]  + dt*(u/H1 * (C_n1[k] - C_n2[k]))  - dt*(ws[k]/H2 * C_n2[k])  
            end 
                        # sed distribution ~  Ns ~ Shear ~ dt                           #  turbulent_shears[i]
                # ssc_ = run_floc_mod(floc_params_list[EID], ssc[EID, :], Ns,  shear, dt) 
                # clamp!(ssc_, 1e-6, 1e10) # Ensure no negative concentrations
                # ssc[ssc .* Volumes' .> 15e-6] .= 100
                # ssc_[.!isfinite.(ssc_)] .= 0 
          
        # ***************************************************************
        if i % isave == 0
            println("$EID Saving @ $(i)")
            index = div(i, isave)
            save_sediment2output_ens(EID, index, ssc2, "ssc2")
            save_sediment2output_ens(EID, index, ssc1, "ssc1")

            if EID==1
                save2output(index, "G", get_shear(time))
                push!(real_times_saved, time)
                flush(stdout)
            end 
        end
    end 
    end 

    println("DONE!")

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
    # println("Length of SSC is ", size(output["ssc"]))

    defDim(ds, "Ds", length(D))
    v = defVar(ds, "Ds", Float32, ("Ds",))
    v[:] = D

    v = defVar(ds, "ssc2", Float64,("N", "Ds", "time"), attrib = OrderedDict(
        "units" =>  "parts/m3", "long_name" => "SHOAL suspended sediment concentration"))
    v[:,:,:] = output["ssc2"]

    v = defVar(ds, "ssc1", Float64,("N", "Ds", "time"), attrib = OrderedDict(
        "units" =>  "parts/m3", "long_name" => "SHOAL suspended sediment concentration"))
    v[:,:,:] = output["ssc1"]



    # "nf", "alpha", "beta", "beta2"]
    v = defVar(ds, "nf", Float64,("N"))
    v[:] = Nfs; 

    v = defVar(ds, "alpha", Float64,("N"))
    v[:] = Alphas; 

    v = defVar(ds, "beta", Float64,("N"))
    v[:] = Betas; 

    v = defVar(ds, "beta2", Float64,("N"))
    v[:] = Beta2s; 

    # D1, M1 = get_particle_density() 
    # v = defVar(ds, "np", Float64, ("Ds",), attrib = OrderedDict(
    #     "units" =>  "particle/floc", "long_name" => "number primary particle per floc"))
    # v[:] = D1

    # v = defVar(ds, "mass", Float64, ("Ds",), attrib = OrderedDict(
    #     "units" =>  "kg/particle", "long_name" => "mass of each floc (fractal!)"))
    # v[:] = M1

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

run_my_model("box_model4.nc", true)
