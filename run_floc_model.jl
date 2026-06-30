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

include("hydro/calculate_physical_variables.jl") 
include("hydro/advance_variables.jl")
include("hydro/forcings.jl") 
include("hydro/output.jl")
include("floc_mod.jl")
using .floc_mod

function run_my_model(file_out_name::String, floc_on::Bool=true)

    @info "Floc on = $floc_on"

    #********************** SPATIAL DOMAIN  ***************************
    N = 10    # number of grid points
    dt = 1    # (seconds) size of time step 
    M  = 3600*5 #3600

    # Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
    isave = 30 # 6 #1000
    var2save = ["G"]

    create_output_dict(M, isave, var2save, N)

    #********************** FIXED CONSTANTS  ***************************
    rhoA = 1.23                     # Density of air, kg/m^3
    rhoW = 1000                     # Density of water, kg/m^3
    specific_heat_water = 4181      # J/kg-degC
    specific_heat_air = 1007        # J/kg-degC x RH

    # ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
    Ns = 40                  # Number of sediment size classes
    ssc0 = zeros(N, Ns)  #.+ 1   # Matrix for sediment concentration (Nz x Ns)
    ssc0[:, 1:30] .= 5e2    # Matrix for sediment concentration (Nz x Ns)
    D = logrange(10e-6, 1500e-6, Ns)      # Sediment grain sizes (\mu m )
    D = collect(D)

    init_params!(D, Ns, ssc0[1,:])

    # Calculate settling velocities for each size class
    ws = floc_mod.ws[]  #.* 10
    for i in 1:Ns
        println("Size class $i: ws = $(ws[i]*100) cm/s")
    end

    add_sediment_to_output(Ns, isave, M, N)

    #***************************************************************************
    #   Initialize variables
    #***************************************************************************

    # Initial dictionary to store variables
    variables = Dict() 
    variables["SSC"] = ssc0

    Times = collect(0:dt:(M*dt))

    save_sediment2output(1, 1, ssc0)
    real_times_saved = [Times[1]]
    #***************************************************************************

    # turbulent_shears = zeros(N) .+ 10 
    turbulent_shears = collect(1:N).^2 

    #*************************** TIME LOOP  *********************************
    for i in 2:(M-1)
        time = Times[i];

        # ***************************************************************
        # [1] Advance sediment concentrations for each size class
        ssc = variables["SSC"]

        if floc_on  
            for ix in 1:N           # sed distribution ~  Ns ~ Shear ~ dt 
                ssc[ix,:] .= run_floc_mod(ssc[ix, :], Ns, turbulent_shears[ix], dt) 
            end
        end

    
        # [2] Pack variables for next timestep 
        variables["G"] = turbulent_shears
        variables["SSC"] = ssc

        if i % isave == 0
            index = div(i, isave) + 1 
            # println(variables["G"])
            save2output(time, index, "G", variables["G"])
            save_sediment2output(time, index, ssc)
            push!(real_times_saved, time)
        end
    end
     
    println("G: ", output["G"])
    # ********************** save data ****************************
    units_dict = Dict("G" => "1/s")
    var2name = Dict("G" => "Turbulent shear")

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

    D1, M1 = get_particle_density() 
    v = defVar(ds, "np", Float64, ("Ds",), attrib = OrderedDict(
        "units" =>  "particle/floc", "long_name" => "number primary particle per floc"))
    v[:] = D1

    v = defVar(ds, "mass", Float64, ("Ds",), attrib = OrderedDict(
        "units" =>  "kg/particle", "long_name" => "mass of each floc (fractal!)"))
    v[:] = M1

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

    print("Saved $file_out_name \n")
    close(ds)

end 

run_my_model("FlocMod_g2.nc", true)
