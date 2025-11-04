


#!/usr/bin/env julia

using Plots
using Printf
using DataStructures: OrderedDict
using NCDatasets
# using Arrow, DataFrames
using CSV, DataFrames
using Colors
using ColorSchemes
using Plots
using Printf
using LaTeXStrings
using Profile
using Statistics 

fp="/global/homes/s/siennaw/scratch/siennaw/two_species/adjoint_phytoplankton/model_code"
include("/global/homes/s/siennaw/scratch/siennaw/two_species/adjoint_phytoplankton/model_code/calculate_physical_variables.jl") 
include("/global/homes/s/siennaw/scratch/siennaw/two_species/adjoint_phytoplankton/model_code/advance_variables.jl")
include("/global/homes/s/siennaw/scratch/siennaw/two_species/adjoint_phytoplankton/model_code/phytoplankton.jl")
include("/global/homes/s/siennaw/scratch/siennaw/two_species/adjoint_phytoplankton/model_code/forcings.jl") 
include("/global/homes/s/siennaw/scratch/siennaw/two_species/adjoint_phytoplankton/model_code/output.jl")
include("/global/homes/s/siennaw/scratch/siennaw/two_species/adjoint_phytoplankton/model_code/define_params.jl")


function run_forward_model(file_out_name::String, adjoint_ds::String)

    #********************** SPATIAL DOMAIN  ***************************
    N = global_params["N"]   # number of grid points
    H = global_params["H"]   # depth (meters)
    dz = global_params["dz"] # grid spacing - may need to adjust to reduce oscillations
    dt = global_params["dt"] # (seconds) size of time step
    # M  = global_params["M"]  # number of time steps
    time_range = global_params["time_range"] # number of time steps


    istart = global_params["istart"] # start time step
    iend = global_params["iend"] # end time step
    M = iend - istart + 1 # number of time steps
    if M <= 0
        error("M must be greater than 0. Check istart and iend values.")
    end
    time_index_vec = collect(istart:(iend+1))


    file_out_name = "$(file_out_name)_$(time_range).nc"
    println("\n\nRunning the FORWARD PHYTOPLANKTON MODEL --> we are going forward in time")
    println("\t Adjusting our growth guess using the gamma from: $(adjoint_ds)")
    println("\t Will be saving phytoplankton output to: $(file_out_name)")

    forcing_folder = "/pscratch/sd/s/siennaw/stockton_field_data/forcing_for_model/2024/august6-28/"
    # Hydrodynamic dataset 
    ds = NCDataset("/pscratch/sd/s/siennaw/two_species/adjoint_phytoplankton/run_hydro/HYDRO_AUGUST6-28.nc")
    

    # Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
    isave = 1 
    var2save = ["floc1", "gamma1", "floc2", "gamma2"]      # Only save growth + algae 

    create_output_dict(M, isave, var2save, N)

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


    #********************** DEFINE PHYTOPLANKTON FORCINGS ***************************

    floc1 = Dict("d50" => 20e-6,             # mean diameter [m]
                "rho_sed" => 1001,          # specific density [kg/m^3]
                "ws" => 1.38e-4,                # vertical velocity [m/s]
                "Li" => 1.38e-5,             # specific loss rate [1/hour]
                 "name" => "sediment_1")                #  name 

   floc2 = Dict("d50" => 20e-6,             # mean diameter [m]
                "rho_sed" => 1001,          # specific density [kg/m^3]
                "ws" => 1.38e-5,                # vertical velocity [m/s]
                "Li" => 1.38e-5,             # specific loss rate [1/hour]
                 "name" => "sediment_2")                #  name


    #***************************************************************************
    #   Initialize variables
    #***************************************************************************

    # Create dictionary to hold important discretization parameters
    discretization = Dict("beta" => (dt/dz^2), "dz" => dz, "dt" => dt, "N" => N, "z"=> z, "H" => H)


    # Test SW for august 13, using the mapping data
    floc1["c"] = zeros(N) .+ init_conc 
    floc2["c"] = zeros(N) .+ init_conc

    # Create vector to hold the time steps 
    Times = collect(1:dt:(M*dt))

    #***************************************************************************
    save2output(1, 1, "floc1", floc1["c"])
    save2output(1, 1, "floc2", floc2["c"])

    variables = Dict("Kz" => ds["Kz"][:,1])

    # Iterate through time 
    for i in 2:M
        index = time_index_vec[i]
        time = index #Times[i];

        # Hydrodynamics
        variables["Kz"] = ds["Kz"][:,index]

        # Estimate source term 
        for j in 1:N 
            growth1[j] = 1
            growth2[j] = 1
        end

        # Split up loss + growth
        gamma1 = growth1 .- floc1["Li"]  # subtract the loss rate
        gamma2 = growth2 .- floc2["Li"]  # subtract the loss rate

        # Algae 
        floc1["c"] = advance_sediment(variables, floc1, gamma1, discretization)  
        floc2["c"] = advance_sediment(variables, floc2, gamma2, discretization) 

        save2output(time, i, "gamma1", growth1)
        save2output(time, i, "gamma2", growth2)
        save2output(time, i, "floc1", floc1["c"])
        save2output(time, i, "floc2", floc2["c"])

        if floc1["c"][1] > 1
            println("Time: $(time) \t floc1: $(floc1["c"][1]) \t gamma: $(gamma1[1])")
        end

    end

    # ********************** save data ****************************
    units_dict = Dict("U" => "m/s", 
        "C" => "deg C", 
        "Kz" => "m\$^2\$ s\$^{-1}\$", 
        "floc1" => L"10$^6$/cm$^3$ cells",
        "floc2" => L"10$^6$/cm$^3$ cells",
        "L" => "Turbulent length scale", 
        "Q2" => "TKE", "Q2L" => "TKE*L",
        "N_BV2" => "Brunt-Vaisala frequency", 
        "Kq" => "Kq", 
        "Nu" => "Nu_t",
        "gamma1" => "Net growth rate for floc1",
        "gamma2" => "Net growth rate for floc2")

    var2name = Dict("U" => "Velocity", 
                "C" => "Temperature", 
                "Kz" => "Turbulent diffusivity", 
                "floc1" => "HAB concentration",
                "floc2" => "Diatom concentration",
                "L" => "Turbulent length scale", 
                "Q2" => "TKE",
                "Q2L" => "TKE*L",
                "N_BV2" => "Brunt-Vaisala frequency", 
                "Kq" => "Kq", 
                "Nu" => "Nu_t", 
                "gamma1" => "Net growth rate [1/s]",
                "gamma2" => "Net growth rate [1/s]")


    fout = "forward_phyto/$(file_out_name)"

    ds = NCDataset(fout,"c")
    nt = div(M,isave) + 1 
    defDim(ds, "z", length(z)) 
    defDim(ds, "t", nt)

    v = defVar(ds, "z", Float32, ("z",))
    v[:] = z

    v = defVar(ds, "t", Int, ("t",), attrib = OrderedDict("units" => "seconds"))
    v[:] = time_index_vec #collect(1:nt)

    for var in var2save
        v = defVar(ds, var, Float64,("z","t"), attrib = OrderedDict(
        "units" =>  units_dict[var], "long_name" => var2name[var]))
        v[:,:] = output[var];
    end

    print("Saved $file_out_name \n")
    close(ds)
    
end 



# file_out_name = "phyto_fake_truth_june22"  
# run_forward_model(file_out_name, "FIRST")

# @profilehtml run_my_model(ws1, ws2, pmax1, pmax2, file_out_name)

# StatProfilerHTML.view()
# Profile.print() 
