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


function run_my_model(file_out_name::String, floc_on::Bool=true)

    #***********************************************************************
    # Wind time series 
    # wind_fn = "/global/homes/s/siennaw/scratch/siennaw/stockton_field_data/forcing_for_model/wind_on_august_10-16.csv"
    # df = CSV.read(wind_fn, DataFrame)
    # wind = df[!,"WindSpeed"]
    # real_time = df[!,"time"]
    # println("Read in wind data ...")
    
    
    # function get_wind_speed(index::Int, wind=wind)
    #     return wind[index]
    # end
    #***********************************************************************

    

    #********************** SPATIAL DOMAIN  ***************************
    N = 50    # number of grid points
    H = 4    # depth (meters)
    dz = H/N  # grid spacing - may need to adjust to reduce oscillations
    dt = 1    # (seconds) size of time step 
    M  = 3600*24*10 #*7 #24 #3600

    # Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
    isave = 40 # 6 #1000
    var2save = ["U", "Kq", "Nu", "C", "Kz", "L", "Q2", "Q2L", "N_BV2"]

    create_output_dict(M, isave, var2save, N)

    # Create depth vector 
    # z = 
    # z = collect(H:-dz:dz) .- dz/2 # depth vector
    z = collect(dz:dz:H) .- dz/2 

    #********************** FIXED CONSTANTS  ***************************
    rhoA = 1.23                     # Density of air, kg/m^3
    rhoW = 1000                     # Density of water, kg/m^3
    specific_heat_water = 4181      # J/kg-degC
    specific_heat_air = 1007        # J/kg-degC x RH
    c_d = 0.05                      # Drag coefficient [-]
    cm2m = 0.01
    hr2s = 1/3600
    base_temp = 22.0                   # Base temperature for density calculation
    #********************** DEFINE HYDRODYNAMIC FORCINGS ***************************
    # (1) PRESSURE 
    Px0 = 1e-4          # Pressure gradient forcing
    T_Px = 12           # Period [hours] on pressure gradient forcing. Set to 0 for steady

    # (2) Wind
    W0 = -0.5 #2.5
    # Wind = 1                       # u_star =m/s >> 0.05 is  drag coefficient, 10 is my wind speed 
    # WIND = (c_d * Wind)^2 * rhoA   # this is rho * u*^2



    #***************************************************************************
    #   Initialize variables
    #***************************************************************************

    # Create dictionary to hold important discretization parameters
    discretization = Dict("beta" => (dt/dz^2), "dz" => dz, "dt" => dt, "N" => N, "z"=> z, "H" => H)

    # Initial dictionary to store variables
    variables = Dict() 
    Times = collect(0:dt:(M*dt))

    variables["U"] = similar(z) .+ 1e-1
    variables["C"] =  collect(LinRange(22, 26, N))#similar(z) .+ 26
    rho_ = calculate_rho(variables["C"], 22)
    variables["N_BV2"] = calculate_brunt_vaisala(rho_, discretization)
    variables["Q2"], variables["Q2L"], 
    variables["Q"], variables["L"], 
    variables["Gh"], variables["Nu"], 
    variables["Kq"], variables["Kz"] = initialize_turbulent_functions(discretization, variables["N_BV2"])

    #***************************************************************************
    # Save initial condition
    save2output(1, "U", variables["U"])
    save2output(1, "Kz", variables["Kz"])
    save2output(1, "C", variables["C"])
    save2output(1, "L", variables["L"])
    save2output(1, "Q2", variables["Q2"])
    save2output(1, "Q2L", variables["Q2L"])
    save2output(1, "N_BV2", variables["N_BV2"])
    save2output(1, "Kq", variables["Kq"])
    save2output(1, "Nu", variables["Nu"])
    real_times_saved = [Times[1]]
    #***************************************************************************


    #*************************** TIME LOOP  *********************************
    for i in 2:(M-1)
        time = Times[i];

        # [1] Get pressure + wind forcing at this time
        pressure = get_pressure_at_timestamp(time, Px0, T_Px)
        ustar = calculate_ustar(variables["U"])

        # [2] Calculate density & stratification from temperature field
        rho = calculate_rho(variables["U"], base_temp)   
        N_BV2 = calculate_brunt_vaisala(rho, discretization)

        # [3] Advance velocity field 
        wind_stress = wind_speed_2_wind_stress(W0, discretization) 
        U = advance_velocity(variables, pressure, discretization, wind_stress)

        # [4] Advance TKE / Q2 / Q2L
        Q2 = advance_Q2(variables, ustar, discretization) 
        Q = @. sqrt(Q2)   
        Q2L = advance_Q2L(variables, ustar, discretization)
        
        # [5] Calculate turbulent shear stress
        # turbulent_shear = (Q2 ./ Q2L) .* Q .*100
        # turbulent_shear[end] = 0.1 

        # [6] Advance temperature 
        C = advance_scalar(variables, discretization) 

        # [7] Semi-implicit: Calculate turbulent lengthscale
        L, Q2L = calculate_lengthscale(Q2, Q2L, N_BV2, discretization)

        # [8] Calculate stability parameter + turbulent diffusivities 
        gh = calculate_Gh(N_BV2, L, Q)
        nu_t, Kq, Kz = calculate_turbulent_functions(gh, Q, L, discretization) 


        
        # [9] Pack variables for next timestep 
        variables["U"] = U
        # variables["C"] = C
        variables["N_BV2"] = N_BV2
        variables["Nu"] = nu_t
        variables["Q2"] = Q2
        variables["Q2L"] = Q2L
        variables["Kq"] = Kq
        variables["Kz"] = Kz
        variables["L"] = L

        if i % isave == 0
            # Save output 
            index = div(i, isave) + 1 
            save2output(index, "U", variables["U"])
            save2output(index, "Kz", variables["Kz"])
            save2output(index, "C", variables["C"])
            save2output(index, "L", variables["L"])
            save2output(index, "Q2", variables["Q2"])
            save2output(index, "Q2L", variables["Q2L"])
            save2output(index, "N_BV2", variables["N_BV2"])
            save2output(index, "Nu", variables["Nu"])
            save2output(index, "Kq", variables["Kq"])
            push!(real_times_saved, time)
        end
    end
     
    # ********************** save data ****************************
    units_dict = Dict("U" => "m/s", 
        "C" => "deg C", 
        "Kz" => "m\$^2\$ s\$^{-1}\$", 
        "L" => "Turbulent length scale", 
        "Q2" => "TKE", "Q2L" => "TKE*L",
        "N_BV2" => "Brunt-Vaisala frequency", "Kq" => "Kq", "Nu" => "Nu_t")

    var2name = Dict("U" => "Velocity", 
                "C" => "Temperature", 
                "Kz" => "Turbulent diffusivity", 
                "L" => "Turbulent length scale", 
                "Q2" => "TKE","Q2L" => "TKE*L",
                "N_BV2" => "Brunt-Vaisala frequency", "Kq" => "Kq", "Nu" => "Nu_t")

    # times_unique = unique(times) 
    nt = div(M,isave) #+ 1
    
    ds = NCDataset(file_out_name,"c")
    ds.attrib["title"] = "testing"

    # model_time = collect(1:M)
    defDim(ds, "z", length(z)) 
    defDim(ds, "time", nt)




    v = defVar(ds, "z", Float32, ("z",))
    v[:] = z

    v = defVar(ds, "time", Float32, ("time",), attrib = OrderedDict("units" => "seconds"))
    v[:] = real_times_saved #real_times_saved #collect(1:nt) #model_time

    for var in var2save
        v = defVar(ds, var, Float64,("z","time"), attrib = OrderedDict(
        "units" =>  units_dict[var], "long_name" => var2name[var]))
        v[:,:] = output[var];
    end

    print("Saved $file_out_name \n")
    close(ds)

end 

# file_out_name = @sprintf("hydro.nc") 
run_my_model("hydro_newZ.nc", true)

