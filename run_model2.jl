#!/usr/bin/env julia

using Printf
using DataStructures: OrderedDict
using NCDatasets
using CSV, DataFrames
using Printf
using Profile
using Statistics 
using Arrow, DataFrames
using LaTeXStrings

include("hydro/calculate_physical_variables.jl") 
include("hydro/advance_variables.jl")
# include("hydro/phytoplankton.jl")
include("hydro/forcings.jl") 
include("hydro/output.jl")


# ws1 = parse(Float64, ARGS[1])
# ws2 = parse(Float64, ARGS[2])
# pmax1 = parse(Float64, ARGS[3])
# pmax2 = parse(Float64, ARGS[4])
# fout_name = ARGS[5]


function run_my_model(ws1::Real, ws2::Real, pmax1::Real, pmax2::Real, file_out_name::String)

    # println("Running model with ws1 = $ws1, ws2 = $ws2, pmax1 = $pmax1, pmax2 = $pmax2.. \n output file name = $file_out_name \n")

    #***********************************************************************



    #***********************************************************************
    # Wind time series 
    # wind_fn = "/global/homes/s/siennaw/scratch/siennaw/stockton_field_data/forcing_for_model/wind_on_august_10-16.csv"
    # df = CSV.read(wind_fn, DataFrame)
    # wind = df[!,"WindSpeed"]
    # real_time = df[!,"time"]
    # println("Read in wind data ...")
    
    
    # function get_wind_speed(index::Int, wind=wind)
    #     return wind[index]
    end
    #***********************************************************************

    


    #********************** SPATIAL DOMAIN  ***************************
    N = 60    # number of grid points
    H = 6    # depth (meters)
    dz = H/N  # grid spacing - may need to adjust to reduce oscillations
    dt = 10   # (seconds) size of time step 
    M  = 51839 #00 #000 # 50000  #500 #

    # Increments for saving profiles. set to 1 to save all; 10 saves every 10th, etc. 
    isave = 6 #1000
    var2save = ["U","Kq", "Nu", "C", "Kz", "L", "Q2", "Q2L", "N_BV2", "ssc"]

    create_output_dict(M, isave, var2save, N)

    # Create depth vector 
    z = collect(H:-dz:dz) .- dz/2 # depth vector
    # println("Length of z is ", length(z))

    #********************** FIXED CONSTANTS  ***************************
    rhoA = 1.23                     # Density of air, kg/m^3
    rhoW = 1000                     # Density of water, kg/m^3
    specific_heat_water = 4181      # J/kg-degC
    specific_heat_air = 1007        # J/kg-degC x RH
    c_d = 0.05                      # Drag coefficient [-]
    cm2m = 0.01
    hr2s = 1/3600



    #********************** DEFINE HYDRODYNAMIC FORCINGS ***************************
    # (1) PRESSURE 
    Px0 = 2e-6          # Pressure gradient forcing
    T_Px = 12           # Period [hours] on pressure gradient forcing. Set to 0 for steady

    # (2) Wind
    # Wind = 1                       # u_star =m/s >> 0.05 is  drag coefficient, 10 is my wind speed 
    # WIND = (c_d * Wind)^2 * rhoA   # this is rho * u*^2


    # ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
    Ns = 5                  # Number of sediment size classes
    ssc = zeros(N, Ns)      # Matrix for sediment concentration (Nz x Ns)

    D = LinRange(1, 20, Ns) .* 1e-6     # Sediment grain sizes (\mu m )
    ρ_s = calculate_density(D)          # Sediment density (kg/m^3) (Ns x 1)
    w_s = calculate_w_s(D, ρ_s)         # Settling velocity (m/s)   (Ns x 1)

    #***************************************************************************
    #   Initialize variables
    #***************************************************************************

    # Create dictionary to hold important discretization parameters
    discretization = Dict("beta" => (dt/dz^2), "dz" => dz, "dt" => dt, "N" => N, "z"=> z, "H" => H)

    # Initial dictionary to store variables
    variables = Dict() 
    # "U" => U, "C" => C, "N_BV2" => N_BV2, 
    #                 "Nu" => nu_t, "Q2" => Q2, "Q2L" => Q2L, 
    #                 "Kq" => Kq, "Kz" => Kz, "L" => L

    Times = collect(0:dt:(M*dt))
    # println("Times = ", Times)

    variables["U"] = similar(z) .+ 1e-1
    variables["C"] = similar(z) .+ 26
    rho_ = calculate_rho(variables["C"], 22)
    variables["N_BV2"] = calculate_brunt_vaisala(rho_, discretization)


    variables["Q2"], variables["Q2L"], 
        variables["Q"], variables["L"], 
        variables["Gh"], variables["Nu"], 
        variables["Kq"], variables["Kz"] = initialize_turbulent_functions(discretization, variables["N_BV2"])

    base_temp = 22
    dtemp = 1.5 
    stretch = 0.25 
    #***************************************************************************

    save2output(1, 1, "U", variables["U"])
    save2output(1, 1, "Kz", variables["Kz"])
    save2output(1, 1, "C", variables["C"])
    save2output(1, 1, "L", variables["L"])
    save2output(1, 1, "Q2", variables["Q2"])
    save2output(1, 1, "Q2L", variables["Q2L"])
    save2output(1, 1, "N_BV2", variables["N_BV2"])
    save2output(1, 1, "Kq", variables["Kq"])
    save2output(1, 1, "Nu", variables["Nu"])

    for i in 2:(M-1)


        time = Times[i];

        # [1] Advance velocity field
        pressure = get_pressure_at_timestamp(time, Px0, T_Px)
        ustar = calculate_ustar(variables["U"])

        W0 = 3 #get_wind_speed(i)

        # C = get_unstrat_temp_field(i)       # [1] Unstratified field @ night 
        rho = calculate_rho(variables["U"], base_temp)   # [2] Calculate density from temperature field
        N_BV2 = calculate_brunt_vaisala(rho, discretization)
        # else
        #     C = get_unstrat_temp_field(i)  ## println("$(real_time[i])  I0 = $I0 --> daytime ")
        #     # C = get_temp_field(i) #get_temp_field(i)              # [1] Observational, sttratified temperature field 
        #     rho = calculate_rho(C, base_temp)  # [2] Calculate density from temperature field  
        #     N_BV2 = calculate_brunt_vaisala(rho, discretization) # [3] Calculate Brunt-Vaisala frequency 
        #     N_BV2 = clamp.(N_BV2, -1e-3, Inf)               # [4] Prevent any unstable stratification during daylight hours
        # end 
               

        # C  = get_temp_field(i)
        # I0 = diurnal_light(time, I_in, 0, DIURNAL_LIGHT)

        # Advance velocity field 
        wind_stress = wind_speed_2_wind_stress(W0, discretization) 
        U = advance_velocity(variables, pressure, discretization, wind_stress)

        #  [2] Advance TKE / Q2 
        Q2 = advance_Q2(variables, ustar, discretization) 
        Q = @. sqrt(Q2)

        #  [3] Advance Q2*L      
        Q2L = advance_Q2L(variables, ustar, discretization)
    
        #  [4] Advance temperature 
        C = advance_scalar(variables, discretization) 

        # [7] Semi-implicit: Calculate turbulent lengthscale
        L, Q2L = calculate_lengthscale(Q2, Q2L, N_BV2, discretization)

        # Calculate stability parameter 
        gh = calculate_Gh(N_BV2, L, Q)
        nu_t, Kq, Kz = calculate_turbulent_functions(gh, Q, L, discretization) 

        # [8] Advance phytoplankton

        # Algae 1 #zeros(N) #
        
        # a1 = advance_algae(variables, algae1, gamma, discretization)  # zeros(N) .+ init_algae  #
        # algae1["c"] =  clamp.(a1, 1e-5, Inf)   

        # println("gamma = ", gamma[end-5:end])
        # println("algae1 = ", algae1["c"][end-5:end])


        # [9] Pack variables for next timestep 
        variables["U"] = U
        variables["C"] = C
        variables["N_BV2"] = N_BV2
        variables["Nu"] = nu_t
        variables["Q2"] = Q2
        variables["Q2L"] = Q2L
        variables["Kq"] = Kq
        variables["Kz"] = Kz
        variables["L"] = L

        if i % isave == 0
            index = div(i, isave) + 1  #(i-1) #div(i, isave)
            save2output(time, index, "U", variables["U"])
            save2output(time, index, "Kz", variables["Kz"])
            save2output(time, index, "C", variables["C"])
            save2output(time, index, "L", variables["L"])
            save2output(time, index, "Q2", variables["Q2"])
            save2output(time, index, "Q2L", variables["Q2L"])
            save2output(time, index, "N_BV2", variables["N_BV2"])
            save2output(time, index, "Nu", variables["Nu"])
            save2output(time, index, "Kq", variables["Kq"])
            # push!(real_times_saved, real_time[i])
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
                "algae1" => "Diatom concentration",
                "algae2" => "HAB concentration",
                "L" => "Turbulent length scale", 
                "Q2" => "TKE","Q2L" => "TKE*L",
                "N_BV2" => "Brunt-Vaisala frequency", "Kq" => "Kq", "Nu" => "Nu_t")

    times_unique = unique(times) 

    ds = NCDataset("hydro.nc" ,"c")
    ds.attrib["title"] = "testing"

    # model_time = collect(1:M)
    defDim(ds, "z", length(z)) 
    defDim(ds, "time", length(times_unique))

    v = defVar(ds, "z", Float32, ("z",))
    v[:] = z

    v = defVar(ds, "time", Float32, ("time",), attrib = OrderedDict("units" => "seconds"))
    v[:] = collect(1:(length(times_unique))) #model_time

    for var in var2save
        # println(var)
        v = defVar(ds, var, Float64,("z","time"), attrib = OrderedDict(
        "units" =>  units_dict[var], "long_name" => var2name[var]))
        v[:,:] = output[var];
    end

    print("Saved $file_out_name \n")
    close(ds)
    
  

end 

file_out_name = @sprintf("hydro.nc") 
run_my_model(1.38e-4, 1.38e-4, 0.04, 0.04, "hydro.nc")



# using StatProfilerHTML 
# # using ProfileView   
# using Profile 




# @profilehtml run_my_model(ws1, ws2, pmax1, pmax2, file_out_name)

# StatProfilerHTML.view()
# Profile.print() 
