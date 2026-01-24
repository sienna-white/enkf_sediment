#!/usr/bin/env julia
    #********************** INITIAL CONDITION ***************************
    # Initialize thermocline based on tanh curve 
    base_temp = 22
    dtemp = 1.5 
    stretch = 0.25 

        # (3) Temperature
    top_temp = 33
    bottom_temp = 30
    bottom_speed = 0 
    top_speed=3.5  


# Initalize velocity
U = similar(z) .+ 1e-1
C = get_unstrat_temp_field(1) # zeros(N) .+ LinRange(bottom_temp, top_temp, N)  
rho = calculate_rho(C, base_temp) 
N_BV2 = calculate_brunt_vaisala(rho, discretization)


Q2, Q2L, Q, L, Gh, nu_t, Kq, Kz = initialize_turbulent_functions(discretization, N_BV2)
