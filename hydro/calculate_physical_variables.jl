
# Physical parameters 
z0 = 0.01         # Bottom roughness [m]
zb = 10*z0        # Bottom height [m]
g  = 9.81         # Gravity [m/s^2]
C_D = 0.0025      # Friction coefficient 
SMALL = 1e-6      # Noise floor for turbulence quantities [m/s?]
kappa = 0.4       # Von Karman constant
nu = 1e-6         # Kinematic viscosity [m^2/s]
rho0 = 1000       # Water density [kg/m^3]
alpha  =  2.1e-4  # Thermal expansivity, set to zero for passive scalar case

function calculate_photic_depth_ind(light, I0)
    light_lim = 0.1 * I0
    search = light .> light_lim
    ind_photic_depth = length(light) - sum(search) 

    # if I0>0.1
    #     println(light[end-10:end])
    #     println("I0 = $I0; Photic depth occurs @ ind = $ind_photic_depth")
    # end 
    return ind_photic_depth
end


function initialize_turbulent_functions(discretization, N_BV2, kappa=kappa, SMALL=SMALL)
    Q2  = fill(SMALL,  discretization["N"])   # "seed" the turbulent field with small values, then let it evolve
    Q2L = fill(SMALL,  discretization["N"])
    Q = sqrt.(Q2)
    z = discretization["z"]
    H = discretization["H"]

    L = similar(Q) 
    for i in 1:length(Q)
        L[i] = Q2L[i]/(Q2[i] + SMALL) #kappa*H*(z[i]/H)*(1-(z[i]/H))    # Q2(n,1)/Q2(n,1) = 1 at initialization
    end
    # Initialize Gh (stratification correction)
    gh = calculate_Gh(N_BV2, L, Q)
    nu_t, Kq, Kz  = calculate_turbulent_functions(gh, Q, L, discretization)
    return Q2, Q2L,Q, L, gh, nu_t, Kq, Kz
end 

function random_seed_vector(discretization, SMALL=SMALL)
   return fill(SMALL,  discretization["N"])
end 

function calculate_rho(C::Vector{<:Real}, base_temp::Real=22, alpha=alpha, rho0=rho0)
    rho = @. rho0*(1 - alpha*(C - base_temp))  # Single scalar, linear equation of state
    return rho
end 

function calculate_ustar(U_past, C_D=C_D)
    ustar = sqrt(C_D) * abs(U_past[1]) 
    return ustar
end

function calculate_Gh(N_BV2, L, Q)
    Gh = similar(L)
    for i in 1:length(L)
        gh = -(N_BV2[i] * L[i]^2)/(Q[i] + SMALL)^2
        gh = clamp(gh, -0.28, 0.0233)
        Gh[i] = gh
    end

    # Gh = -((N_BV2*L)/(Q + SMALL))**2
    # Gh = np.clip(Gh, -0.28, 0.0233)

    return Gh 
end 


function calculate_brunt_vaisala(rho, discretization)
    N = discretization["N"]
    dz = discretization["dz"]

    n_bv2 = similar(rho)

    for i in 1:(N-1)
        dpdz_i = (rho[i+1] - rho[i])/dz
        n_bv2[i] = -g/rho0 * dpdz_i
    end
    n_bv2[end] =(-g/rho0) * (rho[end] - rho[end-1])/dz
    return n_bv2
end 



function add_noise_floor(vector, SMALL=SMALL)
    vector[vector .< 0] .= SMALL
    return vector
end 


function calculate_turbulent_functions(gh, Q, L, discretization)

    N = discretization["N"]
    nu_t = similar(Q)
    Kq = similar(Q)
    Kz = similar(Q) 
    sq = 0.2
    for i in 1:N 

        # Calculate sm + sh (stability functions)
        num = B1^(-1/3) - A1*A2*gh[i]*((B2-3*A2)*(1-6*A1/B1)-3*C1*(B2+6*A1))
        dem = (1 - 3*A2*gh[i]*(B2+6*A1))*(1 - 9*A1*A2*gh[i])
        sm = num/dem
        sh = A2*(1-6*A1/B1)/(1-3*A2*gh[i]*(B2+6*A1))
              
        nu_t[i] = (sm * Q[i] * L[i]) + nu  # Turbulent diffusivity for Q2
        Kq[i]   = (sq * Q[i] * L[i]) + nu  # Turbulent diffusivity for Q
        kz0   = (sh * Q[i] * L[i]) + nu 
        Kz[i] = clamp(kz0, SMALL, Inf)     # Turbulent diffusivity for z


    end
    

    return nu_t, Kq, Kz
end 

function calculate_lengthscale(Q2, Q2L, N_BV2, discretization, SMALL=SMALL, zb=zb)
    N = discretization["N"]
    L = zeros(N)

    for i in 1:N

        L[i] = Q2L[i]/(Q2[i] + SMALL)

        if (L[i]^2 * N_BV2[i]) > 0.281*Q2[i]  # ^2
            Q2L_ = Q2[i]*sqrt(0.281*Q2[i]/(N_BV2[i] + SMALL))
            Q2L[i] = Q2L_ 
            L[i] = Q2L_ / (Q2[i])  
        end 
        if abs(L[i]) <= zb
            L[i] = zb
        end
    end 
    return L, Q2L 
    
end
 
