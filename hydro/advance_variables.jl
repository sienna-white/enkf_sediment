# Physical parameters 
z0 = 0.01         # Bottom roughness [m]
zb = 10*z0        # Bottom height [m]
g  = 9.81         # Gravity [m/s^2]
C_D = 0.0025      # Friction coefficient 
SMALL = 1e-6      # Noise floor for turbulence quantities [m/s?]
kappa = 0.4       # Von Karman constant
nu = 1e-6         # Kinematic viscosity [m^2/s]
rho0 = 1000       # Water density [kg/m^3]
alpha  =  2.1e-4      # Thermal expansivity, set to zero for passive scalar case

# Mellor-Yamada closure parameters. All dimensionless --> no need to change  
A1=0.92 # [-]
A2=0.74 # [-]
B1=16.6 # [-]
B2=10.1 # [-]
C1=0.08 # [-]
E1=1.8  # [-]
E2=1.33 # [-]
E3=0.25 # [-]
Sq=0.2  # [-]


function TDMA(a::Vector{<:Real}, b::Vector{<:Real}, c::Vector{<:Real}, d::Vector{<:Real}, N::Real)
    # Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
    # a = Lower Diag, b = Main Diag, c = Upper Diag, d = solution vector

    x = zeros(N)
    for i in 2:N
        b[i] = b[i] - a[i]/b[i-1]*c[i-1]
        d[i] = d[i] - a[i]/b[i-1]*d[i-1]
    end
    x[N] = d[N]/b[N]
    for i in N-1:-1:1
        x[i] = (1/b[i])*(d[i] - c[i]*x[i+1])
    end
    return x
end

 
function initialize_abcd(N)
    a = zeros(N)
    b = zeros(N)
    c = zeros(N)
    d = zeros(N)
    return a, b, c, d
end

# (U_past, Nut_past, pressure, beta, 0)
function advance_velocity(past_var::Dict, Px::Real, discretization::Dict, W=0::Real)

    U_past = past_var["U"]
    nu_past = past_var["Nu"]

    N = length(U_past)

    aU, bU, cU, dU = initialize_abcd(N)

    beta = discretization["beta"]
    dt = discretization["dt"]
    dz = discretization["dz"]
    
    for i in 2:(N-1)
        aU[i] = -beta/2*(nu_past[i] + nu_past[i-1])
        bU[i] = 1 + beta/2*(nu_past[i+1] + 2*nu_past[i] + nu_past[i-1])
        cU[i] = -beta/2 * (nu_past[i] + nu_past[i+1])
        dU[i] = U_past[i] - dt*Px #[i]
    end

    bU[1] = 1 + beta/2*(nu_past[2] + nu_past[1] + 2*((C_D^0.5)/kappa)*nu_past[1])
    cU[1] = -beta/2*(nu_past[2] + nu_past[1])
    dU[1] = U_past[1] - dt*Px # [1]

    aU[end] = -beta/2*(nu_past[end] + nu_past[end-1])
    bU[end] = 1 + beta/2*(nu_past[end] + nu_past[end-1])
    dU[end] = U_past[end] - dt*Px + W #dz # Px[end]

    U = TDMA(aU, bU, cU, dU, N)

    return U
end 

function advance_algae_tracer(variables, algae, ind_photic_depth, discretization)
    N = discretization["N"]
    beta = discretization["beta"]
    dt = discretization["dt"]
    dz = discretization["dz"]

    photic_zone = zeros(N) 
    if ind_photic_depth < (N-1)
        photic_zone[ind_photic_depth:end] .= 1 
    end 
    aA, bA, cA, dA = initialize_abcd(N)

    ws = algae["ws"]
    Kz_past = variables["Kz"]
    A_past = algae["c"]
    Age_past = algae["age"]

    wsdtdz = abs(ws*dt)/dz

    # If settling speed is UPWARD (swimming!)
    if ws>0
        for i in 2:(N-1)
            aA[i] = -wsdtdz - beta/2 * (Kz_past[i-1] + Kz_past[i])
            bA[i] = 1 + wsdtdz  + beta/2*(Kz_past[i+1] + 2*Kz_past[i] + Kz_past[i-1])
            cA[i] = -beta/2 * (Kz_past[i] + Kz_past[i+1])
            dA[i] = A_past[i] + photic_zone[i]*dt
        end

        # Bottom-Boundary: no flux for scalars
        bA[1] =  1  + beta/2*(Kz_past[2] + Kz_past[1]) + wsdtdz
        cA[1] = -beta/2 * (Kz_past[2] + Kz_past[1])
        dA[1] =  A_past[1] +  photic_zone[1]*dt

        # Top-Boundary: no flux for scalars
        aA[end] = -wsdtdz - beta/2 * (Kz_past[end] + Kz_past[end-1])
        bA[end] = 1 + beta/2 * (Kz_past[end] + Kz_past[end-1])  
        dA[end] = A_past[end] + photic_zone[end]*dt

    # ********** If settling speed is DOWNWARD (sinking!) **********
    else
        for i in 2:(N-1)
            aA[i]  = -wsdtdz - beta/2 * (Kz_past[i-1]+ Kz_past[i]) 
            bA[i]  = 1 + wsdtdz  + beta/2*(Kz_past[i+1] + 2*Kz_past[i] + Kz_past[i-1]) 
            cA[i]  = -beta/2 * (Kz_past[i] + Kz_past[i+1])
            dA[i]  = A_past[i] + photic_zone[i]*dt 
        end 
           
        # Bottom-Boundary: no flux for scalars
        bA[1] =  1 + wsdtdz  + beta/2*(Kz_past[2] + Kz_past[1]) 
        cA[1] =  -wsdtdz -beta/2 * (Kz_past[2] + Kz_past[1])
        dA[1] =  A_past[1] + photic_zone[1]*dt

        # Top-Boundary: no flux for scalars
        aA[end] =  -beta/2 * (Kz_past[end] + Kz_past[end-1])
        bA[end] =  1 + beta/2 * (Kz_past[end] + Kz_past[end-1]) + wsdtdz # okay adding this here 
        dA[end] = A_past[end] + photic_zone[end]*dt
    end 

    # Solve the tridiagonal system
    A = TDMA(aA, bA, cA, dA, N) 
    
    return A
end 


function advance_algae(variables, algae, gamma, discretization)
    N = discretization["N"]
    beta = discretization["beta"]
    dt = discretization["dt"]
    dz = discretization["dz"]

    aA, bA, cA, dA = initialize_abcd(N)

    ws = algae["ws"]
    Kz_past = variables["Kz"]
    A_past = algae["c"]

    wsdtdz = abs(ws*dt)/dz

    # If settling speed is UPWARD (swimming!)
    if ws>0
        for i in 2:(N-1)
            aA[i] = -wsdtdz - beta/2 * (Kz_past[i-1] + Kz_past[i])
            bA[i] = 1 + wsdtdz - gamma[i]*dt + beta/2*(Kz_past[i+1] + 2*Kz_past[i] + Kz_past[i-1])
            cA[i] = -beta/2 * (Kz_past[i] + Kz_past[i+1])
            dA[i] = A_past[i]
        end

        # Bottom-Boundary: no flux for scalars
        bA[1] =  1 - (gamma[1]*dt) + beta/2*(Kz_past[2] + Kz_past[1]) + wsdtdz
        cA[1] = -beta/2 * (Kz_past[2] + Kz_past[1])
        dA[1] =  A_past[1]

        # Top-Boundary: no flux for scalars
        aA[end] = -wsdtdz - beta/2 * (Kz_past[end] + Kz_past[end-1])
        bA[end] = 1 - gamma[end]*dt + beta/2 * (Kz_past[end] + Kz_past[end-1])  
        dA[end] = A_past[end]

    # ********** If settling speed is DOWNWARD (sinking!) **********
    else
        for i in 2:(N-1)
            aA[i]  = -wsdtdz - beta/2 * (Kz_past[i-1]+ Kz_past[i]) 
            bA[i]  = 1 + wsdtdz - gamma[i]*dt  + beta/2*(Kz_past[i+1] + 2*Kz_past[i] + Kz_past[i-1]) 
            cA[i]  = -beta/2 * (Kz_past[i] + Kz_past[i+1])
            dA[i]  = A_past[i]
        end 
           
        # Bottom-Boundary: no flux for scalars
        bA[1] =  1 + wsdtdz - (gamma[1]*dt) + beta/2*(Kz_past[2] + Kz_past[1]) 
        cA[1] =  -wsdtdz -beta/2 * (Kz_past[2] + Kz_past[1])
        dA[1] =  A_past[1]

        # Top-Boundary: no flux for scalars
        aA[end] =  -beta/2 * (Kz_past[end] + Kz_past[end-1])
        bA[end] =  1 - gamma[end]*dt + beta/2 * (Kz_past[end] + Kz_past[end-1]) + wsdtdz # okay adding this here 
        dA[end] = A_past[end]   
    end 

    # Solve the tridiagonal system
    A = TDMA(aA, bA, cA, dA, N) 
    
    return A
end 


function advance_scalar(variables, discretization)

    Kz_past = variables["Kz"]
    C_past = variables["C"]
    N = discretization["N"]
    beta = discretization["beta"]

    aC, bC, cC, dC = initialize_abcd(N) 

    for i in 2:(N-1)
        aC[i] = -0.5*beta*(Kz_past[i] + Kz_past[i-1])
        bC[i] = 1 + 0.5*beta*(Kz_past[i+1] + 2*Kz_past[i] + Kz_past[i-1])
        cC[i] = -0.5*beta*(Kz_past[i] + Kz_past[i+1])
        dC[i] = C_past[i]
    end

    #####  Boundary conditions
    # Bottom-Boundary: no flux for scalars
    bC[1] = 1 + beta/2*(Kz_past[2] + Kz_past[1])
    cC[1] = -beta/2*(Kz_past[2] + Kz_past[1])
    dC[1] =  C_past[1] 

    # Top-Boundary: no flux for scalars
    aC[end] = -0.5*beta*(Kz_past[end] + Kz_past[end-1])
    bC[end] = 1+0.5*beta*(Kz_past[end] + Kz_past[end-1])
    dC[end] = C_past[end]  
    
    C = TDMA(aC, bC, cC, dC, N)
    return C
end


function advance_Q2(past_var::Dict, ustar, discretization::Dict, B1=B1)

    Q2_past = past_var["Q2"]
    L_past = past_var["L"]
    Kq_past = past_var["Kq"]
    nu_t_past = past_var["Nu"]
    U_past = past_var["U"]
    Kz_past = past_var["Kz"]
    N_BV2_past = past_var["N_BV2"]

    beta = discretization["beta"]
    dt = discretization["dt"]

    N = length(Q2_past)
    aQ2, bQ2, cQ2, dQ2 = initialize_abcd(N)

    # diss = @. (2 * dt *(Q2_past[2:end].^0.5))/(B1*L_past[2:end])
  
    for i in 2:(N-1)
        diss = 2*dt * ((Q2_past[i]^0.5)/(B1*L_past[i]))
        aQ2[i] = -0.5*beta*(Kq_past[i] + Kq_past[i-1])
        bQ2[i] = 1 + 0.5*beta*(Kq_past[i+1] + 2*Kq_past[i] + Kq_past[i-1]) + diss
        cQ2[i] = -0.5*beta*(Kq_past[i] + Kq_past[i+1]) # buoyancy production term  
                                                       # (should be negative such that this term is
                                                       # adding TKE when density is unstable)
        dQ2[i] = Q2_past[i] + 0.25*beta*nu_t_past[i]*(U_past[i+1]-U_past[i-1])^2 - dt*Kz_past[i]*N_BV2_past[i]
    end

    # Bottom-Boundary Condition
    Q2bot = B1^(2/3) * ustar^2
    bdryterm = 0.5*beta*Kq_past[1]*Q2bot
    dissipation = 2 * dt *((Q2_past[1]^0.5)/(B1*L_past[1]))
    bQ2[1] = 1 + 0.5*beta*(Kq_past[2] + Kq_past[1]) + dissipation
    cQ2[1] = -0.5*beta*(Kq_past[2] + Kq_past[1])
    dQ2[1] = Q2_past[1] + dt*((ustar^4)/nu_t_past[1]) - dt*Kz_past[1]*(N_BV2_past[1]) + bdryterm

    # Top-Boundary Condition
    dissipation = 2 * dt *((Q2_past[end]^0.5)/(B1*L_past[end]))
    aQ2[end] = -0.5*beta*(Kq_past[end] + Kq_past[end-1])
    bQ2[end] = 1 + 0.5*beta*(Kq_past[end] + 2*Kq_past[end] + Kq_past[end-1]) + dissipation
    dQ2[end] = Q2_past[end] + 0.25*beta*nu_t_past[end] * (U_past[end] - U_past[end-1])^2 - 4*dt*Kz_past[end]*N_BV2_past[end]


    Q2  = TDMA(aQ2, bQ2, cQ2, dQ2, N) 
    Q2  = add_noise_floor(Q2)


return Q2
end 


function advance_Q2L(past_var::Dict, ustar::Real, discretization::Dict, B1=B1, kappa=kappa, zb=zb, E2=E2, E3=E3)

    # Unpack discretization parameters
    N = discretization["N"]
    beta = discretization["beta"]
    z = discretization["z"]
    H = discretization["H"]
    dt = discretization["dt"]

    # Unpack variables
    Q2L_past = past_var["Q2L"]
    Q2_past = past_var["Q2"]
    L_past = past_var["L"]
    Kq_past = past_var["Kq"]
    U_past = past_var["U"]
    Kz_past = past_var["Kz"]
    N_BV2_past = past_var["N_BV2"]
    nut_past = past_var["Nu"]


    aQ2L, bQ2L, cQ2L, dQ2L = initialize_abcd(N)
    
    for i in 2:(N-1)
        dissipation_i = 2*dt*((sqrt(Q2_past[i])) / (B1*L_past[i]))*(1+E2*(L_past[i]/(kappa*abs(H-z[i])))^2 + E3*(L_past[i]/(kappa*z[i]))^2)
        aQ2L[i] = -0.5*beta*(Kq_past[i] + Kq_past[i-1])
        bQ2L[i] = 1 + 0.5*beta*(Kq_past[i+1] + 2*Kq_past[i] + Kq_past[i-1]) + dissipation_i
        cQ2L[i] = -0.5*beta*(Kq_past[i] + Kq_past[i+1])
        dQ2L[i] = Q2L_past[i] + 0.25*beta*nut_past[i]*E1*L_past[i]*(U_past[i+1]-U_past[i-1])^2  - 2*dt*L_past[i]*E1*Kz_past[i]*N_BV2_past[i]
    end 

    # Bottom boundary Condition
    q2lbot = B1^(2/3) * (ustar^2) * kappa * zb
    bdryterm = 0.5*beta*Kq_past[1]*q2lbot
    dissipation =  2 * dt *(Q2_past[1]^0.5)/(B1*L_past[1])*(1+E2*(L_past[1]/(kappa*abs(H-z[1])))^2 
                                                + E3*(L_past[1]/(kappa*abs(z[1])))^2)
    bQ2L[1] = 1+0.5*beta*(Kq_past[2] + Kq_past[1]) + dissipation
    cQ2L[1] = -0.5*beta*(Kq_past[2] + Kq_past[1])      
    dQ2L[1] = Q2L_past[1] + dt*((ustar^4)/nut_past[1])*E1*L_past[1] - dt*L_past[1]*E1*Kz_past[1]*(N_BV2_past[1]) + bdryterm                                      

    # Top boundary condition
    dissipation =  2 * dt *(sqrt(Q2_past[end]))/(B1*L_past[end])*(1+E2*(L_past[end]/(kappa*abs(H-z[end])))^2 
                                                + E3*(L_past[end]/(kappa*abs(z[end])))^2)
    aQ2L[end] = -0.5*beta*(Kq_past[end] + Kq_past[end-1])
    bQ2L[end] = 1+0.5*beta*(Kq_past[end] + 2*Kq_past[end] + Kq_past[end-1]) + dissipation
    dQ2L[end] = Q2L_past[end] + (0.25*beta*nut_past[end]*E1*L_past[end]*(U_past[end]-U_past[end-1])^2) - (2*dt*L_past[end-1]*E1*Kz_past[end]*N_BV2_past[end]) # Lp[end-1 or end]?

    Q2L  = TDMA(aQ2L, bQ2L, cQ2L, dQ2L, N)
    Q2L = abs.(Q2L)
    
    # Q2L  = add_noise_floor(Q2L)
    return  Q2L 
end 