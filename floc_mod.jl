#!/usr/bin/env julia



# **************** Fixed constants ***************
g = 9.81            # gravitational constant [m/s^2]
ν = 1.3e-6          # visosity of water [m^2/s]
ρ_w = 1000          # density of water [kg/m^3]
ρ_w = 2600          # density of sediment kg/m^3
α = 1 

# ************************************************
nf = 2.5                # fractal dimension exponent
Dp = 4 * 1e-6           # reference diameter (m)
Fy = 1e-10              # yield strength (N)
β_2 = 1.5            # winterwerp (2002)
β_3 = 3 - nf         # winterwerp (2002)

N = 200  # number of size classes
D = linspace(1e-5, 1e-4, N)  # particle diameters from 1 micron to 1 mm
n = zeros(N) + linspace(1e5, 1e4, N)  # number concentration in each size class (particles/m^3)



######################### Floc properties #########################
function calculate_w_s(D::Vector{<:Real}, floc_density::Vector{<:Real}, N=N, ρ_w=ρ_w, nu=nu, g=g)
    # settling velocity for particle of density floc_density and diameter D
    # returns : settling velocity (m/s)
    ws_ = zeros(N)
    for i in 1:N
        ws_[i] = g/(18*ν) * (floc_density[i] - ρ_w)/(ρ_w) * D[i]^2 
    end 
    return ws_
end 
    


function calculate_density(D::Vector{<:Real}, ρ_w=ρ_w, ρ_w=ρ_w, Dp=Dp, nf=nf, N=N)
    # calculate the density of a floc in size class i
    # parameters:
    #   ρ_w: particle density (kg/m^3)
    #   D: array of particle diameters (m)
    #   i: index of the particle size class
    #   Dp: reference diameter (m)
    #   nf: fractal dimension exponent
    # returns: density of a particle in size class i (kg/m^3)
    density_ = zeros(N)
    for i in 1:N
        density_[i] = ρ_w + (ρ_w - ρ_w) * (Dp/D[i])^(nf - 3)
    end 
    return density_
end 

function calculate_mass(D::Vector{<:Real}, floc_density::Vector{<:Real}, N=N, Dp=Dp, nf=nf)
    # calculate the mass of a floc in size class i
    # parameters:
    #   floc_density: particle density (kg/m^3)
    #   D: array of particle diameters (m)
    #   i: index of the particle size class
    #   Dp: reference diameter (m)
    #   nf: fractal dimension exponent
    # returns: mass of a particle in size class i (kg)
    mass_ = np.zeros(N)
    for i in 1:N
        mass_[i] = floc_density[i] * π/6 * D[i]^3 * (D[i]/Dp)^nf 
    end 
    return mass_
end 


# calculate once since diameters don't change
collision_matrix = zeros(N,N)
for i in 1:N
    for j in 1:N
        collision_matrix[i,j] = 1/6 * (D[i] + D[j])^3
    end
end 

closest_half_mass = zeros(N)
for i in 1:N
    closest = mass[i] - (2*mass)   # SW adding this since no mass is exactly half of another mass
                                   # instead, we find the class w/ the closest mass to half the mass of floc k
    closest_half_mass[i] = argmin(abs(closest))
end 



function A(G::Real, i::Int, j::Int, collision_matrix=collision_matrix)
    #  Two-body collision probability function A(i,j) is a function of the 
    # shear rate G and the particle diameters D_i and D_j
    # parameters:
    #   G: shear rate (1/s)
    #   D: array of particle diameters (m)
    #   i, j: indices of the particle size classes for particles i and j
    # returns: collision probability between particles i and j (m^3/s) 
    return  G * collision_matrix[i,j]
end

function g1(k::Int, G::Real, n=n, N=N, α=α)
    # Growth due to collisions for class k 
    # Parameters:
    #   k: index of the particle size class
    #   A : collision probability function
    #   n : number concentration array
    #   G: shear rate (1/s)
    # Returns: growth rate due to aggregation for size class k (particles/m^3/s) 
    g1_ = 0.0 
    for j in 1:k
        for i in 1:k
            if i + j == k
                # j = k - i
                g1_ += α * A(G,i,j) * n[i] * n[j] 
            end
        end
    end
    g1_ *= 0.64 #5 SW adjustment for mass conservation. 
    return g1_
end

function l1(k, G, n=n, N=N, α=α)
# need to fix mass balance with aggregation / loss terms here! 
    # Loss due to collisions for class k
    l1_ = 0 
    for i in 1:(N-1)  # note this is N in original flocmod equations 
        l1_ += α * A(G,i,k) * n[i] * n[k] 
    end 
    return l1_
end 

######################### Shear break-up #########################



function FDBS(k::Int, i::Int, mass=mass, N=N, closest_half_mass=closest_half_mass)
    # Break-up distribution function
    # What size classes result from a floc breaking up? Using binary assumption 
    # here: a floc breaks into two equal pieces.
    
    # SW adding this since no mass is exactly half of another mass
    # instead, we find the class w/ the closest mass to half the mass of floc k
    j1 = closest_half_mass[i]
    if k == j1 
        FDBS_ = 2
    else 
        FDBS_ = 0

    end 
    return FDBS_
end 




# Calculate once since diameters don't change
D_ratio_4_B = zeros(N)
for i in 1:N                  # SW adding abs to avoid imaginary numbers
    D_ratio_4_B[i] = D[i] * β * abs((D[i] - Dp)/Dp)^β_3
end 


function B(k::Int, G::Real, D_ratio_4_B=D_ratio_4_B, β_2=β_2)
    # Bi = beta * G**(beta_2) * D[i] * ((D[i] - Dp)/Dp)**(beta_3)
    Bi = D_ratio_4_B[k] * G^(β_2)
    return Bi 
end 

function g2(k::Int, G::Real, n=n, N=N)
    # Growth due to shear break-up for class k
    g2_ = 0
    if k == N
        return g2_
    end

    for i in (k+1):N
        g2_ += FDBS(k,i) * B(i, G) * n[i]
    end
    return g2_
end


function l2(k::Int, G::Real, n=n)
    # Loss due to shear break-up for class k
    l2_ = B(k, G) * n[k]
    return l2_
end
