#!/usr/bin/env julia

# module load  julia/1.11.7
module floc_mod
export init_params!, run_floc_mod 

# **************** Fixed constants ***************
const g = 9.81            # gravitational constant [m/s^2]
const ν = 1.3e-6          # visosity of water [m^2/s]
const ρ_w = 1000          # density of water [kg/m^3]
const ρ_s = 2600          # density of sediment kg/m^3
const α = 1 

# ************************************************
const nf = 2.5                # fractal dimension exponent
const Dp = 4 * 1e-6           # reference diameter (m)
const Fy = 1e-10              # yield strength (N)
const β_2 = 1.5            # winterwerp (2002)
const β_3 = 3 - nf         # winterwerp (2002)
const β = 0.1

const closest_half_mass = Ref{Vector{Int}}()
const collision_matrix = Ref{Matrix{Float64}}()
const D_ratio_4_B = Ref{Vector{Float64}}()
const ws = Ref{Vector{Float64}}()
const mass = Ref{Vector{Float64}}()
const floc_density = Ref{Vector{Float64}}()


function init_params!(D::Vector{<:Real}, N::Int)

    @info "Initializing flocculation model with $N sediment classes ..."

    # Calculate floc density, settling velocity, and mass 
    floc_density[] = calculate_density(D, N)
    ws[] = calculate_w_s(D, N)
    mass[] = calculate_mass(D, N)
    
    @info "\t initialized density, settling velocity, and floc mass..."
    closest_half_mass0 = zeros(N)
    collision_matrix0 = zeros(N,N)
    D_ratio_4_B0 = zeros(N)

    for i in 1:N
        closest = @. mass[][i] - (2*mass[])   # SW adding this since no mass is exactly half of another mass
                                   # instead, we find the class w/ the closest mass to half the mass of floc k
        closest_half_mass0[i] = argmin(broadcast(abs, closest))

        D_ratio_4_B0[i] = D[i] * β * abs((D[i] - Dp)/Dp)^β_3 # SW adding abs to avoid imaginary numbers

        for j in 1:N
            collision_matrix0[i,j] = 1/6 * (D[i] + D[j])^3
        end
    end 

    closest_half_mass[] = closest_half_mass0
    collision_matrix[] = collision_matrix0
    D_ratio_4_B[] = D_ratio_4_B0

    @info "\t initialized collision matrix, half-mass indices, and D ratio for break-up ..."

    return nothing


end



######################### Floc properties #########################
function calculate_w_s(D::Vector{<:Real}, N::Int)
    # settling velocity for particle of density floc_density and diameter D
    # returns : settling velocity (m/s)
    ws_ = zeros(N)
    for i in 1:N
        ws_[i] = g/(18*ν) * (floc_density[][i] - ρ_w)/(ρ_w) * D[i]^2 
    end 
    return ws_
end 
    


function calculate_density(D::Vector{<:Real}, N::Int)
    # calculate the density of a floc in size class i
    # parameters:
    #   ρ_w: particle density (kg/m^3)
    #   D: array of particle diameters (m)
    # returns: density of a particle in size class i (kg/m^3)
    density_ = zeros(N)
    for i in 1:N
        density_[i] = ρ_w + (ρ_s - ρ_w) * (Dp/D[i])^(nf - 3)
    end 
    return density_
end 

function calculate_mass(D::Vector{<:Real}, N=N::Int)
    # calculate the mass of a floc in size class i
    # parameters:
    #   floc_density: particle density (kg/m^3)
    #   D: array of particle diameters (m)
    #   i: index of the particle size class
    #   Dp: reference diameter (m)
    #   nf: fractal dimension exponent
    # returns: mass of a particle in size class i (kg)
    mass_ = zeros(N)
    for i in 1:N
        mass_[i] = floc_density[][i] * π/6 * D[i]^3 * (D[i]/Dp)^nf 
    end 
    return mass_
end 


# ******************** Flocculation functions ************************


function run_floc_mod(n::Vector{<:Float64}, N::Int, G::Real)
    # println("[1] Mass = ", sum(n .* mass[]))  # check mass conservation

    g1_ = zeros(N)
    l1_ = zeros(N)
    g2_ = zeros(N)
    l2_ = zeros(N)

    for k in 1:N
        g1_[k] = g1(n, k, G)
        l1_[k] = l1(n, k, G, N)
        g2_[k] = g2(n, k, G, N)
        l2_[k] = l2(n, k, G)
    end 
    
    change = zeros(N)
    for i in 1:N
      #          agg (+)  agg(-)  shear(+)   shear(-)
        change[i]  =  g1_[i] - l1_[i] + g2_[i]  - l2_[i]
        # println("n[$i] = $(n[i])")
        # println(" g1 = $(g1_[i]), l1 = $(l1_[i]), g2 = $(g2_[i]), l2 = $(l2_[i])")
    end 
    # println("n[1:10] = $(n)")
    n_new = n .+ change
    n_adj = flocmod_mass_redistribute(n_new, N)
    change = n_adj .- n
    # println("[2] Mass change = ", sum(change .* mass[]))  # check mass conservation
    return change

end 


# ****************** Aggregation functions ***********************
function A(G::Real, i::Int, j::Int)
    #  Two-body collision probability function A(i,j) is a function of the 
    # shear rate G and the particle diameters D_i and D_j
    # parameters:
    #   G: shear rate (1/s)
    #   D: array of particle diameters (m)
    #   i, j: indices of the particle size classes for particles i and j
    # returns: collision probability between particles i and j (m^3/s) 
    return  G * collision_matrix[][i,j]
end

function g1(n::Vector{<:Float64}, k::Int, G::Real)
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

function l1(n::Vector{<:Float64}, k::Int, G::Real, N::Int)
# need to fix mass balance with aggregation / loss terms here! 
    # Loss due to collisions for class k
    l1_ = 0 
    for i in 1:(N-1)  # note this is N in original flocmod equations 
        l1_ += α * A(G,i,k) * n[i] * n[k] 
    end 
    return l1_
end 

######################### Shear break-up #########################



function FDBS(k::Int, i::Int,  N::Int)
    # Break-up distribution function
    # What size classes result from a floc breaking up? Using binary assumption 
    # here: a floc breaks into two equal pieces.
    
    # SW adding this since no mass is exactly half of another mass
    # instead, we find the class w/ the closest mass to half the mass of floc k
    j1 = closest_half_mass[][i]
    if k == j1 
        FDBS_ = 2
    else 
        FDBS_ = 0

    end 
    return FDBS_
end 




function B(k::Int, G::Real)
    # Bi = beta * G**(beta_2) * D[i] * ((D[i] - Dp)/Dp)**(beta_3)
    Bi = D_ratio_4_B[][k] * G^(β_2)
    return Bi 
end 

function g2(n::Vector{<:Float64}, k::Int, G::Real,  N::Int)
    # Growth due to shear break-up for class k
    g2_ = 0
    if k == N
        return g2_
    end

    for i in (k+1):N
        g2_ += FDBS(k,i,N) * B(i, G) * n[i]
    end
    return g2_
end


function l2( n::Vector{<:Float64}, k::Int, G::Real)
    # Loss due to shear break-up for class k
    l2_ = B(k, G) * n[k]
    return l2_
end


function flocmod_mass_redistribute(n::Vector{<:Float64}, N::Int)
    # Redistribute negative masses in NN toward positive ones,
    # setting negatives to zero. converted to python from original fortran code 
    # by SW on 1/22/2025 with help from chatgpt. 

    # Parameters
    # ----------
    # n : Number of particles per floc size class 
    # N : Number of particle size classes

    # Returns
    # -------
    # n : modified (no negatives) number of particles per floc size class 



    # Temporary copy
    NNtmp = copy(n)

    # Identify negative and positive entries
    neg_mask = n .< 0.0
    pos_mask = n .> 0.0

    # Toal negative mass (weighted by f_mass)
    mneg = -sum(n[neg_mask] .* mass[][neg_mask])

    # Number of positive bins
    npos = sum(pos_mask)

    # Set negative entries to zero in temporary array
    NNtmp[neg_mask] .= 0.0

    # println("There is negative mass to redistribute: $(mneg)")
    if mneg > 0.0
        if npos == 0
            @error "CAUTION: all floc sizes have negative mass! "
            exit(1)
        total_positive = sum(NNtmp)
        end 

        # Redistribute negative mass linearly over positive classes
        for iv in 1:N
            if n[iv] > 0.0
                n[iv] = (n[iv]  - (mneg / total_positive)* n[iv]/ f_mass[iv])
            else
                n[iv] = 0.0
            end 
        end 
    end 
    return n 
end 


end




# N = 200  # number of size classes
# D = linspace(1e-5, 1e-4, N)  # particle diameters from 1 micron to 1 mm
# n = zeros(N) .+ 100 # linspace(1e5, 1e4, N)  # number concentration in each size class (particles/m^3)


# calculate once since diameters don't change
# collision_matrix = zeros(N,N)
# for i in 1:N
#     for j in 1:N
#         collision_matrix[i,j] = 1/6 * (D[i] + D[j])^3
#     end
# end 

# closest_half_mass = zeros(N)
# for i in 1:N
#     closest = mass[i] - (2*mass)   # SW adding this since no mass is exactly half of another mass
#                                    # instead, we find the class w/ the closest mass to half the mass of floc k
#     closest_half_mass[i] = argmin(abs(closest))
# end 

# Calculate once since diameters don't change
# D_ratio_4_B = zeros(N)
# for i in 1:N                  
#     D_ratio_4_B[i] = D[i] * β * abs((D[i] - Dp)/Dp)^β_3
# end 


