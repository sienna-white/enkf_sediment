#!/usr/bin/env julia
using Printf

# module load  julia/1.11.7
module floc_mod
using Printf
export init_params!, run_floc_mod, return_parameter_space, get_particle_density

# **************** Fixed constants ***************
const g = 9.81            # gravitational constant [m/s^2]
const ν = 1.5e-6 # temp1.3e-6          # visosity of water [m^2/s] 
const ρ_w = 1000          # density of water [kg/m^3]
const ρ_s = 2650          # density of sediment kg/m^3
const α = 0.35 #4.5 #1.5  #0.55


# ************************************************
const nf = 2.3 #2.1 #1.9 #2.0                # fractal dimension exponent
const Dp = 10e-6           # reference diameter (m)
const Fy = 1e-10           # yield strength (N)
const β_2 = 1.5 #          # winterwerp (2002)
const β_3 = 3 - nf         # winterwerp (2002)
const β = 0.08 #12 

# constant vectors 
const total_mass = Ref{Float64}() # total mass of sediment (kg/m^3)
const closest_half_mass = Ref{Vector{Int}}()
const collision_matrix = Ref{Matrix{Float64}}()
const D_ratio_4_B = Ref{Vector{Float64}}()
const ws = Ref{Vector{Float64}}()
const mass = Ref{Vector{Float64}}()
const floc_density = Ref{Vector{Float64}}()
const particle_density = Ref{Vector{Float64}}() # Number of primary particles in each floc size class


function init_params!(D::Vector{<:Real}, N::Int, n::Vector{<:Float64})

    @info "Initializing flocculation model with $N sediment classes ..."
    @info "Primary particle diameter = $(D[1]) m, max particle diameter = $(D[end]) m"

    # Check that the primary particle is equal to the first size class diameter
    if D[1] != Dp
        @error "Error: primary particle diameter Dp must be equal to the first size class diameter D[1]."
        exit(1)
    end

     # ---------------------------------------------
    # Calculate floc density, settling velocity, and mass 
    floc_density[] = calculate_density(D, N)
    ws[] = calculate_w_s(D, N)
    mass[] = calculate_mass(D, N)

    # ---------------------------------------------
    # Calculate number of primary particles in each floc size class
    # a bit complicated due to the fractal formulation ... 
    particle_ = (π/6 * ρ_s * Dp^(3-nf))./mass[][1] 
    particle_density[] = particle_.* D.^nf
    

    # ---------------------------------------------
    # Pre-calculate large matricies 
    @info "\t initialized density, settling velocity, and floc mass..."
    closest_half_mass[] = calculate_closest_half_mass(D, particle_density[], N)
    collision_matrix[] = calculate_collision_matrix(D, N)
    D_ratio_4_B[] = calculate_shear_breakup_matrix(D, N) 

    @info "\t Initialized collision matrix, half-mass indices, and D ratio for break-up ..."
    total_mass[] = sum(mass[] .* n)
    @info "\t Initial total mass of flocculated sediment... $(total_mass[]*1000) g/L"

    return nothing


end


function get_particle_density()
    return particle_density[], mass[]
end

function calculate_shear_breakup_matrix(D::Vector{<:Real}, N::Int)
    # Pre-calculate the shear break-up matrix for all size classes, 
    # which is used in the break-up probability function B(i)

    D_ratio_4_B_ = zeros(N)
    for i in 1:N
        D_ratio_4_B_[i] = D[i] * β * abs((D[i] - Dp)/Dp)^β_3
    end 
    return D_ratio_4_B_
end

function calculate_collision_matrix(D::Vector{<:Real}, N::Int)
    # Pre-calculate the collision matrix for all size classes, 
    # which is used in the collision probability function A(i,j)

    collision_matrix_ = zeros(N,N)
    for i in 1:N
        for j in 1:N
            collision_matrix_[i,j] = 1/6 * (D[i] + D[j])^3
        end
    end 
    return collision_matrix_
end

function calculate_closest_half_mass(D::Vector{<:Real}, particle_density::Vector{<:Real}, N::Int)
    # Used for binary fragmentation: what particle is closest to half the particle count of each floc?
    closest_half_mass_ = zeros(N)
    for i in 1:N
        closest = @. particle_density[i] - (2*particle_density)  
        closest_half_mass_[i] = argmin(abs.(closest))
    end 
    return closest_half_mass_
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
        density_[i] = ρ_w + (ρ_s - ρ_w) * (D[i]/Dp)^(nf - 3)
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
    for i in 1:N # floc_density[][i]
        mass_[i] = ρ_s * π/6 * D[i]^3 * (D[i]/Dp)^nf 
    end 


    return mass_
end 




# ******************** Flocculation functions ************************

function return_parameter_space()
    # Returns a string of the parameter space for the current flocculation model parameters.
    return nf, α, β, β_2, β_3
end 

function run_floc_mod(n::Vector{<:Float64}, N::Int, G::Real, dt::Int)
    # println("[1] Initial number of primary particles = ", sum(n .* particle_density[]))  # check mass conservation

    g1_ = zeros(N)
    l1_ = zeros(N)
    g2_ = zeros(N)
    l2_ = zeros(N)

    total_mass = sum(particle_density[] .* n)
    # println("Total mass = ", total_mass[])  # check mass conservation

    for k in 1:N
        g1_[k] = g1(n, k, G)
        l1_[k] = l1(n, k, G, N)
        g2_[k] = g2(n, k, G, N)
        l2_[k] = l2(n, k, G)
    end 
    # println("\t g1 = ", g1_)
    # println("agg -/+ =", sum((g1_ - l1_) .* particle_density[]))
    # println("\t l1 = ", l1_)
    # println("\t net change: ", ((g1_ - l1_)))
    # println("\t g2 = ", g2_)
    # println("\t l2 = ", l2_)
    # println("breakup -/+ =", sum((g2_ - l2_) .* particle_density[]))

    change = zeros(N)
    for i in 1:N
      #          agg (+)  agg(-)  shear(+)   shear(-)
        change[i]  =  g1_[i]   - l1_[i]  + g2_[i]  - l2_[i]
        # println("\t i=$i n=$(n[i]) del=$(change[i]) // g1 = $(g1_[i]), l1 = $(l1_[i]), g2 = $(g2_[i]), l2 = $(l2_[i])")
    end 
    n_new = n .+ (change .* dt) 

    new_mass =  sum(n_new .* particle_density[]) 

    mass_change = total_mass - new_mass
    # println("\t[1] Change in NP = ", mass_change)  # check mass conservation
    n_new[1] -= mass_change
    # print("PARTICLE_DIST = ", (n_new .* particle_density[]), "\n")
    # for iv in 1:N
    #     # if n_new[iv] > 0.0              # was total_positive
    #     n_new[iv] = n_new[iv]  + (mass_change/N) * 1/particle_density[][iv] #* ((n_new[iv]* mass[][iv])/new_mass) 
    #     # end 
    # end 
    # new_mass_change = total_mass - sum(n_new .* particle_density[])
    # println("\t [2] Mass change = ", new_mass_change)  # check mass conservation

    # gamma = n - n_new
    # n_new = flocmod_mass_redistribute(n_new, N)

    # new_mass_change = total_mass - sum(n_new .* particle_density[])
    # println("[3] Mass change = ", new_mass_change)  # check mass conservation
    n_new = flocmod_mass_redistribute(n_new, N) 

    return n_new

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
    for j in 1:(k-1)
            g1_ += α * A(G,j,k-j) * n[k-j] * n[j] * (particle_density[][j] + particle_density[][k-j])/particle_density[][k] 
    end
    g1_ = g1_* 0.5
    return g1_
end

function l1(n::Vector{<:Float64}, k::Int, G::Real, N::Int)
# need to fix mass balance with aggregation / loss terms here! 
    # Loss due to collisions for class k
    l1_ = 0 
    for i in 1:N  # note this is N in original flocmod equations 
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
        FDBS_ = 1 
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
        g2_ += FDBS(k,i,N) * B(i, G) * n[i]  * (particle_density[][i]/particle_density[][k])
    end

    return g2_
end


function l2( n::Vector{<:Float64}, k::Int, G::Real)
    # Loss due to shear break-up for class k
    l2_ = B(k, G) * n[k] #* mass[][i]/mass[][k] 
    return l2_
end


### **** Collision induced break-up  ****
# function g3(n::Vector{<:Float64}, k::Int, G::Real,  N::Int)
#     g3_ = 0
#     for i in 1:N
#         for j in 1:N
#             g3_ += FDBC(i,j) * A(i,j) * n[i] * n[j]
#             end 
#         end 
#     end
# end 

# function l3(n::Vector{<:Float64}, k::Int, G::Real,  N::Int)
#     l3_ = 0
#     for i in 1:N
#         g3_ += FDBC(i,k) * A(i,k) * n[i] * n[k]      
#     end 
# end 


# function FDBC(i::Int, j::Int, D::Vector{<:Real})
#     # Collision-induced break-up distribution function
#     Fp = 0.1 # Depth of interparticle penetration (estimated to be 0.1)
#     top = 8 * mass[][i] * mass[][j] *  (G/2 * (D[i] + D[j]))^2
#     bottom = pi * Fp * D[i]^2 * (D[i] + D[j]) * (mass[][i] + mass[][j])
#     tau_collision = top / bottom 

#     Fy = 1e-10 
#     tau_i = Fy * ((rho_w - floc_density[][i])/rho_w)^(2/(3-nf))
#     tau_j = Fy * ((rho_w - floc_density[][j])/rho_w)^(2/(3-nf))

#     if tau_collision > tau_i
#         FDBC_ = 1.0 

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
    mneg = -sum(n[neg_mask] .* particle_density[][neg_mask])

    mpos = sum(n[pos_mask] .* particle_density[][pos_mask])
    # Number of positive bins
    npos = sum(pos_mask)

    # Set negative entries to zero in temporary array
    NNtmp[neg_mask] .= 0.0

    if mneg > 0.0
        if npos == 0
            @error "CAUTION: all floc sizes have negative mass! "
            exit(1)
        end 
        
        # Redistribute negative mass linearly over positive classes
        for iv in 1:N
            if n[iv] > 0.0              # was total_positive
                n[iv] = n[iv]  - (mneg) * ((n[iv]* particle_density[][iv])/mpos) /particle_density[][iv] # (n[iv]* / mass[][iv])
            else
                n[iv] = 0.0
            end 
        end 
    end 
    return n 
end 


end
