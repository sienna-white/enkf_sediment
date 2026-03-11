using CUDA
CUDA.set_runtime_version!
using Random, Statistics, Base.Threads
#!/usr/bin/env julia
using Printf

# module load  julia/1.11.7
module floc_mod
using Printf
export init_params!, run_floc_mod 

# **************** Fixed constants ***************
const g = 9.81            # gravitational constant [m/s^2]
const ν = 1.5e-6 # temp1.3e-6          # visosity of water [m^2/s] 
const ρ_w = 1000          # density of water [kg/m^3]
const ρ_s = 2650          # density of sediment kg/m^3
const α = 0.55 #1.5 

# ************************************************
const nf = 1.9 #2.0                # fractal dimension exponent
const Dp = 4 * 1e-6           # reference diameter (m)
const Fy = 1e-10              # yield strength (N)
const β_2 = 1.5            # winterwerp (2002)
const β_3 = 3 - nf         # winterwerp (2002)
const β = 0.12 #0.1
const total_mass = Ref{Float64}() # total mass of sediment (kg/m^3)
const closest_half_mass = Ref{Vector{Int}}()
const collision_matrix = Ref{Matrix{Float64}}()
const D_ratio_4_B = Ref{Vector{Float64}}()
const ws = Ref{Vector{Float64}}()
const mass = Ref{Vector{Float64}}()
const floc_density = Ref{Vector{Float64}}()



mass = CuArray(mass)



function my_kernel( n::CuDeviceVector{<:Real},      # # of flocs in each size class 
                    mass::CuDeviceVector{<:Real},   # Mass of each sediment class
                    N::Int,            # Number of sediment classes
                    G::Float64,        # Shear rate [1/s]
                    dt::Int)           # Time step 

    # Get thread index on the GPU
    i = threadIdx().x;  # What thread am I within the block?
    j = blockIdx().x;   # What block am I in?

    # Calculate the 1D index of what sediment class this thread should process
    k = (j-1)*blockDim().x + i

    # Exit if we're outside the # sediment classes 
    if k > N
        return 

    # *****************************************************************************
    # [1] Calculate growth due to aggregation for sediment class 'k'
    g1_ = 0.0 
    for j in 1:k
        for i in 1:k
            if i + j == k
                #  Two-body collision probability function A(i,j) is a function of the 
                # shear rate G and the particle diameters D_i and D_j
                A = G * collision_matrix[i,j]
                # Growth
                g1_ += α * A* n[i] * n[j] * (mass[i] + mass[j])/mass[k] 
            end
        end
    end
    g1_ = g1_ * 0.5  

    # *****************************************************************************
    # [2] Calculate loss due to aggregation for sediment class 'k'
    l1_ = 0 
    for i in 1:N   
        l1_ += α * A(G,i,k) * n[i] * n[k] * (mass[i]/mass[k])
    end 
    
    # *****************************************************************************
    # [3] Calculate growth due to shear collision for sediment class 'k' 
    g2_ = 0
    if k < N
        for i in (k+1):N
            g2_ += FDBS(k,i,N) * B(i, G) * n[i]  * mass[i]/mass[k] 
        end
    end 

    # *****************************************************************************
    # [4] Calculate loss due to shear collision for sediment class 'k' 
    l2_ = B(k, G) * n[k] * mass[i]/mass[k] 
 
       #   agg (+)  agg(-)  shear(+)   shear(-)
    change_ =  g1_   - l1_    + g2_  - l_2 
    # @inbounds change[k] = change_

    @inbounds n[k] = n + (change_ * dt) 
    return nothing
end 



    n_new = n .+ (change .* dt) 
    new_mass =  sum(n_new .* mass) 
    mass_change = total_mass - new_mass
    # println("[1] Mass change = ", mass_change)  # check mass conservation

    for iv in 1:N
        if n_new[iv] > 0.0              # was total_positive
            n_new[iv] = n_new[iv]  + mass_change/mass[iv] * ((n_new[iv]* mass[iv])/new_mass) 
        end 
    end 
    # new_mass_change = total_mass - sum(n_new .* mass)
    # println("[2] Mass change = ", new_mass_change)  # check mass conservation

    n_new = flocmod_mass_redistribute(n_new, N)
    # new_mass_change = total_mass - sum(n_new .* mass)
    # println("[3] Mass change = ", new_mass_change)  # check mass conservation
    
    return n_new

end 

function my_kernel(xvec::CuDeviceVector{<:Real}, yvec::CuDeviceVector{<:Real},
                grid_x::CuDeviceVector{<:Real}, grid_y::CuDeviceVector{<:Real},
                    h::Float64, kde_values::CuDeviceMatrix{<:Real}, N::Int)
    i = threadIdx().x;  # What thread am I within the block?
    j = blockIdx().x;   # What block am I in?
    # Calculate the 1D index of what data point the thread should process
    ind = (j-1)*blockDim().x + i
    # Convert the 1D index into a 2D index
    ii = (ind% N) + 1
    jj = div(ind, N) + 1
    # # Ensure we do not access out-of-bounds memory
9

    if Int(ii) > N || Int(jj) > N
    return
end
# Grab our x, y values
x = @inbounds grid_x[ii]
y = @inbounds grid_y[jj]
# Build our KDE sum
kde_sum = 0
for k in 1:N
    dx = (x - xvec[k]) / h
    dy = (y - yvec[k]) / h
    kde_sum += exp(-0.5 * (dx^2 + dy^2))  # Gaussian kernel
end
kde_values[ii, jj] = kde_sum ./ (N * 2 * h)  # Normalize
return nothing
end


N = length(utm_x)
println("Processing KDE for $N data points")
# Copy data to GPU
x_val = CuArray(utm_x)
y_val = CuArray(utm_y)
# Create gridded x & y values
grid_x = CuArray(grid_x1)
grid_y = CuArray(grid_y1)
kde_values = CUDA.zeros(N,N)
# Set up the kernel
threads_per_block = 1024
blocks = ceil(Int, N^2 / threads_per_block)
# Set bandwidth
h = 2e5
@info "There are $threads_per_block threads per block and $blocks blocks"


@info "This is a total of $(threads_per_block * blocks) threads for our $N x $N grid"
@cuda threads=threads_per_block blocks=blocks my_kernel(n, mass, N, G, dt)
CUDA.synchronize()
kde_cpu = Array(kde_values)


function run_floc_mod(n::Vector{<:Float64}, N::Int, G::Real, dt::Int)


    change = CUDA.zeros(N)
    
    @info "This is a total of $(threads_per_block * blocks) threads for our $N x $N grid"

    @cuda threads=threads_per_block blocks=blocks my_kernel(n, mass, N, G, dt) 
    CUDA.synchronize()
    
    new_mass =  sum(n .* mass) 
    mass_change = total_mass - new_mass

    for iv in 1:N
        if n_new[iv] > 0.0              # was total_positive
            n_new[iv] = n_new[iv]  + mass_change/mass[iv] * ((n_new[iv]* mass[iv])/new_mass) 
        end 
    end 
    n_new = flocmod_mass_redistribute(n_new, N)

    return n_new

end 




function init_params!(D::Vector{<:Real}, N::Int, n::Vector{<:Float64})

    @info "Initializing flocculation model with $N sediment classes ..."

    # Calculate floc density, settling velocity, and mass 
    floc_density[] = calculate_density(D, N)
    ws[] = calculate_w_s(D, N)
    mass = calculate_mass(D, N)
    
    @info "\t initialized density, settling velocity, and floc mass..."
    closest_half_mass0 = zeros(N)
    collision_matrix0 = zeros(N,N)
    D_ratio_4_B0 = zeros(N)

    for i in 1:N
        closest = @. mass[i] - (2*mass)   # SW adding this since no mass is exactly half of another mass
                                   # instead, we find the class w/ the closest mass to half the mass of floc k
        closest_half_mass0[i] = argmin(broadcast(abs, closest))

        D_ratio_4_B0[i] = D[i] * β * abs((D[i] - Dp)/Dp)^β_3 # SW adding abs to avoid imaginary numbers

        for j in 1:N
            collision_matrix0[i,j] = 1/6 * (D[i] + D[j])^3
        end
    end 

    closest_half_mass = closest_half_mass0
    collision_matrix[] = collision_matrix0
    D_ratio_4_B[] = D_ratio_4_B0
    @info "\t initialized collision matrix, half-mass indices, and D ratio for break-up ..."
    total_mass = sum(mass .* n)
    @info "\t Initial total mass of sediment is $(total_mass) kg/m^3"
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

function FDBS(k::Int, i::Int,  N::Int)
    # Break-up distribution function
    # What size classes result from a floc breaking up? Using binary assumption 
    # here: a floc breaks into two equal pieces.
    
    # SW adding this since no mass is exactly half of another mass
    # instead, we find the class w/ the closest mass to half the mass of floc k
    j1 = closest_half_mass[i]
    if k == j1 #div(i,2) 
        FDBS_ = (mass[i] / mass[k])  
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
    mneg = -sum(n[neg_mask] .* mass[neg_mask])

    mpos = sum(n[pos_mask] .* mass[pos_mask])
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
                n[iv] = n[iv]  - (mneg) * ((n[iv]* mass[iv])/mpos) /mass[iv] # (n[iv]* / mass[iv])
            else
                n[iv] = 0.0
            end 
        end 
    end 
    return n 
end 


end


