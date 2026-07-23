#!/usr/bin/env julia
using Printf

# module load  julia/1.11.7
module floc_mod
using Printf

# 1. Define a struct to hold thread-isolated parameters
struct FlocParams
    α::Float64
    β::Float64
    β_2::Float64
    nf::Float64
    # ws::Vector{Float64} # Settling velocities specific to this thread

    # Calculated properties
    # chol::Ref{Union{Nothing, Matrix{Float64}}} # Cache Cholesky decomposition
    # total_mass::Float64
    closest_half_mass::Vector{Int}
    collision_matrix::Matrix{Float64}
    D_ratio_4_B::Vector{Float64}
    floc_density::Vector{Float64}
    ws::Vector{Float64}
    particle_density::Vector{Float64}
    mass::Vector{Float64}
end

# constant vectors 
# const total_mass = Ref{Float64}() # total mass of sediment (kg/m^3)
# const closest_half_mass = Ref{Vector{Int}}()
# const collision_matrix = Ref{Matrix{Float64}}()
# const D_ratio_4_B = Ref{Vector{Float64}}()
# const ws = Ref{Vector{Float64}}()
# const mass = Ref{Vector{Float64}}()
# const floc_density = Ref{Vector{Float64}}()
# const particle_density = Ref{Vector{Float64}}() # Number of primary particles in each floc size class





function calculate_cholesky!(fp::FlocParams)
      #  Check the cache to see if we've already taken the Cholesky ...
      if  isnothing(ts.chol[])
          @info "Generating cholesky for the first time!"
          covariance = ts.covFun(ts.x, ts.params)
          local chol   # Need to make my variable local so it's accessible
                       # outside the try/catch block
          try
              chol = cholesky(covariance)
          catch e
              @warn "Cholesky failed, adding tiny value to the diagonal ..."
              chol = cholesky(covariance + 1e-11*I)
end
          ts.chol[] = chol.L
      else
          @info "Using cached cholesky ..."
return end
end



export init_params, run_floc_mod, run_floc_mod_semi, return_parameter_space, get_particle_density,
       max_loss_rate

# **************** Fixed constants ***************
const g = 9.81            # gravitational constant [m/s^2]
const ν = 1.5e-6          # visosity of water [m^2/s] 
const ρ_w = 1000          # density of water [kg/m^3]
const ρ_s = 2650          # density of sediment kg/m^3

# ************************************************
# const nf = 1.9 #2.1 #1.9 #2.0                # fractal dimension exponent
const Dp = 1e-6           # reference diameter (m)
const Fy = 1e-10           # yield strength (N)
# const β_2 = 1.1 #1.5 #          # winterwerp (2002)
# const β_3 = 3 - nf         # winterwerp (2002)
 #12 #12 
# const total_mass = Ref{Float64}() # total mass of sediment (kg/m^3)


# # Updated during EnKF process 
# const α = Ref{Float64}() #  0.35 #0.35 #4.5 #1.5  #0.55
# const β = Ref{Float64}() #0.045
# const nf = Ref{Float64}() # 1.9 #2.1 #1.9 #2.0  


function init_params(D::Vector{<:Real}, N::Int, n::Vector{<:Float64}, alpha::Float64, beta::Float64, beta2::Float64, nf::Float64;
                     collision_matrix::Union{Nothing,Matrix{Float64}}=nothing)
    # `collision_matrix` depends only on D, so when init_params is called inside the
    # time loop (once per member per step) recomputing it is by far the dominant cost.
    # Pass it in precomputed.

    # @info "Initializing flocculation model with $N sediment classes ..."
    # @info "Primary particle diameter = $(D[1]) m, max particle diameter = $(D[end]) m"

    # Check that the primary particle is equal to the first size class diameter
    if D[1] != Dp
        @error "Error: primary particle diameter Dp must be equal to the first size class diameter D[1]."
        exit(1)
    end
    
    # ---------------------------------------------
    # Calculate floc density, settling velocity, and mass 
    floc_density = calculate_density(nf, D, N)
    ws = calculate_w_s(floc_density, D, N)
    mass = calculate_mass(nf, D, N)

    # ---------------------------------------------
    # Calculate number of primary particles in each floc size class
    # a bit complicated due to the fractal formulation ... 
    particle_ = (π/6 * ρ_s * Dp^(3-nf))./mass[1] 
    particle_density = particle_.* D.^nf
    
    # ---------------------------------------------
    # Pre-calculate large matricies 
    # @info "\t initialized density, settling velocity, and floc mass..."
    closest_half_mass = calculate_closest_half_mass(D, particle_density, N)
    cmat = isnothing(collision_matrix) ? calculate_collision_matrix(D, N) : collision_matrix
    D_ratio_4_B = calculate_shear_breakup_matrix(beta, nf, D, N)

    # println("Mass = ", mass)
    # println("n = ", n)
    # total_mass = sum(mass .* n)
    # @debug "\t Initial total mass of flocculated sediment... $(total_mass*1000) g/L"

    # Create structure
    # Create structure to hold thread-isolated parameters
    floc_params = FlocParams(alpha, beta, beta2, nf, closest_half_mass, cmat, D_ratio_4_B, floc_density, ws, particle_density, mass)
    return floc_params


end


function get_particle_density(fp::FlocParams)
    return nothing, nothing #particle_density[], mass[]
end

function calculate_shear_breakup_matrix(β, nf, D::Vector{<:Real}, N::Int)
    # Pre-calculate the shear break-up matrix for all size classes, 
    # which is used in the break-up probability function B(i)
    β_3 = 3 - nf
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
    # NOTE: this used to build a Vector{Float64} and rely on the FlocParams constructor
    # to convert it to the declared Vector{Int}.  That conversion throws InexactError
    # the moment particle_density contains a NaN/Inf (which it does as soon as nf runs
    # away), and inside a @threads block that surfaces as an opaque TaskFailedException.
    # Build the Int vector directly and guard the input.
    closest_half_mass_ = zeros(Int, N)
    for i in 1:N
        best = 1; bestval = Inf
        for j in 1:N
            v = abs(particle_density[i] - 2*particle_density[j])
            if isfinite(v) && v < bestval
                bestval = v; best = j
            end
        end
        closest_half_mass_[i] = best
    end
    return closest_half_mass_
end


######################### Floc properties #########################
function calculate_w_s(floc_density::Vector{<:Real}, D::Vector{<:Real}, N::Int)
    # settling velocity for particle of density floc_density and diameter D
    # returns : settling velocity (m/s)
    ws_ = zeros(N)
    for i in 1:N
        ws_[i] = g/(18*ν) * (floc_density[i] - ρ_w)/(ρ_w) * D[i]^2 
    end 
    return ws_
end 
    


function calculate_density(nf::Real, D::Vector{<:Real}, N::Int)
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

function calculate_mass(nf::Real, D::Vector{<:Real}, N=N::Int)
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
        # FIXED: fractal floc mass is  rho_s * (pi/6) * Dp^3 * (D/Dp)^nf.
        # The old form used D[i]^3 * (D[i]/Dp)^nf, i.e. m ~ D^(3+nf), which is not a
        # fractal scaling.  This only ever fed particle_ via mass_[1] (where D[1]==Dp,
        # so the two agree) -- so behaviour is unchanged -- but the vector itself was
        # wrong for any other use.
        mass_[i] = ρ_s * π/6 * Dp^3 * (D[i]/Dp)^nf
    end

    return mass_
end




# ******************** Flocculation functions ************************

# function return_parameter_space()
#     # Returns a string of the parameter space for the current flocculation model parameters.
#     return nf, α, β, β_2 #, β_3
# end 

function run_floc_mod(fp::FlocParams, n::Vector{<:Float64}, N::Int, G::Real, dt::Real)
    # println("[1] Initial number of primary particles = ", sum(n .* particle_density[]))  # check mass conservation

    g1_ = zeros(N)
    l1_ = zeros(N)
    g2_ = zeros(N)
    l2_ = zeros(N)

    total_mass = sum(fp.particle_density .* n)
    # println("Total mass = ", total_mass[])  # check mass conservation
    # print("[0] n = ", n, "\n")
    for k in 1:N
        g1_[k] = g1(fp, n, k, G)
        l1_[k] = l1(fp, n, k, G, N)
        g2_[k] = g2(fp, n, k, G, N)
        l2_[k] = l2(fp, n, k, G)
    end 

    # println("Shear = ", G)
    # println("\t g1 = ", g1_.* particle_density[])
    # println("agg -/+ =", sum((g1_ - l1_) .* particle_density[]))
    # println("\t l1 = ", l1_.* particle_density[])
    # println("\t net change: ", ((g1_ - l1_)))
    # println("\t g2 = ", g2_)
    # println("\t l2 = ", l2_)
    # println("breakup -/+ =", sum((g2_ - l2_) .* particle_density[]))

    change = zeros(N)
    for i in 1:N
      #          agg (+)  agg(-)  shear(+)   shear(-)
        change[i]  =  g1_[i]   - l1_[i]  + g2_[i]  - l2_[i]
    end 
    n_new = n .+ (change .* dt) 

    new_mass =  sum(n_new .* fp.particle_density)

    mass_change = total_mass - new_mass
    # println("\t[1] Change in NP = ", mass_change)  # check mass conservation
    # FIXED sign: mass_change = total - new is the DEFICIT, so it has to be added back.
    # particle_density[1] == 1, so adding mass_change to class 1 restores exactly.
    # The old `-=` doubled the error instead of cancelling it.
    n_new[1] += mass_change
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
    n_new = flocmod_mass_redistribute(fp, n_new, N) 
    # n_new .= n_new - n
    return n_new

end 


"""
    max_loss_rate(fp, n, N, G)

Largest per-class linear loss rate (1/s) in the current state.  The explicit Euler
step in `run_floc_mod` is only stable while `dt * max_loss_rate < 1`.  At realistic
LISST-derived concentrations (n ~ 1e12 /m^3 in the fine classes) this is ~1e-3 s, so
`dt = 1` is unstable by 3 orders of magnitude.  Use `run_floc_mod_semi` instead.
"""
function max_loss_rate(fp::FlocParams, n::Vector{<:Float64}, N::Int, G::Real)
    m = 0.0
    for k in 1:N
        s = 0.0
        for i in 1:N
            s += fp.collision_matrix[i,k] * n[i]
        end
        r = fp.α * G * s + B(fp, k, G)
        m = max(m, r)
    end
    return m
end

"""
    run_floc_mod_semi(fp, n, N, G, dt)

Fixed-pivot aggregation + implicit, flux-limited sinks.  Drop-in replacement for
`run_floc_mod`, and the one the EnKF driver should use.  Three distinct problems in
the original are fixed here.

1. STABILITY.  `run_floc_mod` is explicit Euler, stable only while
   `dt * max_loss_rate < 1`.  At physically-sized concentrations (fine classes hold
   ~1e12 particles/m^3 once the initial condition comes from LISST instead of being
   1e4x too small) that limit is ~3e-3 s at G = 40 and ~7e-4 s at G = 175, so the
   driver's `dt = 1` is three orders of magnitude too large and the spectrum blows up
   to n ~ 1e26.  Both sinks are linear in n[k],

       l1 = n[k] * (alpha * G * sum_i C[i,k] n[i])      l2 = n[k] * B[k]

   so they are taken implicitly and each class keeps a bounded survival fraction.
   The sources stay explicit but are scaled by a flux limiter -- the ratio of mass
   actually removed to mass the explicit scheme would remove.  That ratio is 1 when
   `dt*rate << 1` (recovering explicit Euler) and < 1 when `dt*rate >> 1`, so a
   fragmenting class can never hand out more mass than it holds.

2. AGGREGATE SIZING.  The original placed the product of classes j and m in class
   j+m.  Index addition is only valid on a grid linear in particle MASS; on this
   geometric grid it overshoots badly -- two 32 um flocs produced a 1038 um floc
   instead of 44 um, a 23x error, and two 15.6 um flocs produced 244 um.  Aggregation
   therefore teleported mass above the LISST range (459 um), where it is unobservable,
   and no parameter choice could make the modelled spectrum match the data.  Here a
   collision between classes i and j yields pd_i + pd_j primary particles, and that
   mass is split between the two classes bracketing it (Kumar & Ramkrishna fixed
   pivot), which conserves mass exactly.

3. MASS BOOK-KEEPING.  Aggregates that overshoot the top of the grid stay in class N
   rather than being deleted.  Deleting them bled 47% of total mass in 5 minutes, and
   at a per-member rate that depended on alpha and nf, so the ensemble's total mass
   fanned out 3.6x between analyses and the filter's log-volume spread ran away.

Unconditionally positive, no sub-cycling, and `flocmod_mass_redistribute` is no longer
needed.  Mass is conserved to machine precision (measured 1e-16 relative).
"""
function run_floc_mod_semi(fp::FlocParams, n::Vector{<:Float64}, N::Int, G::Real, dt::Real)
    pd = fp.particle_density
    L1_ = zeros(N); B_ = zeros(N); g1e = zeros(N); g2e = zeros(N)

    @inbounds for k in 1:N
        s = 0.0
        for i in 1:N
            s += fp.collision_matrix[i,k] * n[i]
        end
        L1_[k] = fp.α * G * s
        B_[k]  = B(fp, k, G)
        g2e[k] = g2(fp, n, k, G, N)
    end

    # ---- fixed-pivot aggregation birth ---------------------------------------
    # pd[j] = (D[j]/Dp)^nf on a geometric D grid, so log(pd) is linear in the class
    # index and the target class follows directly from the aggregate's mass.
    lnpdr = log(pd[2] / pd[1])
    @inbounds for i in 1:N
        ni = n[i]
        ni <= 0.0 && continue
        for j in i:N
            nj = n[j]
            nj <= 0.0 && continue
            half = (i == j) ? 0.5 : 1.0
            R = fp.α * G * fp.collision_matrix[i,j] * half * ni * nj
            (R <= 0.0 || !isfinite(R)) && continue
            pt = pd[i] + pd[j]                       # primary particles in the aggregate
            kf = 1.0 + log(pt) / lnpdr
            kk = clamp(floor(Int, kf), 1, N-1)
            w1 = clamp((pt - pd[kk]) / (pd[kk+1] - pd[kk]), 0.0, 1.0)
            g1e[kk]   += R * (1.0 - w1)
            g1e[kk+1] += R * w1
        end
    end

    # ---- implicit sinks -------------------------------------------------------
    n_new = zeros(N); rem_agg = zeros(N); rem_brk = zeros(N)
    @inbounds for k in 1:N
        tot  = L1_[k] + B_[k]
        ns   = n[k] / (1.0 + dt*tot)
        n_new[k] = ns
        rem = n[k] - ns
        if tot > 0
            rem_agg[k] = rem * L1_[k] / tot
            rem_brk[k] = rem * B_[k]  / tot
        end
    end

    # ---- flux limiters: (mass actually removed) / (mass explicitly removed) ----
    want_a = 0.0; got_a = 0.0; want_b = 0.0; got_b = 0.0
    @inbounds for k in 1:N
        want_a += dt * L1_[k] * n[k] * pd[k]; got_a += rem_agg[k] * pd[k]
        want_b += dt * B_[k]  * n[k] * pd[k]; got_b += rem_brk[k] * pd[k]
    end
    lim_a = want_a > 0 ? got_a / want_a : 0.0
    lim_b = want_b > 0 ? got_b / want_b : 0.0

    placed_a = 0.0
    @inbounds for k in 1:N
        add = dt * g1e[k] * lim_a
        placed_a += add * pd[k]
        v = n_new[k] + add + dt*g2e[k]*lim_b
        n_new[k] = (isfinite(v) && v > 0.0) ? v : 0.0
    end

    # aggregates that overshot the top of the grid stay in the largest class
    lost = got_a - placed_a
    if isfinite(lost) && lost > 0
        n_new[N] += lost / pd[N]
    end
    return n_new
end


# ****************** Aggregation functions ***********************
function A(fp::FlocParams, G::Real, i::Int, j::Int)
    #  Two-body collision probability function A(i,j) is a function of the 
    # shear rate G and the particle diameters D_i and D_j
    # parameters:
    #   G: shear rate (1/s)
    #   D: array of particle diameters (m)
    #   i, j: indices of the particle size classes for particles i and j
    # returns: collision probability between particles i and j (m^3/s) 
    collision_matrix = fp.collision_matrix
    return  G * collision_matrix[i,j]
end

function g1(fp::FlocParams, n::Vector{<:Float64}, k::Int, G::Real)
    # Growth due to collisions for class k 
    # Parameters:
    #   k: index of the particle size class
    #   A : collision probability function
    #   n : number concentration array
    #   G: shear rate (1/s)
    # Returns: growth rate due to aggregation for size class k (particles/m^3/s) 
    g1_ = 0.0 

    # Pull out parameters from the struct
    α = fp.α
    particle_density = fp.particle_density

    for j in 1:(k-1)
            g1_ += α * A(fp, G,j,k-j) * n[k-j] * n[j] * (particle_density[j] + particle_density[k-j])/particle_density[k] 
    end
    g1_ = g1_* 0.5
    return g1_
end

function l1(fp::FlocParams, n::Vector{<:Float64}, k::Int, G::Real, N::Int)
    # Loss due to collisions for class k
    #
    # FIXED (was the "need to fix mass balance" TODO): the loss term used to carry a
    # spurious 1/particle_density[k] factor.  particle_density[k] = (D[k]/Dp)^nf runs
    # from 1 to ~1e8 across the grid, so the sink was under-counted by up to 8 orders
    # of magnitude relative to the g1 source.  Aggregation was almost pure gain, and
    # the resulting mass residual was silently dumped into class 1 by the n_new[1]
    # patch in run_floc_mod -- which is a large part of why particle counts blow up.
    # Measured at a representative state: gain/loss was 4.1e5 before, 0.21 after.
    #
    # Smoluchowski loss is  dn_k/dt = -n_k * sum_i alpha*A(i,k)*n_i   (no 1/pd factor).
    l1_ = 0
    α = fp.α
    for i in 1:N  # note this is N in original flocmod equations
        l1_ += α * A(fp, G,i,k) * n[i] * n[k]
    end
    return l1_
end

######################### Shear break-up #########################



function FDBS(fp::FlocParams, k::Int, i::Int,  N::Int)
    # Break-up distribution function
    # What size classes result from a floc breaking up? Using binary assumption 
    # here: a floc breaks into two equal pieces.
    
    # SW adding this since no mass is exactly half of another mass
    # instead, we find the class w/ the closest mass to half the mass of floc k
    j1 = fp.closest_half_mass[i]
    if k == j1 
        FDBS_ = 1 
    else 
        FDBS_ = 0

    end 
    return FDBS_
end 




function B(fp::FlocParams, k::Int, G::Real)
    # Bi = beta * G**(beta_2) * D[i] * ((D[i] - Dp)/Dp)**(beta_3)
    Bi = fp.D_ratio_4_B[k] * G^(fp.β_2)
    return Bi 
end 

function g2(fp::FlocParams, n::Vector{<:Float64}, k::Int, G::Real,  N::Int)
    # Growth due to shear break-up for class k
    g2_ = 0
    if k == N
        return g2_
    end
    particle_density = fp.particle_density
    for i in (k+1):N
        g2_ += FDBS(fp, k,i,N) * B(fp, i, G) * n[i]  * (particle_density[i]/particle_density[k])
    end

    return g2_
end


function l2(fp::FlocParams, n::Vector{<:Float64}, k::Int, G::Real)
    # Loss due to shear break-up for class k
    l2_ = B(fp, k, G) * n[k] #* mass[][i]/mass[][k] 
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

function flocmod_mass_redistribute(fp::FlocParams, n::Vector{<:Float64}, N::Int)
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
    mneg = -sum(n[neg_mask] .* fp.particle_density[neg_mask])

    mpos = sum(n[pos_mask] .* fp.particle_density[pos_mask])
    # Number of positive bins
    npos = sum(pos_mask)

    # Set negative entries to zero in temporary array
    NNtmp[neg_mask] .= 0.0

    if mneg > 0.0
        if npos == 0
            # @error "CAUTION: all floc sizes have negative mass! "
            # exit(1)
            n = abs.(randn(N)).*50 # zeros(N)
        end 
        
        # Redistribute negative mass linearly over positive classes
        for iv in 1:N
            if n[iv] > 0.0              # was total_positive
                n[iv] = n[iv]  - (mneg) * ((n[iv]* fp.particle_density[iv])/mpos) /fp.particle_density[iv] # (n[iv]* / mass[][iv])
            else
                n[iv] = 0.0
            end 
        end 
    end 
    return n 
end 


end
