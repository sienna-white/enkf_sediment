#!/usr/bin/env julia
#
#  Flocculation model + Ensemble Kalman Filter -- FIXED version of run_floc_model_enkf1.jl
#
#  module load julia/1.11.7
#
#  What changed and why is documented inline against tags [F1] ... [F10].
#  Set ENKF_DATA_DIR to point at the csv files (defaults to this script's directory).
#
using Printf
using DataStructures: OrderedDict
using NCDatasets
using CSV, DataFrames, DataInterpolations
using Random
using Statistics
using LinearAlgebra
using Distributions
using Base.Threads
using Compat # for older julia versoins 

include("hydro/calculate_physical_variables.jl")
include("hydro/advance_variables.jl")
include("hydro/forcings.jl")
include("hydro/output.jl")
include("floc_mod_enkf2.jl")
using .floc_mod

const DATA_DIR = get(ENV, "ENKF_DATA_DIR", @__DIR__)

println("Using $(Threads.nthreads()) threads for parallelization.\n")

#  Seed the RNG.  enkf1 left this unseeded, so no two runs were comparable -- which
#  matters a lot here, because the filter is only marginally stable and the outcome
#  varies substantially between seeds.  Fix it while tuning; sweep it when validating.
Random.seed!(parse(Int, get(ENV, "ENKF_SEED", "20240722")))

# ===========================================================================
#  Helpers
# ===========================================================================

"""Gaspari-Cohn (1999) compactly-supported 5th-order correlation function."""
function gaspari_cohn(z::Real)
    z = abs(z)
    if z <= 1
        return (((-0.25z + 0.5)z + 0.625)z - 5/3)z^2 + 1
    elseif z <= 2
        return ((((z/12 - 0.5)z + 0.625)z + 5/3)z - 5)z + 4 - 2/(3z)
    else
        return 0.0
    end
end

"""Log-midpoint edges for a geometric ladder of bin centres."""
function bin_edges_from_centres(c::Vector{Float64})
    lg = log.(c); n = length(c)
    e = zeros(n + 1)
    e[2:n]  .= exp.(0.5 .* (lg[1:n-1] .+ lg[2:n]))
    e[1]     = exp(lg[1] - 0.5*(lg[2] - lg[1]))
    e[n+1]   = exp(lg[n] + 0.5*(lg[n] - lg[n-1]))
    return e
end

# ===========================================================================


"""
    advance_member!(ssc, EID, D, Ns, CMAT, G, dt, alpha, beta, beta2, nf)

Advance one ensemble member in place.

This is deliberately a top-level function rather than the body of the `@threads`
loop.  Inlining it produced a genuine race: variables assigned inside a `@threads`
body can be boxed by the closure conversion and then shared across tasks, so members
silently corrupted each other's state.  The symptom was subtle -- the run completed
without error but the prior volume came out at 1.7e5 uL/L against observations of 25,
and the ensemble log-spread hit 3.0 instead of 1.6.  Running with `julia -t 1` gave
the correct 23.8 uL/L, which is the tell.  Keeping the per-member work behind a
function boundary guarantees every local really is task-local.
"""
function advance_member!(ssc::Matrix{Float64}, EID::Int, D::Vector{Float64}, Ns::Int,
                         CMAT::Matrix{Float64}, G::Real, dt::Real,
                         alpha::Float64, beta::Float64, beta2::Float64, nf::Float64)
    n = ssc[EID, :]                      # this is a copy, so it is task-local
    fp = init_params(D, Ns, n, alpha, beta, beta2, nf; collision_matrix = CMAT)
    n_new = run_floc_mod_semi(fp, n, Ns, G, dt)
    @inbounds for j in 1:Ns
        v = n_new[j]
        ssc[EID, j] = (isfinite(v) && v > 0.0) ? v : 0.0
    end
    return nothing
end

# ===========================================================================

function run_my_model(file_out_name::String)

    #********************** ENSEMBLE / TIME  ***************************
    N  = 65             # number of ensemble members
    dt = 1.0            # (seconds) outer time step -- now safe, see [F2]
    M  = parse(Int, get(ENV, "ENKF_M", string(3600*12)))   # number of steps
    isave = 30
    var2save = ["G", "nf", "alpha", "beta", "beta2", "msrc"]
    create_output_dict(M, isave, var2save, N)

    #*************** SEDIMENT SIZE CLASSES ****************************
    Ns = 50
    D  = collect(logrange(1e-6, 600e-6, Ns))      # m
    Volumes = (4/3)*pi*(D./2).^3                   # m^3 per floc
    VOLU = Volumes .* 1e6                          # uL/L per (particle/m^3)
    Dum  = D .* 1e6                                # um

    Nflux = 0
    Ns_aug = Ns + Nflux + 5   # 4 floc parameters + 1 mass-source rate
    add_sediment_to_output(Ns, isave, M, N)
    # depends only on D -- hoisted out of the time loop (was ~60% of runtime)
    CMAT = [(1/6)*(D[i] + D[j])^3 for i in 1:Ns, j in 1:Ns]

    #***************************************************************************
    #  [F1] PARAMETERS.
    #  alpha0 and beta0 were BOTH set to 0.0 in enkf1.  With alpha = beta = 0 the
    #  aggregation and break-up kernels vanish identically, so run_floc_mod returned
    #  its input unchanged -- the forward model was the identity map (verified: after
    #  200 steps max|n - n0| was exactly 0).  Nothing linked the parameters to the
    #  observations, so every parameter "update" the filter made was pure sampling
    #  noise.  Restored to the values that were commented out on those lines.
    #***************************************************************************
    alpha0 = 0.35      # aggregation efficiency
    beta0  = 0.15      # break-up coefficient
    nf0    = 2.5       # (kept for reference; nf now uses the transform below)
    beta20 = 1.5       # break-up shear exponent

    #  [F3] PARAMETER TRANSFORMS.
    #  enkf1 used  nf = nf0 * sin(Nfs)^2, which is periodic and many-to-one: Nfs and
    #  -Nfs give the same nf, and nf sweeps through 0 (a fractal dimension of zero is
    #  unphysical -- it should live in ~[1.4, 3]).  An EnKF can only move a parameter
    #  through its *linear correlation* with the observations, and a symmetric
    #  many-to-one map drives that correlation to zero, so nf was structurally
    #  unidentifiable.  Replaced with monotone, bounded transforms.
    #  Ranges matter for stability as well as physics.  alpha is a collision
    #  EFFICIENCY, so it belongs in (0,1] -- enkf1's alpha0*exp(x) reached 7.  beta_2
    #  is the shear exponent in B ~ beta*G^beta_2; with G up to 175 1/s in this record,
    #  letting beta_2 wander to 3 gives G^beta_2 ~ 5e6 and the (explicit) break-up
    #  source overwhelms any time step.  Winterwerp-type values are ~1-1.5.
    #  alpha0 = 0.35 is ~100x too large for this collision kernel.  With the
    #  aggregate sizing fixed, alpha = 0.35 drives d50 to ~1050 um within 5 minutes,
    #  while the LISST record has a median d50 of 78 um (range 34-393).  A direct sweep
    #  puts the working value at alpha ~ 3.5e-3 (d50 82-139 um).  Log transform, so the
    #  filter explores it multiplicatively.
    to_alpha(x) = 3.5e-3 * exp(clamp(x, -2.5, 2.5))         # ~3e-4 .. 4e-2
    to_beta(x)  = beta0 * exp(clamp(x, -2.0, 2.0))         # (0.020, 1.11)
    to_beta2(x) = 0.6 + 1.4 / (1 + exp(-0.6x))             # (0.6, 2.0)
    to_nf(x)    = 1.6 + (2.95 - 1.6) / (1 + exp(-x))       # (1.6, 2.95)

    #  [F13] MASS SOURCE / SINK.
    #  This is a 0-D box: run_floc_mod only REDISTRIBUTES sediment between size
    #  classes (and loses a little off the top of the grid).  It has no erosion, no
    #  deposition and no advection, so total suspended mass is fixed by the initial
    #  condition -- but the observed LISST load swings from 25 to 5850 uL/L over this
    #  record.  Asking the four floc parameters to explain that is genuinely
    #  ill-posed: no combination of alpha/beta/nf can create sediment.  Estimating one
    #  extra state variable, a log-mass source rate, gives the filter a legitimate
    #  handle on total load.  (This is what the commented-out `Flux` term in enkf1 was
    #  reaching for.)
    #
    #  OFF BY DEFAULT.  It works, but it is an integrating control -- the state is a
    #  RATE, so any error in it accumulates in total mass between analyses -- and left
    #  as an undamped random walk the filter runs it away (measured: model/observed
    #  volume reached 1.5e9 within three hours).  It needs a mean-reverting prior, so
    #  when enabled below the random walk is Ornstein-Uhlenbeck with timescale
    #  TAU_SRC rather than a pure random walk.  Turn it on only once you are happy
    #  with the rest of the filter, and watch `msrc` in the output.
    ESTIMATE_MSRC = false
    TAU_SRC = 1800.0                                       # s, mean-reversion timescale
    to_src(x)   = clamp(x, -5.0, 5.0) * 1e-4               # 1/s, log-mass source rate

    #***************************************************************************
    #   OBSERVATIONAL DATA
    #***************************************************************************
    @info "Reading observational data from $DATA_DIR ..."

    df = CSV.read(joinpath(DATA_DIR, "adcp_lateral_velocity.csv"), DataFrame)
    get_velocity = LinearInterpolation(df.lateral_velocity, df.seconds) #,
                                    #    extrapolation = ExtrapolationType.Linear)

    df = CSV.read(joinpath(DATA_DIR, "adv_shear_4_model.csv"), DataFrame)
    get_shear = LinearInterpolation(df.smoothed_shear, df.seconds) #,
                                    # extrapolation = ExtrapolationType.Linear)
    turbulent_shears = df[!, "smoothed_shear"]

    Ds_LISST = [  1.21,   1.6 ,   1.89,  2.23 ,  2.63,  3.11,  3.67,   4.33,   5.11 ,  6.03,
                  7.11,   8.39,   9.9,   11.7,   13.8,  16.3,  19.2,   22.7,   26.7,   31.6,
                  37.2,   43.9,   51.9,  61.2,   72.2,  85.2,  101.,   119.,   140.,   165.,
                  195.,   230.,   273.,  324.,   386.,  659.] # 459.
    N_lisst = length(Ds_LISST)

    dfL = CSV.read(joinpath(DATA_DIR, "lisst_data.csv"), DataFrame)
    dfR = CSV.read(joinpath(DATA_DIR, "lisst_variance.csv"), DataFrame)

    #***************************************************************************
    #  [F4] OBSERVATION OPERATOR.
    #  enkf1 built H by walking the LISST centres and assigning each model class to
    #  the first centre >= D.  Two problems:
    #    (a) the model grid runs to 1200 um but LISST stops at 459 um, and the
    #        `if i == N_lisst` fallback swept ALL 7 out-of-range classes into ring 36.
    #        Because H entries scale as D^3, that row ended up 50x larger than any
    #        other (sum 2.5e-3 vs 4.6e-5) and dominated the analysis.
    #    (b) assignment used centres rather than edges, biasing every class upward.
    #  Now: proper log-midpoint edges, and classes outside the LISST range are simply
    #  unobserved (they are still updated, but only through the ensemble covariance).
    #***************************************************************************
    edges = bin_edges_from_centres(Ds_LISST)
    bin_of = fill(0, Ns)                     # 0 == outside the LISST range
    for j in 1:Ns
        if Dum[j] >= edges[1]  && Dum[j] < edges[end]
            bin_of[j] = searchsortedlast(edges, Dum[j])
            println("Model class $(Dum[j]) um -> LISST bin $(bin_of[j])")
        end
    end


    n_in_bin = zeros(Int, N_lisst)
    for j in 1:Ns
        if bin_of[j] > 0
            n_in_bin[bin_of[j]] += 1
        end
    end
    n_out = count(==(0), bin_of)
    @info "Model classes outside the LISST range ($(round(edges[1],digits=2))-$(round(edges[end],digits=1)) um): $n_out of $Ns"


    #####
    # linear (volume-space) operator, kept for output / plotting
    H = zeros(N_lisst, Ns)
    for j in 1:Ns
        if bin_of[j] > 0
            H[bin_of[j], j] = VOLU[j]
        end
    end

    # H = zeros((N_lisst, Ns_aug))
    
    # # Create volume-weighted observation operator matrix H
    # for j in 1:Ns
    #     @debug "Sorting floc size class: ", D[j]*1e6
    #     for i in 1:N_lisst
    #         if ((D[j]*1e6) <= Ds_LISST[i])
    #             @debug "-> placing in LISST size class: ", Ds_LISST[i]
    #             H[i, j] = VOLU[j] # Volumes[j]*1e6
    #             break
    #         end 
    #         if i==N_lisst
    #             @debug "Larger than max(LISST) -> placing in LISST size class: ", Ds_LISST[i]
    #             H[i, j] = VOLU[j] #Volumes[j]*1e6
    #             break
    #         end 
    #     end
    # end 

    #  [F5] CHANNEL QC.
    #  Rings 1-3 (1.21 / 1.60 / 1.89 um) carry a CONSTANT placeholder variance of
    #  ~8.3e-9 uL/L^2 (std 9e-5) at every single record, while the model sits at
    #  ~0.25 uL/L there.  That is a 2800-sigma innovation, and it dominated the whole
    #  analysis: removing 0.25 uL/L from a bin whose particles have V = 5.2e-19 m^3
    #  requires deleting ~5e11 particles/m^3, which is exactly the enormous increment
    #  the filter was applying.  These rings are also the least trustworthy on a
    #  LISST.  Dropped, along with any ring that contains no model class.
    active = trues(N_lisst)
    active[1:3] .= false
    for i in 1:N_lisst
        if n_in_bin[i] == 0
            active[i] = false
        end
    end
    idx_active = findall(active)
    p = length(idx_active)
    @info "Assimilating $p of $N_lisst LISST rings"

    #  [F5b] R conditioning: relative floor + absolute floor.  This is Tikhonov
    #  regularisation of the inverse in S = HPH' + R -- it bounds the gain instead of
    #  letting near-zero reported variances drive it to infinity.
    R_REL = 0.10        # 10% of the reported volume
    R_ABS = 0.02        # uL/L, roughly the LISST noise floor

    function get_observation_row(time_stamp::Real)
        row_idx = findfirst(dfL.seconds .== time_stamp)
        isnothing(row_idx) && return false, nothing, nothing
        y = Vector{Float64}(dfL[row_idx, Not(:seconds)])
        r = Vector{Float64}(dfR[row_idx, Not(:seconds)])
        #  [F6] reject dead LISST records.  22% of the rows in lisst_data.csv are
        #  identically zero across all 36 rings.  enkf1 assimilated them as "the
        #  water is perfectly clear, and I am certain", which is catastrophic.
        if !any(y .> 0)
            return false, nothing, nothing
        end
        return true, y, r
    end

    #***************************************************************************
    #  [F7] INITIAL CONDITION.
    #  enkf1 used  ssc0[:,1:4] = |randn|*1e3  plus |randn| elsewhere, which is a total
    #  volume of 2.6e-3 uL/L -- about 10,000x smaller than the first LISST record
    #  (25.3 uL/L).  The commented-out `ic = [...]` vector on line 64 of enkf1 was in
    #  fact the right magnitude (29.3 uL/L); it had been divided by 100 and disabled.
    #  Consequence: essentially ALL of the sediment in the old run was manufactured by
    #  the additive `randn .* (1e-8 ./ Volumes')` noise, which is a random walk, so the
    #  load grew like sqrt(t) and was never anchored to anything physical.
    #  Now: invert the first valid LISST spectrum onto the model grid.
    #***************************************************************************
    first_valid = findfirst(i -> any(Vector(dfL[i, Not(:seconds)]) .> 0), 1:nrow(dfL))
    y_ic = Vector{Float64}(dfL[first_valid, Not(:seconds)])
    vol_ic = zeros(Ns)
    for i in 1:N_lisst
        if n_in_bin[i] > 0 && y_ic[i] > 0
            for j in 1:Ns
                if bin_of[j] == i
                    vol_ic[j] = y_ic[i] / n_in_bin[i]
                end
            end
        end
    end
    vfloor = 1e-6 * maximum(vol_ic)
    vol_ic .= max.(vol_ic, vfloor)
    n_ic = vol_ic ./ VOLU
    @info @sprintf("Initial condition from LISST record %d: %.2f uL/L, max %.3e particles/m3",
                   first_valid, sum(vol_ic), maximum(n_ic))

    IC_SPREAD = 0.35                      # lognormal spread across members
    ssc = zeros(N, Ns)
    for e in 1:N, j in 1:Ns
        ssc[e, j] = n_ic[j] * exp(randn()*IC_SPREAD - 0.5*IC_SPREAD^2)
    end

    #***************************************************************************
    #  [F8] MODEL ERROR.
    #  enkf1 added  ssc += abs.(randn(N, Ns))  every step.  abs() of a normal has mean
    #  0.798, so that is a deterministic positive source of ~0.8 particles/class/step,
    #  not noise -- 2900 particles per class over an hour.  Combined with
    #  clamp!(ssc, 0, Inf) after the perturbation (which biases the mean upward again)
    #  the ensemble mean was being pushed around by its own "noise".
    #  Now: zero-mean multiplicative (lognormal) model error, which is unbiased in log
    #  space and cannot produce negatives.
    #***************************************************************************
    SIG_MODEL = 0.01                      # 1% per step
    PAR_NOISE = 2e-3                      # parameter random-walk std per step

    #  [F9] LOCALISATION.  With N = 65 members and a 54-dimensional augmented state,
    #  sample correlations have a noise floor of ~1/sqrt(N) = 0.12; measured spurious
    #  |corr| between the parameters and the observations reached 0.28 even when the
    #  forward model was the identity (i.e. when the true correlation was exactly 0).
    #  Taper the state-observation covariance by distance in log10-diameter so a
    #  1 um class cannot be updated by a 400 um ring, and damp the parameter block.
    #
    #  IMPORTANT: taper BOTH Cxy (state-obs) and Cyy (obs-obs).  Tapering only Cxy
    #  leaves the gain inconsistent with the innovation covariance it is inverted
    #  against, and the filter diverges hard -- in testing that configuration blew the
    #  volume up by 17 orders of magnitude, while the correct two-block version was the
    #  single largest improvement in the whole set (misfit reduction per analysis went
    #  from -18% to -42%, and model/observed volume from 0.63 to 0.90).
    LOC_DECADES = 0.75
    PARAM_LOC   = 0.5
    LOC_xy = ones(Ns_aug, N_lisst)
    for j in 1:Ns, i in 1:N_lisst
        LOC_xy[j, i] = gaspari_cohn(abs(log10(Dum[j]) - log10(Ds_LISST[i])) / LOC_DECADES)
    end
    LOC_xy[Ns+1:end, :] .= PARAM_LOC
    LOC_yy = [gaspari_cohn(abs(log10(Ds_LISST[i]) - log10(Ds_LISST[j])) / LOC_DECADES)
              for i in 1:N_lisst, j in 1:N_lisst]

    #  RTPP inflation (Zhang et al. 2004): a convex blend of analysis and prior
    #  perturbations.  Deliberately NOT RTPS: the state here is a LOG, so RTPS's
    #  unbounded inflation factor gets exponentiated on the back-transform and blows
    #  the ensemble up (measured: log-volume spread 0.15 -> 62 in three hours).
    #  RTPP cannot exceed the prior spread by construction.
    RTPP = 0.3
    SPREAD_CAP = 1.0     # max log-volume ensemble sd (bias exp(sd^2/2) <= 1.65)

    #  [F12] LOG-SPACE OBSERVATIONS.
    #  Reported LISST std/obs runs 0.5-2.2 across the rings, i.e. the observation error
    #  is essentially proportional to the signal, not additive.  Assimilating volume
    #  directly against a log-volume state also overshoots badly: the increment is
    #  linear in log v, so an innovation that is a large fraction of the state maps to
    #  an exponential correction (measured: misfit 21 -> 839 uL/L in one analysis).
    #  Compare log volumes instead, with R mapped by the delta method, var(log y) =
    #  var(y)/y^2.
    YFLOOR = 0.05        # uL/L; below this a ring carries no usable information
    RLOG_MIN = 0.01      # floor on log-space observation variance

    #***************************************************************************
    #  Parameters (unbounded space) + initial save
    #***************************************************************************
    #  Parameter prior spread.  randn(N) (sd 1) spans essentially the whole
    #  admissible range of every parameter, so members' spectra diverge over the 300 s
    #  between analyses and the log-volume spread reaches 2.3 -- which the exp()
    #  back-transform then turns into a 14x high bias on the ensemble-mean volume.
    #  sd 0.35 keeps observation-space spread at 0.66 and the bias at 1.25x.
    PAR_SPREAD = 0.35
    Alphas = randn(N).*PAR_SPREAD; Betas  = randn(N).*PAR_SPREAD
    Nfs    = randn(N).*PAR_SPREAD; Beta2s = randn(N).*PAR_SPREAD
    Msrcs  = randn(N) .* 0.1

    Times = collect(0:dt:(M*dt))
    save2output(1, "alpha", to_alpha.(Alphas))
    save2output(1, "beta",  to_beta.(Betas))
    save2output(1, "beta2", to_beta2.(Beta2s))
    save2output(1, "nf",    to_nf.(Nfs))
    save2output(1, "msrc",  to_src.(Msrcs))
    save2output(1, "G", get_shear(1))
    save_sediment2output(1, ssc, "ssc")
    real_times_saved = [Times[1]]

    #*************************** TIME LOOP  *********************************
    @info "Starting time loop for $M steps with dt = $dt s"
    X = zeros(Ns_aug, N)
    Y = zeros(p, N)
    n_analyses = 0

    for i in 2:M
        time = Times[i]

        # ---- parameter random walk ---------------------------------------
        Alphas .+= randn(N) .* PAR_NOISE
        Betas  .+= randn(N) .* PAR_NOISE
        Beta2s .+= randn(N) .* PAR_NOISE
        if ESTIMATE_MSRC
            Msrcs .*= (1 - dt/TAU_SRC)                     # mean-reverting, not a free walk
            Msrcs .+= randn(N) .* PAR_NOISE
        end
        Nfs    .+= randn(N) .* PAR_NOISE

        # ---- model error (unbiased, positivity preserving) ----------------
        @inbounds for e in 1:N
            src = ESTIMATE_MSRC ? exp(to_src(Msrcs[e]) * dt) : 1.0   # [F13]
            for j in 1:Ns
                ssc[e, j] *= src * exp(randn()*SIG_MODEL - 0.5*SIG_MODEL^2)
            end
        end

        shear = get_shear(time)

        # ---- forward step -------------------------------------------------
        #  [F2] STABILITY.  run_floc_mod is explicit Euler.  Its stability limit is
        #  dt * max_k(loss rate_k) < 1.  With a physically-sized initial condition the
        #  fine classes hold ~1e12 particles/m^3, which puts that limit at ~3e-3 s at
        #  G = 40 and ~7e-4 s at G = 175 -- the driver's dt = 1 s is 3 orders of
        #  magnitude too large.  (The old run never tripped over this only because its
        #  initial condition was 1e4x too small, which kept the rates artificially
        #  tiny.  Restoring a correct IC alone makes the explicit scheme blow up to
        #  n ~ 1e26.)  run_floc_mod_semi treats the two linear sinks implicitly and is
        #  unconditionally stable and positive, so dt = 1 s is fine.
        @threads for EID in 1:N
            advance_member!(ssc, EID, D, Ns, CMAT, shear, dt,
                            to_alpha(Alphas[EID]), to_beta(Betas[EID]),
                            to_beta2(Beta2s[EID]), to_nf(Nfs[EID]))
        end

        # ================= EnKF ANALYSIS =================================
        run_analysis, y_all, r_all = get_observation_row(time)

        if run_analysis
            n_analyses += 1
            y = y_all[idx_active]
            Rv = max.(r_all[idx_active], (R_REL .* y).^2 .+ R_ABS^2)

            #  [F10] STATE TRANSFORM (the regularisation that actually matters).
            #  The state is a particle-number spectrum spanning ~12 orders of
            #  magnitude, and H entries scale as D^3 over a 1.7e9 range.  A Gaussian
            #  (linear) update in that space is badly ill-posed: measured on the
            #  shipped configuration it drove 20% of the state NEGATIVE at every
            #  analysis, with increments of ~3e10 particles/m^3 -- the size of the
            #  whole state.  Assimilating log(volume) instead makes the operator a
            #  plain 0/1 binning matrix, compresses the dynamic range to ~30 in log
            #  units, and enforces positivity structurally.  H is then nonlinear, so
            #  use the ensemble-space form  K = Cxy (Cyy + R)^-1  with the predicted
            #  observations evaluated member by member.
            @inbounds for e in 1:N
                for j in 1:Ns
                    X[j, e] = log(max(ssc[e, j] * VOLU[j], 1e-12))
                end
                X[Ns+1, e] = Alphas[e]
                X[Ns+2, e] = Betas[e]
                X[Ns+3, e] = Nfs[e]
                X[Ns+4, e] = Beta2s[e]
                X[Ns+5, e] = Msrcs[e]
            end

            fill!(Y, 0.0)
            @inbounds for e in 1:N
                for j in 1:Ns
                    b = bin_of[j]
                    if b > 0 && active[b]
                        Y[searchsortedfirst(idx_active, b), e] += ssc[e, j] * VOLU[j]
                    end
                end
            end
            # log-space observation comparison [F12]
            Yl = log.(Y .+ YFLOOR)
            yl = log.(y .+ YFLOOR)
            Rl = max.(Rv ./ (y .+ YFLOOR).^2, RLOG_MIN)

            Xm = mean(X, dims=2); Ym = mean(Yl, dims=2)
            Xa = X .- Xm;         Ya = Yl .- Ym
            Cxy = (Xa * Ya') ./ (N - 1)
            Cyy = (Ya * Ya') ./ (N - 1)
            Cxy .*= LOC_xy[:, idx_active]                     # localisation, both blocks
            Cyy .*= LOC_yy[idx_active, idx_active]
            S = Cyy + Diagonal(Rl)

            #  [F11] PERTURBED OBSERVATIONS.  enkf1 gave every member the SAME
            #  innovation (observations .- H*x_e).  That is the classic stochastic-EnKF
            #  bug: without perturbing the observations the analysis covariance is
            #  systematically too small and the ensemble collapses.  Measured on the
            #  shipped code the spread fell to 0.33x of its prior value at every single
            #  analysis; it only looked "alive" because the biased abs(randn) noise
            #  re-inflated it between steps.
            E = randn(p, N) .* sqrt.(Rl)
            E .-= mean(E, dims=2)                        # keep the sample mean unbiased
            Dmat = (yl .+ E) .- Yl

            # diagnostics before updating
            dpre = yl .- vec(Ym)
            chi2 = dot(dpre, S \ dpre) / p

            X .+= Cxy * (S \ Dmat)

            # RTPP inflation: x' <- (1-g) x'_analysis + g x'_prior
            Xm2 = mean(X, dims=2)
            @inbounds for k in 1:Ns_aug, e in 1:N
                X[k, e] = Xm2[k] + (1 - RTPP)*(X[k, e] - Xm2[k]) + RTPP*Xa[k, e]
            end
            # Spread cap.  The state is a log, so the back-transform to volume carries
            # a bias of exp(sd^2/2) -- at sd = 3 that is a factor of 90 on the
            # ensemble-mean volume.  Shrink any element whose spread exceeds the cap.
            Xm3 = mean(X, dims=2)
            @inbounds for k in 1:Ns
                sd = std(@view X[k, :])
                if sd > SPREAD_CAP
                    f = SPREAD_CAP / sd
                    for e in 1:N
                        X[k, e] = Xm3[k] + f*(X[k, e] - Xm3[k])
                    end
                end
            end

            # back-transform
            @inbounds for e in 1:N
                for j in 1:Ns
                    ssc[e, j] = exp(clamp(X[j, e], -30.0, 40.0)) / VOLU[j]
                end
                Alphas[e] = X[Ns+1, e]
                Betas[e]  = X[Ns+2, e]
                Nfs[e]    = X[Ns+3, e]
                Beta2s[e] = X[Ns+4, e]
                if ESTIMATE_MSRC
                    Msrcs[e] = X[Ns+5, e]
                end
            end

            if n_analyses % 5 == 1
                mod_after = zeros(p)
                for e in 1:N, j in 1:Ns
                    b = bin_of[j]
                    if b > 0 && active[b]
                        mod_after[searchsortedfirst(idx_active, b)] += ssc[e, j]*VOLU[j]/N
                    end
                end
                mod_before = vec(mean(Y, dims=2))
                spread = mean(std(X[1:Ns, :], dims=2))
                @printf("t=%6.2f h | chi2/p %7.2f | misfit %7.2f -> %7.2f uL/L | vol mod/obs %6.3f | log-spread %5.3f | a=%.3f b=%.3f nf=%.2f b2=%.2f\n",
                        time/3600, chi2, sum(abs.(y .- mod_before)), sum(abs.(y .- mod_after)),
                        sum(mod_after)/sum(y), spread,
                        mean(to_alpha.(Alphas)), mean(to_beta.(Betas)),
                        mean(to_nf.(Nfs)), mean(to_beta2.(Beta2s)))
            end
        end
        # -------------------------------------------------------------
        if i % isave == 0
            index = div(i, isave)
            save2output(index, "alpha", to_alpha.(Alphas))
            save2output(index, "beta",  to_beta.(Betas))
            save2output(index, "beta2", to_beta2.(Beta2s))
            save2output(index, "nf",    to_nf.(Nfs))
            save2output(index, "msrc",  to_src.(Msrcs))
            save2output(index, "G", get_shear(time))
            save_sediment2output(index, ssc, "ssc")
            push!(real_times_saved, time)
        end
    end

    @info "Completed $n_analyses analyses"

    # ********************** save data ****************************
    units_dict = Dict("G" => "1/s", "alpha" => "-", "flux" => "-", "beta" => "1/s",
                      "beta2" => "-", "nf" => "-", "sed_mass" => "kg/m^3", "msrc" => "1/s")
    var2name = Dict("G" => "Turbulent shear", "alpha" => "Aggregation efficiency",
                    "flux" => "sediment influx",
                    "beta" => "Fragmentation coefficient", "beta2" => "Fragmentation exponent",
                    "nf" => "Fractal dimension", "sed_mass" => "Sediment mass concentration",
                    "msrc" => "Estimated log-mass source rate")

    nt = div(M, isave)
    ds = NCDataset(file_out_name, "c")
    ds.attrib["title"] = "floc mod + EnKF (fixed)"

    defDim(ds, "N", N); defDim(ds, "time", nt)
    defDim(ds, "Ny", N_lisst); defDim(ds, "Ds", length(D))

    v = defVar(ds, "Ds", Float32, ("Ds",)); v[:] = D
    v = defVar(ds, "ssc", Float64, ("N", "Ds", "time"), attrib = OrderedDict(
        "units" => "parts/m3", "long_name" => "suspended sediment concentration"))
    v[:,:,:] = output["ssc"]
    v = defVar(ds, "H", Float64, ("Ny", "Ds"),
               attrib = OrderedDict("obs operator" => "particle count -> LISST volume (uL/L)"))
    v[:,:] = H
    v = defVar(ds, "active", Int64, ("Ny",),
               attrib = OrderedDict("long_name" => "1 if LISST ring was assimilated"))
    v[:] = Int.(active)
    v = defVar(ds, "time", Float32, ("time",), attrib = OrderedDict("units" => "seconds"))
    v[:] = real_times_saved[1:nt]

    for var in var2save
        v = defVar(ds, var, Float64, ("N", "time"), attrib = OrderedDict(
            "units" => units_dict[var], "long_name" => var2name[var]))
        v[:,:] = output[var]
    end
    close(ds)
    print("Saved $file_out_name \n")
end


run_my_model("FlocMod_ADV_G_fixed2.nc")
