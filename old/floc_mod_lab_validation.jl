#!/usr/bin/env julia

#  module load  julia/1.11.7 
using Printf
using DataStructures: OrderedDict
using NCDatasets
using CSV, DataFrames
using Printf
using Profile
using Statistics 
# using Arrow, DataFrames
using LaTeXStrings
using LinearAlgebra
using Plots
using StatsBase

using Base.Threads

println("Using $(Threads.nthreads()) threads") 


include("floc_mod.jl")
using .floc_mod
# Pkg.add(["LaTeXStrings", "LinearAlgebra", "Profile"])
# println("hello")

# ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
Ns = 60                  # Number of sediment size classes

# D = LinRange(4, 1500, Ns) .* 1e-6     # Sediment grain sizes (\mu m )
D = logrange(1e-6, 330e-6, Ns) #.* 1e-6     # Sediment grain sizes (\mu m )
D = collect(D)

ssc0 = zeros(Ns)
IC = 4
if IC == 1
        ssc0 .= 5   # Matrix for sediment concentration (Nz x Ns)
        ssc0[10:20] .= 350   # Matrix for sediment concentration (Nz x Ns)
elseif IC == 2
        ssc0 .= 5   # Matrix for sediment concentration (Nz x Ns)
        ssc0[25:40] .= 250   # Matrix for sediment concentration (Nz x Ns)
elseif IC == 3
        ssc0 .= 5   # Matrix for sediment concentration (Nz x Ns)
        ssc0[20:38] .= 250   # Matrix for sediment concentration (Nz x Ns)
elseif IC == 4
        ssc0 .= 5   # Matrix for sediment concentration (Nz x Ns)
        ssc0[10:30] .= 350   # Matrix for sediment concentration (Nz x Ns) 
end 


# ssc0[10:15] .= 1e3    # Matrix for sediment concentration (Nz x Ns)

# initialize once
init_params!(D, Ns, ssc0)

# Calculate settling velocities for each size class
ws = floc_mod.ws[] 
# println("Settling velocities (m/s) = ", ws)


experiments = [
    "Exp01_2020-06-26_Tank1",  "Exp07_2020-06-25_Tank2",  "Exp13_2021-01-08_Tank1",  "Exp19_2021-01-08_Tank2",
    "Exp02_2020-06-27_Tank1",  "Exp08_2020-06-26_Tank2",  "Exp14_2021-01-09_Tank1",  "Exp20_2021-01-09_Tank2",
    "Exp03_2020-06-28_Tank1",  "Exp09_2020-06-27_Tank2",  "Exp15_2021-01-10_Tank1",  "Exp21_2021-01-10_Tank2",
    "Exp04_2020-06-29_Tank1",  "Exp10_2020-06-29_Tank2",  "Exp16_2021-01-11_Tank1",  "Exp22_2021-01-11_Tank2",
    "Exp05_2020-06-30_Tank1",  "Exp11_2020-06-30_Tank2",  "Exp17_2021-01-13_Tank1",  "Exp23_2021-01-13_Tank2",
    "Exp06_2020-07-02_Tank1",  "Exp12_2020-07-02_Tank2",  "Exp18_2021-01-14_Tank1",  "Exp24_2021-01-14_Tank2"
]
# Exp01_2020-06-26_Tank1  Exp07_2020-06-25_Tank2  Exp13_2021-01-08_Tank1  Exp19_2021-01-08_Tank2
# Exp02_2020-06-27_Tank1  Exp08_2020-06-26_Tank2  Exp14_2021-01-09_Tank1  Exp20_2021-01-09_Tank2
# Exp03_2020-06-28_Tank1  Exp09_2020-06-27_Tank2  Exp15_2021-01-10_Tank1  Exp21_2021-01-10_Tank2
# Exp04_2020-06-29_Tank1  Exp10_2020-06-29_Tank2  Exp16_2021-01-11_Tank1  Exp22_2021-01-11_Tank2
# Exp05_2020-06-30_Tank1  Exp11_2020-06-30_Tank2  Exp17_2021-01-13_Tank1  Exp23_2021-01-13_Tank2
# Exp06_2020-07-02_Tank1  Exp12_2020-07-02_Tank2  Exp18_2021-01-14_Tank1  Exp24_2021-01-14_Tank2

# Load your data

@threads for exp in experiments
        # exp = "Exp07_2020-06-25_Tank2"
        fn = "/global/homes/s/siennaw/scratch/siennaw/scripts/Data-Abolfazli-et-al-Mississippi/Data_01_Time_Series/$(exp)/G_S_data.csv"
        df = CSV.read(fn, DataFrame)
        @info "Data loaded from $fn"

        # Pre-convert minutes to seconds once to save speed
        times_sec = df.min .* 60
        total_time = maximum(times_sec)
        @info "Total simulation time (s): $total_time"
        g_values = df.G_Hz

        function get_Gval(t)
                # Find the index of the last time entry that is <= t
                idx = findlast(x -> x <= t, times_sec)
                
                # If t is before the first entry, return the first value
                if idx === nothing
                        return g_values[1]
                end
                
                return g_values[idx]
        end


        NFRAMES= total_time #159120 #159120
        # output = zeros(Ns, NFRAMES)
        # output[:,1] = ssc0
        N0 = ssc0 
        dt = 1 

        SKIP = 1200
        output_csv = zeros(div(NFRAMES, SKIP), 4)

        for i in 2:NFRAMES
                
                Gval = get_Gval(i)
                # N0 = output[:,i-1]
                N1 = run_floc_mod(N0, Ns, Gval, dt)
                # output[:,i] = N1 
                N0 = N1 
                
                if i%SKIP ==0
                        # print("Frame $i , gval=$(Gval) \n")
                        avg_dist = mean(D.*1e6, weights(N1))
                        # println("Average particle size at time $i is $avg_dist um")
                        d50 = quantile(D.*1e6, weights(N1), 0.5)
                        d84 = quantile(D.*1e6, weights(N1), 0.84)
                        # println("D50 @  $i is $d50 um / d84 is $d84 um")
                        output_csv[div(i, SKIP), 1] = i
                        output_csv[div(i, SKIP), 2] = d50
                        output_csv[div(i, SKIP), 3] = d84
                        output_csv[div(i, SKIP), 4] = Gval
                end 
        end 

        nf, α, α_2, β, β_2, β_3 = return_parameter_space()
        # Get unique runID for this run
        hash_input = string(exp, nf, α, α_2, β, β_2, β_3, IC)
        runid = hash(hash_input)

        df = DataFrame(run_id = runid,
                        experiment=exp,
                        nf=nf, 
                        alpha=α, 
                        alpha_2=α_2, 
                        beta=β, 
                        beta_2=β_2, 
                        beta_3=β_3, 
                        IC=IC)
        CSV.write("run_log.csv", df, append=true, writeheader=false) #(Threads.threadid()==1))

        df = DataFrame(
                time = output_csv[:, 1],
                d50  = output_csv[:, 2],
                d84  = output_csv[:, 3],
                g = output_csv[:, 4],)
        CSV.write("output/$(runid).csv", df)
        @info "CSV saved: $(runid).csv"




        # skip = 60*10 #60*5
        # anim = @animate for i in 2:(div(NFRAMES, skip)) 
        #         frame = i*skip
        #         time = frame*dt/60 
        #         shear = get_Gval2(frame)
        #         time_string = @sprintf("%2.1f min (Gval=%d)", time, shear)
        #         scatter(D.*1e6, output[:,frame],  title=time_string, legend=false, xscale = :log10) #xlim=(0,1), ylim=(0,1),
        #         xlabel!("Particle Diameter (um)")
        #         ylabel!("N. particles")
        #         # xscale!(:log)
        #         avg_dist = mean(D.*1e6, weights(output[:,frame]))
        #         # println("Average particle size at time $time_string is $avg_dist um")
        # # plot(avg_dist)
        # # save
        # # savefig("average_size_distribution.png")
        # end 

        # gif(anim, "verney_test.gif", fps = 20)

end 