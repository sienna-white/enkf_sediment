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


include("floc_mod.jl")
using .floc_mod
# Pkg.add(["LaTeXStrings", "LinearAlgebra", "Profile"])
# println("hello")

# ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
Ns = 30                  # Number of sediment size classes

D = LinRange(4, 200, Ns) .* 1e-6     # Sediment grain sizes (\mu m )
D = logrange(4e-6, 400e-6, Ns) #.* 1e-6     # Sediment grain sizes (\mu m )
D = collect(D)
# println("Sediment grain sizes (m) = ", D)


ssc0 = zeros(Ns)
ssc0[1:15] .= 6.5e2    # Matrix for sediment concentration (Nz x Ns)


# initialize once
init_params!(D, Ns, ssc0)

# Calculate settling velocities for each size class
ws = floc_mod.ws[] 
# println("Settling velocities (m/s) = ", ws)

turbulent_shear = 35 #2 #0.5


NFRAMES=2000
output = zeros(Ns, NFRAMES)
output[:,1] = ssc0
dt = 1 

for i in 2:NFRAMES
        # print("Frame $i \n")
        N0 = output[:,i-1]
        N1 = run_floc_mod(N0, Ns, turbulent_shear, dt)
        output[:,i] = N1 
end 



skip = 20
anim = @animate for i in 2:(div(NFRAMES, skip)) 
        frame = i*skip
        time = frame*dt/60 
        time_string = @sprintf("%2.1f min", time)
        scatter(D.*1e6, output[:,frame],  title=time_string, legend=false, xscale = :log10) #xlim=(0,1), ylim=(0,1),
        xlabel!("Particle Diameter (um)")
        ylabel!("N. particles")
        # xscale!(:log)
        avg_dist = mean(D.*1e6, weights(output[:,frame]))
        println("Average particle size at time $time_string is $avg_dist um")
# plot(avg_dist)
# save
# savefig("average_size_distribution.png")
end 

gif(anim, "animated_scatter.gif", fps = 10)
 
# Plot average size distribution 
# Weighted average

