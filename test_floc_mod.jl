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


include("floc_mod.jl")
using .floc_mod
# Pkg.add(["LaTeXStrings", "LinearAlgebra", "Profile"])
# println("hello")

# ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
Ns = 100                  # Number of sediment size classes

D = LinRange(1, 200, Ns) .* 1e-6     # Sediment grain sizes (\mu m )
D = collect(D)
# println("Sediment grain sizes (m) = ", D)

# initialize once
init_params!(D, Ns)

# Calculate settling velocities for each size class
ws = floc_mod.ws[] 
# println("Settling velocities (m/s) = ", ws)

ssc0 = zeros(Ns) .+ 1e6     # Matrix for sediment concentration (Nz x Ns)

turbulent_shear = 0.5


NFRAMES=6 
output = zeros(Ns, NFRAMES)
output[:,1] = ssc0

anim = @animate for i in 2:NFRAMES
        print("Frame $i \n")
        N0 = output[:,i-1]
        N1 = run_floc_mod(N0, Ns, turbulent_shear)
        scatter(D, N1,  title="Frame $i", legend=false) #xlim=(0,1), ylim=(0,1),
        output[:,i] = N1 
end 

gif(anim, "animated_scatter.gif", fps = 1)
 
