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
Ns = 60                  # Number of sediment size classes

# D = LinRange(4, 1500, Ns) .* 1e-6     # Sediment grain sizes (\mu m )
D = logrange(1e-6, 330e-6, Ns) #.* 1e-6     # Sediment grain sizes (\mu m )
D = collect(D)
# println("Sediment grain sizes (m) = ", D)

ssc0 = zeros(Ns)
ssc0[1:13] .= 0.1   # Matrix for sediment concentration (Nz x Ns)
ssc0[47:58] .= 26   # Matrix for sediment concentration (Nz x Ns)

# ssc0[10:15] .= 1e3    # Matrix for sediment concentration (Nz x Ns)

# initialize once
init_params!(D, Ns, ssc0)

# Calculate settling velocities for each size class
ws = floc_mod.ws[] 
# println("Settling velocities (m/s) = ", ws)

turbulent_shear = 35 #2 #0.5


# NFRAMES=6000
# output = zeros(Ns, NFRAMES)
# output[:,1] = ssc0
# dt = 1 

# for i in 2:NFRAMES
#         # print("Frame $i \n")
#         N0 = output[:,i-1]
#         N1 = run_floc_mod(N0, Ns, turbulent_shear, dt)
#         output[:,i] = N1 
# end 


function get_Gval2(i)
    if (i < 21600)
            Gval=95.0 #Gval=95.0
    elseif (i < 43200)
            Gval=50.0
    elseif (i < 64800)
            Gval=20.0
    elseif (i < 86400)
            Gval=50.0
    elseif (i < 108000)            
            Gval=95.0
    elseif (i < 159120)
            Gval=50.0
    else 
       Gval=50.0 
    end 
    return Gval  #+ randn()*5.0
end 
    
function get_Gval(i)
       if (i < 7201.0) 
                Gval=1.0
        elseif (i < 8401.0) 
                Gval=2.0
        elseif (i < 9601.0) 
                Gval=3.0  
        elseif (i < 10801.0) 
                Gval=4.0
        elseif (i < 12601.0) 
                Gval=12.0
        elseif (i < 13801.0) 
                Gval=4.0
        elseif (i < 15001.0) 
                Gval=3.0
        elseif (i < 16201.0) 
                Gval=2.0
        elseif (i < 21601.0) 
                Gval=1.0
        elseif (i < 25201.0) 
                Gval=0.0
        elseif (i < 30601.0) 
                Gval=1.0
        elseif (i < 31801.0) 
                Gval=2.0                     
        elseif (i < 33001.0) 
                Gval=3.0       
        elseif (i < 34201.0) 
                Gval=4.0
        elseif (i < 36001.0) 
                Gval=12.0
        elseif (i < 37201.0) 
                Gval=4.0
        elseif (i < 38401.0) 
                Gval=3.0
        elseif (i < 39601.0) 
                Gval=2.0
        elseif (i < 45001.0) 
                Gval=1.0
        elseif (i < 48601.0) 
                Gval=0.0
        elseif (i < 54001.0) 
                Gval=1.0
        elseif (i < 55201.0) 
                Gval=2.0                     
        elseif (i < 56401.0) 
                Gval=3.0       
        elseif (i < 57601.0) 
                Gval=4.0
        else 
                Gval=12.0
       end 
return Gval 
end 


# skip = 30
# anim = @animate for i in 2:(div(NFRAMES, skip)) 
#         frame = i*skip
#         time = frame*dt/60 
#         time_string = @sprintf("%2.1f min", time)
#         scatter(D.*1e6, output[:,frame],  title=time_string, legend=false, xscale = :log10) #xlim=(0,1), ylim=(0,1),
#         xlabel!("Particle Diameter (um)")
#         ylabel!("N. particles")
#         # xscale!(:log)
#         avg_dist = mean(D.*1e6, weights(output[:,frame]))
#         println("Average particle size at time $time_string is $avg_dist um")
# # plot(avg_dist)
# # save
# # savefig("average_size_distribution.png")
# end 

# gif(anim, "animated_scatter.gif", fps = 20)


#         ! reproducing flocculation experiment Verney et al., 2011

NFRAMES= 159120 #159120
output = zeros(Ns, NFRAMES)
output[:,1] = ssc0
dt = 1 

SKIP = 2500
output_csv = zeros(div(NFRAMES, SKIP), 3)

for i in 2:NFRAMES
        
        Gval = get_Gval2(i)
        N0 = output[:,i-1]
        N1 = run_floc_mod(N0, Ns, Gval, dt)
        output[:,i] = N1 
        
        if i%SKIP ==0
                print("Frame $i , gval=$(Gval) \n")
                avg_dist = mean(D.*1e6, weights(N1))
                # println("Average particle size at time $i is $avg_dist um")
                d50 = quantile(D.*1e6, weights(N1), 0.5)
                d84 = quantile(D.*1e6, weights(N1), 0.84)
                println("D50 @  $i is $d50 um / d84 is $d84 um")
                output_csv[div(i, SKIP), 1] = i
                output_csv[div(i, SKIP), 2] = d50
                output_csv[div(i, SKIP), 3] = d84

        end 
end 

df = DataFrame(
    time = output_csv[:, 1],
    d50  = output_csv[:, 2],
    d84  = output_csv[:, 3]
)
CSV.write("output_nf=2.2.csv", df)
println("CSV saved.")

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

