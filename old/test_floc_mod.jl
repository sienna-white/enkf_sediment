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
# using LaTeXStrings
using LinearAlgebra
using Plots
# using StatsBase


include("floc_mod.jl")
using .floc_mod


# ********************** DEFINE SEDIMENT SIZE CLASSES ****************************
Ns = 45                  # Number of sediment size classes

D = logrange(10e-6, 1500e-6, Ns) # Sediment grain sizes (\mu m )
D = collect(D)

ssc0 = zeros(Ns) #.+ 1e8
ssc0[1:30] .= 1e3    # Matrix for sediment concentration (Nz x Ns)
# ssc0[7:Ns] .= 150
# initialize once
init_params!(D, Ns, ssc0)

# Calculate settling velocities for each size class
ws = floc_mod.ws[] 
turbulent_shear = 6.5 


NFRAMES=60 #3600*7 #8600*6
output = zeros(Ns, NFRAMES)
output[:,1] = ssc0
dt = 5
println("CONCENTRATIONS = ", ssc0, "\n")

D1, M1 = get_particle_density() 
for i in 2:NFRAMES
        # print("Frame $i \n")
        N0 = output[:,i-1]
        N1 = run_floc_mod(N0, Ns, turbulent_shear, dt)
        output[:,i] = N1 
        if mod(i, 10) == 0
                println("($i) CONCENTRATIONS = ", N1.*D1, "\n")
        end
                
end 


skip = 1 #60*1
anim = @animate for i in 2:(div(NFRAMES, skip)) 
        frame = i*skip
        time = frame*dt/3600 
        shear = turbulent_shear
        time_string = @sprintf("%2.1f hr (Gval=%d)", time, shear)
        scatter(D.*1e6, output[:,frame].*M1,  title=time_string, legend=false, xscale = :log10) #xlim=(0,1), ylim=(0,1),
        xlabel!("Particle Diameter (um)")
        ylabel!("Mass in particle class")
        # xscale!(:log)
        # avg_dist = mean(D.*1e6, weights(output[:,frame]))
        # println("Average particle size at time $time_string is $avg_dist um")
# plot(avg_dist)
# save
# savefig("average_size_distribution.png")
end 

gif(anim, "verney_test.gif", fps = 15)



exit() 

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

NFRAMES=100
output = zeros(Ns, NFRAMES)
output[:,1] = ssc0
dt = 1 

for i in 2:NFRAMES
        
        Gval = get_Gval(i)
        N0 = output[:,i-1]
        N1 = run_floc_mod(N0, Ns, Gval, dt)
        output[:,i] = N1 
        
        if i%500 ==0
                print("Frame $i , gval=$(Gval) \n")
        end 

end 



   