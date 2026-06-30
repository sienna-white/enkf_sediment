


# N = 60    # number of grid points
# H = 6    # depth (meters)
# dz = H/N  # grid spacing - may need to adjust to reduce oscillations
# dt = 10   # (seconds) size of time step 
# M  = 10000 #00 #000 # 50000  #500 #

global_params = Dict(
    "N" => 60,   # number of grid points
    "H" => 6,    # depth (meters)
    "dz" => 6/60, # grid spacing - may need to adjust to reduce oscillations
    "dt" => 10,  # (seconds) size of time step
    "M" => 103580, #,86400, # Auugst 6-16   #8640, august 13
    "time_range" => "august_6",
    "istart" => 1440,
    "iend" => 7560) # number of time steps
# i0:  70560  i1:  76680




