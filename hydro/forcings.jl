
#********************** FIXED CONSTANTS  ***************************
rhoA = 1.23  # DENSITY OF AIR, kg / m^3
rhoW = 1000  # Density of Water
specific_heat_water = 4181 # J/kg-degC
specific_heat_air = 1007 # J/kg-degCxrh
c_d = 0.05   # Drag coefficient 

function get_pressure_at_timestamp(time::Real, Px0::Real, T_Px::Real, phase_shift=0)

    if T_Px == 0
        return Px0
    else
        time = time/3600
        period = (2*pi)/T_Px
        return Px0 * cos(period * (time - phase_shift))
    end
end 

function diurnal_light(t, I_max, phase_shift=0, diurnal=true)
    # Function to generate estimate of light according to diurnal cycle.
    # Inputs: 
        #  * t (in seconds)
        #  * I_max (maximum light intensity/ light at noon)
    # return I_in
    if diurnal
        hour = t/3600
        period = (2*pi)/24
        light = I_max * cos(period * (hour - phase_shift))
        clamp!(light, 0, Inf) #During night, light is zero
        return light
    else
        return I_max
    end
end





function wind_speed_2_wind_stress(Wind::Real, discretization, c_d=c_d, rhoA=rhoA, rho0=rhoW)
    # c_d = 0.05
    dt = discretization["dt"]
    dz = discretization["dz"]
    W = (c_d * Wind)^2 * rhoA * dt/(dz*rho0) 
    return W # Changed 2/21 to be W^2 instead of (0.5*W)^2
end

# Wind = model.wind_speed(time_index, bottom_speed, top_speed, phase_shift=WIND_PHASE_SHIFT)
# wind = (c_d * Wind)**2 * rhoA