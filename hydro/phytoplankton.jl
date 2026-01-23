
function calculate_net_growth(algae, I, discretization)
    # Inputs: 
    #     pmax : maximum specific growth rate [1/hour]
    #     I    : light intensity [mu mol photons/m^2/s]
    #     Hi   : half-saturation of light-limited growth [mu mol photons * m^2/s]
    # Output:
    #     pi   : specific growth rate [1/hour converted to 1/second]
    N = discretization["N"]
    pmax = algae["pmax"]
    Hi = algae["Hi"]
    loss = algae["Li"]

    growth = zeros(N)
    for i in 1:N 
        growth[i] = pmax * I[i]/(Hi + I[i]) - loss
    end

    return growth
end 

function self_shading(algae1, algae2, I_in, turbidity, discretization)
    #  Use Lambert-Beer's Law to calculate light intensity at each depth

    # Inputs:
    z = discretization["z"]
    dz = discretization["dz"]
    N = discretization["N"]

    # We want water depth [cm] to be zero at the top, 20 at the bottom (pos. numbers)
    # corrected_z = @. -z 

    k1 = algae1["k"]
    k2 = algae2["k"]

    c1 = algae1["c"]
    c2 = algae2["c"]

    # Vector to store light intensity at each depth
    I = zeros(N)
    I2 = zeros(N)

    for i in 1:N
        I[i] = k1*c1[i] + k2*c2[i] + turbidity #*corrected_z[i])
    end 

    I2[end] = I[end]
    for i in (N-1):-1:1
        I2[i] = I[i+1] + I[i]
    end

    for i in 1:N
        I2[i] = I_in * exp(-I2[i] * z[i]) 
    end


    return I2 
end 