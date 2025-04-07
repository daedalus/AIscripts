import math

# Constants for conversions
IN_PER_MB = 1 / 33.86389
MB_PER_IN = 33.86389
M_PER_FT = 0.304800
FT_PER_M = 1 / 0.304800

# Function to calculate saturation vapor pressure using the Wobus polynomial
def calc_vapor_pressure_wobus(t):
    eso = 6.1078
    c0 = 0.99999683
    c1 = -0.90826951e-02
    c2 = 0.78736169e-04
    c3 = -0.61117958e-06
    c4 = 0.43884187e-08
    c5 = -0.29883885e-10
    c6 = 0.21874425e-12
    c7 = -0.17892321e-14
    c8 = 0.11112018e-16
    c9 = -0.30994571e-19

    pol = c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * (c6 + t * (c7 + t * (c8 + t * c9))))))))
    es = eso / (pol ** 8)
    return es

# Function to calculate absolute air pressure
def calc_abs_press(pressure, altitude):
    k1 = 0.190284
    k2 = 8.4288e-5
    p1 = pressure ** k1
    p2 = altitude * k2
    p3 = 0.3 + (p1 - p2) ** (1 / k1)
    return p3

# Function to calculate air density
def calc_density(abs_press_mb, e, tc):
    Rv = 461.4964
    Rd = 287.0531
    tk = tc + 273.15
    pv = e * 100
    pd = (abs_press_mb - e) * 100
    d = (pv / (Rv * tk)) + (pd / (Rd * tk))
    return d

# Function to calculate ISA altitude for a given density
def calc_altitude(d):
    g = 9.80665
    Po = 101325
    To = 288.15
    L = 6.5
    R = 8.314320
    M = 28.9644
    D = d * 1000

    p2 = ((L * R) / (g * M - L * R)) * math.log((R * To * D) / (M * Po))
    H = -(To / L) * (math.exp(p2) - 1)
    h = H * 1000
    return h

# Function to calculate geometric altitude from geopotential altitude
def calc_z(h):
    r = 6369e3
    return (r * h) / (r - h)

# Function to calculate geopotential altitude from geometric altitude
def calc_h(z):
    r = 6369e3
    return (r * z) / (r + z)

# Function to calculate actual pressure from altimeter setting and geopotential altitude
def calc_as2_press(As, h):
    k1 = 0.190263
    k2 = 8.417286e-5
    p = (As ** k1 - k2 * h) ** (1 / k1)
    return p

# Function to validate input
def validate_input(value, prompt):
    try:
        float(value)
        return True
    except ValueError:
        print(prompt)
        return False

# Main function to perform calculations
def calculate_density_altitude(elevation, elevation_unit, temperature, temp_unit, alt_setting, alt_setting_unit, dew_point, dp_unit):
    # Validate inputs
    if not all([
        validate_input(elevation, "Invalid entry for Elevation"),
        validate_input(temperature, "Invalid entry for Temperature"),
        validate_input(alt_setting, "Invalid entry for Altimeter Setting"),
        validate_input(dew_point, "Invalid entry for Dew Point")
    ]):
        return None

    # Convert elevation to meters
    if elevation_unit == "feet":
        zm = float(elevation) * M_PER_FT
    else:
        zm = float(elevation)

    # Convert altimeter setting to mb
    if alt_setting_unit == "inHg":
        altset_mb = float(alt_setting) * MB_PER_IN
    else:
        altset_mb = float(alt_setting)

    # Convert temperature to Celsius
    if temp_unit == "degF":
        tc = (float(temperature) - 32) * 5 / 9
    else:
        tc = float(temperature)

    # Convert dew point to Celsius
    if dp_unit == "degF":
        tdpc = (float(dew_point) - 32) * 5 / 9
    else:
        tdpc = float(dew_point)

    # Calculate vapor pressure
    emb = calc_vapor_pressure_wobus(tdpc)

    # Calculate geopotential altitude
    hm = calc_h(zm)

    # Calculate absolute pressure
    actpress_mb = calc_as2_press(altset_mb, hm)

    # Calculate air density
    density = calc_density(actpress_mb, emb, tc)
    relden = 100 * (density / 1.225)

    # Calculate density altitude
    densalt_m = calc_altitude(density)
    densalt_zm = calc_z(densalt_m)

    # Convert units for output
    actpress_in = actpress_mb * IN_PER_MB
    densalt_ft = densalt_zm * FT_PER_M

    # Check for valid range
    if densalt_ft > 36090 or densalt_ft < -15000:
        print("Out of range for Troposphere Algorithm: Altitude =", round(densalt_ft, 0), "feet")
        return None

    # Calculate estimated AWOS density altitude
    nws = 145442.16 * (1 - ((17.326 * actpress_mb) / (tc + 273.15 + 32)) ** 0.235)
    awos_ft = round(nws / 100, 0) * 100
    awos_m = awos_ft * M_PER_FT

    # Return results
    return {
        "Density Altitude (ft)": round(densalt_ft, 0),
        "Density Altitude (m)": round(densalt_zm, 0),
        "Absolute Pressure (inHg)": round(actpress_in, 2),
        "Absolute Pressure (hPa)": round(actpress_mb, 1),
        "Air Density (lb/ft3)": round(density * 0.062428, 4),
        "Air Density (kg/m3)": round(density, 3),
        "Relative Density (%)": round(relden, 2),
        "Estimated AWOS (ft)": awos_ft,
        "Estimated AWOS (m)": round(awos_m, 0)
    }

# Example usage
if __name__ == "__main__":
    elevation = input("Enter elevation: ")
    elevation_unit = input("Enter elevation unit (feet/meters): ")
    temperature = input("Enter temperature: ")
    temp_unit = input("Enter temperature unit (degF/degC): ")
    alt_setting = input("Enter altimeter setting: ")
    alt_setting_unit = input("Enter altimeter setting unit (inHg/hPa): ")
    dew_point = input("Enter dew point: ")
    dp_unit = input("Enter dew point unit (degF/degC): ")

    results = calculate_density_altitude(elevation, elevation_unit, temperature, temp_unit, alt_setting, alt_setting_unit, dew_point, dp_unit)
    if results:
        for key, value in results.items():
            print(f"{key}: {value}")
