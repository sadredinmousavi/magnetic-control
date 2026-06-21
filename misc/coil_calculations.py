import numpy as np

mu0 = 4 * np.pi * 1e-7
rho_cu = 1.68e-8
rho_copper = 8960

# thermal resistance (free air assumption)
R_th = 3.5  # K/W


def coil_pair(N_w, N_h, D0_mm, wire_d_mm, I):

    R0 = D0_mm / 2 / 1000
    d = wire_d_mm / 1000

    # =========================
    # TOTAL TURNS
    # =========================
    # N = N_w * N_h
    N = sum(N_w - (i % 2) for i in range(N_h))

    # =========================
    # WIRE GEOMETRY
    # =========================
    # axial height
    h = (N_h - 1) * d * np.sqrt(3)/2 + d
    w = N_w * d

    # radial build thickness (THIS IS tr)
    t_r = N_h * d * np.sqrt(3)/2

    # inner / outer diameter
    D_inner = 2 * (R0 - t_r)
    D_outer = 2 * (R0 + t_r)

    # =========================
    # ELECTRICAL
    # =========================
    A_wire = np.pi * (d / 2)**2

    L_wire = 2 * np.pi * R0 * N

    R_coil = rho_cu * L_wire / A_wire
    R = 2 * R_coil  

    L_coil = (mu0 * N**2 * np.pi * R0**2) / (2 * R0)
    L_ind = 2 * L_coil

    tau = L_ind / R
    fc = R / (2 * np.pi * L_ind)

    # impedance at 60 Hz
    f = 60
    w_ = 2 * np.pi * f
    Z = np.sqrt(R**2 + (w_ * L_ind)**2)

    V_60 = I * Z
    P = I**2 * R

    # =========================
    # MAGNETIC FIELD
    # =========================
    B = (8 * mu0 * N * I) / (5 * np.sqrt(5) * R0)

    # =========================
    # THERMAL
    # =========================
    delta_T = P * R_th

    # =========================
    # MASS OF COIL
    # =========================
    volume = L_wire * A_wire
    mass_coil = volume * rho_copper
    mass = mass_coil * 2

    # =========================
    # RETURN
    # =========================
    return {
        "N": N,
        "R": R,
        "L": L_ind,
        "tau": tau,
        "fc": fc,
        "B": B,
        "V_60": V_60,
        "P": P,
        "delta_T": delta_T,
        "mass": mass,
        "h": h,
        "w": w,
        "t_r": t_r,
        "D_inner": D_inner,
        "D_outer": D_outer
    }


# =========================
# INPUTS
# =========================
wire_d_mm = 1.36
I = 3
#
# N_w = 12
# N_h = 10
# D0_mm = 90
#
N_w = 16
N_h = 10
D0_mm = 150
#
# N_w = 16
# N_h = 14
# D0_mm = 220
#


# =========================
# RUN
# =========================
res = coil_pair(N_w, N_h, D0_mm, wire_d_mm, I)


# =========================
# PRINT RESULTS
# =========================
print("\n===== HELMHOLTZ COIL FULL MODEL =====")

print(f"Total Turns        : {res['N']}")
print(f"Resistance (Ω)     : {res['R']:.4f}")
print(f"Inductance (mH)    : {res['L']*1000:.4f}")
print(f"Time constant (ms) : {res['tau']*1000:.4f}")
print(f"Corner freq (Hz)   : {res['fc']:.2f}")
print(f"Magnetic field (mT): {res['B']*1000:.4f}")

print(f"\nVoltage @ 60Hz (V) : {res['V_60']:.2f}")
print(f"Power loss (W)     : {res['P']:.2f}")

print(f"\nTemperature rise ΔT: {res['delta_T']:.2f} °C")

print(f"Mass (kg)          : {res['mass']:.4f}")

print(f"\nInner diameter (mm): {res['D_inner']*1000:.2f}")
print(f"Outer diameter (mm): {res['D_outer']*1000:.2f}")
print(f"Winding width w0 (mm): {res['w']*1000:.2f}")
print(f"Winding height h0 (mm): {res['h']*1000:.2f}")