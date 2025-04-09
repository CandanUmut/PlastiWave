import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Constants and Parameters
# -----------------------------

# Vibrational frequency (cm^-1 to Hz conversion)
# 1 cm^-1 = 29.9792458 GHz
freq_CO_cm1 = 1150  # C–O bond
freq_CN_cm1 = 1250  # C–N bond
freq_CO_Hz = freq_CO_cm1 * 29.9792458e9
freq_CN_Hz = freq_CN_cm1 * 29.9792458e9

# Effective reduced mass (kg)
# approx carbon + oxygen and carbon + nitrogen system
mass_CO = 2.7e-26  # kg

# Time domain
duration = 0.1  # seconds
samples = 50000
time = np.linspace(0, duration, samples)
dt = time[1] - time[0]

# Amplitude envelope: start small, grow smoothly to 1 nm
base_amp = 5.0e-11  # 0.05 nm
max_amp = 1.0e-9    # 1 nm
envelope = np.linspace(base_amp, max_amp, samples)

# Frequency transition curve (smooth transition from C–O to C–N)
transition = np.linspace(0, 1, samples)
omega_CO = 2 * np.pi * freq_CO_Hz
omega_CN = 2 * np.pi * freq_CN_Hz

# -----------------------------
# 2. Resonant Shift Signal
# -----------------------------

# Adaptive frequency sweep: vibrate with gradually shifting frequency
vibration = np.sin((1 - transition) * omega_CO * time + transition * omega_CN * time)

# Apply amplitude envelope
oscillation = envelope * vibration

# -----------------------------
# 3. Derive Velocity and Energy
# -----------------------------

velocity = np.gradient(oscillation, dt)
kinetic_energy = 0.5 * mass_CO * velocity**2

# Extract simulation results
max_disp = np.max(np.abs(oscillation))
max_vel = np.max(np.abs(velocity))
max_energy = np.max(kinetic_energy)
bond_break_energy = 3.6e-19  # J

# -----------------------------
# 4. Output Results
# -----------------------------

print("==== Resonant Bond Shift Simulation ====")
print(f"Transition: C–O ({freq_CO_cm1} cm⁻¹) → C–N ({freq_CN_cm1} cm⁻¹)")
print(f"Max Displacement: {max_disp:.2e} m")
print(f"Max Velocity:     {max_vel:.2e} m/s")
print(f"Max Energy:       {max_energy:.2e} J")
print(f"Bond Breakage Threshold: {bond_break_energy:.2e} J")
print(f"Bond Broken?      {'YES' if max_energy > bond_break_energy else 'NO'}")
print(f"Resonant Shift Success? {'YES' if max_disp > 1e-10 and max_energy < bond_break_energy else 'LIKELY'}")

# -----------------------------
# 5. Optional: Visualization
# -----------------------------

plt.figure(figsize=(10, 5))
plt.plot(time, oscillation, label="Bond Oscillation")
plt.title("Vibrational Guidance: C–O → C–N")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
