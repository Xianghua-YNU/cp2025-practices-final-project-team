import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Physical parameters setup
m = 150000  # Mass (kg) = 150 tons
P = 2e7     # Power (W)
k1 = 5.90e-4  # Linear resistance coefficient (kg/s)
k2 = 5.86    # Quadratic resistance coefficient (kg/m)
f0_maglev = 0  # Constant resistance for maglev train (N)
f0_train = 4000  # Constant resistance for conventional train (N)

# Define the differential equation
def train_model(v, f0):
    """Train motion differential equation"""
    if v < 1e-6:  # Avoid division by zero
        return (P - f0) / m
    return (P/v - f0 - k1*v - k2*v**2) / m

# 4th-order Runge-Kutta solver
def runge_kutta4(f, v0, t_start, t_end, dt, f0):
    """4th-order Runge-Kutta method for solving differential equations"""
    num_steps = int((t_end - t_start) / dt)
    t = np.zeros(num_steps)
    v = np.zeros(num_steps)
    
    t[0] = t_start
    v[0] = v0
    
    for i in range(num_steps - 1):
        k1_val = dt * f(v[i], f0)
        k2_val = dt * f(v[i] + k1_val/2, f0)
        k3_val = dt * f(v[i] + k2_val/2, f0)
        k4_val = dt * f(v[i] + k3_val, f0)
        
        v[i+1] = v[i] + (k1_val + 2*k2_val + 2*k3_val + k4_val)/6
        t[i+1] = t[i] + dt
        
    return t, v

# Calculate terminal velocity (steady-state solution)
def terminal_velocity(f0, P, k1, k2):
    """Calculate terminal velocity (when drag equals thrust)"""
    def equation(v):
        return f0*v + k1*v**2 + k2*v**3 - P
    
    # Solve the cubic equation numerically
    # Initial guess (approximation ignoring f0 and k1)
    v_guess = (P / k2)**(1/3)
    v_term = fsolve(equation, v_guess)[0]
    return v_term

# Calculate terminal velocities for both cases
v_term_maglev = terminal_velocity(f0_maglev, P, k1, k2)
v_term_train = terminal_velocity(f0_train, P, k1, k2)

print(f"Maglev terminal velocity: {v_term_maglev:.2f} m/s ({v_term_maglev*3.6:.2f} km/h)")
print(f"Conventional train terminal velocity: {v_term_train:.2f} m/s ({v_term_train*3.6:.2f} km/h)")

# Solve the differential equations numerically
t_start = 0      # Start time (s)
t_end = 1000     # End time (s)
dt = 0.1         # Time step (s)
v0 = 1e-19         # Initial velocity (m/s) - avoid division by zero

# Solve for maglev train (f0=0)
t_maglev, v_maglev = runge_kutta4(train_model, v0, t_start, t_end, dt, f0_maglev)

# Solve for conventional train (f0=4000)
t_train, v_train = runge_kutta4(train_model, v0, t_start, t_end, dt, f0_train)

# Create the plot
plt.figure(figsize=(14, 8))

# Plot v-t curves
plt.plot(t_maglev, v_maglev, 'b-', linewidth=2, label='Maglev Train (f0=0)')
plt.plot(t_train, v_train, 'r-', linewidth=2, label='Conventional Train (f0=4000N)')

# Plot terminal velocity reference lines
plt.axhline(y=v_term_maglev, color='b', linestyle='--', 
            label=f'Maglev terminal: {v_term_maglev:.2f} m/s ({v_term_maglev*3.6:.2f} km/h)')
plt.axhline(y=v_term_train, color='r', linestyle='--', 
            label=f'Conventional terminal: {v_term_train:.2f} m/s ({v_term_train*3.6:.2f} km/h)')

# Add legend and labels
plt.legend(fontsize=12, loc='lower right')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Velocity (m/s)', fontsize=14)
plt.title('Train Velocity vs Time for Different Resistance Models', fontsize=16)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xlim([0, 1000])
plt.ylim([0, 160])  # Set appropriate y-axis limits

# Add secondary Y-axis (km/h)
ax2 = plt.twinx()
y_ticks = np.arange(0, 300, 50)  # km/h range
ax2.set_yticks(y_ticks/3.6)  # Convert km/h to m/s for positioning
ax2.set_yticklabels([f'{y:.0f}' for y in y_ticks])
ax2.set_ylabel('Velocity (km/h)', fontsize=14)

# Add physical parameters information
param_text = (f"Parameters:\n"
              f"Mass m = {m/1000:.0f} tons\n"
              f"Power P = {P/1e6:.0f} MW\n"
              f"k₁ = {k1:.2e} kg/s\n"
              f"k₂ = {k2:.2f} kg/m")
plt.annotate(param_text, xy=(0.95, 0.15), xycoords='axes fraction', 
             ha='right', va='bottom', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()
