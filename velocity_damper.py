import numpy as np
import matplotlib.pyplot as plt

# Define preferences
auto_xi = False     # Automatically calculate xi
use_lambda = False  # Calculate xi with lambda

# Define constants
xi = 10.5            # Damping coefficient
ds = 1.0            # Security distance
di = 10.0           # Influence distance
e_initial = 20.0    # Initial distance
t_max = 20.0        # Maximum time
dt = 0.01           # Time step
acc = -20.5          # Acceleration MUST BE NEGATIVE
lambda_ = 1.0       # Damping coefficient for lambda, only used if use_lambda is True
xi_off = 0.5        # Only used if auto_xi is True

# Flags
under_di = False    # Under influence distance
xi_flag = False     # Calculate xi one time only


if use_lambda:
    xi = lambda_*(di-ds)/4
    print(f"From Lambda={lambda_}, xi={xi}")


# Initialize arrays
time = np.arange(0, t_max, dt)
e = np.zeros_like(time)
e_dot = np.zeros_like(time)
e_ddot = np.zeros_like(time)
e_dot_constraint = np.zeros_like(time) # First order velocity damper, calculated to show it only

# Set initial conditions
e[0] = e_initial
e_ddot_constraint = 0.0

print("New implementation")
# Euler integration
for i in range(1, len(time)):

    # Calculate xi if auto_xi is True
    if under_di and xi_flag and auto_xi:
        xi = xi_off + (ds-di)/(e[i-1]-ds) * e_dot[i-1]
        xi_flag = False
        print(f'Auto xi: {xi}')


    if e[i-1] > di:
        e_ddot[i-1] = acc
        if under_di is True:
            under_di = False
            print(f"Not under influence distance at t={time[i]}")
        
    else:
        if under_di is False:
            under_di = True
            xi_flag = True
            print(f"Under influence distance at t={time[i]}")
        
        e_ddot_constraint = (-4 * xi * e_dot[i-1] / (di - ds) 
                    - 4 * xi**2 * e[i-1] / (di - ds)**2 
                    + (4 * xi**2 + 1) * ds / (di - ds)**2)

        if e_ddot_constraint > acc:
            e_ddot[i-1] = e_ddot_constraint
        else:
            e_ddot[i-1] = acc
    
    # Update e using Euler method 
    e[i] = e[i-1] + e_dot[i-1] * dt
    e_dot[i] = e_dot[i-1] + e_ddot[i-1] * dt 
    e_dot_constraint[i] = -xi * (e[i-1] - ds) / (di - ds)

# ---------------------------- Old implementation ----------------------------

print("Old implementation")

# Initialize arrays
e_old = np.zeros_like(time)
e_dot_old = np.zeros_like(time)
e_ddot_old = np.zeros_like(time)
e_dot_constraint_old = np.zeros_like(time) 

# Set initial conditions
e_old[0] = e_initial
e_ddot_old_constraint = 0.0

# Reset flags
under_di = False # Under influence distance
xi_flag = False  # Calculate xi one time only

# Euler integration for the old implementation
for i in range(1, len(time)):

    if under_di and xi_flag and auto_xi:
        xi = xi_off + (ds-di)/(e_old[i-1]-ds) * e_dot_old[i-1]
        xi_flag = False
        print(f'Auto xi: {xi}')

    if e_old[i-1] > di:
        e_ddot_old[i-1] = acc
        if under_di is True:
            under_di = False
            print(f"Not under influence distance at t={time[i]}")
    else:
        if under_di is False:
            under_di = True
            xi_flag = True
            print(f"Under influence distance at t={time[i]}")
        
        e_ddot_old_constraint = (-xi/dt)*((e_old[i-1] - ds)/(di - ds)) - e_dot_old[i-1]/dt
        # print(f"e_ddot_old_constraint={e_ddot_old_constraint} at t={time[i]}")
        
        if e_ddot_old_constraint > acc:
            e_ddot_old[i-1] = e_ddot_old_constraint
        else:
            e_ddot_old[i-1] = acc
    
    # Update e using Euler method 
    e_old[i] = e_old[i-1] + e_dot_old[i-1] * dt
    e_dot_old[i] = e_dot_old[i-1] + e_ddot_old[i-1] * dt
    e_dot_constraint_old[i] = -xi * (e_old[i-1] - ds) / (di - ds)

plt.figure(figsize=(12, 8))

if use_lambda:
    plt.suptitle(f"Second Order Velocity Damper comparison between new and old implementation with λ = {lambda_}")
else:
    plt.suptitle(f"Second Order Velocity Damper comparison between new and old implementation with ξ = {xi}")
# Plots for the new implementation (left column)

# Plot distance e(t)
plt.subplot(3, 2, 1)
plt.plot(time, e, label='Distance e(t)')
plt.axhline(y=ds, color='r', linestyle='--', label='Security distance ds')
plt.axhline(y=di, color='g', linestyle='--', label='Influence distance di')
plt.title('Distance Over Time (New Implementation)')
plt.xlabel('Time [s]')
plt.ylabel('Distance e(t)')
plt.legend()

# Plot acceleration e_ddot(t)
plt.subplot(3, 2, 3)
plt.plot(time, e_ddot, label="Acceleration $\ddot{e}(t)$", color='m')
plt.title('Acceleration Over Time (New Implementation)')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration $\ddot{e}(t)$')
plt.legend()

# Plot velocity e_dot(t)
plt.subplot(3, 2, 5)
plt.plot(time, e_dot, label="Velocity $\dot{e}(t)$", color='b')
plt.plot(time, e_dot_constraint, label="Velocity Constraint $\dot{e}_{constraint}(t)$", color='r')
plt.title('Velocity Over Time (New Implementation)')
plt.xlabel('Time [s]')
plt.ylabel('Velocity $\dot{e}(t)$')
plt.legend()

# Plots for the old implementation (right column)

# Plot distance old implementation e_old(t)
plt.subplot(3, 2, 2)
plt.plot(time, e_old, label='Distance Old Implementation e_old(t)')
plt.axhline(y=ds, color='r', linestyle='--', label='Security distance ds')
plt.axhline(y=di, color='g', linestyle='--', label='Influence distance di')
plt.title('Distance Over Time (Old Implementation)')
plt.xlabel('Time [s]')
plt.ylabel('Distance e_old(t)')
plt.legend()

# Plot acceleration old implementation e_ddot_old(t)
plt.subplot(3, 2, 4)
plt.plot(time, e_ddot_old, label="Acceleration Old Implementation $\ddot{e}_{old}(t)$", color='m')
plt.title('Acceleration Over Time (Old Implementation)')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration $\ddot{e}_{old}(t)$')
plt.legend()

# Plot velocity e_dot(t)
plt.subplot(3, 2, 6)
plt.plot(time, e_dot_old, label="Velocity Old Implementation $\dot{e}_{old}(t)$", color='b')
plt.plot(time, e_dot_constraint_old, label="Velocity Constraint Old Implementation $\dot{e}_{constraint_old}(t)$", color='r')
plt.title('Velocity Over Time (Old Implementation)')
plt.xlabel('Time [s]')
plt.ylabel('Velocity $\dot{e}(t)$')
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make space for suptitle

# Show the figure
plt.show()