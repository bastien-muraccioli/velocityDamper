import numpy as np
import matplotlib.pyplot as plt

def velocity_damper(xi = 0.0, M = 1.0, lambda_ = 0.0):
    # Define constants
    #xi = 1.0            # Damping coefficient
    ds = 1.0            # Security distance
    di = 10.0           # Influence distance
    e_initial = 20.0    # Initial distance
    t_max = 20.0        # Maximum time
    dt = 0.01           # Time step
    acc = -0.5          # Acceleration MUST BE NEGATIVE
    velocity_noise_std = 0.01 # standard deviation of the Gaussian noise
    velocity_noise_mean = 0.0 # mean of the Gaussian noise, assume unbiased noise
    # M = 2.0             # Constant of amortization margin
    e_dot_min = -3.0    # Velocity limit


    # Initialize arrays
    time = np.arange(0, t_max, dt)
    e = np.zeros_like(time)
    e_dot = np.zeros_like(time)
    e_dot_real = np.zeros_like(time)
    e_ddot = np.zeros_like(time)
    e_dot_constraint = np.zeros_like(time) # First order velocity damper, calculated to show it only
    e_ddot_constraint_p = np.zeros_like(time) # Second order velocity damper, position constraint in acceleration
    e_ddot_constraint_v = np.zeros_like(time) # Second order velocity damper, velocity constraint in acceleration
    e_ddot_constraint = np.zeros_like(time) # Second order velocity damper constraint
    
    # Set initial conditions
    e[0] = e_initial

    print("New implementation")
    # Euler integration
    for i in range(1, len(time)):
        e_ddot_constraint_p[i-1] = -4 * M**2 * xi * e_dot[i-1] / ds - 4 * M**2 * xi**2 * (e[i-1]-ds) / ds**2
        e_ddot_constraint_v[i-1] = (-4 * M**2 * xi/ds) * (e_dot[i-1] - e_dot_min)
        e_ddot_constraint[i-1] = max(e_ddot_constraint_p[i-1], e_ddot_constraint_v[i-1])
        if e_ddot_constraint[i-1] > acc:
            e_ddot[i-1] = e_ddot_constraint[i-1]
        else:
            e_ddot[i-1] = acc
            
        
        # Update e using Euler method 
        e[i] = e[i-1] + e_dot[i-1] * dt
        noise = np.random.normal(velocity_noise_mean, velocity_noise_std)
        e_dot_real[i] = e_dot[i-1] + e_ddot[i-1] * dt
        e_dot[i] = e_dot_real[i] + noise # Add noise to the velocity like a measurement
        e_dot_constraint[i] = -(xi/ds) * (e[i-1] - ds)

    # ---------------------------- soa implementation ----------------------------
    print("soa implementation")

    # Initialize arrays
    xi_soa = (di-ds)*xi/ds
    print(f"xi_soa={xi_soa}")
    e_soa = np.zeros_like(time)
    e_dot_soa = np.zeros_like(time)
    e_dot_real_soa = np.zeros_like(time)
    e_ddot_soa = np.zeros_like(time)
    e_dot_constraint_soa = np.zeros_like(time) 
    e_ddot_soa_constraint = np.zeros_like(time)

    # Set initial conditions
    e_soa[0] = e_initial     

    # Euler integration for the soa implementation
    for i in range(1, len(time)):
        
        e_ddot_soa_constraint[i-1] = (-xi_soa/dt)*((e_soa[i-1] - ds)/(di - ds)) - e_dot_soa[i-1]/dt


        if e_soa[i-1] > di:
            e_ddot_soa[i-1] = acc

        else:         
            if e_ddot_soa_constraint[i-1] > acc:
                e_ddot_soa[i-1] = e_ddot_soa_constraint[i-1]
            else:
                e_ddot_soa[i-1] = acc
        
        # Update e using Euler method 
        e_soa[i] = e_soa[i-1] + e_dot_soa[i-1] * dt
        noise = np.random.normal(velocity_noise_mean, velocity_noise_std)
        e_dot_real_soa[i] = e_dot_soa[i-1] + e_ddot_soa[i-1] * dt
        e_dot_soa[i] = e_dot_real_soa[i] + noise # Add noise to the velocity
        e_dot_constraint_soa[i] = -xi_soa * (e_soa[i-1] - ds) / (di - ds)

    plt.figure(figsize=(12, 8))

    plt.suptitle(f"Second Order Velocity Damper comparison between new and SoA implementation with ξ = {xi} and M = {M}")
    # Plots for the new implementation (left column)

    # Plot distance e(t)
    plt.subplot(3, 2, 1)
    plt.plot(time, e, label='Distance e(t)')
    plt.axhline(y=ds, color='r', linestyle='--', label='Security distance ds')
    plt.axhline(y=di, color='g', linestyle='--', label='Influence distance di')
    plt.title('Distance Over Time (New Implementation)')
    plt.xlabel('Time [s]')
    plt.ylabel('Distance e(t)')
    plt.ylim(ds-1,e_initial+1)
    plt.legend()

    # Plot acceleration e_ddot(t)
    plt.subplot(3, 2, 3)
    plt.plot(time[:-1], e_ddot_constraint_v[:-1], label="Acceleration Constraint for the velocity $\ddot{e}_{cv}(t)$", color='g')
    plt.plot(time[:-1], e_ddot_constraint_p[:-1], label="Acceleration Constraint for the position $\ddot{e}_{cp}(t)$", color='y')
    plt.plot(time[:-1], e_ddot_constraint[:-1], label="Acceleration Constraint $\ddot{e}_{c}(t)$", color='r')
    plt.plot(time[:-1], e_ddot[:-1], label="Acceleration $\ddot{e}(t)$", color='m')
    plt.title('Acceleration Over Time (New Implementation)')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration $\ddot{e}(t)$')
    plt.ylim(min(e_ddot+e_ddot_soa)-5,max(e_ddot+e_ddot_soa)+1)
    plt.legend()

    # Plot velocity e_dot(t)
    plt.subplot(3, 2, 5)
    plt.plot(time[1:], e_dot_constraint[1:], label="Velocity Constraint $\dot{e}_{c}(t)$", color='r')
    plt.plot(time[1:], e_dot[1:], label="Velocity Measurement (Noised) $\dot{e}(t)$", color='b')
    plt.plot(time[1:], e_dot_real[1:], label="Velocity $\dot{e}_{real}(t)$", color='g')
    plt.title('Velocity Over Time (New Implementation)')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity $\dot{e}(t)$')
    plt.ylim(min(e_dot+e_dot_soa)-5,max(e_dot+e_dot_soa)+1)
    plt.legend()

    # Plots for the soa implementation (right column)

    # Plot distance soa implementation e_soa(t)
    plt.subplot(3, 2, 2)
    plt.plot(time, e_soa, label='Distance SoA Implementation e(t)')
    plt.axhline(y=ds, color='r', linestyle='--', label='Security distance ds')
    plt.axhline(y=di, color='g', linestyle='--', label='Influence distance di')
    plt.title('Distance Over Time (SoA Implementation)')
    plt.xlabel('Time [s]')
    plt.ylabel('Distance e(t)')
    plt.ylim(ds-1,e_initial+1)
    plt.legend()

    # Plot acceleration soa implementation e_ddot_soa(t)
    plt.subplot(3, 2, 4)
    plt.plot(time[:-1], e_ddot_soa_constraint[:-1], label="Acceleration Constraint SoA Implementation $\ddot{e}_{c}(t)$", color='r')
    plt.plot(time[:-1], e_ddot_soa[:-1], label="Acceleration SoA Implementation $\ddot{e}(t)$", color='m')
    plt.title('Acceleration Over Time (SoA Implementation)')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration $\ddot{e}(t)$')
    plt.ylim(min(e_ddot+e_ddot_soa)-5,max(e_ddot+e_ddot_soa)+1)
    plt.legend()

    # Plot velocity e_dot(t)
    plt.subplot(3, 2, 6)
    plt.plot(time[1:], e_dot_constraint_soa[1:], label="Velocity Constraint SoA Implementation $\dot{e}_{c}(t)$", color='r')
    plt.plot(time[1:], e_dot_soa[1:], label="Velocity SoA Measurement (Noised) $\dot{e}(t)$", color='b')
    plt.plot(time[1:], e_dot_real_soa[1:], label="Velocity SoA Implementation $\dot{e}_{real}(t)$", color='g')
    plt.title('Velocity Over Time (SoA Implementation)')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity $\dot{e}(t)$')
    plt.ylim(min(e_dot+e_dot_soa)-5,max(e_dot+e_dot_soa)+1)
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make space for suptitle

    # Show the figure
    # if xi == 10.0:
    #     plt.show()
    plt.show()

    # Save the figure in plot fsoaer
    # plt.savefig(f'plot/noise_velocity_damper_xi{xi}.png')
    
def main():
    test_xi = [0.5, 1.0, 2.0, 5.0] #, 25.0, 30.0]
    test_M = [2.0]#, 2.0, 5.0] 
    for xi in test_xi:
        for M in test_M:
            velocity_damper(xi=xi, M=M)
if __name__ == "__main__":
    main()
        