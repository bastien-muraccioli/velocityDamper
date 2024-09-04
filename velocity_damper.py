import numpy as np
import matplotlib.pyplot as plt

class VelocityDamper:
    def __init__(self, ds, di, e_initial, t_max, dt, acc, velocity_noise_std, velocity_noise_mean, e_dot_min):
        self.lambda_ = 0.0               # Our method gain
        self.xi = 0.0                    # SoA method gain
        self.M = 1.0                     # Constant of amortization margin
        self.ds = ds                    # Security distance
        self.di = di                    # Influence distance
        self.e_initial = e_initial             # Initial distance
        self.t_max = t_max                 # Maximum time
        self.dt = dt                    # Time step
        self.acc = acc                   # Acceleration MUST BE NEGATIVE
        self.velocity_noise_std = velocity_noise_std    # standard deviation of the Gaussian noise
        self.velocity_noise_mean = velocity_noise_mean   # mean of the Gaussian noise, assume unbiased noise
        self.e_dot_min = e_dot_min             # Velocity limit
        np.random.seed(42)
        self.noise = np.random.normal(self.velocity_noise_mean, self.velocity_noise_std, int(self.t_max/self.dt))

    def simulation(self, soa_method=False):
        # Initialize arrays
        time = np.arange(0, self.t_max, self.dt)
        e = np.zeros_like(time)
        e_dot = np.zeros_like(time)
        e_dot_state = np.zeros_like(time)
        e_ddot = np.zeros_like(time)
        e_dot_constraint = np.zeros_like(time) # First order velocity damper, calculated to show it only
        e_ddot_constraint_p = np.zeros_like(time) # Second order velocity damper, position constraint in acceleration
        e_ddot_constraint_v = np.zeros_like(time) # Second order velocity damper, velocity constraint in acceleration
        e_ddot_constraint = np.zeros_like(time) # Second order velocity damper constraint
        e_raw = np.zeros_like(time)
        e_dot_raw = np.zeros_like(time)
        lambda_save = self.lambda_
        if soa_method:
            if self.xi != 0.0:
                self.lambda_ = (self.xi*4*self.M**2/(self.dt*(self.di-self.ds)))**0.5
            else:
                print("Please provide xi, to use SoA method")
        
        # Set initial condition
        e[0] = self.e_initial
        e_raw[0] = self.e_initial
        
        # Simulation
        for t in range(1, len(time)):
            e_ddot_constraint_p[t-1] = -self.lambda_ * e_dot[t-1] - self.lambda_**2/(4 * self.M**2) * (e[t-1]-self.ds)
            e_ddot_constraint_v[t-1] = -self.lambda_ * (e_dot[t-1] - self.e_dot_min)
            e_ddot_constraint[t-1] = max(e_ddot_constraint_p[t-1], e_ddot_constraint_v[t-1])
            # if e[t-1] > self.di:
            #     e_ddot[t-1] = self.acc
            # else:
            if e_ddot_constraint[t-1] > self.acc:
                e_ddot[t-1] = e_ddot_constraint[t-1]
            else:
                e_ddot[t-1] = self.acc
            
            # Update e using Euler Integration
            e[t] = e[t-1] + e_dot[t-1] * self.dt
            e_dot_state[t] = e_dot_state[t-1] + e_ddot[t-1] * self.dt
            e_dot[t] = e_dot_state[t] + self.noise[t]
            e_dot_constraint[t] = -self.xi * (e[t-1] - self.ds) / (self.di - self.ds)
            e_raw[t] = e_raw[t-1] + e_dot_raw[t-1] * self.dt
            e_dot_raw[t] = e_dot_raw[t-1] + self.acc * self.dt  # + self.noise[t]
        self.lambda_ = lambda_save
        return time, e, e_dot, e_dot_state, e_ddot, e_dot_constraint, e_ddot_constraint_p, e_ddot_constraint_v, e_ddot_constraint, e_raw, 
            
    def plot_comparison_simulation(self, lambda_, M):
        self.lambda_ = lambda_
        self.M = M
        self.xi = (self.di-self.ds)/(4*self.dt*self.M**2) # Calculate xi for the SoA method
        
        time_new, e_new, e_dot_new, e_dot_real_new, e_ddot_new, e_dot_constraint_new, e_ddot_constraint_p_new, e_ddot_constraint_v_new, e_ddot_constraint_new, e_raw_new = self.simulation()
        time_soa, e_soa, e_dot_soa, e_dot_real_soa, e_ddot_soa, e_dot_constraint_soa, e_ddot_constraint_p_soa, e_ddot_constraint_v_soa, e_ddot_constraint_soa, e_raw_soa = self.simulation(soa_method=True)
        
        plt.figure(figsize=(12, 8))
        
        plt.suptitle(f"Second Order Velocity Damper comparison between new and SoA implementation with λ = {self.lambda_}, ξ = {self.xi} and M = {self.M}")
        
        # for the new implementation (left column)
        
        # Plot distance e(t)
        plt.subplot(3, 2, 1)
        plt.plot(time_new, e_new, label='Distance e(t)')
        plt.axhline(y=self.ds, color='r', linestyle='--', label='Security distance ds')
        plt.axhline(y=self.di, color='g', linestyle='--', label='Influence distance di')
        plt.title('Distance Over Time (New Implementation)')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance e(t)')
        plt.ylim(self.ds-1,self.e_initial+1)
        plt.legend()
        
        # Plot acceleration e_ddot(t)
        plt.subplot(3, 2, 3)
        plt.plot(time_new[:-1], e_ddot_constraint_v_new[:-1], label="Acceleration Constraint for the velocity $\ddot{e}_{cv}(t)$", color='g')
        plt.plot(time_new[:-1], e_ddot_constraint_p_new[:-1], label="Acceleration Constraint for the position $\ddot{e}_{cp}(t)$", color='y')
        plt.plot(time_new[:-1], e_ddot_constraint_new[:-1], label="Acceleration Constraint $\ddot{e}_{c}(t)$", color='r')
        plt.plot(time_new[:-1], e_ddot_new[:-1], label="Acceleration $\ddot{e}(t)$", color='m')
        plt.title('Acceleration Over Time (New Implementation)')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration $\ddot{e}(t)$')
        plt.ylim(min(e_ddot_new+e_ddot_soa)-5,max(e_ddot_new+e_ddot_soa)+1)
        plt.legend()
        
        # Plot velocity e_dot(t)
        plt.subplot(3, 2, 5)
        plt.plot(time_new[1:], e_dot_constraint_new[1:], label="Velocity Constraint $\dot{e}_{c}(t)$", color='r')
        plt.plot(time_new[1:], e_dot_new[1:], label="Velocity Measurement (Noised) $\dot{e}(t)$", color='b')
        plt.plot(time_new[1:], e_dot_real_new[1:], label="Velocity $\dot{e}_{real}(t)$", color='g')
        plt.title('Velocity Over Time (New Implementation)')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity $\dot{e}(t)$')
        plt.ylim(min(e_dot_new+e_dot_soa)-5,max(e_dot_new+e_dot_soa)+1)
        plt.legend()
        
        # for the soa implementation (right column)
        
        # Plot distance soa implementation e_soa(t)
        plt.subplot(3, 2, 2)
        plt.plot(time_soa, e_soa, label='Distance SoA Implementation e(t)')
        plt.axhline(y=self.ds, color='r', linestyle='--', label='Security distance ds')
        plt.axhline(y=self.di, color='g', linestyle='--', label='Influence distance di')
        plt.title('Distance Over Time (SoA Implementation)')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance e(t)')
        plt.ylim(self.ds-1,self.e_initial+1)
        plt.legend()
        
        # Plot acceleration soa implementation e_ddot_soa(t)
        plt.subplot(3, 2, 4)
        plt.plot(time_soa[:-1], e_ddot_constraint_soa[:-1], label="Acceleration Constraint SoA Implementation $\ddot{e}_{c}(t)$", color='r')
        plt.plot(time_soa[:-1], e_ddot_soa[:-1], label="Acceleration SoA Implementation $\ddot{e}(t)$", color='m')
        plt.title('Acceleration Over Time (SoA Implementation)')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration $\ddot{e}(t)')
        plt.ylim(min(e_ddot_new+e_ddot_soa)-5,max(e_ddot_new+e_ddot_soa)+1)
        plt.legend()
        
        # Plot velocity e_dot(t)
        plt.subplot(3, 2, 6)
        plt.plot(time_soa[1:], e_dot_constraint_soa[1:], label="Velocity Constraint SoA Implementation $\dot{e}_{c}(t)$", color='r')
        plt.plot(time_soa[1:], e_dot_soa[1:], label="Velocity SoA Measurement (Noised) $\dot{e}(t)$", color='b')
        plt.plot(time_soa[1:], e_dot_real_soa[1:], label="Velocity SoA Implementation $\dot{e}_{real}(t)$", color='g')
        plt.title('Velocity Over Time (SoA Implementation)')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity $\dot{e}(t)')
        plt.ylim(min(e_dot_new+e_dot_soa)-5,max(e_dot_new+e_dot_soa)+1)
        plt.legend()
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make space for suptitle
        
        # Show the figure
        plt.show()
        
    def plot_logspace_dt(self, lambda_min: float, nb_lambda: int):
        lambdas = np.logspace(np.log10(lambda_min), np.log10(1/self.dt), nb_lambda) # Generate points between 1 and 1/dt on a logarithmic scale
        print(f"lambdas: {lambdas}")
        self.M = 1.1
        e_new_table = []
        e_dot_new_table = []
        e_ddot_new_table = []
        for lambda_ in lambdas:
            self.lambda_ = lambda_
            
            # self.xi = self.lambda_*(self.di-self.ds)/(4*self.M**2)
            time_new, e_new, e_dot_new, e_dot_real_new, e_ddot_new, e_dot_constraint_new, e_ddot_constraint_p_new, e_ddot_constraint_v_new, e_ddot_constraint_new, e_raw = self.simulation()
            e_new_table.append(e_new), e_dot_new_table.append(e_dot_new), e_ddot_new_table.append(e_ddot_new)
        
        plt.figure(figsize=(12, 8))
        
        plt.suptitle(f"Second Order Velocity Damper comparison between new and SoA implementation with λ and ξ from {lambda_min} to 1/dt:{1/self.dt}, and M = {self.M}")
        
        # Plot distance e(t)
        plt.subplot(3, 1, 1)
        for e_new in e_new_table:
            plt.plot(time_new, e_new)
        plt.plot(time_new, e_raw, color='black', linestyle='--', label='Distance e(t) without VelocityDamper')
        plt.axhline(y=self.ds, color='r', linestyle='--', label='Security distance ds')
        plt.axhline(y=self.di, color='g', linestyle='--', label='Influence distance di')
        plt.title('Distance Over Time (New Implementation)')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance e(t)')
        plt.ylim(self.ds-1,self.e_initial+1)
        plt.legend()
        
        # Plot acceleration e_ddot(t)
        plt.subplot(3, 1, 2)
        for e_ddot_new in e_ddot_new_table:
            plt.plot(time_new[:-1], e_ddot_new[:-1])
        plt.title('Acceleration Over Time (New Implementation)')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration $\ddot{e}(t)$')
        # plt.ylim(-2,5)
        plt.legend()
        
        # Plot velocity e_dot(t)
        plt.subplot(3, 1, 3)
        i = 0
        for e_dot_new in e_dot_new_table:
            percent_value = lambdas[i]*self.dt*100
            plt.plot(time_new[1:], e_dot_new[1:], label=f"λ = {percent_value:.1f}% of 1/dt")
            i+=1
        plt.title('Velocity Over Time (New Implementation)')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity $\dot{e}(t)$')
        plt.legend()
        
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Show the figure
        plt.show()
        
        
        
 
def main():
    velocity_damper = VelocityDamper(ds=1.0, di=10.0, e_initial=20.0, t_max=20.0, dt=0.001, acc=-0.5, velocity_noise_std=0.02, velocity_noise_mean=0.0, e_dot_min=-3.0)
    velocity_damper.plot_logspace_dt(lambda_min=1.0, nb_lambda=10)
    #velocity_damper.plot_comparison_simulation(lambda_=1.0, M=1.1)
if __name__ == "__main__":
    main()