import mc_log_ui
import numpy as np
import matplotlib.pyplot as plt

class LogPlotter:
    def __init__(self, log):
        self.log = log
        self.freq = 1000 # Hz
        eps_data_sec = 0.002 # sec # time epsilon for data filtering
        self.eps_data = self.convert_time_to_index(eps_data_sec) # index to remove data before and after the desired time
        
        # Values taken from mc_rtc
        joint_limit_4 = [-2.45, 2.45]     # Joint 4 limits
        ds_percentage = 0.01              # Safety distance percentage
        di_percentage = 0.1               # Influence distance percentage
        self.ds = [joint_limit_4[0] * -ds_percentage + joint_limit_4[0] , joint_limit_4[1] * -ds_percentage + joint_limit_4[1]] # Safety distance  
        self.di = [joint_limit_4[0] * -di_percentage + joint_limit_4[0] , joint_limit_4[1] * -di_percentage + joint_limit_4[1]] # Influence distance
        
        self.q_in = log['qIn_3'] # joint 4 position measurement
        self.q_out = log['qOut_3'] # joint 4 position command
        self.alpha_in = log['alphaIn_3'] # joint 4 velocity measurement
        self.alpha_out = log['alphaOut_3'] # joint 4 velocity command
        self.tau_in = log['tauIn_3'] # joint 4 torque measurement
        self.tau_out = log['tauOut_3'] # joint 4 torque command
        # executor contains the different states of the log: ['', 'RALExpController_VelLimitEF', 'RALExpController_VelLimitNoEF', 'RALExpController_VelLimitPose']
        self.executor = log['Executor_RALExp_FSM_DropExp']
    
    def convert_time_to_index(self, time: float) -> int:
        return int(time * self.freq)
    
    def find_data_in_state(self, data: np.ndarray, state: str, time_min: float, time_max: float):
        '''
        Find data in a specific state between time_min and time_max.
        '''
        idx_min = self.convert_time_to_index(time_min)
        idx_max = self.convert_time_to_index(time_max)
        
        # Select data in the state between time_min and time_max
        data_in_state = []
        for i in range(idx_min, idx_max):
            if self.executor[i] == state:
                data_in_state.append(data[i])
        
        # Remove data before and after the desired time
        data_in_state = data_in_state[self.eps_data:-self.eps_data]        
        return np.array(data_in_state)     

    def plot(self, time_ef_new_vd, time_ef_old_vd, time_no_ef_new_vd, time_no_ef_old_vd):
        # Find data in EF state
        new_vd_q_in_ef = self.find_data_in_state(self.q_in, 'RALExpController_VelLimitEF', time_ef_new_vd[0], time_ef_new_vd[1])
        old_vd_q_in_ef = self.find_data_in_state(self.q_in, 'RALExpController_VelLimitEF', time_ef_old_vd[0], time_ef_old_vd[1])
        new_vd_alpha_in_ef = self.find_data_in_state(self.alpha_in, 'RALExpController_VelLimitEF', time_ef_new_vd[0], time_ef_new_vd[1])
        old_vd_alpha_in_ef = self.find_data_in_state(self.alpha_in, 'RALExpController_VelLimitEF', time_ef_old_vd[0], time_ef_old_vd[1])
        new_vd_tau_in_ef = self.find_data_in_state(self.tau_in, 'RALExpController_VelLimitEF', time_ef_new_vd[0], time_ef_new_vd[1])
        old_vd_tau_in_ef = self.find_data_in_state(self.tau_in, 'RALExpController_VelLimitEF', time_ef_old_vd[0], time_ef_old_vd[1])
        new_vd_tau_in_ef_fft = np.fft.fft(new_vd_tau_in_ef)
        old_vd_tau_in_ef_fft = np.fft.fft(old_vd_tau_in_ef)
        
        new_vd_q_out_ef = self.find_data_in_state(self.q_out, 'RALExpController_VelLimitEF', time_ef_new_vd[0], time_ef_new_vd[1])
        old_vd_q_out_ef = self.find_data_in_state(self.q_out, 'RALExpController_VelLimitEF', time_ef_old_vd[0], time_ef_old_vd[1])
        new_vd_alpha_out_ef = self.find_data_in_state(self.alpha_out, 'RALExpController_VelLimitEF', time_ef_new_vd[0], time_ef_new_vd[1])
        old_vd_alpha_out_ef = self.find_data_in_state(self.alpha_out, 'RALExpController_VelLimitEF', time_ef_old_vd[0], time_ef_old_vd[1])
        new_vd_tau_out_ef = self.find_data_in_state(self.tau_out, 'RALExpController_VelLimitEF', time_ef_new_vd[0], time_ef_new_vd[1])
        old_vd_tau_out_ef = self.find_data_in_state(self.tau_out, 'RALExpController_VelLimitEF', time_ef_old_vd[0], time_ef_old_vd[1])
        new_vd_tau_out_ef_fft = np.fft.fft(new_vd_tau_out_ef)
        old_vd_tau_out_ef_fft = np.fft.fft(old_vd_tau_out_ef)
        
        # Find data in NoEF state
        new_vd_q_in_no_ef = self.find_data_in_state(self.q_in, 'RALExpController_VelLimitNoEF', time_no_ef_new_vd[0], time_no_ef_new_vd[1])
        old_vd_q_in_no_ef = self.find_data_in_state(self.q_in, 'RALExpController_VelLimitNoEF', time_no_ef_old_vd[0], time_no_ef_old_vd[1])
        new_vd_alpha_in_no_ef = self.find_data_in_state(self.alpha_in, 'RALExpController_VelLimitNoEF', time_no_ef_new_vd[0], time_no_ef_new_vd[1])
        old_vd_alpha_in_no_ef = self.find_data_in_state(self.alpha_in, 'RALExpController_VelLimitNoEF', time_no_ef_old_vd[0], time_no_ef_old_vd[1])
        new_vd_tau_in_no_ef = self.find_data_in_state(self.tau_in, 'RALExpController_VelLimitNoEF', time_no_ef_new_vd[0], time_no_ef_new_vd[1])
        old_vd_tau_in_no_ef = self.find_data_in_state(self.tau_in, 'RALExpController_VelLimitNoEF', time_no_ef_old_vd[0], time_no_ef_old_vd[1])
        new_vd_tau_in_no_ef_fft = np.fft.fft(new_vd_tau_in_no_ef)
        old_vd_tau_in_no_ef_fft = np.fft.fft(old_vd_tau_in_no_ef)
        
        new_vd_q_out_no_ef = self.find_data_in_state(self.q_out, 'RALExpController_VelLimitNoEF', time_no_ef_new_vd[0], time_no_ef_new_vd[1])
        old_vd_q_out_no_ef = self.find_data_in_state(self.q_out, 'RALExpController_VelLimitNoEF', time_no_ef_old_vd[0], time_no_ef_old_vd[1])
        new_vd_alpha_out_no_ef = self.find_data_in_state(self.alpha_out, 'RALExpController_VelLimitNoEF', time_no_ef_new_vd[0], time_no_ef_new_vd[1])
        old_vd_alpha_out_no_ef = self.find_data_in_state(self.alpha_out, 'RALExpController_VelLimitNoEF', time_no_ef_old_vd[0], time_no_ef_old_vd[1])
        new_vd_tau_out_no_ef = self.find_data_in_state(self.tau_out, 'RALExpController_VelLimitNoEF', time_no_ef_new_vd[0], time_no_ef_new_vd[1])
        old_vd_tau_out_no_ef = self.find_data_in_state(self.tau_out, 'RALExpController_VelLimitNoEF', time_no_ef_old_vd[0], time_no_ef_old_vd[1])
        new_vd_tau_out_no_ef_fft = np.fft.fft(new_vd_tau_out_no_ef)
        old_vd_tau_out_no_ef_fft = np.fft.fft(old_vd_tau_out_no_ef)
        
        
        plt.figure(figsize=(12, 8))
        
        # Plot q in EF state
        plt.subplot(4, 2, 1)
        time = np.arange(0, len(new_vd_q_in_ef)/self.freq, 1/self.freq)
        if len(time) != len(new_vd_q_in_ef):
            time = time[:len(new_vd_q_in_ef)]
        plt.plot(time, old_vd_q_in_ef, label='Distance e(t) with SoA velocity damper')
        plt.plot(time, new_vd_q_in_ef, label='Distance e(t) with closed-loop velocity damper')
        # plt.plot(time, old_vd_q_out_ef, label='Distance Command e(t) with SoA velocity damper')
        # plt.plot(time, new_vd_q_out_ef, label='Distance Command e(t) with closed-loop velocity damper')
        plt.axhline(y=self.ds[0], color='r', linestyle='--', label='Safety distance ds')
        plt.axhline(y=self.di[0], color='g', linestyle='--', label='Influence distance di')
        plt.title('Joint 4 position measurement in EF state')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [rad]')
        plt.legend()
        
        # Plot alpha in EF state
        plt.subplot(4, 2, 3)
        time = np.arange(0, len(new_vd_alpha_in_ef)/self.freq, 1/self.freq)
        if len(time) != len(new_vd_alpha_in_ef):
            time = time[:len(new_vd_alpha_in_ef)]
        plt.plot(time, old_vd_alpha_in_ef, label='Velocity e(t) with SoA velocity damper')
        plt.plot(time, new_vd_alpha_in_ef, label='Velocity e(t) with closed-loop velocity damper')
        # plt.plot(time, old_vd_alpha_out_ef, label='Velocity Command e(t) with SoA velocity damper')
        # plt.plot(time, new_vd_alpha_out_ef, label='Velocity Command e(t) with closed-loop velocity damper')
        plt.title('Joint 4 velocity measurement in EF state')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [rad/s]')
        plt.legend()
        
        # Plot tau in EF state
        plt.subplot(4, 2, 5)
        time = np.arange(0, len(new_vd_tau_in_ef)/self.freq, 1/self.freq)
        if len(time) != len(new_vd_tau_in_ef):
            time = time[:len(new_vd_tau_in_ef)]
        plt.plot(time, old_vd_tau_out_ef, label='Torque Command e(t) with SoA velocity damper')
        plt.plot(time, new_vd_tau_out_ef, label='Torque Command e(t) with closed-loop velocity damper')
        plt.plot(time, old_vd_tau_in_ef, label='Torque e(t) with SoA velocity damper')
        plt.plot(time, new_vd_tau_in_ef, label='Torque e(t) with closed-loop velocity damper')
        plt.title('Joint 4 torque measurement in EF state')
        plt.xlabel('Time [s]')
        plt.ylabel('Torque [Nm]')
        plt.legend()
        
        # Plot tau FFT in EF state
        plt.subplot(4, 2, 7)
        # Compute the frequency bins
        freq = np.fft.fftfreq(len(new_vd_tau_in_ef), 1/self.freq)
        # Filter out negative frequencies
        positive_freqs = freq > 0
        # Plot only positive frequencies and corresponding FFT magnitudes
        plt.plot(freq[positive_freqs], np.abs(old_vd_tau_out_ef_fft)[positive_freqs], label='Torque Command e(t) with SoA velocity damper')
        plt.plot(freq[positive_freqs], np.abs(new_vd_tau_out_ef_fft)[positive_freqs], label='Torque Command e(t) with closed-loop velocity damper')
        plt.plot(freq[positive_freqs], np.abs(old_vd_tau_in_ef_fft)[positive_freqs], label='Torque e(t) with SoA velocity damper')
        plt.plot(freq[positive_freqs], np.abs(new_vd_tau_in_ef_fft)[positive_freqs], label='Torque e(t) with closed-loop velocity damper')
        plt.title('Joint 4 torque measurement FFT in EF state')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Torque [Nm]')
        plt.legend()
        
        # Plot q in NoEF state
        plt.subplot(4, 2, 2)
        time = np.arange(0, len(new_vd_q_in_no_ef)/self.freq, 1/self.freq)
        if len(time) != len(new_vd_q_in_no_ef):
            time = time[:len(new_vd_q_in_no_ef)]
        plt.plot(time, old_vd_q_in_no_ef, label='Distance e(t) with SoA velocity damper')
        plt.plot(time, new_vd_q_in_no_ef, label='Distance e(t) with closed-loop velocity damper')
        #plt.plot(time, old_vd_q_out_no_ef, label='Distance Command e(t) with SoA velocity damper')
        #plt.plot(time, new_vd_q_out_no_ef, label='Distance Command e(t) with closed-loop velocity damper')
        plt.axhline(y=self.ds[0], color='r', linestyle='--', label='Safety distance ds')
        plt.axhline(y=self.di[0], color='g', linestyle='--', label='Influence distance di')
        plt.title('Joint 4 position measurement in NoEF state')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [rad]')
        plt.legend()
        
        # Plot alpha in NoEF state
        plt.subplot(4, 2, 4)
        time = np.arange(0, len(new_vd_alpha_in_no_ef)/self.freq, 1/self.freq)
        if len(time) != len(new_vd_alpha_in_no_ef):
            time = time[:len(new_vd_alpha_in_no_ef)]
        plt.plot(time, old_vd_alpha_in_no_ef, label='Velocity e(t) with SoA velocity damper')
        plt.plot(time, new_vd_alpha_in_no_ef, label='Velocity e(t) with closed-loop velocity damper')
        # plt.plot(time, old_vd_alpha_out_no_ef, label='Velocity Command e(t) with SoA velocity damper')
        # plt.plot(time, new_vd_alpha_out_no_ef, label='Velocity Command e(t) with closed-loop velocity damper')
        plt.title('Joint 4 velocity measurement in NoEF state')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [rad/s]')
        plt.legend()
        
        # Plot tau in NoEF state
        plt.subplot(4, 2, 6)
        time = np.arange(0, len(new_vd_tau_in_no_ef)/self.freq, 1/self.freq)
        if len(time) != len(new_vd_tau_in_no_ef):
            time = time[:len(new_vd_tau_in_no_ef)]
        plt.plot(time, old_vd_tau_out_no_ef, label='Torque Command e(t) with SoA velocity damper')
        plt.plot(time, new_vd_tau_out_no_ef, label='Torque Command e(t) with closed-loop velocity damper')
        plt.plot(time, old_vd_tau_in_no_ef, label='Torque e(t) with SoA velocity damper')
        plt.plot(time, new_vd_tau_in_no_ef, label='Torque e(t) with closed-loop velocity damper')
        plt.title('Joint 4 torque measurement in NoEF state')
        plt.xlabel('Time [s]')
        plt.ylabel('Torque [Nm]')
        plt.legend()
        
        # Plot tau FFT in NoEF state
        plt.subplot(4, 2, 8)
        freq = np.fft.fftfreq(len(new_vd_tau_in_no_ef), 1/self.freq)
        # Filter out negative frequencies
        positive_freqs = freq > 0
        # Plot only positive frequencies and corresponding FFT magnitudes
        plt.plot(freq[positive_freqs], np.abs(old_vd_tau_out_no_ef_fft)[positive_freqs], label='Torque Command e(t) with SoA velocity damper')
        plt.plot(freq[positive_freqs], np.abs(new_vd_tau_out_no_ef_fft)[positive_freqs], label='Torque Command e(t) with closed-loop velocity damper')
        plt.plot(freq[positive_freqs], np.abs(old_vd_tau_in_no_ef_fft)[positive_freqs], label='Torque e(t) with SoA velocity damper')
        plt.plot(freq[positive_freqs], np.abs(new_vd_tau_in_no_ef_fft)[positive_freqs], label='Torque e(t) with closed-loop velocity damper')
        plt.title('Joint 4 torque measurement FFT in NoEF state')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Torque [Nm]')
        plt.legend()
        
        plt.tight_layout()
        
        plt.show()
        
def main():
    log = mc_log_ui.read_log('mylog.bin')
    plotter = LogPlotter(log)
    # time_ef_new_vd = [start_time, end_time] for the closed-loop velocity damper in VelLimitEF state
    # time_ef_old_vd = [start_time, end_time] for the SoQ velocity damper in VelLimitEF state
    # time_no_ef_new_vd = [start_time, end_time] for the closed-loop velocity damper in VelLimitNoEF state
    # time_no_ef_old_vd = [start_time, end_time] for the SoQ velocity damper in VelLimitNoEF state
    # The time intervals are in second, you can give the intervals approximately (outside of the state VelLimitEF or VelLimitNoEF), based on the mc_log_ui x values. Then the data will be filtered based to save only the data in the desired state.
    plotter.plot(time_ef_new_vd=[65, 80], time_ef_old_vd=[40, 58], time_no_ef_new_vd=[139, 150], time_no_ef_old_vd=[155, 170])
    
if __name__ == '__main__':
    main()
