import stochpy
import numpy as np
import matplotlib.pyplot as plt

# Initialize SSA Model
smod = stochpy.SSA()
smod.Model('chain3')  # Load the stochastic model

# Simulation Parameters
lag_vals = np.arange(0, 25, 1)  # Array of lag values
n_sim = 10000  # Number of simulations
sim_time = 5.2  # End time for each simulation

# Initialize Arrays for Species Counts
E1_init = np.zeros(n_sim)  # Initial species count for E1
ES_init = np.zeros(n_sim)  # Initial species count for ES
E2_init = np.zeros(n_sim)  # Initial species count for E2

E1_lags = np.zeros((len(lag_vals), n_sim))  # Species count for E1 over lags
ES_lags = np.zeros((len(lag_vals), n_sim))  # Species count for ES over lags
E2_lags = np.zeros((len(lag_vals), n_sim))  # Species count for E2 over lags

# Array to store time values
time_values = np.zeros(len(lag_vals))

# Run Simulations
for j in range(n_sim):
    smod.DoStochSim(end = sim_time, mode='time')  # Perform stochastic simulation
#    ds = smod.data_stochsim.getSimData('Time', 'E1', 'ES', 'E2', 'P')  # Retrieve simulation data
    ds = smod.data_stochsim.getSimData('Time', 'E', 'ES', 'ES2')  # Retrieve simulation data
    
    # Process data for each lag value
    for l in range(len(lag_vals)+1):
        dsb = ds[:, 0] > (l * 0.45 * sim_time) / len(lag_vals) + 0.5 * sim_time
        dstemp = ds[dsb]
        
        if l > 0:
            time_values[l -1] = dstemp[0, 0]  # Store time value
            E1_lags[l - 1, j] = dstemp[0, 1]
            ES_lags[l - 1, j] = dstemp[0, 2]
            E2_lags[l - 1, j] = dstemp[0, 3]
        else:
            E1_init[j] = dstemp[0, 1]
            ES_init[j] = dstemp[0, 2]
            E2_init[j] = dstemp[0, 3]
    
    print("Simulation "+str(j)+" completed.")
 
# Compute Cross-Correlation Coefficients
corr_mat_E1_ES = np.corrcoef(ES_lags,E1_init)
corr_E1_ES = corr_mat_E1_ES[:-1, len(lag_vals)]
line_E1_ES, = plt.plot(lag_vals, corr_E1_ES, 'r-')

corr_mat_ES_E1 = np.corrcoef(E1_lags,ES_init)
corr_ES_E1 = corr_mat_ES_E1[:-1, len(lag_vals)]
line_ES_E1, = plt.plot(lag_vals, corr_ES_E1, 'r^')

corr_mat_E2_ES = np.corrcoef(E2_lags, ES_init)
corr_E2_ES = corr_mat_E2_ES[:-1, len(lag_vals)]
line_E2_ES, = plt.plot(lag_vals, corr_E2_ES, 'b-')

corr_mat_ES_E2 = np.corrcoef(ES_lags, E2_init)
corr_ES_E2 = corr_mat_ES_E2[:-1, len(lag_vals)]
line_ES_E2, = plt.plot(lag_vals, corr_ES_E2, 'bo')

corr_mat_E1_E2 = np.corrcoef(E1_lags, E2_init)
corr_E1_E2 = corr_mat_E1_E2[:-1, len(lag_vals)]
line_E1_E2, = plt.plot(lag_vals, corr_E1_E2, 'g-')

corr_mat_E2_E1 = np.corrcoef(E2_lags, E1_init)
corr_E2_E1 = corr_mat_E2_E1[:-1, len(lag_vals)]
line_E2_E1, = plt.plot(lag_vals, corr_E2_E1, 'gs')

# Set plot labels and legend
line_E1_ES.set_label('Cov(E1(p) ES(p+q))')
line_ES_E1.set_label('Cov(ES(p) E1(p+q))')
line_E2_ES.set_label('Cov(ES(p) E2(p+q))')
line_ES_E2.set_label('Cov(E2(p) ES(p+q))')
line_E1_E2.set_label('Cov(E2(p) E1(p+q))')
line_E2_E1.set_label('Cov(E1(p) E2(p+q))')

plt.legend(loc=0)
plt.xlabel('Lags')
plt.ylabel('CrossCorrelation coefficient')
plt.show()
