import stochpy
import numpy as np
import matplotlib.pyplot as plt

# Initialize SSA Model
smod = stochpy.SSA()
smod.Model('MM')  # Load the model

# Simulation Parameters
n_lags = 10        # Number of lags for correlation calculation
n_sim = 1000       # Number of simulations
steps = 1000        # Number of steps per simulation
n_invalid = 0           # Repetition counter for invalid simulations

lag_vals = np.arange(0, n_lags, 1)  # Lag values
fwt = np.zeros(n_sim)  # First waiting time array
wt = np.zeros((n_lags, n_sim))  # Waiting time array

# Run Simulations
for j in range(n_sim):
    while True:
        smod.DoStochSim(end=steps)  # Run simulation
        ds = smod.data_stochsim.getSimData('P')  # Retrieve simulation data
        if ds[steps - 1, 1] > n_lags + 1:  # Check for valid simulation
            break
        else:
            n_invalid += 1  # Track invalid simulations
            print("Simulation "+str(j)+": Repeating simulation, count = "+str(n_invalid)+", P = "+ str(ds[-1, 1]))

    # Extract waiting times and first waiting time
    for i in range(len(lag_vals) + 1):
        dsb = ds[:, 1] == i + 1  # Mask to extract waiting times
        dstemp = ds[dsb]  # Filtered dataset
        if i < 1:
            fwt[j] = dstemp[0, 0]  # First waiting time
            rt = dstemp[0, 0]  # Reset reference time
        else:
            wt[i - 1, j] = dstemp[0, 0] - rt  # Compute waiting time difference
            rt = dstemp[0, 0]  # Update reference time

    print("Simulation "+str(j)+" completed.")

# Compute correlation coefficient between waiting times and first waiting time
correlation_Matrix = np.corrcoef(wt, fwt)
correlation = correlation_Matrix[:-1, n_lags]

# Plot results
plt.plot(lag_vals, correlation, 'go', label='Waiting Time Correlation')
plt.xlabel('Lag')
plt.ylabel('Correlation Coefficient')
plt.legend(loc='best')
# plt.ylim(-0.3, 0.1)  # Uncomment to set y-axis limits
plt.show()
