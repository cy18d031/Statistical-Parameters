# Ranomness Parameter vs turnover(p)
import stochpy
import numpy as np
import matplotlib.pyplot as plt

# Initialize SSA Model
smod = stochpy.SSA()
smod.Model('MM')  # Load the model named 'MM'

# Simulation Parameters
n_sim = 1000     # Number of Monte Carlo simulations
n_steps = 3000  # Total simulation time steps
n_pts = 20      # Number of data points
p_max = 40      # Maximum turnover number
p_min = 1       # Minimum turnover number
retry_count = 0 # Invalid simulation counter

# Pre-allocate Arrays for Performance Optimization
times = np.zeros((n_pts, n_sim))    # Time values for each turnover number and simulation
times_sq = np.zeros((n_pts, n_sim)) # Squared time values for variance calculation
mean_t = np.zeros(n_pts)            # Mean time for each turnover number
mean_t_sq = np.zeros(n_pts)         # Mean squared time for each turnover number
rand_param = np.zeros(n_pts)        # Randomness parameter for each turnover number
turnovers = np.zeros(n_pts)         # Turnover number array

# Initialize Turnover Number (p)
turnovers = np.ceil(np.linspace(p_min, p_max, n_pts))                      # Linear scaling for turnover numbers
turnovers = np.ceil(np.logspace(np.log10(p_min), np.log10(p_max), n_pts))  # Log scaling for turnover numbers
turnovers[1:] = np.maximum(turnovers[1:], turnovers[:-1] + 1)  # Ensure p values are strictly increasing

# Run Stochastic Simulations
for j in range(n_sim):
    while True:
        smod.DoStochSim(end=n_steps)  # Perform SSA simulation
        ds = smod.data_stochsim.getSimData('Time', 'P', 'E1', 'ES', 'E2')  # Retrieve simulation data
        
        if ds[-1, 1] > turnovers[-1] + 1:  # Ensure valid simulation results
            break  # Exit loop when valid simulation is obtained
        else:
            retry_count += 1  # Track invalid simulations
            print("Simulation "+str(j)+": Repeating simulation, count = "+str(retry_count)+", P = "+ str(ds[-1, 1]))

    # Extract and Store Simulation Data for Each Turnover Number
    for l in range(len(turnovers)):
        dsb = ds[:, 1] == turnovers[l]      # Boolean mask for extracting data where P == turnovers[i]
        dstemp = ds[dsb]                    # Filtered data

        times[l, j]    = dstemp[0, 0]       # Capture time of first occurrence
        times_sq[l, j] = dstemp[0, 0] ** 2  # Store squared time value

    print("Simulation "+str(j)+" completed.")

# Compute Mean and Randomness Parameter
mean_t = np.mean(times, axis=1)  # Mean time per turnover value
mean_t_sq = np.mean(times_sq, axis=1)  # Mean squared time for each turnover value
rand_param = (turnovers * ((mean_t_sq - mean_t**2) / (mean_t**2)))  # Calculate randomness parameter

# Plot Results
plt.figure(figsize=(8, 5))
plt.plot(turnovers, rand_param, 'bo', label='Randomness Parameter')  # Scatter plot
plt.xlabel('Turnover Number (p)')
plt.ylabel('Randomness Parameter')
plt.title('Randomness Parameter vs. Turnover Number')
plt.legend()
plt.show()
