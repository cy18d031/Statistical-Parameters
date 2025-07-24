# Fano Factor vs time(t)
import stochpy
import numpy as np
import matplotlib.pyplot as plt

# Initialize SSA Model
smod = stochpy.SSA()
smod.Model('MM')  # Load the model named 'MM'

# Simulation Parameters
n_steps = 1.400   # Time for the simulation
n_sim = 10000       # Number of Monte Carlo simulations
n_pts = 20        # Number of data points

t_max = 1.0
t_min = 1e-6

# Pre-allocate Arrays for Performance Optimization
time_vals = np.zeros(n_pts)  # Array to store time values
np_vals = np.zeros((n_pts, n_sim))  # Array to store "p" values for each simulation
np2_vals = np.zeros((n_pts, n_sim))  # Array to store squared "p" values for each simulation

# Initialize Time Values 
#time_vals = np.ceil(np.linspace(t_min, t_max, n_pts))               # Linear scaling for turnover numbers
time_vals = np.logspace(np.log10(t_min), np.log10(t_max), n_pts)  # Log scaling for turnover numbers

# Run Stochastic Simulations
for j in range(n_sim):
    smod.DoStochSim(end=n_steps, mode='time')  # Run the simulation
    ds = smod.data_stochsim.getSimData('Time', 'P', 'E', 'ES', 'E2')  # Retrieve simulation data
    
    # Extract and store the data
    for i in range(n_pts):
        dsb = ds[:, 0] >= time_vals[i]  # Mask for times greater than the current time value
        dstemp = ds[dsb]  # Filter the dataset
        
        np_vals[i, j] = dstemp[0, 1]  # Store "p" value (turnover)
        np2_vals[i,j] = dstemp[0, 1]**2  # Store squared "p" value

    print("Simulation "+str(j)+" completed.")

# Compute Mean and Fano Factor
mean_p = np.mean(np_vals, axis=1)  # Mean "p" per time value
mean_p2 = np.mean(np2_vals, axis=1)  # Mean squared "p"
fano_factor = (mean_p2 - mean_p**2) / mean_p  # Fano factor calculation

# Plot Results
plt.figure(figsize=(8, 6))
plt.plot(time_vals, fano_factor, 'b^', label="Fano Factor")
plt.xscale("log")  # Log scale for x-axis
plt.xlabel('Time')
plt.ylabel('Intensity (Fano Factor)')
plt.legend(loc='best')
plt.show()
