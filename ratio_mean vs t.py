# Fano Factor vs Time (t)
import stochpy
import numpy as np
import matplotlib.pyplot as plt

# Initialize SSA Model
smod = stochpy.SSA()
smod.Model('MM')  # Load the model named 'MM'

# Simulation Parameters
sim_time = 1.4  # Total simulation time
n_sim = 100  # Number of simulations
n_pts = 20  # Number of time points

t_max, t_min = 1.0, 1e-6

# Pre-allocate Arrays
t_vals = np.zeros(n_pts)  # Time values
E1_vals = np.zeros((n_pts, n_sim))  # E counts
ES_vals = np.zeros((n_pts, n_sim))  # ES counts
E2_vals = np.zeros((n_pts, n_sim))  # E2 counts

# Initialize Time Values (Logarithmic Scale)
#time_vals = np.ceil(np.linspace(t_min, t_max, n_pts))               # Linear scaling for turnover numbers
t_vals = np.logspace(np.log10(t_min), np.log10(t_max), n_pts)        # Log scaling for turnover numbers

# Run Simulations
for j in range(n_sim):
    smod.DoStochSim(end=sim_time, mode='time')  # Run simulation
    ds = smod.data_stochsim.getSimData('Time', 'P', 'E', 'ES', 'E2')  # Retrieve data
    
    # Store species counts at different time points
    for l in range(n_pts):
        dsb = ds[:, 0] >= t_vals[l]  # Mask for time threshold
        dstemp = ds[dsb]  # Filter dataset
        
        E1_vals[l, l] = dstemp[0, 1]  # Store E count
        ES_vals[l, l] = dstemp[0, 2]  # Store ES count
        E2_vals[l, l] = dstemp[0, 3]  # Store E2 count
   
    print("Simulation "+str(j)+" completed.")


# Compute Mean Ratios
R_E1_ES = np.mean(E1_vals, axis=1)/ np.mean(ES_vals, axis=1)
R_ES_E2 = np.mean(ES_vals, axis=1)/ np.mean(E2_vals, axis=1)
R_E2_E1 = np.mean(E2_vals, axis=1)/ np.mean(E1_vals, axis=1)

# Plot Results
plt.figure(figsize=(8, 6))
plt.plot(t_vals, R_E1_ES, 'bo', label="E/ES Ratio")
plt.plot(t_vals, R_ES_E2, 'rs', label="ES/E2 Ratio")
plt.plot(t_vals, R_E2_E1, 'g^', label="E2/E Ratio")
plt.xscale("log")  # Log scale for x-axis
plt.xlabel('Time (log scale)', fontsize=12)
plt.ylabel('Mean Ratio', fontsize=12)
plt.title("Species Ratios Over Time", fontsize=14)
plt.legend(loc='best')
plt.show()
