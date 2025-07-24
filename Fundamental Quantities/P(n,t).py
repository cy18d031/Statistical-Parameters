import stochpy
import numpy as np
import matplotlib.pyplot as plt

# Initialize SSA Model
smod = stochpy.SSA()
smod.Model('MM.psc')  # Load the stochastic model

# Simulation Parameters
n_sim = 10000  # Total number of simulations
t_th = 0.02       # Time at which measurements are taken
sim_time = 0.3             # Simulation end time
retry_count = 0      # Counter for discarded simulations

# List of species to track (modifiable)
sp_list = ['E', 'ES' ]
num_sp = len(sp_list)

# Pre-allocate Arrays to Store Results
sp_data = np.zeros((n_sim, num_sp))  # Stores species counts + P values

# Run Stochastic Simulations
for j in range(n_sim):
    while True:
        smod.DoStochSim(end=sim_time, mode='time')  # Run stochastic simulation
        ds = smod.data_stochsim.getSimData('Time', 'E', 'ES', 'P')  # Retrieve time-series data

        # Ensure valid simulation by checking if final P value exceeds threshold
        if (ds[len(ds)-1, 0] > t_th):
            break  
        retry_count += 1  # Track invalid runs

    # Extract time when P equals the threshold
    dsb = ds[:, 0] >= t_th
    dstemp = ds[dsb]
    
    # Store extracted species data
    sp_data[j, :] = dstemp[0, 1:-1]  # Species values
    print("Simulation "+str(j)+" completed.")


# Compute Mean and Standard Deviation for Each Species
sp_mean = np.mean(sp_data, axis=0)

# Plot Histograms with Normal Distribution Fit
for l in range(len(sp_list)):
    plt.figure(figsize=(8, 5))
    plt.hist(sp_data[:, l], bins= 10, density=True, alpha=0.7, color='blue', edgecolor='black')

    # Formatting the Plot
    plt.title('Distribution of '+sp_list[l], fontsize=14, fontweight='bold')
    plt.xlabel(sp_list[l], fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.xlim(-0.55,50)
    plt.show()

# Print Mean and Standard Deviation for Each Species
for l in range(len(sp_list)):
    species_name = sp_list[l] 
    print("Mean " + species_name + ": " + str(round(sp_mean[l], 4)))

