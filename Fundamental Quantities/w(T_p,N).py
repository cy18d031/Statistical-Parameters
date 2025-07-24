import stochpy
import numpy as np
import matplotlib.pyplot as plt

# Initialize Model
smod = stochpy.SSA()
smod.Model('MM.psc')

# Simulation Parameters
num_sim = 10000  # Total number of simulations
p_th = 5         # Threshold for P
retry_count = 0  # Counter for simulation retries
sim_time = 2.0  # End time for each simulation

# Pre-allocate Arrays
turn_time = np.zeros(num_sim)  # Stores turnover times

# Run Stochastic Simulations
for j in range(num_sim):
    while True:
        smod.DoStochSim(end=sim_time, mode='time')  # Run the stochastic simulation
        ds = smod.data_stochsim.getSimData('Time', 'P', 'E', 'ES')  # Retrieve simulation data

        # Check if the simulation meets the threshold condition
        if ds[-1, 1] > p_th:
            break  # Exit loop if P exceeds the threshold
        else:
            retry_count += 1
            print("Sim " + str(j) + ": Retrying (count=" + str(retry_count) + "), P = " + str(ds[-1, 1]))

    # Extract time when P equals the threshold
    dsb = ds[:, 1] == p_th
    dstemp = ds[dsb]

    if len(dstemp) > 0:
        turn_time[j] = dstemp[0, 0]  # Store Turnover-Time value
    
    print("Sim " + str(j) + " completed.")

mean_t = np.mean(turn_time)      # Compute Statistics

# Plot Histogram: Turnover-Time Distribution
plt.figure(figsize=(9, 6))
plt.hist(turn_time, bins=30, density=True, alpha=0.75, color='blue', edgecolor='black')

# Formatting and Labels
plt.title('Turnover-Time Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('Frequency Density', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(left=min(turn_time[turn_time > 0]))
# Show the plot
plt.show()

# Print the mean turnover time (converted to string and rounded to 4 decimal places)
print("Mean Turnover Time: " + str(round(mean_t, 4)))