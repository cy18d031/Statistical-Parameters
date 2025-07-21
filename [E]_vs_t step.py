# Run the stochastic simulation in 'steps' mode
import stochpy
import numpy as np
import matplotlib.pyplot as plt

# Initialize the StochPy SSA model
smod = stochpy.SSA()
smod.Model('MM.psc')

# Simulation parameters
num_intervals = 25  # Number of time intervals
time_points = np.arange(0, num_intervals, 1)
num_simulations = 30  # Number of stochastic simulations
simulation_end_step = 200  # Steps for each simulation

# Arrays to store results: concentration of 'E', 'ES', and time points
enzyme_count = np.zeros((num_intervals, num_simulations))
complex_count = np.zeros((num_intervals, num_simulations))
time_stamps = np.zeros((num_intervals, num_simulations))

# Arrays to store average values for each time interval
avg_enzyme = np.zeros(num_intervals)
avg_complex = np.zeros(num_intervals)
avg_time_stamps = np.zeros(num_intervals)

# Run stochastic simulations for 'num_simulations' times
for sim_idx in range(num_simulations):
    # Set initial conditions and parameters for each simulation
    smod.ChangeInitialSpeciesCopyNumber('E', 40)
    smod.ChangeParameter('k1', 25)
    
    # Run the stochastic simulation in 'steps' mode
    smod.DoStochSim(end=simulation_end_step, mode='steps')
    simulation_data = smod.data_stochsim.getSimData('Time', 'E', 'ES')
    
    # Extract data for each time interval
    for interval_idx in range(num_intervals):
        # Index the relevant data from the simulation output
        data_point = simulation_data[(simulation_end_step / len(time_points)) * interval_idx + 1]
        time_stamps[interval_idx, sim_idx] = data_point[0]  # Time
        enzyme_count[interval_idx, sim_idx] = data_point[1]  # Concentration of 'E'
        complex_count[interval_idx, sim_idx] = data_point[2]  # Concentration of 'ES'
    
    # Print progress
    print("Simulation {}/{} completed.".format(sim_idx + 1, num_simulations))

# Calculate the average values for each time interval
for interval_idx in range(num_intervals):
    avg_enzyme[interval_idx] = np.mean(enzyme_count[interval_idx])
    avg_complex[interval_idx] = np.mean(complex_count[interval_idx])
    avg_time_stamps[interval_idx] = np.mean(time_stamps[interval_idx])

# Plot individual simulations and their average values
for sim_idx in range(num_simulations):
    plt.plot(time_stamps[:, sim_idx], enzyme_count[:, sim_idx], 'r-', linewidth=0.6, alpha=0.1)
    plt.plot(time_stamps[:, sim_idx], complex_count[:, sim_idx], 'b-', linewidth=0.6, alpha=0.1)

# Plot the average values of 'E' and 'ES' concentrations
plt.plot(avg_time_stamps, avg_enzyme, 'r-o', linewidth=3, label='[E]')
plt.plot(avg_time_stamps, avg_complex, 'b-s', linewidth=3, label='[ES]')

# Add labels and legend
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.xlim(0.0, 0.25)  # Time range for the plot

# Show the plot
plt.show()
