# Run the stochastic simulation in 'time' mode
import stochpy
import numpy as np
import matplotlib.pyplot as plt

# Initialize the StochPy model
smod = stochpy.SSA()
smod.Model('MM.psc')  # Load the model file

# Simulation parameters
num_intervals = 25           # Number of time intervals
num_simulations = 300    # Number of stochastic simulations/trajectories
simulation_end_time = 0.26  # Total simulation duration

# Initialize arrays to store results
time_intervals = np.arange(0,num_intervals, 1)
enzyme_counts = np.zeros((num_intervals, num_simulations))
complex_counts = np.zeros((num_intervals, num_simulations))
time_points = np.zeros((num_intervals, num_simulations))

# Averages for each interval
avg_enzyme = np.zeros(num_intervals)
avg_complex = np.zeros(num_intervals)
avg_time = np.zeros(num_intervals)

# Run stochastic simulations
for sim_idx in range(num_simulations):
    # Set initial conditions and parameters for the model
    smod.ChangeInitialSpeciesCopyNumber('E', 40)  # Initial enzyme count
    smod.ChangeParameter('k1', 25)  # Reaction rate constant

        # Run the stochastic simulation in 'time' mode
    smod.DoStochSim(end=simulation_end_time, mode='Time')
    sim_data = smod.data_stochsim.getSimData('Time', 'E', 'ES')

    # Extract data for each time interval
    for loop_idx in range(num_intervals):
        # Get data points within the current time interval
        time_filter = sim_data[:, 0] >= (simulation_end_time / len(time_intervals)) * loop_idx
        filtered_data = sim_data[time_filter]

        # Store the first data point of the interval
        time_points[loop_idx, sim_idx] = filtered_data[0, 0]
        enzyme_counts[loop_idx, sim_idx] = filtered_data[0, 1]
        complex_counts[loop_idx, sim_idx] = filtered_data[0, 2]

    # Print progress
    print("Simulation {}/{} completed.".format(sim_idx + 1, num_simulations))

# Calculate average values for each time interval
for loop_idx in range(num_intervals):
    avg_enzyme[loop_idx] = np.mean(enzyme_counts[loop_idx])
    avg_complex[loop_idx] = np.mean(complex_counts[loop_idx])
    avg_time[loop_idx] = np.mean(time_points[loop_idx])

# Plotting the results
for sim_idx in range(num_simulations):
    # Plot individual stochastic trajectories with low opacity
    plt.plot(time_points[:, sim_idx], enzyme_counts[:, sim_idx], 'r-', linewidth=0.6, alpha=0.1)
    plt.plot(time_points[:, sim_idx], complex_counts[:, sim_idx], 'b-', linewidth=0.6, alpha=0.1)

# Plot average trajectories
avg_enzyme_line, = plt.plot(avg_time, avg_enzyme, 'r-o', linewidth=3, label='[E] (Average)')
avg_complex_line, = plt.plot(avg_time, avg_complex, 'b-s', linewidth=3, label='[ES] (Average)')

# Add labels and legend
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.xlim(0.0, simulation_end_time)
plt.legend()
plt.title('Stochastic Simulation of Enzyme Kinetics')
plt.show()
