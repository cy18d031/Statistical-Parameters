# Theta vs p
import stochpy
import numpy as np
import matplotlib.pyplot as plt

# Initialize Stochastic Model
smod = stochpy.SSA()
smod.Model('MM.psc')

# Simulation Parameters
num_sim = 100#0  # Number of simulations
retry_count = 0  # Count of re-simulations
num_steps = 2000  # Number of time steps
p_min = 1
p_max = 25
n_pts = 25

# Turnover Number (p) Initialization
#p_vals = np.ceil(np.linspace(p_min, p_max, n_pts))                      # Linear scaling for turnover numbers
p_vals = np.ceil(np.logspace(np.log10(p_min), np.log10(p_max), n_pts))  # Logarithmic scaling
p_vals[1:] = np.maximum(p_vals[1:], p_vals[:-1] + 1)                    # Ensure strictly increasing values

# Pre-allocate Arrays
turn_time = np.zeros((n_pts, num_sim))  # Turnover times
turn_vel  = np.zeros_like(p_vals)  # Mean turnover time per turnover

#Steady-State Velocity
Vss = 1.0   #Velocity at steady-state evaluated mathematically

# Run Simulations
for j in range(num_sim):
    while True:
        smod.DoStochSim(end=num_steps)  # Run stochastic simulation
        ds = smod.data_stochsim.getSimData('Time', 'P', 'E1', 'ES', 'E2')  # Retrieve data
        
        if ds[-1, 1] > p_vals[-1]:  # Check if final P count exceeds last turnover value
            break
        else:
            retry_count += 1
            print("Simulation "+str(j)+": Repeating simulation, count = "+str(retry_count)+", P = "+ str(ds[-1, 1]))
    
    # Extract turnover times for each turnover point
    for l in range(n_pts):
        dsb = ds[:, 1] == p_vals[l]  # Mask for exact turnover points
        dstemp = ds[dsb]  # Filter dataset
        
        if dstemp.size > 0:
            turn_time[l, j] = dstemp[0, 0]  # Store first turnover time

    print("Simulation "+str(j)+" completed.")

# Compute Mean Turnover Times
for l in range(n_pts):
    turn_vel[l] = p_vals[l] / np.mean(turn_time[l])  

theta = turn_vel/Vss  # θp : Ratio of mean velocity

# Plot Results
plt.figure(figsize=(8, 6))
plt.plot(p_vals, turn_vel, 'go', label='')
plt.xlabel('p', fontsize=12)
plt.ylabel('Theta p', fontsize=12)
plt.legend()
plt.title('Ratio of Mean-Velocity vs Turnover Number')
plt.show()
