# region imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#endregion

# Define constants (example values—adjust to your problem)
M = 1.473  # kg (mass of the spool)
c = 100.0  # N·s/m (damping)
K = 200.0  # N/m (spring constant)
p1 = 70000.0  # Pa (inlet pressure, constant at 70 kPa)
A = 0.01  # m^2 (effective piston area, example)
v = 0.002  # m^3 (fluid volume behind the spool, example)

# Define the ODE system in first-order form
def valve_system(t, X):
    """
    X = [ x, xdot, p2 ]
    Returns [ dx/dt, d^2x/dt^2, dp2/dt ].
    """
    x, xdot, p2 = X

    # dx/dt = xdot
    dx_dt = xdot

    # d^2x/dt^2 = [ (p1 - p2)*A - c*xdot ] / M
    xddot = ((p1 - p2) * A - c * xdot) / M

    # dp2/dt = [ K*x - p2*A ] / v
    dp2_dt = (K * x - p2 * A) / v

    return [dx_dt, xddot, dp2_dt]


# Initial conditions and integration
x0 = 0.0  # initial displacement (m)
xdot0 = 0.0  # initial velocity (m/s)
p2_0 = 0.0  # initial downstream pressure (Pa), or whatever your problem states

X0 = [x0, xdot0, p2_0]

t_start = 0.0
t_end = 2.0  # seconds
t_eval = np.linspace(t_start, t_end, 200)  # points at which we want the solution

# Solve the system using solve_ivp
sol = solve_ivp(valve_system, [t_start, t_end], X0, t_eval=t_eval)

# Extract solutions
t = sol.t
x = sol.y[0, :]
xdot = sol.y[1, :]
p2 = sol.y[2, :]

# We can also compute p2dot if we like, from the same formula:
p2dot = (K * x - p2 * A) / v


# Plot results
plt.figure(figsize=(10, 8))

# (a) x(t) and p2(t)
plt.subplot(2, 1, 1)  # Create a 2-row by 1-column figure and select the first subplot
plt.plot(t, x, label='x(t) [m]') # Plot spool displacement x(t) on the current subplot, labeling the line
plt.plot(t, p2, label='p2(t) [Pa]')  # Plot downstream pressure p2(t) on the same subplot, labeling the line
plt.title('Spool displacement and downstream pressure vs. time') # Set the title for the first subplot
plt.xlabel('Time [s]')  # Label the x-axis as time in seconds
plt.grid(True) # Turn on the grid for better readability
plt.legend() # Show the legend for the plotted lines

# (b) xdot(t) and p2dot(t)
plt.subplot(2, 1, 2)  # Create/select the second subplot in the same 2x1 figure layout
plt.plot(t, xdot, label='xdot(t) [m/s]') # Plot spool velocity xdot(t) on this subplot, labeling the line
plt.plot(t, p2dot, label='p2dot(t) [Pa/s]') # Plot rate of change of downstream pressure p2dot(t) on the same subplot
plt.title('Spool velocity and rate of change of downstream pressure vs. time') # Title for the second subplot
plt.xlabel('Time [s]')  # Label the x-axis as time in seconds
plt.grid(True)  # Turn on the grid
plt.legend()  # Show the legend for these plotted lines

plt.tight_layout()
plt.show()

#endregion