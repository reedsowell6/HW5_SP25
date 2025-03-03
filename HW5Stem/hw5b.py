# region imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sys
#endregion

# GLOBAL PARAMETERS (for water in English units)
# Typical kinematic viscosity of water around room temp
# (units: ft^2/s). Adjust as needed.
nu = 1.22e-5

# Acceleration due to gravity in ft/s^2
g = 32.174

# HELPER FUNCTIONS
def colebrook_equation(x, Re, eD):
    """
    Colebrook equation in terms of x = 1/sqrt(f).
    eD is the relative roughness, epsilon/D.
    """
    # Colebrook form:  x + 2.0 * log10( eD/3.7 + 2.51*x / Re ) = 0
    return x + 2.0 * np.log10(eD / 3.7 + (2.51 * x) / Re)


def friction_factor(Re, eD):
    """
    Compute the Darcy-Weisbach friction factor f given:
    - Reynolds number Re
    - Relative roughness eD = epsilon / D
    We handle laminar, transitional, and turbulent regimes:
      - Laminar: f = 64/Re
      - Transitional (2000 <= Re <= 4000):
          We do a linear interpolation between f_lam and f_cb
          plus some randomness from a normal distribution
      - Turbulent: solve Colebrook implicitly for f
    Returns: (f_value, regime)
      regime = 'laminar', 'transitional', or 'turbulent'
    """
    if Re < 2000:
        # Laminar
        return (64.0 / Re, "laminar")
    elif Re > 4000:
        # Turbulent -> solve Colebrook for x = 1/sqrt(f)
        x0 = 1.0 / np.sqrt(0.02)  # initial guess
        sol, = fsolve(colebrook_equation, x0, args=(Re, eD))
        f_val = 1.0 / (sol * sol)
        return (f_val, "turbulent")
    else:
        # Compute laminar friction factor at this Re
        f_lam = 64.0 / Re
        # Compute turbulent friction factor at this Re (like Re>4000 approach)
        x0 = 1.0 / np.sqrt(0.02)
        sol, = fsolve(colebrook_equation, x0, args=(Re, eD))
        f_cb = 1.0 / (sol * sol)

        # Interpolate linearly between f_lam and f_cb:
        # mu_f = f_lam + (f_cb - f_lam)*((Re - 2000)/2000)
        # Then add randomness from a normal distribution with std = 0.2*mu_f
        mu_f = f_lam + (f_cb - f_lam) * ((Re - 2000.0) / 2000.0)
        sigma_f = 0.2 * mu_f

        # Draw one random sample for f
        f_rand = np.random.normal(mu_f, sigma_f)

        # If the random value goes negative (rare), clamp it:
        f_val = max(f_rand, 1e-6)
        return (f_val, "transitional")


def head_loss_per_foot(f, V, D):
    """
    Returns the head loss per foot, hf/L, given:
      f: friction factor
      V: flow velocity (ft/s)
      D: pipe diameter (ft)
    Uses Darcy-Weisbach:
      hf = f * (L/D) * (V^2/(2*g))
    so:
      hf/L = f * (V^2/(2*g*D))
    """
    return f * (V ** 2) / (2.0 * g * D)



# Moody Diagram Base
def moody_diagram_base():
    """
    Plots a basic Moody diagram background (laminar, transitional, and
    a set of turbulent curves for different relative roughnesses).
    """
    # Creating a logspace of Re for laminar, transitional, and turbulent
    Re_lam = np.logspace(np.log10(600), np.log10(2000), 50)
    Re_trans = np.logspace(np.log10(2000), np.log10(4000), 50)
    Re_turb = np.logspace(np.log10(4000), np.log10(1e8), 200)

    # Laminar friction factor (f=64/Re) for the laminar range
    f_lam = 64.0 / Re_lam
    # Plot laminar
    plt.loglog(Re_lam, f_lam, 'b-', label='Laminar')

    # Transitional (just show dashed line from end of laminar to start of turbulent)
    f_trans = 64.0 / Re_trans  # purely laminar formula for illustration
    plt.loglog(Re_trans, f_trans, 'r--', label='Transitional (approx)')

    # A set of relative roughnesses for the turbulent region
    rrVals = [0.0, 1e-6, 5e-6, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 4e-3, 1e-2]

    # For each relative roughness, compute friction factor using Colebrook
    for rr in rrVals:
        f_vals = []
        for Re in Re_turb:
            # Solve Colebrook
            x0 = 1.0 / np.sqrt(0.02)
            sol, = fsolve(colebrook_equation, x0, args=(Re, rr))
            f_val = 1.0 / (sol * sol)
            f_vals.append(f_val)
        plt.loglog(Re_turb, f_vals, 'k-', linewidth=1)
        # Label the curve at the right end
        plt.text(Re_turb[-1] * 1.05, f_vals[-1], f"{rr:.1e}", fontsize=8)

    # Axes limits and labels
    plt.xlim(600, 1e8) # Set the x-axis limits from 600 to 100,000,000 (1e8)
    plt.ylim(0.008, 0.1) # Set the y-axis limits from 0.008 to 0.1
    plt.xlabel("Reynolds number (Re)", fontsize=12)  # Label the x-axis and set the font size to 12
    plt.ylabel("Friction factor (f)", fontsize=12)  # Label the y-axis and set the font size to 12
    plt.title("Moody Diagram (Basic)", fontsize=14)  # Set the plot title with a font size of 14
    plt.grid(which='both') # Enable grid lines for both major and minor ticks
    plt.legend(loc='upper right', fontsize=10)  # Display the legend at the upper right with font size 10


# Main Driver
def main():
    """
    Continuously prompt the user for pipe diameter, roughness, and flow rate,
    compute friction factor and head loss per foot, and plot on Moody diagram.
    """
    # Use an interactive approach: keep track of all (Re, f, marker)
    # so we can re-plot them each time
    user_points = []

    print("Welcome to the pipe-flow calculator with a Moody diagram!\n")
    print("Press Ctrl+C or close the plot to quit at any time.\n")

    while True:
        try:
            # Gather user input
            D_in = float(input("Pipe diameter (inches): "))  # Prompt user for the pipe diameter in inches and convert the input to a float
            e_mic = float(input("Pipe roughness (micro-inches): ")) # Prompt user for the pipe roughness in micro-inches and convert the input to a float
            Q_gpm = float(input("Flow rate (gallons per minute): ")) # Prompt user for the flow rate in gallons per minute and convert the input to a float

            # Convert to consistent units
            # Diameter in feet:
            D_ft = D_in / 12.0
            # Relative roughness e/D (dimensionless):
            # e in inches = e_mic * 1e-6, so e/D = (e_mic*1e-6)/D_in
            eD = (e_mic * 1e-6) / D_in

            # Compute velocity in ft/s
            # 1 gallon = 231 in^3 = 0.13368 ft^3
            # Q (ft^3/min) = Q_gpm * 0.13368
            # Convert to ft^3/s by dividing by 60
            Q_ft3_s = (Q_gpm * 0.13368) / 60.0

            # Cross-sectional area in ft^2
            area = np.pi * (D_ft ** 2) / 4.0
            if area < 1e-12:
                print("Error: Pipe diameter is too small or zero. Try again.\n")
                continue

            V = Q_ft3_s / area  # ft/s

            # Compute Reynolds number Re = (V * D) / nu
            Re = (V * D_ft) / nu

            # Compute friction factor (f)
            f, regime = friction_factor(Re, eD)

            # Compute head loss per foot (hf/L)
            hf_per_ft = f * (V ** 2) / (2.0 * g * D_ft)

            # Print results
            print(f"\nResults:")  # Print a newline and the heading "Results:" to start the output block
            print(f"  Reynolds number = {Re:,.2f}")  # Print the Reynolds number formatted with commas and 2 decimal places
            print(f"  Flow regime     = {regime}") # Print the flow regime (e.g., laminar, transitional, turbulent)
            print(f"  Friction factor = {f:.5f}")  # Print the friction factor formatted to 5 decimal places
            print(f"  Head loss/ft    = {hf_per_ft:.5f} ft/ft\n") # Print the head loss per foot formatted to 5 decimal places and add "ft/ft", ending with a newline

            # Decide marker for plotting
            #    Upward triangle if transitional, circle otherwise
            marker = '^' if regime == 'transitional' else 'o'

            # Store the point so we can re-plot
            user_points.append((Re, f, marker))

            # Plot the Moody diagram base, then plot all user points
            plt.figure(figsize=(9, 6))
            moody_diagram_base()

            # Plot each point
            for (x, y, mk) in user_points:
                plt.loglog(x, y, marker=mk, markersize=10,
                           markerfacecolor='none', markeredgecolor='red')

            plt.tight_layout()
            plt.show()

            # Optionally, ask user if they want to continue
            cont = input("Enter 'q' to quit, or any other key to continue: ")
            if cont.strip().lower() == 'q': # Check if the trimmed, lowercase input equals 'q
                print("Exiting program.")  # Inform the user that the program is exiting
                break # Break out of the loop to stop further execution

        except KeyboardInterrupt:  # Catch a KeyboardInterrupt
            print("\nUser interrupted. Exiting.")  # Inform the user that the interruption occurred
            sys.exit(0)  # Exit the program immediately with a success status

        except Exception as ex: # Catch any other exceptions that might occur
            print(f"An error occurred: {ex}") # Print out the error message
            print("Please try again.\n") # Prompt the user to try again

if __name__ == "__main__":
    main()

# endregion