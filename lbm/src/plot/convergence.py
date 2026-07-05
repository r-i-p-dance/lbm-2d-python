import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(Ny_values, L2_errors, save_path):
    """
    Plot the convergence of the L2 error with respect to grid resolution.

    Parameters:
    Ny_values (list): List of grid resolutions in the y-direction.
    L2_errors (list): Corresponding L2 errors for each resolution.
    save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(8, 6))
    plt.loglog(Ny_values, L2_errors, 'o', markersize=6)
    
    plt.xlabel('Grid Resolution (Ny)', fontsize=14)
    plt.ylabel('L2 Error', fontsize=14)
    plt.title('Method Convergence Rate Study', fontsize=16)
    plt.grid(True, which="both", ls="--")
    
    # Fit a line to the log-log data to estimate the convergence rate
    coeffs = np.polyfit(np.log(Ny_values), np.log(L2_errors), 1)
    convergence_rate = coeffs[0]
    intercept = coeffs[1]
    
    # Calculate the y-values for the fitted line: y = exp(c) * x^m
    fitted_errors = np.exp(intercept) * np.array(Ny_values)**convergence_rate
    
    # Plot the fitted line as a dashed line
    plt.loglog(Ny_values, fitted_errors, '--', color='tab:orange', linewidth=2, label='Fitted Line')
    
    # Annotate the convergence rate on the plot
    plt.annotate(f'Convergence Rate: {convergence_rate:.2f}', 
                 xy=(0.5, 0.9), xycoords='axes fraction', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    
    plt.savefig(save_path, dpi=300)
    plt.close()