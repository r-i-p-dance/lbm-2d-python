import numpy as np
import matplotlib.pyplot as plt

def plot_profile_comparison(u_numerical, u_analytical, Ny, save_path):
    """
    Plot the comparison between numerical and analytical velocity profiles.

    Parameters:
    u_numerical (numpy.ndarray): The numerical velocity profile (1D array).
    u_analytical (numpy.ndarray): The analytical velocity profile (1D array).
    Ny (int): Number of grid points in the y-direction.
    save_path (str): Path to save the plot image.
    """
    y = np.arange(1, Ny-1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    
    # =========================================================================
    # Subplot 1: Velocity Profiles
    # =========================================================================
    ax1.plot(y, u_numerical, 'o-', label='LBM', ms=3, linewidth = 2)
    ax1.plot(y, u_analytical, '-', label='Analytical', linewidth = 2)
    
    ax1.set_xlabel('y', fontsize=14)
    ax1.set_ylabel('Velocity', fontsize=14)
    ax1.set_title('Velocity Profile Comparison', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True)
    ax1.set_box_aspect(1)  # Forces the plot area to be perfectly square

    # =========================================================================
    # Subplot 2: Residuals
    # =========================================================================
    ax2.plot(y, u_numerical - u_analytical, 'o-', ms=3, color='tab:red', linewidth = 2)
    
    ax2.set_xlabel('y', fontsize=14)
    ax2.set_ylabel('Error', fontsize=14)
    ax2.set_title('Residual (Numerical - Analytical)', fontsize=16)
    ax2.grid(True)
    ax2.set_box_aspect(1)  # Forces the plot area to be perfectly square
    
    fig.tight_layout() 
    
    fig.savefig(save_path, dpi=300)
    plt.close(fig)