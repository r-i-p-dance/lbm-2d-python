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
    
    plt.figure(figsize=(8, 6))
    plt.plot(y, u_numerical, 'o-', label='LBM', ms=3)
    plt.plot(y, u_analytical, '-', label='Analytical')
    
    plt.xlabel('Velocity (u)', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('Velocity Profile Comparison', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(y, u_numerical - u_analytical, 'o-', ms=3)
    plt.title('Residual (numerical - analytical)')
    
    plt.savefig(save_path, dpi=300)
    plt.close()