import numpy as np

def poiseuille_from_force(Ny, g_x, nu):
    """
    Compute the analytical Poiseuille flow profile for a given number of grid points in the y-direction (Ny),
    body force in the x-direction (g_x), and kinematic viscosity (nu).
    
    Parameters:
    Ny (int): Number of grid points in the y-direction.
    g_x (float): Body force in the x-direction.
    nu (float): Kinematic viscosity.
    
    Returns:
    numpy.ndarray: The analytical Poiseuille flow profile as a 1D array of length Ny.
    """
    y = np.arange(1, Ny-1)
    u_profile = (g_x/(2*nu)) * (y - 0.5) * (Ny - 1.5 - y)
    
    return u_profile
