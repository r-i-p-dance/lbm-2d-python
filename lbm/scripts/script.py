import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Add it to sys.path so Python can find 'lbm'
sys.path.insert(0, root_dir)

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from lbm.src.core.lattice import *

plt.style.use('dark_background')
plt.ion()

ltc = lattice()

# --- Initialize visualization
fig, ax = plt.subplots(figsize=(10, 3))
vel_mag = np.zeros((ltc.nx, ltc.ny))
img = ax.imshow(vel_mag.T, cmap='RdYlBu', vmin=0.0, vmax=0.1)
title = ax.set_title("Time Step: 0")

old_ux = np.zeros(ltc.ny)

for step in range(ltc.Nt):
    ltc.macro()
    ltc.equilibrium()
    ltc.collision()
    ltc.bounce_back_obstacle()
    ltc.stream()

    if step % 50 == 0:
        vel_mag = np.sqrt(ltc.ux**2 + ltc.uy**2)
        vel_mag[ltc.obstacle] = 0.0 
        img.set_data(vel_mag.T)
        title.set_text(f"Time Step: {step}")
        plt.pause(0.01)


