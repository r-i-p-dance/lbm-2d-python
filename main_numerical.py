from lbm.src.plot.convergence import plot_convergence
from lbm.src.study.grid_convergence import run_grid_convergence_study

from lbm.src.plot.animation import Recorder
from lbm.src.plot.animation import record_development

# Cases
from lbm.src.cases.two_rectangles import TwoRectangles
from lbm.src.cases.backward_step import BackwardStep
from lbm.src.cases.cylinder import Cylinder


RESOLUTIONS = [32, 64, 128, 256]
Re          = 10.0
lbm_tau     = 0.6
tol         = 1e-8

# Run the study
for case in [TwoRectangles, BackwardStep, Cylinder]:
    print(f"\nRunning grid convergence study for {case.__name__} at Re={Re}…")
    results = run_grid_convergence_study(case, RESOLUTIONS, Re=Re, lbm_tau=lbm_tau, tol=tol)

    L2_errors = []

    # Produce the reports
    for i, r in enumerate(results):
        L2_errors.append(r.L2_error)

        plot_convergence(RESOLUTIONS[:i+1], L2_errors, save_path=f"results/plots/{case.__name__}/convergence_{case.__name__}_Re{int(Re)}.png")

Re = 20.0
RESOLUTIONS = [128]
vmax_factors = [4.0, 0.8, 0.4]

for i, case in enumerate([TwoRectangles, BackwardStep, Cylinder]):
    for r in RESOLUTIONS:

        record_development(case, nx=5*r, ny=r, Re=Re, tau_lbm=0.6, max_steps=100_000, every=100, 
                           path=f"results/animations/{case.__name__}/anim_Ny{r}_{case.__name__}_Re{int(Re)}.gif",
                           vmax_factor=vmax_factors[i], cmap='RdBu_r', interpolation='bilinear', 
                           print_progress=False)

