import numpy as np

from lbm.src.plot.convergence import plot_convergence
from lbm.src.plot.profile import plot_profile_comparison
from lbm.src.cases.forced import ForcedPoiseuille
from lbm.src.study.convergence import run_convergence_study


CASES = [ForcedPoiseuille]
RESOLUTIONS = [16, 32, 64, 128]
Re          = 1.0
tau_lbm     = 0.933
tol         = 1e-12

# Run the study
for case in CASES:
    print(f"\nRunning grid convergence study for {case.__name__} at tol={tol}, tau={tau_lbm}, Re={Re}…")
    results = run_convergence_study(case, RESOLUTIONS,
                                         Re=Re, tau_lbm=tau_lbm, tol=tol)

    Ny_values, L2_errors = [], []

    # Produce the reports
    for r in results:
        Ny_values.append(r.Ny)
        L2_errors.append(r.L2_error)

        plot_profile_comparison(r.u_numerical, r.u_analytical, r.Ny, 
                                save_path=f"results/plots/{case.__name__}/profile_Ny{r.Ny}_{case.__name__}_dark.png",
                                title=f"Velocity profile comparison against analytical solution at Re={int(Re)}")

        plot_convergence(Ny_values, L2_errors,
                         save_path=f"results/plots/{case.__name__}/convergence_{case.__name__}_dark.png",
                         title=f"Method convergence study \nagainst analytical solution at Re={int(Re)}")