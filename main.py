from lbm.src.cases.forced import ForcedPoiseuille
from lbm.src.cases.pressure import PressurePoiseuille
from lbm.src.plot.animation import record_development
from lbm.src.plot.profile import plot_profile_comparison
from lbm.src.plot.convergence import plot_convergence
from lbm.src.study.convergence import run_convergence_study


RESOLUTIONS = [16, 32, 64]

# Run the study
for case in [ForcedPoiseuille, PressurePoiseuille]:
    results = run_convergence_study(case, RESOLUTIONS, tau_lbm=0.933, u_max=0.04)

    L2_errors = []
    # Produce the reports
    for i, r in enumerate(results):
        plot_profile_comparison(r.u_numerical, r.u_analytical, r.Ny,
                                save_path=f"results/plots/profile_Ny{r.Ny}_{case.__name__}.png")
        L2_errors.append(r.L2_error)

        plot_convergence(RESOLUTIONS[:i+1], L2_errors, save_path=f"results/plots/convergence_{case.__name__}.png")

for case in [ForcedPoiseuille, PressurePoiseuille]:
    for r in RESOLUTIONS:
        record_development(case, nx=5*r, ny=r, tol=1e-12, every=100,
                           path=f"results/animations/anim_Ny{r}_{case.__name__}.gif",
                           tau_lbm=0.933, u_max=0.04)