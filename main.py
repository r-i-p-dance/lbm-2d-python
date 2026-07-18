from lbm.src.cases.forced import ForcedPoiseuille
from lbm.src.study.convergence import run_convergence_study
from lbm.src.plot.profile import plot_profile_comparison
from lbm.src.plot.convergence import plot_convergence

RESOLUTIONS = [16, 32, 64]

# Run the study
results = run_convergence_study(ForcedPoiseuille, RESOLUTIONS, tau_lbm=0.933, u_max=0.04)

L2_errors = []
# Produce the reports
for i, r in enumerate(results):
    plot_profile_comparison(r.u_numerical, r.u_analytical, r.Ny,
                            save_path=f"results/plots/profile_Ny{r.Ny}.png")
    L2_errors.append(r.L2_error)

    plot_convergence(RESOLUTIONS[:i+1], L2_errors, save_path="results/plots/convergence.png")

