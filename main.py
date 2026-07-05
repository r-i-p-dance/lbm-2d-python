from lbm.src.study.convergence import run_convergence_study
from lbm.src.plot.profile import plot_profile_comparison
from lbm.src.plot.convergence import plot_convergence

RESOLUTIONS = [16, 32, 64, 128]

# Run the study
results = run_convergence_study(RESOLUTIONS, tau=0.8, u_max=0.04)

L2_errors = []
# Produce the reports
for r in results:
    plot_profile_comparison(r.u_numerical, r.u_analytical, r.Ny,
                            save_path=f"results/profile_Ny{r.Ny}.png")
    L2_errors.append(r.L2_error)

plot_convergence(RESOLUTIONS, L2_errors, save_path="results/convergence.png")