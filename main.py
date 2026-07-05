from lbm.src.study.convergence import run_convergence_study
from lbm.src.plot.profile import plot_profile_comparison

RESOLUTIONS = [16]

# Run the study
results = run_convergence_study(RESOLUTIONS, tau=0.8, u_max=0.04)

# Produce the reports
for r in results:
    plot_profile_comparison(r.u_numerical, r.u_analytical, r.Ny,
                            save_path=f"results/profile_Ny{r.Ny}.png")
