from lbm.src.plot.convergence import plot_convergence
from lbm.src.plot.animation import record_development
from lbm.src.study.grid_convergence import run_grid_convergence_study
from lbm.src.cases.two_rectangles import TwoRectangles
from lbm.src.plot.animation import Recorder


RESOLUTIONS = [16, 32, 64, 128]
Re          = 10.0
tol         = 1e-8

# Run the study
for case in [TwoRectangles]:
    print(f"Running grid convergence study for {case.__name__}…")
    results = run_grid_convergence_study(case, RESOLUTIONS, Re=Re, tol=tol)

    L2_errors = []

    # Produce the reports
    for i, r in enumerate(results):
        L2_errors.append(r.L2_error)

        plot_convergence(RESOLUTIONS[:i+1], L2_errors, save_path=f"results/plots/two_rectangles/convergence_{case.__name__}.png")

steps = 3000

for case in [TwoRectangles]:
    for r in RESOLUTIONS:
        ltc = case(ny=r, Re=Re)
        rec = Recorder(ltc, path=f"results/animations/two_rectangles/anim_Ny{r}_{case.__name__}.gif", every=5, interpolation='none')

        for step in range(steps):
            ltc.step()
            if rec.maybe_capture(step):
                print(f"\r{step}/{steps}", end="")
        rec.close()

        # record_development(case, nx=5*r, ny=r, Re = Re, tol=tol, every=1,
        #                    path=f"results/animations/two_rectangles/anim_Ny{r}_{case.__name__}.gif",
        #                    interpolation='none')

