from dataclasses import dataclass
import numpy as np
import time

from lbm.src.core import analytical
from lbm.src.core.lattice import BaseLattice
from lbm.src.plot.animation import Recorder

@dataclass
class ConvergenceResult:
    Ny: int
    L2_error: float
    u_numerical: np.ndarray
    u_analytical: np.ndarray
    iterations: int
    runtime_seconds: int

def run_convergence_study(case_class, resolutions, **case_kwargs):
    total = len(resolutions)
    study_start = time.perf_counter()

    for idx, r in enumerate(resolutions, start=1):
        print(f"[{idx}/{total}] Ny={r:<4} running…", end = "", flush=True)

        ltc = case_class(ny=r, **case_kwargs)
        t0 = time.perf_counter()
        ltc.converge(tol=1e-12)
        runtime = time.perf_counter() - t0

        u_numerical  = ltc.ux[ltc.nx // 2, 1:-1]
        u_analytical = ltc.analytical_profile()
        L2_error     = np.sqrt(np.sum((u_numerical - u_analytical)**2) 
                             / np.sum(u_analytical**2))

        print(f"\r[{idx}/{total}] Ny={r:<4} done in {runtime:7.2f}s "
              f"({ltc.it:>8} iters | L2={L2_error:.2e})")

        yield ConvergenceResult(
            Ny=r,
            L2_error=L2_error,
            u_numerical=u_numerical,
            u_analytical=u_analytical,
            iterations=ltc.it,
            runtime_seconds=runtime,
        )

    total_runtime = time.perf_counter() - study_start
    print(f"\nTotal study time: {total_runtime:.2f}s")
    