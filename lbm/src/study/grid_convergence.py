from dataclasses import dataclass
import numpy as np
import time


@dataclass
class ConvergenceResult:
    Ny: int
    L2_error: float
    iterations: int
    runtime_seconds: float


def upsample(field, factor):
    """Nearest-neighbor block upsampling: each cell -> factor x factor block."""
    return np.repeat(np.repeat(field, factor, axis=0), factor, axis=1)


def run_grid_convergence_study(case_class, resolutions, Re, tol=1e-8, **case_kwargs):
    total = len(resolutions)
    resolutions = sorted(resolutions)
    max_r = resolutions[-1]
    
    study_start = time.perf_counter()  

    print(f"[1/{total}] Ny={max_r:<4} running…", end = "", flush=True)

    ltc = case_class(ny=max_r, Re=Re, **case_kwargs)
    t0 = time.perf_counter()
    ltc.converge(tol=tol)
    runtime = time.perf_counter() - t0
    u_ref = ltc.ux**2 + ltc.uy**2
    
    print(f"\r[1/{total}] Ny={max_r:<4} done in {runtime:7.2f}s "
          f"({ltc.it:>8} iters |            )")

    for idx, r in enumerate(resolutions[:-1], start=2):
        print(f"[{idx}/{total}] Ny={r:<4} running…", end = "", flush=True)

        ltc = case_class(ny=r, Re=Re, **case_kwargs)
        t0 = time.perf_counter()
        ltc.converge(tol=tol)
        runtime = time.perf_counter() - t0
        u_c = ltc.ux**2 + ltc.uy**2

        # Upsample coarser field to the finest resolution for error comparison
        factor = max_r // r
        u_up   = upsample(u_c, factor)
         
        L2_error    = np.sqrt(np.sum((u_up - u_ref)**2) 
                            / np.sum(u_ref**2)
                            )

        print(f"\r[{idx}/{total}] Ny={r:<4} done in {runtime:7.2f}s "
              f"({ltc.it:>8} iters | L2={L2_error:.2e})")

        yield ConvergenceResult(
            Ny=r,
            L2_error=L2_error,
            iterations=ltc.it,
            runtime_seconds=runtime,
        )

    total_runtime = time.perf_counter() - study_start
    print(f"\nTotal study time: {total_runtime:.2f}s")