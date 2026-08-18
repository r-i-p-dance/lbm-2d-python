from dataclasses import dataclass
import numpy as np
import time

from lbm.src.plot.difference import plot_field_comparison


@dataclass
class ConvergenceResult:
    Ny: int
    L2_error: float
    iterations: int
    runtime_seconds: float


def upsample(field, factor):
    """Nearest-neighbor block upsampling: each cell -> factor x factor block."""
    return np.repeat(np.repeat(field, factor, axis=0), factor, axis=1)


def run_grid_convergence_study(case_class, resolutions, Re, lbm_tau=0.6, tol=1e-8, **case_kwargs):
    total = len(resolutions)
    resolutions = sorted(resolutions)
    max_r = resolutions[-1]
    study_start = time.perf_counter()  

    print(f"[1/{total}] Ny={max_r:<4} running…", end = "", flush=True)

    ltc_reference = case_class(ny=max_r, Re=Re, tau_lbm=lbm_tau, **case_kwargs)
    t0 = time.perf_counter()
    ltc_reference.converge(tol=tol)
    runtime = time.perf_counter() - t0

    u_ref_norm = np.sqrt(ltc_reference.ux**2 + ltc_reference.uy**2) / ltc_reference.u_max
    
    print(f"\r[1/{total}] Ny={max_r:<4} done in {runtime:7.2f}s "
          f"({ltc_reference.it:>8} iters |            )")

    for idx, r in enumerate(resolutions[:-1], start=2):
        print(f"[{idx}/{total}] Ny={r:<4} running…", end = "", flush=True)

        ltc = case_class(ny=r, Re=Re, tau_lbm=lbm_tau, **case_kwargs)
        t0 = time.perf_counter()
        ltc.converge(tol=tol)
        runtime = time.perf_counter() - t0

        u_c_norm = np.sqrt(ltc.ux**2 + ltc.uy**2) / ltc.u_max

        # Upsample coarser field to the finest resolution for error comparison
        factor = max_r // r
        u_up_norm = upsample(u_c_norm, factor)
        obstacle_up = upsample(ltc.obstacle, factor)
        
        # Create a mask for the fluid region 
        fluid_mask = (~ltc_reference.obstacle) & (~obstacle_up)
        
        L2_error  = np.sqrt(np.sum((u_up_norm - u_ref_norm)[fluid_mask]**2) 
                            / np.sum(u_ref_norm[fluid_mask]**2))

        print(f"\r[{idx}/{total}] Ny={r:<4} done in {runtime:7.2f}s "
              f"({ltc.it:>8} iters | L2={L2_error:.2e} )")

        plot_field_comparison(
                u_ref_norm, u_up_norm, 
                save_path=f"results/plots/{case_class.__name__}/diff_{case_class.__name__}_{max_r}_vs_{r}_Re{int(Re)}_tol{tol}",
                title=f"{case_class.__name__}: Ny={max_r} vs Ny={r} (L2={L2_error:.3e})",
            )
        
        yield ConvergenceResult(
            Ny=r,
            L2_error=L2_error,
            iterations=ltc.it,
            runtime_seconds=runtime,
        )

    total_runtime = time.perf_counter() - study_start
    print(f"\nTotal study time: {total_runtime:.2f}s")