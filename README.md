<p align="center">
  <img src="results\animations\anim_Ny32_ForcedPoiseuille.gif" width="100%"/>
</p>

# 2D Lattice Boltzmann Solver — Verification Study via Poiseuille Flow

A Python implementation of the D2Q9 lattice Boltzmann method for 2D incompressible flow, with Numba-accelerated kernels. Includes two Poiseuille flow cases used to verify the method's second-order accuracy against analytical solutions. Two driving mechanisms — a body force and a pressure difference — were implemeted and compared.

This repository accompanies a URSS summer research project at the University of
Warwick under supervision of Dr. Radu Cimpeanu and is intended both as a verification study and as a foundation for
subsequent topology-optimisation work.

---

## Table of Contents

1. [The two verification cases](#the-two-cases)
2. [Error analysis: magnitude and profile shape](#error-analysis)
3. [Why the two cases are equivalent](#equivalence)
4. [Minimum sensible resolution](#minimum-resolution)
5. [Reproducing the results](#reproducing)
6. [References](#references)

---

<a name="the-two-cases"></a>
## 1. The Two Verification Cases

### Case A — Force-driven periodic channel

<p align="center">
  <img src="results\animations\anim_Ny32_ForcedPoiseuille.gif" width="97%"/>
</p>

<p align="center">
  <img src="results\plots\profile_Ny64_ForcedPoiseuille.png" width="54%"/>
  <img src="results\plots\convergence_ForcedPoiseuille.png" width="42%"/>
</p>

The solver recovers analytical Poiseuille flow to floating-point precision. Convergence is clean second-order in Ny, as expected for BGK collision with half-way bounce-back at $\tau = 0.933$ (where the effective wall sits exactly at $y = 0.5$).

A uniform body force `g_x` is applied at every fluid node, mimicking a constant
pressure gradient. The channel is **periodic** in the flow direction. Because the flow is translationally
invariant along the channel, a very short domain suffices. This is the cleaner of the two
tests and isolates the interior physics.

The body force required to achieve a target `u_max` is

```
g_x = 8 * nu * u_max / H_eff^2,   with H_eff = Ny - 2
```


### Case B — Pressure-driven channel

<p align="center">
  <img src="results\animations\anim_Ny32_PressurePoiseuille.gif" width="97%"/>
</p>

<p align="center">
  <img src="results\plots\profile_Ny64_PressurePoiseuille.png" width="54%"/>
  <img src="results\plots\convergence_PressurePoiseuille.png" width="42%"/>
</p>

A density (pressure) difference is imposed between inlet and outlet using
**Zou-He** boundary conditions; the walls use half-way bounce-back. This channel
is **not periodic**. Pressure in LBM is related
to density by `p = c_s^2 * rho = rho/3`, so prescribing a density difference
prescribes a pressure difference.

The density difference required to achieve the same target `u_max` is

```
delta_rho = 24 * nu * L * u_max / H_eff^2,   with L = Nx - 1
rho_in  = 1 + delta_rho/2
rho_out = 1 - delta_rho/2
```

Keeping the mean density at 1 (symmetric split around 1) maximises BGK stability.

Imposing a pressure difference on a fluid at
rest sends acoustic waves inward from both ends simultaneously (pressure
information propagates at the lattice sound speed in both directions). These waves
meet, reflect, and slowly damp. Only after the acoustic ringing subsides does the
flow settle into steady Poiseuille flow — which is why this case requires many
more iterations to converge than the force-driven case.

---

<a name="error-analysis"></a>
## 2. Error Analysis: Magnitude and Profile Shape

### Magnitude

The force-driven periodic case reaches residuals of `1e-8` or smaller. The pressure-driven case reaches residuals of `1e-4` to `1e-5` — several orders
of magnitude larger — because it carries genuine physical discretisation error
from its open boundaries and density variation.

### Profile shape

<p align="center">
  <img src="results\plots\profile_Ny16_PressurePoiseuille.png" width="49%"/>
  <img src="results\plots\profile_Ny32_PressurePoiseuille.png" width="49%"/>
</p>
<p align="center">
  <img src="results\plots\profile_Ny64_PressurePoiseuille.png" width="49%"/>
  <img src="results\plots\profile_Ny128_PressurePoiseuille.png" width="49%"/>
</p>

Whereas the force-driven residual is flat (a constant offset at the tolerance level), the pressure-driven residual has structure that changes with resolution:

- At low `Ny`, the error is entirely positive (LBM slightly exceeds analytical).
- At intermediate `Ny`, the error changes sign across the channel.
- At high `Ny`, the error becomes predominantly negative.

This is the signature of two competing error sources:

1. **Compressibility error (positive).** In the pressure-driven channel the
   density varies along the flow direction from `rho_in` to `rho_out`. Since
   `u_x = momentum / rho`, the velocity sampled where density is lower is slightly
   *higher* than the incompressible analytical solution assumes. This error scales
   with `delta_rho`, which shrinks as the grid is refined — so it dominates at low
   resolution and vanishes at high resolution.

2. **Bounce-back wall slip (negative).** A small residual slip velocity at the
   walls reduces the near-wall velocity. As a fraction of `u_max` this is roughly
   independent of resolution.

At low `Ny` the compressibility term dominates (positive error). As `Ny` grows,
compressibility falls off while the slip term persists, so the two cross over
(mixed-sign error) and eventually the slip term dominates (negative error).

### Convergence rate transition

The same competition explains the convergence-rate behaviour. Compressibility
error scales as `~1/Ny^2` (second order); wall slip scales as `~1/Ny` (first
order). At low-to-moderate resolution the second-order compressibility term
dominates and the measured slope is steep (approaching or exceeding `-2`). Once
compressibility drops below the slip floor, the first-order slip term takes over
and the slope flattens towards `-1`. The pressure-driven convergence plot
therefore shows a **regime transition**, not a single clean slope. Restricting the fit to the
compressibility-limited regime recovers a second-order rate.

---

<a name="equivalence"></a>
## 3. Why the Two Cases Are Equivalent (and Comparable)

A uniform body force and a uniform pressure gradient are physically identical
drivers of channel flow. In the momentum equation they enter the same way:

```
rho * g_x   <-->   -dp/dx
```

Since `p = c_s^2 * rho`, a pressure gradient `dp/dx = c_s^2 * delta_rho / L`
corresponds to a body force `g_x = c_s^2 * delta_rho / (rho_0 * L)`. Both produce
the *same* steady parabolic profile

```
u(y) = (driver / 2nu) * y * (H - y)
```

Because both cases are configured from a single input `u_max` (with `g_x` and
`delta_rho` derived from the formulas above), they target the **same** peak
velocity and are therefore directly comparable. Any difference in the measured
error is attributable to the *numerics of the driving mechanism and boundary
conditions*, not to a difference in the underlying flow. This is what makes the
comparison meaningful.

---

<a name="minimum-resolution"></a>
## 4. Minimum Sensible Resolution

We run convergence studies starting at `Ny = 16` and treat anything
below `Ny = 8` as outside the method's regime of validity.

**Knudsen-number argument.** LBM is an expansion of the Boltzmann equation valid
at small Knudsen number `Kn` (the ratio of mean free path to channel width). For
this solver, `Kn ~ c_s * tau / Ny`. The Chapman–Enskog expansion that connects
LBM to Navier–Stokes requires `Kn << 1`; below `Ny ≈ 8` the expansion is only
marginally valid.

**Resolution of the parabola.** A parabolic profile needs a handful of points
across the channel simply to be represented. 16+ are needed for a sufficient match.

**Bounce-back wall placement.** Half-way bounce-back places the no-slip wall
exactly midway between the last fluid node and the solid node *only at a specific
viscosity*. The effective wall offset scales as `1/Ny`, so at `Ny = 4` the wall
is mislocated by ~25% of the channel width; at `Ny = 64` this drops to ~1.5%.

---

<a name="reproducing"></a>
## 5. Reproducing the Results

```bash
pip install -e .
python main.py       # forced-Poiseuille and pressure-Poiseuille convergence study + animation
```

Outputs are written to `results/plots/` and
`results/animations/`. The convergence study prints per-resolution timings,
iteration counts, and L2 errors to the terminal.

---

<a name="references"></a>
## 6. References

- T. Krüger et al., *The Lattice Boltzmann Method: Principles and Practice*,
  Springer, 2017.
- P. Bhatnagar, E. Gross, M. Krook, "A model for collision processes in gases,"
  *Phys. Rev.* 94(3), 1954. *(BGK collision operator.)*
- Y. Qian, D. d'Humières, P. Lallemand, "Lattice BGK models for Navier–Stokes
  equation," *Europhys. Lett.* 17(6), 1992. *(D2Q9 lattice.)*
- Q. Zou, X. He, "On pressure and velocity boundary conditions for the lattice
  Boltzmann BGK model," *Phys. Fluids* 9(6), 1997. *(Zou-He boundaries.)*
- Z. Guo, C. Zheng, B. Shi, "Discrete lattice effects on the forcing term in the
  lattice Boltzmann method," *Phys. Rev. E* 65(4), 2002. *(Guo forcing.)*
- S. Chen, G. Doolen, "Lattice Boltzmann method for fluid flows," *Annu. Rev.
  Fluid Mech.* 30, 1998. *(Chapman–Enskog / validity regime.)*