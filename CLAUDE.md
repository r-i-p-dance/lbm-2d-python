# lbm-2d-python

Verified D2Q9 LBM solver. Used by the sibling repo `topopt-lbm-python`,
whose `CLAUDE.md` holds the project conventions — read it first.

Changes to kernels, boundary conditions or `converge()` affect the verified
adjoint. Run the transpose tests in `../topopt-lbm-python/tests/` after any
such change.