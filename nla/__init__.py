"""Hand-written numerical linear algebra routines.

Everything in this package is implemented from scratch, following the
Baruch Pre-MFE "Numerical Linear Algebra for Financial Engineering" course.
NumPy is used only for array storage and elementwise arithmetic; no library
solvers or factorizations are called. The test suite checks every routine
against NumPy/SciPy reference implementations.

Modules
-------
triangular   Forward / backward substitution (Session 2)
basic_operations  Non-singularity check (more basic operations to come)
lu           LU decomposition with and without pivoting (Sessions 2-3)
tridiagonal  Linear-time LU and solver for tridiagonal matrices (Session 3)
spd          Gershgorin discs, diagonal dominance, Sylvester's criterion (Session 5)
"""
