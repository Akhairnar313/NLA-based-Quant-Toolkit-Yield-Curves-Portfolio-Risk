"""Hand-written numerical linear algebra routines.

Everything in this package is implemented from scratch, following the
Baruch Pre-MFE "Numerical Linear Algebra for Financial Engineering" course.
NumPy is used only for array storage and elementwise arithmetic; no library
solvers or factorizations are called. The test suite checks every routine
against NumPy/SciPy reference implementations.

Modules
-------
triangular_matrix_operations  Forward / backward substitution, bidiagonal variants (Session 2)
basic_operations  Non-singularity check
lu           LU decomposition with and without pivoting, solve, inverse, determinant (Sessions 2-3)
tridiagonal  Linear-time LU and solver for tridiagonal matrices (Session 3)
spd_checks   Gershgorin discs, diagonal dominance, Sylvester's criterion (Session 5)
cholesky     Cholesky decomposition, SPD solve / inverse / determinant / test (Session 6)
eigen        Power method, inverse and Rayleigh quotient iteration, QR, condition number (Session 8)
"""
