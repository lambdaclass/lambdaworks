# lambdaworks Polynomials

Contains all the relevant tools for polynomials. Supports:
- [Univariate polynomials](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/polynomial/mod.rs)
- [Dense Multivariate polynomials](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/polynomial/dense_multilinear_poly.rs) and [Sparse Multilinear polynomials](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/polynomial/sparse_multilinear_poly.rs)

lambdaworks's polynomials work over [Finite Fields](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/field).

## Univariate polynomials

Univariate polynomials are expressions of the form $p(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n$, where $x$ is the indeterminate and $a_0, a_1 , ... , a_n$ take values over a finite field.
