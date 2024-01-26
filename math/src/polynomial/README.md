# lambdaworks Polynomials

Contains all the relevant tools for polynomials. Supports:
- [Univariate polynomials](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/polynomial/mod.rs)
- [Dense Multivariate polynomials](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/polynomial/dense_multilinear_poly.rs) and [Sparse Multilinear polynomials](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/polynomial/sparse_multilinear_poly.rs)

lambdaworks's polynomials work over [Finite Fields](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/field).

## Univariate polynomials

Univariate polynomials are expressions of the form $p(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n$, where $x$ is the indeterminate and $a_0, a_1 , ... , a_n$ take values over a finite field. A univariate polynomial is represented by means of the following struct:
```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polynomial<FE> {
    pub coefficients: Vec<FE>,
}
```
it contains the coefficients in increasing order (we start with the independent term, $a_0$, then $a_1$, and so on and so forth). To create a new polynomial,
```rust
let my_poly = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3)])
```
This creates the polynomial $p(x) = 1 + 2 x + 3 x^2$. If we provide additional zeros to the right, the `new` method will remove those unnecessary zeros. For example,
```rust
let my_poly = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3), FE::ZERO])
```
generates the same polynomial as before. We can also create a monomial, such as $5 x^4$ or $27 x^{120}$, which can be simpler sometimes (instead of providing a long list of zeros). To define a monomial, simply
```rust
let my_monomial = Polynomial::new_monomial(FE::new(27),6)
```
generates the monomial $p(x) = 27 x^6$, which has a representation as polynomial $(0,0,0,0,0,0,27)$.

Univariate polynomials have a [ring structure](https://en.wikipedia.org/wiki/Ring_(mathematics)): we can add, subtract, multiply and divide as we did with integers. For example,
```rust
let p_1 = Polynomial::new(&[FE::new(3), FE::new(4), FE::new(5)])
let p_2 = Polynomial::new(&[FE::new(4), FE::new(6), FE::new(8)])
let p_a = p_1 + p_2
```
