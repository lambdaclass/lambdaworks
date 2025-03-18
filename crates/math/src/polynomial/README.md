# lambdaworks Polynomials

Contains all the relevant tools for polynomials. Supports:
- [Univariate polynomials](./mod.rs)
- [Dense Multivariate polynomials](../polynomial/dense_multilinear_poly.rs) and [Sparse Multilinear polynomials](../polynomial/sparse_multilinear_poly.rs)

lambdaworks's polynomials work over [Finite Fields](../field/README.md).

## Univariate polynomials

Univariate polynomials are expressions of the form $p(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n$, where $x$ is the indeterminate and $a_0, a_1 , ... , a_n$ take values over a finite field. The power with the highest non-zero coefficient is called the degree of the polynomial. A univariate polynomial is represented by means of the following struct:
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

Univariate polynomials have a [ring structure](https://en.wikipedia.org/wiki/Ring_(mathematics)): we can add, subtract, multiply and divide as we did with integers. For example, to add two polynomials,
```rust
let p_1 = Polynomial::new(&[FE::new(3), FE::new(4), FE::new(5)])
let p_2 = Polynomial::new(&[FE::new(4), FE::new(6), FE::new(8)])
let p_a = p_1 + p_2
```
Polynomial multiplication,
```rust
let p1 = Polynomial::new(&[FE::new(3), FE::new(3), FE::new(2)]);
let p2 = Polynomial::new(&[FE::new(4), FE::new(1)]);
assert_eq!(
    p2 * p1,
    Polynomial::new(&[FE::new(12), FE::new(15), FE::new(11), FE::new(2)])
    );
```
Division,
```rust
let p1 = Polynomial::new(&[FE::new(1), FE::new(3)]);
let p2 = Polynomial::new(&[FE::new(1), FE::new(3)]);
let p3 = p1.mul_with_ref(&p2);
assert_eq!(p3 / p2, p1);
```
Note that, in the case of polynomial division, it may have a remainder. If you want to divide a polynomial $p(x)$ by $x - b$, you can use faster alternatives, such as `ruffini_division` or `ruffini_division_inplace`.

Polynomials can also be evaluated at a point $x_0$ using `evaluate`. This provides the evaluation $p( x_0 ) = a_0 + a_1 x_0 + a_2 x_0^2 + ... + a_n x_0^n$. For example,
```rust
let p = Polynomial::new(&[FE::new(3), -FE::new(2), FE::new(4)]);
assert_eq!(p.evaluate(&FE::new(2)), FE::new(15));
```
evaluates the polynomial $p(x) = 3 - 2 x + 4 x^2$ at $2$ to yield $15$. If you need to evaluate at several points, you can use `evaluate_slice`.

Alternatively, polynomials of degree $n$ can be defined by providing exactly $n + 1$ evaluations. For example, $p(1) = 1$ and $p(0) = 2$ defines a unique polynomial of degree $1$, $p(x) = 2 - x$. To obtain the coefficients of $p(x)$ we need to use the function `interpolate`, which takes to vectors, of equal length: the first contains the $x$ coordinates $(0,1)$ and the second, the $y$ components $(2,1)$ (note that we have to provide the evaluation points in the same order as their corresponding evaluations):
```rust
let p = Polynomial::interpolate(&[FE::new(0), FE::new(1)], &[FE::new(2), FE::new(1)]).unwrap();
```

Many polynomial operations can go faster by using the [Fast Fourier Transform](../fft/polynomial.rs).
