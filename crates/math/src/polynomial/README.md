# lambdaworks Polynomials

Contains all the relevant tools for polynomials. Supports:
- [Dense univariate polynomials](./mod.rs)
- [Sparse univariate polynomials](./sparse.rs) - memory-efficient for polynomials with few non-zero terms
- [Dense multilinear polynomials](../polynomial/dense_multilinear_poly.rs) and [Sparse multilinear polynomials](../polynomial/sparse_multilinear_poly.rs)

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

## Sparse Univariate Polynomials

When a polynomial has few non-zero coefficients relative to its degree (e.g., `X^1000 - 1`), using a sparse representation is more memory-efficient. Sparse polynomials store only non-zero terms as `(degree, coefficient)` pairs.

```rust
use lambdaworks_math::polynomial::sparse::SparsePolynomial;

// Create polynomial 3*X^100 + 2*X^50 + 1
let poly = SparsePolynomial::from_coefficients(vec![
    (0, FE::from(1)),
    (50, FE::from(2)),
    (100, FE::from(3)),
]);

assert_eq!(poly.degree(), 100);
assert_eq!(poly.num_terms(), 3);
```

Sparse polynomials support arithmetic operations (add, subtract, multiply) and can be converted to/from dense representation when needed. They are useful for:
- Vanishing polynomials like `X^n - 1`
- R1CS constraint systems
- Custom gates in PLONK-like systems

## Quotient Ring Polynomials

For lattice-based cryptography, we need to work in polynomial quotient rings of the form `Zq[X]/(X^N + 1)`. The [`PolynomialRingElement<F, N>`](./quotient_ring.rs) type provides this, wrapping a `Polynomial<FieldElement<F>>` with automatic reduction modulo `X^N + 1`.

In the quotient ring, `X^N ≡ -1`: any polynomial of degree ≥ N wraps around with a sign flip on its coefficients. This negacyclic structure is central to schemes like Dilithium and Kyber.

```rust
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::dilithium_prime::DilithiumField;
use lambdaworks_math::polynomial::quotient_ring::PolynomialRingElement;

type FE = FieldElement<DilithiumField>;
type R256 = PolynomialRingElement<DilithiumField, 256>;

let a = R256::new(&[FE::from(1u64), FE::from(2u64)]);
let b = R256::new(&[FE::from(3u64), FE::from(4u64)]);

// Schoolbook multiplication with automatic reduction mod X^256 + 1
let product = a.mul_schoolbook(&b);

// FFT-based multiplication (faster for large N, requires FFT-friendly field)
let product_ntt = a.mul_ntt(&b);
assert_eq!(product, product_ntt);
```

The type also provides `infinity_norm()`, `centered_coefficient()`, and `is_small(bound)` for working with short vectors, which are central to lattice-based security. See the [lattice cryptography README](../../../crypto/src/lattice/README.md) for full mathematical background.

## Multilinear polynomials

Multilinear polynomials are useful to define multilinear extensions of functions, which then play an important role in proof systems involving the [sumcheck protocol](../../../provers/sumcheck/README.md). There are two ways to define multilinear polynomials:
- Dense
- Sparse

Sparse is more convenient whenever the number of non-zero coefficients in the polynomial is small (compared to the length of the polynomial), avoiding the storage of unnecessary zeros. For dense multilinear polynomials we have the following structure, working over some field $F$:
```rust
pub struct DenseMultilinearPolynomial<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    evals: Vec<FieldElement<F>>,
    n_vars: usize,
    len: usize,
}
```
The polynomial is assumed to be given in evaluation form over the binary strings of length $\{0 , 1 \}^{n_{vars}}$. We can also interpret this as the coefficients of the polynomial with respect to the Lagrange basis polynomials over $\{0 , 1 \}^{n_{vars}}$. There are $2^{n_{vars}}$ Lagrange polynomials, given by the formula:
$L_k (x_0 , x_1 , ... , x_{n_{vars} - 1}) = \prod (x_j b_{kj} + (1 - x_j ) (1 - b_{kj} ))$
where $b_{kj}$ are given by the binary decomposition of $k$, that is $k = \sum_j b_{kj} 2^j$. We can see that each such polynomial is equal to one over $\{b_{k0}, b_{k1} , ... b_{k (n_{vars} - 1)}}$ and zero for any other element in $\{0 , 1 \}^{n_{vars}}$. The polynomial is thus defined as
$p (x_0 , x_1, ... , x_{n_{vars} - 1} ) = \sum_k p(b_{k0}, b_{k1} , ... , b_{k (n_{vars} - 1)}) L_k (x_0 , x_1, ... , x_{n_{vars} - 1} )$
Sometimes, we will use $L_k (j)$ to refer to the evaluation of $L_k$ at the binary decomposition of $j$, that is $j = \sum_k b_{k}2^k$.

An advantage of Lagrange basis polynomials is that we can evaluate all $2^{n_{vars}}$ polynomials at a point $(r_0 , r_1 ... , r_{n_{vars} - 1})$ in $\mathcal{O}(2^{n_{vars}})$ operations (linear in the size of the number of polynomials). Refer to [Thaler's book](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf) for more information.

To create a new polynomial, provide a list of evaluations of $p$; the length of this list should be a power of 2.
```rust
pub fn new(mut evals: Vec<FieldElement<F>>) -> Self {
    while !evals.len().is_power_of_two() {
        evals.push(FieldElement::zero());
    }
    let len = evals.len();
    DenseMultilinearPolynomial {
        n_vars: log_2(len),
        evals,
        len,
    }
}
```

Dense multilinear polynomials allow you to access the fields `n_vars`, `len` and `evals` with the methods `pub fn num_vars(&self) -> usize`, `pub fn len(&self) -> usize` and `pub fn evals(&self) -> &Vec<FieldElement<F>>`.

If you want to evaluate outside $\{0 , 1 \}^{n_{vars}}$, you can use the functions `pub fn evaluate(&self, r: Vec<FieldElement<F>>)` and `pub fn evaluate_with(evals: &[FieldElement<F>], r: &[FieldElement<F>])`, providing the point $r$ whose length must be $n_{vars}$. For evaluations over $\{0 , 1 \}^{n_{vars}}$, you can get the value directly from the list of evaluations defining the polynomial. For example, 
```rust
// Example: Z = [1, 2, 1, 4]
let z = vec![FE::one(), FE::from(2u64), FE::one(), FE::from(4u64)];
// r = [4, 3]
let r = vec![FE::from(4u64), FE::from(3u64)];
let eval_with_lr = evaluate_with_lr(&z, &r);
let poly = DenseMultilinearPolynomial::new(z);
let eval = poly.evaluate(r).unwrap();
assert_eq!(eval, FE::from(28u64));
```

An important functionality is `pub fn to_univariate(&self) -> Polynomial<FieldElement<F>>`, which converts a multilinear polynomial into a univariate polynomial, by summing over all variables over $\{0 , 1 \}^{n_{vars} - 1}$, leaving $x_{n_{vars} - 1}$ as the only variable, 
$$f(x) = \sum_{(x_0 , x_1, ... , x_{n_{vars} - 2} ) \in \{0 , 1 \}^{n_{vars} - 1}} p(x_0 , x_1, ... , x_{n_{vars} - 2} , x)$$. For example,
```rust
let univar0 = prover.poly.to_univariate();
```
is used in the sumcheck protocol.

Multilinear polynomials can be added and multiplied by scalars.