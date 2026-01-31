# Polynomials

Polynomials are central to zero-knowledge proof systems. They encode computations, represent constraints, and enable efficient verification. This document covers polynomial concepts and their implementation in lambdaworks.

## Univariate Polynomials

A univariate polynomial over a finite field is an expression of the form:

$$p(x) = a_0 + a_1 x + a_2 x^2 + \cdots + a_n x^n$$

where the coefficients $a_i$ are field elements. The largest $k$ with $a_k \neq 0$ is the degree of the polynomial.

### Creating Polynomials

```rust
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

type F = U64PrimeField<65537>;
type FE = FieldElement<F>;

// From coefficients: p(x) = 1 + 2x + 3x^2
let p = Polynomial::new(&[FE::from(1), FE::from(2), FE::from(3)]);

// Create a monomial: 5x^4
let monomial = Polynomial::new_monomial(FE::from(5), 4);

// Zero polynomial
let zero = Polynomial::<FE>::zero();

// Constant polynomial
let constant = Polynomial::new(&[FE::from(42)]);
```

### Basic Operations

```rust
let p1 = Polynomial::new(&[FE::from(1), FE::from(2)]);  // 1 + 2x
let p2 = Polynomial::new(&[FE::from(3), FE::from(4)]);  // 3 + 4x

// Addition: (1 + 2x) + (3 + 4x) = 4 + 6x
let sum = &p1 + &p2;

// Subtraction
let diff = &p1 - &p2;

// Multiplication: (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2
let product = &p1 * &p2;

// Scalar multiplication
let scaled = p1.scale(&FE::from(5));

// Polynomial division
let (quotient, remainder) = p1.div_with_ref(&p2);
```

### Evaluation

Evaluating a polynomial at a point computes $p(x_0)$:

```rust
let p = Polynomial::new(&[FE::from(1), FE::from(2), FE::from(3)]);
// p(x) = 1 + 2x + 3x^2

let result = p.evaluate(&FE::from(2));
// p(2) = 1 + 4 + 12 = 17
```

For multiple evaluations, use `evaluate_slice`:

```rust
let points = vec![FE::from(0), FE::from(1), FE::from(2)];
let evaluations = p.evaluate_slice(&points);
```

### Interpolation

Lagrange interpolation reconstructs a polynomial from its evaluations:

```rust
// Given points and values
let xs = vec![FE::from(0), FE::from(1), FE::from(2)];
let ys = vec![FE::from(1), FE::from(4), FE::from(9)];

// Find p such that p(0) = 1, p(1) = 4, p(2) = 9
let p = Polynomial::interpolate(&xs, &ys).unwrap();
```

This is fundamental for STARKs, where the execution trace is interpolated into polynomials.

### Division and Remainders

Polynomial division is important for constraint checking:

```rust
// Divide by (x - r) using Ruffini's method
let root = FE::from(5);
let (quotient, remainder) = p.ruffini_division(&root);

// If remainder is zero, r is a root of p
if remainder == FE::zero() {
    println!("5 is a root of p");
}

// In-place Ruffini division (faster for repeated operations)
let mut p_clone = p.clone();
let remainder = p_clone.ruffini_division_inplace(&root);
```

### Degree and Properties

```rust
let p = Polynomial::new(&[FE::from(1), FE::from(2), FE::from(3)]);

// Degree: highest power with non-zero coefficient
let deg = p.degree();  // 2

// Leading coefficient
let lead = p.leading_coefficient();  // 3

// Check if zero polynomial
let is_zero = p.is_zero();

// Number of coefficients
let len = p.coeff_len();
```

## Fast Polynomial Operations with FFT

The Fast Fourier Transform enables polynomial multiplication in $O(n \log n)$ time instead of $O(n^2)$. lambdaworks provides FFT operations for FFT-friendly fields.

### FFT Basics

The FFT evaluates a polynomial at the $n$-th roots of unity simultaneously:

```rust
use lambdaworks_math::fft::polynomial::evaluate_fft;
use lambdaworks_math::fft::cpu::roots_of_unity::get_powers_of_primitive_root;

// Get roots of unity for a domain of size 8
let order = 8u64;
let roots = get_powers_of_primitive_root::<Stark252PrimeField>(order, 8).unwrap();

// Evaluate polynomial at all roots of unity
let evaluations = evaluate_fft(&p).unwrap();
```

### FFT-Based Multiplication

```rust
use lambdaworks_math::fft::polynomial::{multiply_fft, evaluate_fft, interpolate_fft};

// Multiply polynomials using FFT
let product = multiply_fft(&p1, &p2).unwrap();

// This is equivalent to p1 * p2, but faster for large polynomials
```

### Inverse FFT

The inverse FFT converts evaluations back to coefficients:

```rust
// Interpolate from evaluations at roots of unity
let coefficients = interpolate_fft(&evaluations).unwrap();
```

## Multilinear Polynomials

Multilinear polynomials have multiple variables, each appearing with degree at most 1. They are essential for sumcheck-based protocols:

$$p(x_0, x_1, \ldots, x_{n-1}) = \sum_{k=0}^{2^n - 1} c_k \cdot L_k(x_0, x_1, \ldots, x_{n-1})$$

where $L_k$ are the Lagrange basis polynomials over $\{0, 1\}^n$.

### Dense Multilinear Polynomials

```rust
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

// Create from evaluations over the Boolean hypercube
// For n_vars = 2, we need 4 evaluations: f(0,0), f(0,1), f(1,0), f(1,1)
let evals = vec![
    FE::from(1),  // f(0, 0)
    FE::from(2),  // f(0, 1)
    FE::from(3),  // f(1, 0)
    FE::from(4),  // f(1, 1)
];

let poly = DenseMultilinearPolynomial::new(evals);

// Properties
let num_vars = poly.num_vars();  // 2
let len = poly.len();            // 4

// Evaluate at any point (not just Boolean)
let point = vec![FE::from(2), FE::from(3)];
let result = poly.evaluate(point).unwrap();
```

### Converting to Univariate

For sumcheck, we convert a multilinear polynomial to univariate by summing over all but one variable:

```rust
// Sum over all variables except the last one
let univariate = poly.to_univariate();
```

## Polynomial Commitment Schemes

Polynomials can be "committed" cryptographically, allowing proofs about their evaluations without revealing the full polynomial.

### KZG Commitments

KZG commitments use elliptic curve pairings:

```rust
use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;

// Assume kzg is initialized with an SRS
let commitment = kzg.commit(&polynomial);

// Prove evaluation at point z
let proof = kzg.open(&z, &y, &polynomial);

// Verify: does polynomial(z) = y?
let is_valid = kzg.verify(&z, &y, &commitment, &proof);
```

### FRI (Fast Reed-Solomon IOP)

FRI is used in STARKs for polynomial commitments:

```rust
// FRI is integrated into the STARK prover
// The polynomial is committed via Merkle trees of its evaluations
```

## Common Polynomial Patterns

### Vanishing Polynomial

A polynomial that is zero at all points in a set $H$:

$$Z_H(x) = \prod_{h \in H} (x - h)$$

For a multiplicative subgroup (roots of unity), $Z_H(x) = x^n - 1$.

### Quotient Polynomial

If $p(x)$ equals $y$ at $z$, then $(p(x) - y)$ is divisible by $(x - z)$:

$$q(x) = \frac{p(x) - y}{x - z}$$

This is the basis of polynomial evaluation proofs.

### Lagrange Basis

The $i$-th Lagrange basis polynomial evaluates to 1 at $x_i$ and 0 at all other interpolation points:

$$L_i(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$$

Any polynomial can be written as $p(x) = \sum_i p(x_i) \cdot L_i(x)$.

## Performance Tips

1. **Use FFT** for multiplying polynomials of degree > 64.

2. **Use Ruffini division** instead of general division when dividing by $(x - r)$.

3. **Batch evaluations** using `evaluate_slice` rather than repeated `evaluate` calls.

4. **Sparse polynomials**: If most coefficients are zero, consider using sparse representations.

5. **Enable parallelism** with the `parallel` feature for large polynomial operations.

## Further Reading

1. [Anatomy of a STARK, Part 3: FRI](https://aszepieniec.github.io/stark-anatomy/fri) - Deep dive into polynomial commitments
2. [Proofs, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf) - Justin Thaler's book
3. [Fast amortized KZG proofs](https://eprint.iacr.org/2023/033) - Efficient batch opening
