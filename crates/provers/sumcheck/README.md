# Sumcheck Protocol

A naive implementation of the Sumcheck Protocol with support for linear, quadratic, and cubic polynomials.

## Overview

The Sumcheck Protocol allows a prover to convince a verifier that the sum of a multivariate polynomial over the Boolean hypercube equals a claimed value without the verifier having to compute the entire sum.

It is an essential building block for many SNARK protocols, given that it reduces the complexity of computing the sum to performing $O(\nu)$ additions, plus an evaluation at a random point.

The protocol proceeds in rounds, with one round per variable of the multivariate polynomial. In each round, the prover sends a univariate polynomial, and the verifier responds with a random challenge. This process reduces a claim about a multivariate polynomial to a claim about a single evaluation point.

### Convenience Wrapper Functions

This implementation provides the following convenience **wrapper functions** for common interaction degrees:
- `prove_linear` / `verify_linear` (Sum of P(x))
- `prove_quadratic` / `verify_quadratic` (Sum of P₁(x)P₂(x))
- `prove_cubic` / `verify_cubic` (Sum of P₁(x)P₂(x)P₃(x))



## Example

Here's a simple example of how to use the Sumcheck Protocol:

```rust
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_sumcheck::{prove, verify};

// Define the field
type F = U64PrimeField<17>;
type FE = FieldElement<F>;

// Create a multilinear polynomial
// P(x1, x2) = evals [3, 4, 5, 7]
let evaluations = vec![
    FE::from(3),
    FE::from(4),
    FE::from(5),
    FE::from(7),
];
let poly = DenseMultilinearPolynomial::new(evaluations);
let num_vars = poly.num_vars();

// Generate a proof using the linear wrapper
let prove_result = prove_linear(poly.clone());
assert!(prove_result.is_ok());
let (claimed_sum, proof_polys) = prove_result.unwrap();

// Verify the proof using the linear wrapper
let verification_result = verify_linear(num_vars, claimed_sum, proof_polys, poly);
assert!(verification_result.is_ok() && verification_result.unwrap());
println!("Simple verification successful!");
```
To use the quadratic wrapper, you can use the `prove_quadratic` and `verify_quadratic` functions. 

```rust
let prove_result = prove_quadratic(poly1, poly2);
let verification_result = verify_quadratic(num_vars, claimed_sum, proof_polys, poly1, poly2);
```

To use the cubic wrapper, you can use the `prove_cubic` and `verify_cubic` functions.

```rust
let prove_result = prove_cubic(poly1, poly2, poly3);
let verification_result = verify_cubic(num_vars, claimed_sum, proof_polys, poly1, poly2, poly3);
```


## References

- [Proofs, Arguments, and Zero-Knowledge. Chapter 4](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)
- [Lambdaclass Blog Post: Have you checked your sums?](https://blog.lambdaclass.com/have-you-checked-your-sums/)
