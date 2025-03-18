# Sumcheck Protocol

A naive implementation of the Sumcheck Protocol. 

## Overview

The Sumcheck Protocol allows a prover to convince a verifier that the sum of a multivariate polynomial over the Boolean hypercube equals a claimed value without the verifier having to compute the entire sum.

It is an essential building block for many SNARK protocols, given that it reduces the complexity of computing the sum to performing $O(\nu)$ additions, plus an evaluation at a random point.

The protocol proceeds in rounds, with one round per variable of the multivariate polynomial. In each round, the prover sends a univariate polynomial, and the verifier responds with a random challenge. This process reduces a claim about a multivariate polynomial to a claim about a single evaluation point.


## Example

Here's a simple example of how to use the Sumcheck Protocol:

```rust
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_sumcheck::{prove, verify};

// Define the field
type F = U64PrimeField<17>;

// Create a multilinear polynomial
let evaluations = vec![
    FieldElement::<F>::from(3),
    FieldElement::<F>::from(4),
    FieldElement::<F>::from(5),
    FieldElement::<F>::from(7),
];
let poly = DenseMultilinearPolynomial::new(evaluations);

// Generate a proof
let (claimed_sum, proof) = prove(poly);

// Verify the proof
let result = verify(poly.num_vars(), claimed_sum, proof, Some(poly));
assert!(result.is_ok() && result.unwrap());
```


## References

- [Proofs, Arguments, and Zero-Knowledge. Chapter 4](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)
- [Lambdaclass Blog Post: Have you checked your sums?](https://blog.lambdaclass.com/have-you-checked-your-sums/)
