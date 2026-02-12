# Sumcheck Protocol

A comprehensive implementation of the Sumcheck Protocol with multiple optimized prover variants for different use cases.

## Overview

The Sumcheck Protocol allows a prover to convince a verifier that the sum of a multivariate polynomial over the Boolean hypercube equals a claimed value without the verifier having to compute the entire sum.

It is an essential building block for many SNARK protocols, given that it reduces the complexity of computing the sum to performing $O(\nu)$ additions, plus an evaluation at a random point.

The protocol proceeds in rounds, with one round per variable of the multivariate polynomial. In each round, the prover sends a univariate polynomial, and the verifier responds with a random challenge. This process reduces a claim about a multivariate polynomial to a claim about a single evaluation point.

## Prover Implementations

This crate provides multiple sumcheck prover implementations optimized for different scenarios:

| Prover | Time Complexity | Space Complexity | Best For | Reference |
|--------|----------------|------------------|----------|-----------|
| **Naive** | O(n·2^2n) | O(2^n) | Learning, small polynomials | - |
| **Optimized** | O(d·2^n) | O(2^n) | General use (default) | [VSBW13](https://eprint.iacr.org/2012/303) |
| **Parallel** | O(d·2^n / cores) | O(2^n) | Multi-core systems, large polynomials | - |
| **Sparse** | O(n·k) | O(k) | Sparse polynomials (k non-zero entries) | [Lasso](https://eprint.iacr.org/2023/1216) |
| **Blendy** | O(k·2^n) | O(2^(n/k)) | Memory-constrained environments | [Chiesa et al.](https://eprint.iacr.org/2024/524) |
| **Batched** | Single proof for m instances | O(2^n) | Multiple sumcheck instances | - |

**Legend:**
- `n` = number of variables
- `d` = degree of product (number of polynomials)
- `k` = number of non-zero entries (for sparse) or stages (for Blendy)
- `cores` = available CPU cores

### Performance Comparison

For a 12-variable polynomial (2^12 = 4,096 elements):

| Implementation | Time | Throughput | Speedup |
|----------------|------|------------|---------|
| Naive | 1.027 s | 4.0 Kelem/s | 1x |
| Optimized | 427 µs | 9.6 Melem/s | **2,400x** |
| Parallel | 473 µs | 8.7 Melem/s | **2,172x** |

### Choosing a Prover

**Use Optimized** (default) if:
- General-purpose proving
- Polynomials are dense (most entries non-zero)
- Single-threaded or moderate parallelism

**Use Parallel** if:
- Large polynomials (n ≥ 14)
- Multi-core CPU available (8+ cores)
- Maximizing throughput

**Use Sparse** if:
- Polynomial has few non-zero entries (< 10% density)
- Memory-efficient lookup arguments
- Lasso-style protocols

**Use Blendy** if:
- Memory is severely constrained
- Can accept 2x slowdown for 650x memory reduction
- Very large polynomials that don't fit in RAM

**Use Batched** if:
- Proving multiple sumcheck instances
- Want to reduce total proof size
- GKR protocol or batched commitments

**Use Naive** if:
- Learning the protocol
- Small polynomials (n < 8)
- Debugging or reference implementation

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

## Using Optimized Provers

### Optimized Prover (Streaming Algorithm)

The optimized prover uses the VSBW13 streaming algorithm for ~2,400x speedup:

```rust
use lambdaworks_sumcheck::prove_optimized;

// Use the optimized prover instead of the naive one
let (claimed_sum, proof_polys) = prove_optimized(vec![poly1, poly2])?;

// Verification is the same
let is_valid = verify(num_vars, claimed_sum, proof_polys, vec![poly1, poly2])?;
```

### Parallel Prover

Enable the `parallel` feature in your `Cargo.toml`:

```toml
lambdaworks-sumcheck = { version = "0.1", features = ["parallel"] }
```

Then use the parallel prover:

```rust
use lambdaworks_sumcheck::prove_parallel;

// Automatically uses all available CPU cores
let (claimed_sum, proof_polys) = prove_parallel(vec![poly])?;
```

### Sparse Prover

For polynomials with few non-zero entries:

```rust
use lambdaworks_sumcheck::sparse_prover::SparseProver;
use std::collections::HashMap;

// Create a sparse polynomial representation
let mut sparse_evals = HashMap::new();
sparse_evals.insert(0, FE::from(5));    // Only index 0
sparse_evals.insert(100, FE::from(7));  // and index 100 are non-zero

let sparse_prover = SparseProver::new(num_vars, sparse_evals, vec![]);
// ... use with run_sumcheck_protocol
```

### Batched Sumcheck

Prove multiple instances efficiently:

```rust
use lambdaworks_sumcheck::batched::{prove_batched, verify_batched};

// Prove multiple polynomials in a single sumcheck
let instances = vec![
    vec![poly1_a, poly1_b],  // Instance 1: P1_a * P1_b
    vec![poly2_a, poly2_b],  // Instance 2: P2_a * P2_b
];

let (claimed_sums, proof) = prove_batched(instances.clone())?;
let is_valid = verify_batched(num_vars, claimed_sums, proof, instances)?;
```

## References

### Academic Papers

- **[Proofs, Arguments, and Zero-Knowledge. Chapter 4](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)** - Justin Thaler's comprehensive textbook on interactive proofs
- **[VSBW13: Efficient RAM and Control Flow in Verifiable Outsourced Computation](https://eprint.iacr.org/2012/303)** - Vu, Setty, Blumberg, and Walfish (NDSS 2013) - Streaming algorithm for O(d·2^n) complexity
- **[Lasso: Unlocking the Lookup Singularity](https://eprint.iacr.org/2023/1216)** - Setty, Thaler, and Wahby - Sparse sumcheck for lookup arguments
- **[Blendy: A Time-Space Tradeoff for the Sumcheck Prover](https://eprint.iacr.org/2024/524)** - Chiesa, Fedele, and Fenzi - Memory-efficient prover with configurable stages

### Blog Posts & Tutorials

- [Lambdaclass Blog Post: Have you checked your sums?](https://blog.lambdaclass.com/have-you-checked-your-sums/)

### Implementation References

The implementations in this crate were informed by the following production codebases:

- **[arkworks/sumcheck](https://github.com/arkworks-rs/sumcheck)** - Reference implementation with clean API design
- **[microsoft/Spartan](https://github.com/microsoft/Spartan)** - Production-quality sumcheck with R1CS optimizations
- **[microsoft/Nova](https://github.com/microsoft/Nova)** - Recursive SNARK with efficient parallelization strategies
- **[a16z/jolt](https://github.com/a16z/jolt)** - Lasso/Jolt implementation with sparse polynomial handling
- **[EspressoSystems/hyperplonk](https://github.com/EspressoSystems/hyperplonk)** - Multilinear polynomial commitments
- **[scroll-tech/ceno](https://github.com/scroll-tech/ceno)** - Memory-efficient GKR with time-space tradeoffs
- **[nexus-xyz/nexus-zkvm](https://github.com/nexus-xyz/nexus-zkvm)** - Production sparse sumcheck for zkVM
