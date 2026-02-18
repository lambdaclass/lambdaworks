# LogUp-GKR

An implementation of the LogUp-GKR protocol for efficient lookup arguments, based on ["Improving logarithmic derivative lookups using GKR"](https://eprint.iacr.org/2023/1284) by Papini and Haböck.

## Overview

LogUp-GKR combines logarithmic derivative lookups with the GKR interactive proof protocol. Instead of committing to intermediate accumulator columns (as in standard LogUp), the prover only commits to a single multiplicities column, reducing the commitment cost significantly.

The core idea: to prove that a set of values comes from a valid table, express the lookup as a fractional sum identity

$$\sum_i \frac{1}{z - a_i} = \sum_j \frac{m_j}{z - t_j}$$

where $a_i$ are the accessed values, $t_j$ are table entries, and $m_j$ are multiplicities. This fractional sum is computed via a binary tree of fraction additions, which forms the GKR circuit.

## Circuit Structure

The circuit is a binary tree where each layer has half the elements of the layer below. The input layer sits at the bottom and the output (a single fraction) at the root.

```
Output:     n/d              <- 1 element (the total sum)
           /   \
          n/d   n/d          <- 2 elements
         / \   / \
        .   . .   .          <- ...
       / \ / \ / \ / \
Input: fractions             <- 2^k elements
```

### Layer types

| Layer | Gate operation | Columns | Use case |
|-------|---------------|---------|----------|
| `GrandProduct` | `a * b` | 1 | Product arguments |
| `LogUpGeneric` | `n_a/d_a + n_b/d_b` | 2 (num, den) | General fraction addition |
| `LogUpMultiplicities` | same as Generic | 2 (base-field num, den) | Table side with integer multiplicities |
| `LogUpSingles` | `1/a + 1/b = (a+b)/(a*b)` | 2 (implicit num=1, den) | Access side where all numerators are 1 |

The fraction addition gate computes `(n_a * d_b + n_b * d_a) / (d_a * d_b)`, kept in projective form to avoid field inversions.

## Protocol

For each layer (from output to input), the prover runs a sumcheck over:

$$g(x) = \text{eq}(x, y) \cdot \big(\text{numer}(x) + \lambda \cdot \text{denom}(x)\big)$$

where $\lambda$ is a random challenge that combines the numerator and denominator columns into a single polynomial, and $\text{eq}(x, y)$ is the multilinear equality polynomial linking the current layer's evaluation point to the previous one. Round polynomials have degree 3.

The verifier checks each layer by:
1. Verifying the sumcheck (round consistency: `g(0) + g(1) == claim`, degree $\leq 3$)
2. Checking the circuit gate locally using the **mask** (evaluations at 0 and 1 for each column)
3. Sampling a challenge to reduce the mask into claims for the next layer

After processing all layers, the verifier obtains an out-of-domain point and claimed evaluations at the input layer, which must be checked externally against the actual input data.

## API

### Single instance

```rust
use lambdaworks_gkr_logup::{prove, verify, Gate, Layer};
use lambdaworks_gkr_logup::mle::Mle;

// Build a GrandProduct layer: proves that product = 1*2*3*4 = 24
let input = Layer::GrandProduct(Mle::new(vec![
    FE::from(1), FE::from(2), FE::from(3), FE::from(4),
]));

// Prove
let mut prover_channel = DefaultTranscript::<F>::new(&[]);
let (proof, prover_artifact) = prove(&mut prover_channel, input);

// Verify (partial — returns claims to check against the input layer)
let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
let result = verify(Gate::GrandProduct, &proof, &mut verifier_channel)?;
// result.ood_point, result.claims_to_verify
```

### Batch (multiple instances)

```rust
use lambdaworks_gkr_logup::{prove_batch, verify_batch, Gate, Layer};

// Two instances can have different sizes
let layers = vec![
    Layer::GrandProduct(Mle::new(values_a)),  // 2^5 elements
    Layer::LogUpSingles { denominators: Mle::new(dens_b) },  // 2^3 elements
];

let mut prover_channel = DefaultTranscript::<F>::new(&[]);
let (proof, _) = prove_batch(&mut prover_channel, layers);

let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
let result = verify_batch(
    &[Gate::GrandProduct, Gate::LogUp],
    &proof,
    &mut verifier_channel,
)?;
// result.claims_to_verify_by_instance[0], result.claims_to_verify_by_instance[1]
```

## Examples

### Read-only memory check

Proves that a set of memory accesses all read from a valid ROM table:

```
cargo run -p lambdaworks-gkr-logup --example read_only_memory
```

Two batch instances: `LogUpSingles` for accesses ($\sum 1/(z - a_i)$) and `LogUpMultiplicities` for the table ($\sum m_j/(z - t_j)$). If accesses are valid, both sides produce the same fraction.

### Range check

Proves that values are in the range $[0, 2^n)$ by treating the range as a ROM table:

```
cargo run -p lambdaworks-gkr-logup --example range_check
```

## References

- [Papini and Haböck. "Improving logarithmic derivative lookups using GKR" (2023)](https://eprint.iacr.org/2023/1284)
- [Gruen. "Some Improvements for the PIOP for ZeroCheck" (2024)](https://eprint.iacr.org/2024/108) — `correct_sum_as_poly_in_first_variable` optimization (section 3.2)
- [stwo reference implementation (StarkWare)](https://github.com/starkware-libs/stwo)
- [Haböck. "Multivariate lookups based on logarithmic derivatives" (2022)](https://eprint.iacr.org/2022/1530) — original LogUp paper
