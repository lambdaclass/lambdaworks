# LogUp-GKR

Implementation of the LogUp-GKR protocol for efficient lookup arguments, following ["Improving logarithmic derivative lookups using GKR"](https://eprint.iacr.org/2023/1284) by Papini and Haböck.

## Overview

LogUp-GKR replaces the intermediate accumulator columns of standard LogUp with a single multiplicities column, using the GKR interactive proof protocol to verify the accumulation. This cuts commitment cost significantly.

The lookup is expressed as a fractional sum identity:

$$\sum_i \frac{1}{z - a_i} = \sum_j \frac{m_j}{z - t_j}$$

where $a_i$ are accessed values, $t_j$ are table entries, and $m_j$ are multiplicities. The prover evaluates this fractional sum through a binary tree of fraction additions, which forms the GKR circuit.

## Circuit Structure

The circuit is a binary tree. Each layer has half the elements of the layer below, with the input fractions at the bottom and a single accumulated fraction at the root.

```
Output:     n/d              <- 1 element (the total sum)
           /   \
          n/d   n/d          <- 2 elements
         / \   / \
        .   . .   .          <- ...
       / \ / \ / \ / \
Input: fractions             <- 2^k elements
```

### Layer Types

| Layer | Gate operation | Columns | Use case |
|-------|---------------|---------|----------|
| `GrandProduct` | `a * b` | 1 | Product arguments |
| `LogUpGeneric` | `n_a/d_a + n_b/d_b` | 2 (num, den) | General fraction addition |
| `LogUpMultiplicities` | same as Generic | 2 (base-field num, den) | Table side with integer multiplicities |
| `LogUpSingles` | `1/a + 1/b = (a+b)/(a*b)` | 2 (implicit num=1, den) | Access side where all numerators are 1 |

Fraction addition computes `(n_a * d_b + n_b * d_a) / (d_a * d_b)`, kept in projective form to avoid field inversions.

## Protocol

### Multilinear GKR

The prover works layer by layer from output to input. At each layer, it runs a sumcheck over:

$$g(x) = \text{eq}(x, y) \cdot \big(\text{numer}(x) + \lambda \cdot \text{denom}(x)\big)$$

Here $\lambda$ is a random challenge that combines the numerator and denominator columns into a single polynomial, and $\text{eq}(x, y)$ is the multilinear equality polynomial linking the current layer's evaluation point to the previous one. The resulting round polynomials have degree 3.

For each layer, the verifier:
1. Checks the sumcheck (round consistency: `g(0) + g(1) == claim`, degree $\leq 3$)
2. Checks the circuit gate locally using the **mask** (evaluations at 0 and 1 for each column)
3. Samples a challenge to reduce the mask into claims for the next layer

After all layers are processed, the verifier holds an out-of-domain point and claimed evaluations at the input layer. These claims must be checked externally against the actual input data.

### Univariate IOP

The multilinear protocol leaves one problem open: how to verify the input layer claims. If the input columns are committed as multilinear extensions over $\{0,1\}^n$, the verifier can check them directly. But in practice, polynomials are committed in univariate form on a cyclic domain $H = \{\omega^i : i \in [0, N)\}$ -- the natural setting for FRI and KZG.

The univariate IOP bridges this gap. Given a multilinear polynomial $f$ with univariate representation $u_f(X)$ on $H$, the evaluation $f(t_0, \ldots, t_{n-1})$ equals the inner product

$$f(t) = \sum_{i=0}^{N-1} u_f(\omega^i) \cdot c_i(t)$$

where $c_i(t) = \text{eq}(\iota(i), t)$ is the **Lagrange column** -- it maps the cyclic domain to the Boolean hypercube via the bit-decomposition map $\iota$. This reduces the GKR input check to an inner product between the committed polynomial and a column the verifier can compute on its own.

Two phases provide different tradeoffs:

**Phase 1 (Transparent):** The prover sends raw polynomial values as Fiat-Shamir commitments. Proof size is $O(N)$.

**Phase 2 (PCS-based):** The prover commits via a polynomial commitment scheme (e.g., FRI with Merkle roots) and uses a **univariate sumcheck** to reduce the inner product to a single point evaluation. Proof size is $O(\log^2 N)$.

The univariate sumcheck relies on the identity:

$$u_f(X) \cdot C_t(X) - \frac{v}{N} = q(X) \cdot (X^N - 1) + X \cdot r'(X)$$

where $q(X)$ and $r'(X)$ are auxiliary polynomials of degree $\leq N-2$. This holds if and only if $\sum_{x \in H} u_f(x) \cdot C_t(x) = v$.

**Phase 2 prover flow:**
1. Commit each input column via PCS
2. Run multilinear GKR, obtaining evaluation claims at a random point $r$
3. Sample $\lambda$, combine claims via random linear combination
4. Compute the Lagrange column, combine columns with powers of $\lambda$
5. Run univariate sumcheck: compute $q(X)$ and $r'(X)$, commit via PCS
6. Sample challenge $z$, batch-open all polynomials at $z$

**Phase 2 verifier flow:**
1. Absorb commitments, run GKR verification
2. Sample $\lambda$, combine claims to get $v$
3. Absorb $q, r'$ commitments, sample $z$
4. Verify batch opening at $z$
5. Compute $C_t(z)$ via barycentric Lagrange interpolation
6. Check: $u_f(z) \cdot C_t(z) - v/N = q(z) \cdot (z^N - 1) + z \cdot r'(z)$

## Architecture

```
gkr-logup/
├── src/
│   ├── lib.rs                  # Public API and re-exports
│   ├── prover.rs               # GKR prover (LayerOracle, sumcheck adapter)
│   ├── verifier.rs             # GKR verifier (Gate types, proof verification)
│   ├── layer.rs                # Input layer types (GrandProduct, LogUp variants)
│   ├── fraction.rs             # Fraction arithmetic for LogUp gates
│   ├── eq_evals.rs             # Equality polynomial evaluations
│   ├── utils.rs                # RLC, Horner evaluation, MLE folding
│   ├── univariate/
│   │   ├── mod.rs              # Module exports
│   │   ├── iop.rs              # Univariate IOP prover/verifier (Phase 1 + Phase 2)
│   │   ├── types.rs            # Proof types (UnivariateIopProof, UnivariateIopProofV2)
│   │   ├── pcs.rs              # PCS trait + TransparentPcs implementation
│   │   ├── sumcheck.rs         # Univariate sumcheck (prove/verify, barycentric eval)
│   │   ├── lagrange_column.rs  # Lagrange column computation + constraint verification
│   │   ├── lagrange.rs         # Univariate Lagrange form + FFT transforms
│   │   ├── domain.rs           # Cyclic domain (roots of unity, bit decomposition)
│   │   └── commitment.rs       # Legacy commitment trait (Phase 1)
│   ├── univariate_layer.rs     # Univariate layer types wrapping UnivariateLagrange
│   └── fri/
│       ├── mod.rs              # FRI prover entry point (commit + query)
│       ├── types.rs            # FriConfig, FriProof, FriQueryRound, FriError
│       ├── fold.rs             # Polynomial folding (even/odd split + challenge)
│       ├── commit.rs           # FRI commit phase (iterative fold + Merkle)
│       ├── query.rs            # FRI query decommitments
│       ├── verify.rs           # FRI verification (fold consistency + Merkle proofs)
│       └── pcs.rs              # FRI PCS adapter (DEEP technique for arbitrary openings)
└── examples/
    ├── logup_gkr.rs                # Multilinear LogUp-GKR (singles + read-only memory)
    ├── univariate_logup_gkr.rs     # Univariate IOP (grand product, singles, multiplicities)
    └── univariate_to_multilinear.rs # IFFT + bit-reversal: univariate -> multilinear -> GKR
```

## API

### Multilinear GKR

```rust
use lambdaworks_gkr_logup::{prove, verify, Gate, Layer};
use lambdaworks_math::polynomial::DenseMultilinearPolynomial;

// Build a GrandProduct layer: proves that product = 1*2*3*4 = 24
let input = Layer::GrandProduct(DenseMultilinearPolynomial::new(vec![
    FE::from(1), FE::from(2), FE::from(3), FE::from(4),
]));

// Prove
let mut prover_channel = DefaultTranscript::<F>::new(&[]);
let (proof, prover_artifact) = prove(&mut prover_channel, input)?;

// Verify (partial — returns claims to check against the input layer)
let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
let result = verify(Gate::GrandProduct, &proof, &mut verifier_channel)?;
// result.ood_point, result.claims_to_verify
```

### Batch (Multiple Instances)

```rust
use lambdaworks_gkr_logup::{prove_batch, verify_batch, Gate, Layer};
use lambdaworks_math::polynomial::DenseMultilinearPolynomial;

// Two instances can have different sizes
let layers = vec![
    Layer::GrandProduct(DenseMultilinearPolynomial::new(values_a)),  // 2^5 elements
    Layer::LogUpSingles { denominators: DenseMultilinearPolynomial::new(dens_b) },  // 2^3 elements
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

### Univariate IOP -- Phase 1 (Transparent)

```rust
use lambdaworks_gkr_logup::univariate::iop::{prove_univariate, verify_univariate};
use lambdaworks_gkr_logup::univariate_layer::UnivariateLayer;
use lambdaworks_gkr_logup::univariate::domain::CyclicDomain;
use lambdaworks_gkr_logup::univariate::lagrange::UnivariateLagrange;
use lambdaworks_gkr_logup::verifier::Gate;

// Build a univariate grand product layer on a cyclic domain of size 8
let values: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
let domain = CyclicDomain::new(3)?; // log2(8) = 3
let uni = UnivariateLagrange::new(values, domain)?;

let layer = UnivariateLayer::GrandProduct { values: uni, commitment: None };

// Prove (transparent commitments: raw values in Fiat-Shamir transcript)
let mut prover_transcript = DefaultTranscript::<F>::new(b"grand_product");
let (proof, result) = prove_univariate(&mut prover_transcript, layer)?;

// Verify
let mut verifier_transcript = DefaultTranscript::<F>::new(b"grand_product");
verify_univariate(Gate::GrandProduct, &proof, &mut verifier_transcript)?;
```

### Univariate IOP -- Phase 2 (FRI PCS)

```rust
use lambdaworks_gkr_logup::univariate::iop::{prove_with_pcs, verify_with_pcs};
use lambdaworks_gkr_logup::fri::pcs::FriPcs;
use lambdaworks_gkr_logup::verifier::Gate;

// Same layer construction as Phase 1...
let layer = UnivariateLayer::GrandProduct { values: uni, commitment: None };

// Prove with FRI PCS (Merkle root commitments, succinct proofs)
let mut prover_transcript = DefaultTranscript::<F>::new(b"grand_product_v2");
let (proof, result) = prove_with_pcs::<F, _, FriPcs>(&mut prover_transcript, layer)?;

// Verify
let mut verifier_transcript = DefaultTranscript::<F>::new(b"grand_product_v2");
verify_with_pcs::<F, _, FriPcs>(Gate::GrandProduct, &proof, &mut verifier_transcript)?;
```

### Custom PCS

The `UnivariatePcs` trait accepts any polynomial commitment scheme:

```rust
pub trait UnivariatePcs<F: IsFFTField> {
    type Commitment: Clone + Debug;
    type ProverState;
    type BatchOpeningProof: Clone + Debug;

    fn commit(evals_on_h: &[FieldElement<F>], transcript: &mut T)
        -> Result<(Self::Commitment, Self::ProverState), PcsError>;
    fn batch_open(states: &[&Self::ProverState], z: &FieldElement<F>, transcript: &mut T)
        -> Result<(Vec<FieldElement<F>>, Self::BatchOpeningProof), PcsError>;
    fn verify_batch_opening(commitments: &[&Self::Commitment], z: &FieldElement<F>,
        values: &[FieldElement<F>], proof: &Self::BatchOpeningProof, transcript: &mut T)
        -> Result<(), PcsError>;
}
```

Two implementations are provided:
- **`TransparentPcs`** -- raw values appended to transcript (Phase 1, $O(N)$ proof size)
- **`FriPcs`** -- FRI with DEEP technique for arbitrary-point openings ($O(\log^2 N)$ proof size)

## Examples

### Read-Only Memory Check

Proves that a set of memory accesses all hit a valid ROM table:

```
cargo run -p lambdaworks-gkr-logup --example logup_gkr
```

This runs two batch instances: `LogUpSingles` for the access side ($\sum 1/(z - a_i)$) and `LogUpMultiplicities` for the table side ($\sum m_j/(z - t_j)$). When the accesses are valid, both sides produce the same fraction.

### Univariate IOP

```
cargo run -p lambdaworks-gkr-logup --example univariate_logup_gkr
```

Runs three examples -- grand product, LogUp singles, and LogUp multiplicities -- all using univariate commitments on a cyclic domain.

### Univariate-to-Multilinear Transform

```
cargo run -p lambdaworks-gkr-logup --example univariate_to_multilinear
```

Takes polynomial evaluations in cyclic domain order (as committed by FRI or KZG), converts them to multilinear extension layout via IFFT + bit-reversal, and runs the standard multilinear GKR prover. This is the alternative to the univariate IOP: instead of bridging at the protocol level, convert the data upfront.

## Running Tests

```bash
# All tests (100 tests)
cargo test -p lambdaworks-gkr-logup

# Only FRI tests
cargo test -p lambdaworks-gkr-logup fri

# Only univariate sumcheck tests
cargo test -p lambdaworks-gkr-logup sumcheck

# Only univariate IOP tests (Phase 1 + Phase 2)
cargo test -p lambdaworks-gkr-logup univariate::iop
```

## References

- [Papini and Haböck. "Improving logarithmic derivative lookups using GKR" (2023)](https://eprint.iacr.org/2023/1284)
- [Gruen. "Some Improvements for the PIOP for ZeroCheck" (2024)](https://eprint.iacr.org/2024/108) -- `correct_sum_as_poly_in_first_variable` optimization (section 3.2)
- [stwo reference implementation (StarkWare)](https://github.com/starkware-libs/stwo)
- [Haböck. "Multivariate lookups based on logarithmic derivatives" (2022)](https://eprint.iacr.org/2022/1530) -- original LogUp paper
- [Ben-Sasson et al. "Aurora: Transparent Succinct Arguments for R1CS" (2019)](https://eprint.iacr.org/2018/828) -- univariate sumcheck technique
