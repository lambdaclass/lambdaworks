# STARK Prover Optimization Findings

## Profiling Summary

Profiled with trace length 2^14 (16,384 rows):
- **Total allocations**: 200 MB in 952K blocks
- **Peak memory**: 33 MB
- **Proving time**: ~1.3s (release, single-threaded)

## Top Allocation Hotspots

| Rank | Function | Total MB | Peak MB | Issue |
|------|----------|----------|---------|-------|
| 1 | `ConstraintEvaluator::evaluate` | 33+ | 12 | Multiple allocations inside hot loop |
| 2 | `MerkleTree::build` | 12 | 6 | Hash buffer allocations |
| 3 | `evaluate_offset_fft` | 7.5 | 4 | FFT buffer allocations |
| 4 | `inplace_batch_inverse` | 4 | 2 | Temporary buffer |
| 5 | `Polynomial::add/sub` | 6 | 1.5 | Intermediate polynomial allocations |

## Detailed Analysis

### 1. Hot Loop Allocations in ConstraintEvaluator (HIGH IMPACT)

**Location**: `constraints/evaluator.rs:186-226`

The main evaluation loop iterates over every point in the LDE domain (trace_length × blowup_factor).
For a 16K trace with 4x blowup = 65,536 iterations.

**Allocations per iteration**:
```rust
// Line 191-194: Allocates Vec for periodic values
let periodic_values: Vec<_> = lde_periodic_columns
    .iter()
    .map(|col| col[i].clone())
    .collect();

// Line 202: Allocates Vec for transition evaluations
let evaluations_transition = air.compute_transition(&transition_evaluation_context);
```

**Estimated allocation**: 65,536 × (periodic_cols + num_constraints) × 32 bytes ≈ several MB

**Proposed fix**:
- Pre-allocate buffers outside the loop
- Add `compute_transition_into()` that takes a mutable slice
- Index into `lde_periodic_columns` directly instead of collecting

### 2. compute_transition Allocates New Vec (HIGH IMPACT)

**Location**: `traits.rs:122-133`

```rust
fn compute_transition(&self, ...) -> Vec<FieldElement<Self::FieldExtension>> {
    let mut evaluations = vec![FieldElement::zero(); self.num_transition_constraints()];
    // ... populate evaluations
    evaluations
}
```

**Issue**: Called 65,536+ times, allocates new Vec each time.

**Proposed fix**: Add a `compute_transition_into()` method:
```rust
fn compute_transition_into(
    &self,
    context: &TransitionEvaluationContext<...>,
    evaluations: &mut [FieldElement<Self::FieldExtension>],
) {
    evaluations.fill(FieldElement::zero());
    self.transition_constraints()
        .iter()
        .for_each(|c| c.evaluate(context, evaluations));
}
```

### 3. evaluate_offset_fft Allocations (MEDIUM IMPACT)

**Location**: `prover.rs:184-200`

The `evaluate_polynomial_on_lde_domain` function calls `evaluate_offset_fft` which allocates a new buffer for each polynomial evaluation.

**Proposed fix**: Use the new `evaluate_offset_fft_with_buffer` function (already implemented in lambdaworks-math) to reuse buffers when evaluating multiple polynomials.

### 4. MerkleTree::build Allocations (MEDIUM IMPACT)

**Location**: `lambdaworks_crypto::merkle_tree::merkle::MerkleTree::build`

Merkle tree construction allocates buffers for hashing.

**Proposed fix**: Pre-allocate hash buffers and reuse across tree construction.

### 5. Polynomial Arithmetic Allocations (LOW-MEDIUM IMPACT)

**Location**: Various polynomial operations in prover.rs

Operations like `poly_a + poly_b` and `poly - constant` create new polynomial instances.

**Proposed fix**: Use in-place operations where possible:
- `poly.add_assign(&other)` instead of `poly + other`
- Consider mutable accumulator patterns

## Recommended Implementation Order

1. **[HIGH]** Pre-allocate `periodic_values` buffer outside loop
2. **[HIGH]** Add `compute_transition_into()` with buffer reuse
3. **[MEDIUM]** Use `evaluate_offset_fft_with_buffer` in LDE evaluation
4. **[MEDIUM]** Optimize MerkleTree hash buffer allocation
5. **[LOW]** In-place polynomial arithmetic

## Expected Impact

Conservative estimates for 2^14 trace:
- Hot loop optimizations (1, 2): 30-50% reduction in allocations
- FFT buffer reuse (3): 10-15% reduction
- Overall: 40-60% fewer allocations, potential 10-20% speedup

## Verification

After implementing optimizations, re-run profiling:
```bash
cargo build --release -p stark-platinum-prover --bench prover_profile --features dhat-heap
./target/release/deps/prover_profile-* --trace-length 14
```

Compare dhat output to baseline.
