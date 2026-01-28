# STARK Prover Optimization Findings

## Profiling Summary

### Trace 2^14 (16,384 rows)
- **Total allocations**: 198 MB in 887K blocks (after initial optimization)
- **Peak memory**: 33 MB
- **Proving time**: ~350ms (release, single-threaded)

### Trace 2^16 (65,536 rows)
- **Total allocations**: 792 MB in 3.5M blocks
- **Peak memory**: 133 MB
- **Proving time**: ~1.5s (release, single-threaded)

---

## Peak Memory Analysis (t-gmax)

The highest peak memory contributors (2^16 trace):

| Rank | Peak MB | Total MB | Location | Issue |
|------|---------|----------|----------|-------|
| 1 | **32** | 32 | `IsStarkProver::prove` (grow_one) | Vec growing without pre-allocation |
| 2 | 16 | 16 | `ConstraintEvaluator::evaluate` | collect() operations |
| 3 | 16 | 20 | `IsStarkProver::prove` | LDE trace buffer |
| 4 | 16 | 24 | `MerkleTree::build` | Tree node allocation |
| 5 | 8 | 8 | `columns2rows` | Matrix transpose |
| 6 | 8 | 8 | `evaluate_offset_fft` | FFT buffer |
| 7 | 8 | 16 | `inplace_batch_inverse` | Temporary buffer |
| 8 | 8 | 8 | `new_domain` | Domain roots |
| 9 | 8 | 8 | `transition_zerofier_evaluations` | Zerofier evals |

**Key insight**: The 32 MB allocation via `grow_one` indicates a vector being grown incrementally instead of pre-allocated. This is the single largest peak memory contributor.

---

## CPU Time Analysis (instruments)

### Time Breakdown (2^16 trace, 1.49s total)

| Round | Time | % | Description |
|-------|------|---|-------------|
| Round 2 | 560ms | **49%** | Composition polynomial |
| Round 1 | 299ms | **26%** | RAP (interpolation + commitment) |
| Round 4 | 279ms | **24%** | FRI protocol |
| Round 0 | 143ms | 12% | Air initialization |
| Round 3 | 7ms | 1% | OOD evaluations |

### Round 2 Breakdown (Composition Polynomial)

| Operation | Time | % of Round 2 |
|-----------|------|--------------|
| Transitions evaluation | 87ms | **15.5%** |
| Transition zerofiers | 63ms | **11.2%** |
| Boundary evaluation | 27ms | 4.8% |
| Boundary polynomials | 3ms | 0.6% |

---

## Optimization Opportunities

### HIGH PRIORITY - Peak Memory

#### 1. Pre-allocate Vec in `prove()` (32 MB savings potential)

**Location**: `prover.rs` - `IsStarkProver::prove`

The `grow_one` pattern suggests incremental Vec growth. Identify which vector is growing and pre-allocate with capacity.

```rust
// Instead of:
let mut result = Vec::new();
for item in items {
    result.push(compute(item));
}

// Use:
let mut result = Vec::with_capacity(items.len());
for item in items {
    result.push(compute(item));
}
```

#### 2. Avoid `lde_trace_evaluations.clone()` (16 MB savings)

**Location**: `prover.rs:269`

```rust
// Current (wasteful):
let mut lde_trace_permuted = lde_trace_evaluations.clone();
for col in lde_trace_permuted.iter_mut() {
    in_place_bit_reverse_permute(col);
}

// Better: Do bit-reverse on original, then transpose
// Or: Combine with columns2rows to avoid intermediate allocation
```

#### 3. Optimize `columns2rows` (8 MB savings)

**Location**: `trace.rs`

The transpose operation allocates a new matrix. Consider:
- In-place transpose for square matrices
- Streaming/iterative approach that doesn't materialize full matrix

### MEDIUM PRIORITY - CPU Time

#### 4. Parallelize FFT operations in Round 1

**Location**: `compute_lde_trace_evaluations`

Each column's FFT is independent - can be parallelized:
```rust
#[cfg(feature = "parallel")]
let evaluations: Vec<_> = trace_polys
    .par_iter()
    .map(|poly| evaluate_polynomial_on_lde_domain(poly, ...))
    .collect();
```

#### 5. Cache zerofier evaluations more efficiently

**Location**: `traits.rs:208-244`

Currently clones cached zerofier evaluations. Use `Rc<Vec<>>` or indices instead:
```rust
// Current: clones the evaluation
evals[c.constraint_idx()] = zerofier_evaluations.clone();

// Better: Use shared reference or index into shared storage
```

#### 6. Use `evaluate_offset_fft_with_buffer` for batch evaluations

**Location**: `prover.rs:184-200`

When evaluating multiple polynomials, reuse a single buffer:
```rust
let mut buffer = Vec::with_capacity(expected_size);
for poly in &trace_polys {
    Polynomial::evaluate_offset_fft_with_buffer(poly, ..., &mut buffer)?;
    // Process buffer...
}
```

### LOW PRIORITY

#### 7. In-place polynomial arithmetic

Replace `poly_a + poly_b` with in-place operations where possible.

#### 8. Optimize MerkleTree hash buffer allocation

Pre-allocate hash buffers in MerkleTree::build.

---

## Implementation Status

| Optimization | Status | Impact |
|-------------|--------|--------|
| `compute_transition_into()` buffer reuse | ✅ Done | 6.9% fewer blocks |
| Pre-allocate periodic_values buffer | ✅ Done | Included above |
| Pre-allocate Vec in prove() | ⬜ TODO | Est. 32 MB peak reduction |
| Avoid lde_trace clone | ⬜ TODO | Est. 16 MB peak reduction |
| Parallelize FFT in Round 1 | ⬜ TODO | Est. 20-30% speedup |
| Zerofier caching optimization | ⬜ TODO | Est. 5-10% speedup |
| evaluate_offset_fft_with_buffer | ⬜ TODO | Est. 8 MB reduction |

---

## How to Profile

### Memory (dhat)
```bash
cargo build --release -p stark-platinum-prover --bench prover_profile --features dhat-heap
./target/release/deps/prover_profile-* --trace-length 16
# View dhat-heap.json at https://nnethercote.github.io/dh_view/dh_view.html
```

### CPU (instruments timing)
```bash
cargo build --release -p stark-platinum-prover --bench prover_profile --features instruments
./target/release/deps/prover_profile-* --trace-length 16
```

### CPU (samply)
```bash
cargo build --release -p stark-platinum-prover --bench prover_profile
samply record ./target/release/deps/prover_profile-* --trace-length 16
```
