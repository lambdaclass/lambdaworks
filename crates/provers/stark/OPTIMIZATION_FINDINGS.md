# STARK Prover Optimization Findings

## Current State (After All Optimizations)

### Trace 2^16 (65,536 rows)
- **Total allocations**: 729 MB in 3.54M blocks
- **Peak memory**: 108 MB
- **Proving time**: ~1.43s (release, single-threaded)
- **Proving time (parallel)**: ~1.37s (with `parallel` feature)

### Improvements Achieved (from baseline)
- Peak memory: 133 MB → 108 MB (**18.9% reduction**)
- Total allocations: 792 MB → 729 MB (**8.0% reduction**)
- Parallel mode: 65-67% faster boundary/transition evaluation

---

## CPU Time Analysis (instruments)

### Time Breakdown (2^16 trace, 1.43s total)

| Round | Time | % | Description |
|-------|------|---|-------------|
| Round 2 | 542ms | **50.2%** | Composition polynomial |
| Round 1 | 311ms | **28.7%** | RAP (interpolation + commitment) |
| Round 4 | 220ms | **20.4%** | FRI protocol |
| Round 0 | 189ms | 17.5% | Air initialization |
| Round 3 | 8ms | 0.7% | OOD evaluations |

### Round 2 Breakdown (Composition Polynomial) - **Primary CPU Bottleneck**

| Operation | Time | % of Round 2 |
|-----------|------|--------------|
| Transitions + accumulation | 88ms | **16.2%** |
| Transition zerofiers | 62ms | **11.4%** |
| Boundary evaluation | 29ms | 5.3% |
| Boundary polynomials | 3ms | 0.5% |

---

## Peak Memory Analysis (t-gmax) - 108 MB Total

| Rank | Peak MB | Total MB | Location | Issue |
|------|---------|----------|----------|-------|
| 1 | **16** | 16 | `ConstraintEvaluator::evaluate` | Boundary zerofiers inverse |
| 2 | **16** | 20 | `IsStarkProver::prove` | LDE trace buffer |
| 3 | **16** | 16 | `ConstraintEvaluator::evaluate` | Boundary polys evaluations |
| 4 | **16** | 24 | `MerkleTree::build` | Tree node allocation |
| 5 | 8 | 8 | `Table::from_columns` | LDE trace table |
| 6 | 8 | 8 | `evaluate_offset_fft` | FFT in zerofier eval |
| 7 | 8 | 8 | `transition_zerofier_evaluations` | Zerofier storage |
| 8 | 8 | 8 | `fft::ops::fft` | FFT buffer |
| 9 | 8 | 8 | `fft::ops::fft` | FFT buffer |
| 10 | 8 | 8 | `columns2rows` | Matrix transpose |

---

## Total Allocation Analysis - 757 MB

| Rank | Total MB | Peak MB | Blocks | Location |
|------|----------|---------|--------|----------|
| 1-3 | 132 | 0 | - | `ConstraintEvaluator::evaluate` (boundary constraints) |
| 4 | 24 | 16 | - | `MerkleTree::build` (main trace) |
| 5 | 20 | 16 | - | `IsStarkProver::prove` (LDE buffer) |
| 6 | 16 | 16 | - | `ConstraintEvaluator::evaluate` |
| 7 | 16 | 8 | - | `inplace_batch_inverse` |
| 8 | 16 | 16 | - | `ConstraintEvaluator::evaluate` |
| 9-11 | 36 | 14 | - | `MerkleTree::build` (aux + FRI) |
| 12-13 | 24 | 6 | - | Polynomial add/sub (deep composition) |
| 14-15 | 20 | 12 | - | `evaluate_offset_fft` (LDE eval) |

---

## Remaining Optimization Opportunities

### HIGH PRIORITY - Peak Memory

#### 1. Avoid `lde_trace_evaluations.clone()` (~16 MB savings)

**Location**: `prover.rs:269, 321` in `interpolate_and_commit_main/aux`

```rust
// Current (wasteful):
let mut lde_trace_permuted = lde_trace_evaluations.clone();
for col in lde_trace_permuted.iter_mut() {
    in_place_bit_reverse_permute(col);
}

// Better Option A: Bit-reverse in-place on original, then transpose
// (requires adjusting downstream code to account for permuted order)

// Better Option B: Fused bit-reverse + transpose
fn columns2rows_bit_reversed<F: Clone>(columns: &[Vec<F>]) -> Vec<Vec<F>> {
    let num_rows = columns[0].len();
    let num_cols = columns.len();
    (0..num_rows)
        .map(|row_index| {
            let permuted_index = bit_reverse(row_index, num_rows);
            (0..num_cols)
                .map(|col_index| columns[col_index][permuted_index].clone())
                .collect()
        })
        .collect()
}
```

#### 2. Optimize `columns2rows` transpose (~8 MB savings)

**Location**: `trace.rs:373-387`

The transpose allocates a completely new matrix. Options:
- Fuse with bit-reverse permutation (as shown above)
- Streaming approach that builds Merkle tree rows on-the-fly

#### 3. Optimize boundary constraint evaluation (~32 MB total reduction)

**Location**: `evaluator.rs:59-73, 104-124`

The boundary zerofier inverse evaluations and boundary poly evaluations both allocate
large vectors via `collect()` for each constraint. Consider:

```rust
// Pre-allocate all boundary data in a single flat buffer
let total_evals = num_constraints * domain_size;
let mut all_boundary_zerofiers = Vec::with_capacity(total_evals);
let mut all_boundary_polys = Vec::with_capacity(total_evals);
```

### MEDIUM PRIORITY - CPU Time

#### 4. ✅ Parallelize zerofier computation in Round 2 (DONE)

**Location**: `traits.rs:305-475` - `transition_zerofier_evaluations`

**Implemented**: Zerofier computation now parallelizes both base zerofier and end exemptions
calculations when the `parallel` feature is enabled.

**Results** (2^16 trace with parallel feature):
- Boundary evaluation: 27ms → 8.9ms (**67% faster**)
- Transitions evaluation: 81ms → 31ms (**65% faster**)

**Note**: For simple AIRs with 1 constraint (like Fibonacci), parallel overhead exceeds
benefits. Optimization primarily helps complex AIRs with many unique zerofiers.

#### 5. ✅ In-place polynomial arithmetic in `compute_deep_composition_poly` (DONE)

**Location**: `prover.rs:683-710`

**Implemented**: The h_terms accumulation now pre-allocates a coefficient buffer and
accumulates `gamma * (part - h_i_eval)` directly into coefficients, avoiding
intermediate polynomial allocations.

**Results** (2^16 trace):
- Total allocations: 748 MB → 729 MB (**2.5% reduction**, ~19 MB savings)
- Peak memory: unchanged (optimization targets total, not peak)

#### 6. Use `evaluate_offset_fft_with_buffer` in LDE evaluation

**Location**: `prover.rs:184-200`

The new buffer-reuse FFT function can reduce allocations when evaluating multiple polynomials.

### LOW PRIORITY

#### 7. Optimize MerkleTree hash buffer pre-allocation

**Location**: `lambdaworks_crypto::merkle_tree::merkle::MerkleTree::build`

#### 8. Zerofier evaluation caching with `Rc<Vec<>>`

**Location**: `traits.rs:240`

Currently clones zerofier evaluations. Using `Rc` would avoid copies:

```rust
use std::rc::Rc;
let mut zerofier_groups: HashMap<ZerofierGroupKey, Rc<Vec<FieldElement<F>>>> = HashMap::new();
// ...
evals[c.constraint_idx()] = Rc::clone(zerofier_groups.get(&key).unwrap());
```

---

## Implementation Status

| Optimization | Status | Impact |
|-------------|--------|--------|
| `compute_transition_into()` buffer reuse | ✅ Done | 6.9% fewer blocks |
| Pre-allocate periodic_values buffer | ✅ Done | Included above |
| Pre-allocate Vec in `commit_composition_polynomial` | ✅ Done | 18.9% peak reduction |
| Fused bit-reverse + transpose | ✅ Done | 1% total allocation reduction |
| Zerofier base/exemptions split caching | ✅ Done | Benefits multi-constraint AIRs |
| Parallelize zerofier computation | ✅ Done | 65-67% faster (parallel mode) |
| In-place polynomial arithmetic (h_terms) | ✅ Done | 19 MB total reduction (2.5%) |
| Boundary zerofier caching by step | ✅ Done | Benefits AIRs with multi-step constraints |
| Pre-compute z_shifted values | ✅ Done | Avoids redundant pow() per trace column |
| evaluate_offset_fft_with_buffer in LDE | ⬜ TODO | Est. 10 MB reduction (requires math crate changes) |

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

---

## Summary of Optimization Strategy

### Phase 1 - Quick Wins (Memory) ✅ DONE
1. Pre-allocate vectors with known capacity
2. Buffer reuse in hot loops

### Phase 2 - Memory Reduction ✅ MOSTLY DONE
1. ✅ Fuse bit-reverse with transpose
2. ✅ Optimize boundary constraint evaluation (zerofier caching)
3. ⬜ Eliminate `lde_trace_evaluations.clone()` (complex, requires API changes)

### Phase 3 - CPU Optimization ✅ MOSTLY DONE
1. ✅ Parallelize zerofier computation
2. ✅ In-place polynomial arithmetic (h_terms)
3. ✅ Pre-compute z_shifted values
4. ⬜ Better FFT buffer management (requires math crate changes)

### Expected Final Results
With all optimizations:
- Peak memory: ~70-80 MB (from 108 MB)
- Total allocations: ~500 MB (from 757 MB)
- CPU time: ~1.0-1.1s (from 1.43s) for 2^16 trace
