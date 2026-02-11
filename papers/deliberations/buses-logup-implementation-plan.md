# Implementation Plan: Buses and LogUp Lookup Argument for lambdaworks

## Executive Summary

This document describes a phased implementation strategy for porting the bus abstraction and LogUp lookup argument from `lambda_vm` into the lambdaworks STARK prover crate. The goal is to provide a first-class, ergonomic lookup argument that integrates with the existing `AIR` trait, `TransitionConstraint`, `TraceTable`, and prover infrastructure without breaking backward compatibility.

The plan is organized into five phases with clear deliverables, test criteria, and risk mitigations.

---

## 1. Gap Analysis: lambda_vm vs lambdaworks

### 1.1 API Differences

| Aspect | lambdaworks (current) | lambda_vm (source) |
|--------|----------------------|-------------------|
| `AIR::new` signature | `fn new(trace_length, pub_inputs, proof_options)` | `fn new(proof_options)` |
| `AIR::build_auxiliary_trace` return | `()` (side-effect only) | `Option<BusPublicInputs<E>>` |
| `AIR::boundary_constraints` args | `(rap_challenges)` | `(pub_inputs, rap_challenges, bus_public_inputs, trace_length)` |
| `AIR::composition_poly_degree_bound` | `fn(&self) -> usize` | `fn(&self, trace_length: usize) -> usize` |
| `TraceTable` | Fixed aux columns at construction | `allocate_aux_table` method for lazy allocation |
| `multi_prove` | Sequential loop, returns last proof | Phased: all main commits, shared challenges, all aux commits |
| Preprocessed tables | Not supported | `is_preprocessed`, `precomputed_commitment`, `num_precomputed_columns` |
| `Frame` access | via `get_evaluation_step` -> `get_main/aux_evaluation_element` | Same, but also uses `TableView` directly |

### 1.2 What lambda_vm Has That lambdaworks Lacks

1. **Core bus types**: `BusInteraction`, `BusValue`, `Packing`, `Multiplicity`, `LinearTerm`
2. **`AirWithBuses` wrapper**: auto-generates LogUp constraints (term + accumulated)
3. **`BusPublicInputs`**: carries initial/final accumulated values for boundary constraints
4. **Auto-generated constraints**: `LookupTermConstraint` and `LookupAccumulatedConstraint`
5. **Debug tracker**: `BusDebugTracker` with orphan/mismatch analysis
6. **Phased multi-table proving**: shared challenges across tables
7. **Preprocessed table support**: hardcoded commitments for lookup tables

### 1.3 What lambdaworks Already Has

1. A working RAP (Randomized AIR with Preprocessing) flow via `build_auxiliary_trace` / `build_rap_challenges`
2. A manual LogUp example (`read_only_memory_logup.rs`) proving the math works
3. A manual permutation example (`read_only_memory.rs`) and RAP example (`fibonacci_rap.rs`)
4. `multi_prove` / `multi_verify` (sequential, but the infrastructure exists)
5. Stone prover compatibility tests

---

## 2. Design Decisions

### 2.1 Non-Breaking Integration

The lambdaworks `AIR` trait has a different signature from lambda_vm. Rather than modifying the existing trait (which would break all existing AIRs), we will:

1. **Add new optional methods** with default implementations to the `AIR` trait where needed
2. **Create a separate `lookup` module** (`crates/provers/stark/src/lookup/`) that contains bus types and the `AirWithBuses` wrapper
3. **Extend `AirContext`** with an optional field for bus-related metadata
4. **Keep `AirWithBuses` as a concrete struct implementing `AIR`**, not a new trait

### 2.2 Simplifications Over lambda_vm

1. **No VM-specific packings initially**: `Packing` variants like `DWordBL`, `QuadHL`, etc. are VM-specific. Phase 1 will include `Direct`, `Word2L`, `Word4L` only, deferring compound packings.
2. **No `TableView` refactor**: lambdaworks uses `Frame::get_evaluation_step` while lambda_vm uses `TableView` directly. We will keep the lambdaworks pattern.
3. **No preprocessed tables in v1**: Preprocessed table support requires changes to the verifier. Defer to Phase 5.

### 2.3 `AIR` Trait Changes

The minimal required changes to the `AIR` trait:

```rust
// New: build_auxiliary_trace returns optional bus public inputs
fn build_auxiliary_trace(
    &self,
    _main_trace: &mut TraceTable<Self::Field, Self::FieldExtension>,
    _rap_challenges: &[FieldElement<Self::FieldExtension>],
) -> Option<BusPublicInputs<Self::FieldExtension>> {
    // Default: no bus public inputs (backward compatible)
    None
}
```

This is the only signature change. The current signature returns `()`, so existing implementations that override it will get a compile error. The fix is trivial: add `None` as the return value. Alternatively, we can keep the current signature and add a separate method:

```rust
fn bus_public_inputs(
    &self,
    _main_trace: &TraceTable<Self::Field, Self::FieldExtension>,
    _rap_challenges: &[FieldElement<Self::FieldExtension>],
) -> Option<BusPublicInputs<Self::FieldExtension>> {
    None
}
```

**Recommendation**: Use the second approach (separate method) to avoid breaking existing code. The prover calls `bus_public_inputs` after `build_auxiliary_trace` to get the bus data.

### 2.4 `TraceTable` Changes

Add `allocate_aux_table` method for lazy allocation:

```rust
impl TraceTable {
    pub fn allocate_aux_table(&mut self, num_aux_columns: usize) {
        let num_rows = self.num_rows();
        self.aux_table = Table::new(
            vec![FieldElement::<E>::zero(); num_rows * num_aux_columns],
            num_aux_columns,
        );
        self.num_aux_columns = num_aux_columns;
    }
}
```

This is additive and non-breaking.

---

## 3. Phased Implementation Plan

### Phase 1: Core Types and Single-Table LogUp (2-3 weeks)

**Goal**: Introduce the bus abstraction types and prove a single-table LogUp works end-to-end.

#### Deliverables

1. **New module**: `crates/provers/stark/src/lookup/mod.rs` with submodules:
   - `types.rs`: `BusInteraction`, `BusValue`, `Packing`, `Multiplicity`, `LinearTerm`, `BusPublicInputs`
   - `constraints.rs`: `LookupTermConstraint`, `LookupAccumulatedConstraint`
   - `air.rs`: `AirWithBuses<F, E, B, PI>`, `AuxiliaryTraceBuildData`, `BoundaryConstraintBuilder`, `NullBoundaryConstraintBuilder`
   - `trace.rs`: `build_logup_term_column`, `build_accumulated_column`

2. **`TraceTable` extension**: Add `allocate_aux_table` method

3. **AIR trait extension**: Add `bus_public_inputs` method with default `None` return

4. **Boundary constraint change**: `boundary_constraints` needs bus public inputs. Add a new method `boundary_constraints_with_bus` that takes the extra parameter, with a default that delegates to the existing method.

5. **Register in `lib.rs`**: `pub mod lookup;`

#### Files Created/Modified

| File | Action |
|------|--------|
| `src/lookup/mod.rs` | Create |
| `src/lookup/types.rs` | Create |
| `src/lookup/constraints.rs` | Create |
| `src/lookup/air.rs` | Create |
| `src/lookup/trace.rs` | Create |
| `src/lib.rs` | Add `pub mod lookup;` |
| `src/trace.rs` | Add `allocate_aux_table` |
| `src/traits.rs` | Add `bus_public_inputs` default method |

#### Test Criteria

- All existing tests pass (backward compatibility)
- Port `read_only_memory_logup.rs` to use `AirWithBuses`: prove + verify succeeds
- Unit tests for `Packing::combine`, `BusValue::combine_from`, `Multiplicity` evaluation
- Unit test: single interaction (sender only) produces correct term column
- Unit test: accumulated column is correct prefix sum

#### Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| `AirWithBuses` signature mismatch with `AIR` trait | `AirWithBuses::new` as a standalone constructor, `AIR::new` unreachable |
| Boundary constraint signature difference | New optional method with default delegation |
| `FieldElement<F>` clone issues in generic code | Follow `.clone()` pattern from MEMORY.md |

---

### Phase 2: Multi-Table Prover Restructuring (2-3 weeks)

**Goal**: Restructure `multi_prove` to support phased execution with shared challenges, enabling correct cross-table bus balancing.

#### Deliverables

1. **New `multi_table_prover.rs`**: Rewrite with phased protocol:
   - Phase A: Commit all main traces to transcript
   - Phase B: Sample shared RAP challenges (z, alpha) once
   - Phase C: Build all auxiliary traces using shared challenges
   - Phase D: Commit all auxiliary traces
   - Phase E: Continue STARK protocol per table

2. **Multi-table proof structure**: Either a combined proof or a vector of per-table proofs

3. **Bus balance check**: After all aux traces are built, verify that the sum of all `BusPublicInputs.final_accumulated` values equals zero

4. **Multi-table verifier update**: Corresponding changes to `multi_table_verifier.rs`

#### Files Created/Modified

| File | Action |
|------|--------|
| `src/multi_table_prover.rs` | Rewrite |
| `src/multi_table_verifier.rs` | Rewrite |
| `src/proof/stark.rs` | Possibly extend for multi-table proofs |

#### Test Criteria

- Port the `multi_table_lookup` example from lambda_vm (CPU + ADD + MUL tables)
- Prove and verify succeeds
- Tamper with a multiplicity and verify fails (soundness)
- Tamper with a value and verify fails (soundness)
- Existing single-table tests still pass

#### Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Breaking the prover protocol flow | Phase the changes carefully; keep `prove` for single tables unchanged |
| Transcript ordering differences | Extensive compatibility tests with fixed seeds |
| Performance regression from phased approach | Benchmark before/after; the phased approach may actually improve parallelism |

---

### Phase 3: Constraint Generation Improvements (1-2 weeks)

**Goal**: Optimize constraint evaluation and add compound packing support.

#### Deliverables

1. **Compound packings**: Add `DWordWL`, `DWordHHW`, `DWordWHH`, `DWordHL`, `DWordBL`, `QuadHL`, `QuadWL`

2. **Alpha power caching**: Precompute alpha powers in `LookupTermConstraint::evaluate` rather than recomputing per row

3. **Batch fingerprint computation**: For the auxiliary trace build, compute all fingerprints with a single batch inverse rather than per-row inverse

4. **Constraint evaluation optimization**: Avoid redundant allocation in the transition constraint evaluation hot path

#### Files Modified

| File | Action |
|------|--------|
| `src/lookup/types.rs` | Add compound packings |
| `src/lookup/constraints.rs` | Add alpha power caching |
| `src/lookup/trace.rs` | Add batch fingerprint computation |

#### Test Criteria

- All existing tests pass
- Benchmark: batch fingerprint vs per-row inverse on trace sizes 2^10 to 2^20
- Unit tests for all compound packing combinations
- Correctness regression test: same proof output with/without optimization

---

### Phase 4: Debug Support (1 week)

**Goal**: Port the bus debug tracker for development-time debugging of bus imbalances.

#### Deliverables

1. **`bus_debug.rs`**: Port `BusDebugTracker`, `BusInteractionLog`, `BusMismatchReport` with the analysis logic
2. **Feature flag**: `debug-checks` cargo feature
3. **Integration with `build_logup_term_column`**: Log interactions when feature is enabled
4. **Per-bus sum tracking**: In `BusPublicInputs`, carry per-bus sums when debug enabled
5. **`DEBUG_BUS_ID` env var filtering**: Runtime filter for specific bus

#### Files Created/Modified

| File | Action |
|------|--------|
| `src/lookup/debug.rs` | Create |
| `src/lookup/trace.rs` | Add debug logging calls |
| `src/lookup/types.rs` | Add debug fields to `BusPublicInputs` |
| `Cargo.toml` | Add `debug-checks` feature |

#### Test Criteria

- Balanced bus: report is empty
- Orphan sender: report identifies the orphan
- Multiplicity mismatch: report identifies the mismatch with table/row/multiplicity details
- Truncation guard: does not OOM on large traces

---

### Phase 5: Preprocessed Tables and Advanced Features (2-3 weeks)

**Goal**: Support preprocessed (precomputed) lookup tables and deferred features.

#### Deliverables

1. **Preprocessed table support**: `AirWithBuses::with_preprocessed(commitment, num_cols)`
2. **Verifier changes**: Use hardcoded commitment for preprocessed tables
3. **Prover changes**: Skip main trace commitment for preprocessed columns (use hardcoded)
4. **`read_only_memory_logup.rs` migration**: Convert the existing manual example to use buses, demonstrating API

#### Files Modified

| File | Action |
|------|--------|
| `src/lookup/air.rs` | Add preprocessed support |
| `src/prover.rs` | Handle preprocessed tables |
| `src/verifier.rs` | Handle preprocessed commitments |
| `src/examples/read_only_memory_logup.rs` | Rewrite using buses |

#### Test Criteria

- Preprocessed table: prove + verify succeeds
- Verifier uses hardcoded commitment, not prover-provided one
- Tampered preprocessed commitment: verify fails
- `read_only_memory_logup` example works with bus abstraction

---

## 4. Testing Strategy

### 4.1 Test Layers

```
Layer 4: End-to-end (prove + verify)
  - Single table LogUp
  - Multi-table LogUp (CPU + ADD + MUL)
  - Read-only memory with buses
  - Preprocessed tables

Layer 3: Integration (constraint evaluation)
  - AirWithBuses constraint evaluation on LDE domain
  - Boundary constraint correctness
  - Auxiliary trace shape (correct number of columns)

Layer 2: Component (individual functions)
  - build_logup_term_column correctness
  - build_accumulated_column correctness
  - LookupTermConstraint evaluation
  - LookupAccumulatedConstraint evaluation

Layer 1: Unit (pure types)
  - Packing::combine for all variants
  - BusValue::combine_from for Packed and Linear
  - Multiplicity evaluation for all variants
  - LinearTerm signed/unsigned arithmetic
```

### 4.2 Test Porting from lambda_vm

The following test files should be ported:

| lambda_vm file | lambdaworks target |
|---------------|-------------------|
| `tests/bus_tests/bus_value_tests.rs` | `src/lookup/tests/bus_value_tests.rs` |
| `tests/bus_tests/packing_tests.rs` | `src/lookup/tests/packing_tests.rs` |
| `tests/bus_tests/multiplicity_tests.rs` | `src/lookup/tests/multiplicity_tests.rs` |
| `tests/bus_tests/completeness_tests.rs` | `src/lookup/tests/completeness_tests.rs` |
| `tests/bus_tests/soundness_tests.rs` | `src/lookup/tests/soundness_tests.rs` |
| `examples/multi_table_lookup.rs` | `src/examples/multi_table_lookup.rs` |
| `bus_debug.rs` tests | `src/lookup/tests/debug_tests.rs` |

### 4.3 Soundness Tests

Every soundness test should verify that tampering causes verification failure:

- **Wrong multiplicity**: Change a multiplicity value; verify fails
- **Wrong value**: Change a bus value in the main trace; verify fails
- **Missing interaction**: Remove one side of a sender/receiver pair; verify fails (bus imbalance)
- **Wrong bus_id**: Use different bus_id for sender and receiver; verify fails
- **Challenge manipulation**: Ensure challenges are properly bound to transcript

### 4.4 Regression Tests

- The existing `read_only_memory_logup` test must continue to pass throughout all phases
- Stone prover compatibility tests must not be affected
- All Fibonacci, quadratic, periodic, bit flags examples must pass

### 4.5 Migration Path for `read_only_memory_logup.rs`

**Decision**: Option (a) -- Rewrite it using the bus abstraction in Phase 5.

Rationale:
- Keeping the manual implementation during Phases 1-4 serves as a regression test
- Rewriting it in Phase 5 serves as the "proof that the API works" milestone
- The manual version documents the math clearly; the bus version documents the API clearly

Steps:
1. Phases 1-4: Keep `read_only_memory_logup.rs` as-is; it tests the raw RAP mechanism
2. Phase 5: Create `read_only_memory_bus.rs` using `AirWithBuses`; verify both produce same proof
3. After Phase 5 validation: Replace the original with the bus version, keeping the old one as a comment/reference

---

## 5. Risk Assessment

### 5.1 Technical Risks

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Breaking existing `AIR` implementations | High | Medium | Use additive-only trait changes (new methods with defaults) |
| Soundness bug in LogUp constraint porting | Critical | Low | Port tests directly from lambda_vm; cross-validate proofs |
| Performance regression in prover | Medium | Medium | Benchmark at each phase; batch inverse optimization in Phase 3 |
| `FieldElement<F>` clone/copy issues | Low | High | Follow MEMORY.md patterns; use `.clone()` everywhere in generic code |
| Multi-table transcript ordering mismatch | High | Medium | Fixed-seed deterministic tests; compare transcript states |
| `AirWithBuses` cannot satisfy `AIR` trait | High | Low | Already proven in lambda_vm; verify trait bounds early in Phase 1 |

### 5.2 Novelty Risks

| Risk | Assessment |
|------|-----------|
| Prior art overlap | Not applicable -- this is a port of our own code, not a research contribution |
| Incorrect LogUp math | Low -- the math is well-established (Haboeck 2022, Polygon Miden) and already validated in lambda_vm |
| API complexity explosion | Medium -- mitigated by keeping `AirWithBuses` as the only entry point; users do not touch constraints |

### 5.3 Scope Risks

| Risk | Mitigation |
|------|-----------|
| Phase 2 (multi-table) is larger than estimated | Can split into 2a (shared challenges) and 2b (proof structure) |
| Phase 5 (preprocessed) requires verifier changes | Can defer entirely; preprocessed tables are not needed for basic bus usage |
| VM-specific packings creep into Phase 1 | Strictly limit Phase 1 to primitive packings only |

---

## 6. Module Organization

```
crates/provers/stark/src/
  lookup/
    mod.rs              -- Public API re-exports
    types.rs            -- BusInteraction, BusValue, Packing, Multiplicity, LinearTerm, BusPublicInputs
    constraints.rs      -- LookupTermConstraint, LookupAccumulatedConstraint
    air.rs              -- AirWithBuses, AuxiliaryTraceBuildData, BoundaryConstraintBuilder
    trace.rs            -- build_logup_term_column, build_accumulated_column
    debug.rs            -- BusDebugTracker (behind feature flag)
    tests/
      mod.rs
      bus_value_tests.rs
      packing_tests.rs
      multiplicity_tests.rs
      completeness_tests.rs
      soundness_tests.rs
      debug_tests.rs
  examples/
    multi_table_lookup.rs    -- Phase 2 example (CPU + ADD + MUL)
    read_only_memory_bus.rs  -- Phase 5 migration
```

---

## 7. Effort Estimates

| Phase | Duration | Dependencies |
|-------|----------|-------------|
| Phase 1: Core Types + Single-Table | 2-3 weeks | None |
| Phase 2: Multi-Table Prover | 2-3 weeks | Phase 1 |
| Phase 3: Constraint Optimization | 1-2 weeks | Phase 1 |
| Phase 4: Debug Support | 1 week | Phase 1 |
| Phase 5: Preprocessed Tables | 2-3 weeks | Phase 2 |
| **Total** | **8-12 weeks** | |

Phases 3 and 4 can run in parallel with Phase 2 since they depend only on Phase 1.

---

## 8. Definition of Done

### v1 (Phases 1-2 complete)

- [ ] `AirWithBuses` compiles and implements `AIR`
- [ ] Single-table LogUp proof: prove + verify succeeds
- [ ] Multi-table LogUp proof with shared challenges: prove + verify succeeds
- [ ] Bus balance check: sum of all `final_accumulated` values is zero
- [ ] All existing lambdaworks tests pass (zero regressions)
- [ ] Soundness tests: tampered proofs are rejected
- [ ] Unit tests for all core types

### v2 (Phases 3-5 complete)

- [ ] All compound packings implemented and tested
- [ ] Batch fingerprint optimization benchmarked
- [ ] Debug tracker functional behind feature flag
- [ ] Preprocessed table support with hardcoded commitments
- [ ] `read_only_memory_logup.rs` migrated to bus abstraction
- [ ] Documentation in code and module-level doc comments

---

## 9. Open Questions

1. **Should `AirWithBuses` live in the STARK crate or in a separate crate?**
   Recommendation: STARK crate. It is tightly coupled to `AIR`, `TransitionConstraint`, and `TraceTable`. A separate crate would create circular dependencies.

2. **Should we change `build_auxiliary_trace` signature or add a new method?**
   Recommendation: Add a new method `bus_public_inputs` to avoid breaking changes. The prover calls both.

3. **Should `multi_prove` return a single combined proof or a vector of proofs?**
   Recommendation: Vector of proofs (like lambda_vm). A combined proof would require significant proof structure changes. The bus balance is checked via the public inputs, not the proof structure.

4. **Should `Packing` be extensible (trait) or closed (enum)?**
   Recommendation: Closed enum. The combining formulas are hardcoded arithmetic; a trait would add complexity without clear benefit. Users needing custom combining can use `BusValue::Linear`.
