# Deliberation Report: Porting Buses and LogUp Argument into Lambdaworks

**Date:** 2026-02-10
**Branch:** `feat/buses-logup-argument`
**Committee Members:** Chairman, Theorist, Analyst, Research Planner, Reviewer
**Status:** Consensus reached with action plan

---

## 1. Question Addressed

How should we port, improve, and incorporate the buses and LogUp lookup argument from `lambda_vm` (a VM-specific implementation) into `lambdaworks` (a general-purpose cryptographic library)?

The lambda_vm has a mature 1417-line implementation in `lookup.rs` with `AirWithBuses`, multi-table phased execution, debug tracking, and preprocessed table support. Lambdaworks has a STARK prover with RAP support and a hand-rolled LogUp example, but no reusable bus abstraction.

---

## 2. Agent Perspectives

### 2.1 Chairman (Framing)

Identified 5 key architectural tensions:
- **Wrapper vs. Trait Extension** for AIR integration
- **Phased Multi-Table Proving** vs. current sequential model
- **VM-Specific vs. Library-Generic** design (Packing enum)
- **Auto-Generated vs. User-Composed** constraints
- **Module Placement** (submodule vs. separate crate)

Recommended dispatching: Theorist + Reviewer in parallel, then Planner, then Experimentalist. Noted that `TransitionEvaluationContext` Prover/Verifier duplication is a pain point the bus module should solve.

### 2.2 Theorist (Formal Analysis)

**Key findings:**

1. **Bus ID soundness is correct.** Placement as constant term in fingerprint ensures cross-bus collision probability is exactly 0 (not probabilistic) when bus IDs differ, regardless of column values.

2. **CRITICAL: Packing with powers of 2 is UNSOUND on small fields.** For BabyBear (p ~ 2^31) and Mersenne-31 (p = 2^31-1), Word2L packing `h_0 + 2^16 * h_1` can produce values up to ~2^47, which wraps modulo p, creating false collisions. This is a soundness bug in lambda_vm when used with small fields.

3. **Term constraint is irreducibly degree 2.** The existing lambdaworks LogUp example uses degree-3 constraints (30% worse for FRI proof size). The separate term column approach reduces to degree 2 at the cost of one extra auxiliary column per interaction -- almost always worth it.

4. **Per-interaction accumulated columns recommended** for modularity and debuggability.

5. **LogUp-GKR proposed as Phase 2** -- eliminates auxiliary columns entirely by reducing the fractional sum to a sumcheck instance, leveraging existing `crates/provers/sumcheck/`.

6. **Soundness bound:** epsilon <= (M*N + w) / |F_{p^k}| where M = interactions, N = rows, w = tuple width. For BabyBear with degree-4 extension: ~95 bits of security.

### 2.3 Analyst (Architecture)

**Recommendations:**

1. **Module structure:** `crates/provers/stark/src/lookup/` with 5 files (~2130 LOC total)
2. **Replace Packing with `ColumnCombiner` trait** for extensibility
3. **`AirWithLogUp` trait extension** (not wrapper struct) for composability
4. **Decompose `prove()` into phases;** expose `multi_prove_with_buses()` as public API
5. **Extend `ProvingError`** with `BusImbalance` and `ZeroFingerprint` variants
6. **Performance metrics:** 6 measurements across computation, memory, and comparison axes

### 2.4 Research Planner (Strategy)

**Five-phase plan:**

| Phase | Focus | Key Deliverable |
|-------|-------|----------------|
| 1 | Core types + single-table LogUp | `lookup/` module with constraints, packing, bus types |
| 2 | Multi-table prover restructuring | Phased `multi_prove` with shared challenges |
| 3 | Constraint optimization | Batch fingerprint, alpha power caching |
| 4 | Debug support | `BusDebugTracker` behind feature flag |
| 5 | Preprocessed tables + migration | Rewrite `read_only_memory_logup.rs` with buses |

**Key decisions:** Backward compatible additive-only AIR trait changes; keep `read_only_memory_logup.rs` as regression test; closed `Packing` enum.

### 2.5 Reviewer (Cross-Examination)

**Strongest aspects:**
- S1: Small-field packing soundness bug is the most valuable finding
- S2: Existing LogUp example provides ground-truth regression test
- S3: Degree-2 constraint improvement is significant (halves FRI proof size vs degree-3)
- S4: ~2130 LOC estimate is realistic
- S5: Per-interaction accumulated columns is correct modularity decision

**Weakest aspects:**
- W1: Wrapper-vs-trait disagreement has concrete prover pipeline consequences (Critical)
- W2: Multi-table protocol is dangerously underspecified (Critical)
- W3: `ColumnCombiner` trait invites silent unsoundness -- closed enum with validation is safer
- W4: LogUp-GKR is out of scope -- sumcheck and STARK transcripts are incompatible today
- W5: LOC estimate excludes verifier changes

**Resolved disagreements:**
- Use **trait extension with helpers** (neither pure wrapper nor pure trait)
- Use **closed `Packing` enum** with runtime validation (not open trait)
- Use **per-interaction accumulated columns** as default

**Unresolved issues:**
- Multi-table transcript ordering needs a concrete protocol specification
- Extension field strategy for small fields needs explicit documentation
- Zero fingerprint recovery (resample challenges vs. error)

---

## 3. Points of Agreement

All agents agreed on:

1. **The bus/LogUp module belongs in `crates/provers/stark/src/lookup/`** -- not a separate crate (too tightly coupled to AIR/TraceTable).

2. **Packing with powers of 2 must be validated against field size.** On small fields (BabyBear, Mersenne-31), only `Direct` (single column) or extension-field fingerprinting is safe.

3. **Degree-2 constraints (separate term column) should be the default.** The existing degree-3 approach in `read_only_memory_logup.rs` is suboptimal.

4. **Per-interaction accumulated columns** for modularity (with a folding optimization available for production use).

5. **The `read_only_memory_logup.rs` example should be kept as regression test** during development, then rewritten using the new API.

6. **The multi-table prover must be restructured** for phased execution with shared challenges. This is a soundness requirement, not just an optimization.

7. **LogUp-GKR should be deferred** to a separate future effort. The current sumcheck infrastructure is not compatible with the STARK transcript system.

---

## 4. Points of Disagreement (Resolved)

### 4.1 Wrapper Struct vs. Trait Extension

| Proposal | Advocate | Resolution |
|----------|----------|------------|
| `AirWithBuses<A>` wrapper (lambda_vm style) | Planner | Rejected |
| `AirWithLogUp: AIR` trait extension | Analyst | Modified |
| Hybrid: trait extension with composable helpers | Reviewer | **Adopted** |

**Resolution:** The reviewer identified that the wrapper has a construction ordering problem (it can't implement `AIR::new()` cleanly) while the pure trait extension is error-prone. The adopted approach:

- Define `AirWithLogUp: AIR` trait with methods: `bus_interactions()`, `logup_aux_column_count()`, `logup_transition_constraints()`, `build_logup_auxiliary_trace()`
- Provide default implementations that users call from their `AIR` trait methods
- Users retain full control of their AIR impl but get composable LogUp building blocks

### 4.2 Packing Enum Scope

| Proposal | Advocate | Resolution |
|----------|----------|------------|
| Generic `ColumnCombiner` trait | Analyst | Rejected |
| Closed `Packing` enum | Planner | Rejected |
| No Packing -- users handle combining | User feedback | **Adopted** |

**Resolution:** The `Packing` enum is VM-specific syntactic sugar. The library provides `BusValue` as column references (direct or linear combination). Users who need to combine columns (byte packing, etc.) do so in their trace generation and pass the combined value as a column. Soundness of combining is the user's responsibility, not the library's.

### 4.3 Single vs. Per-Interaction Accumulated Columns

| Proposal | Advocate | Resolution |
|----------|----------|------------|
| Single accumulated column (lambda_vm) | Planner (by default) | Rejected as default |
| Per-interaction accumulated columns | Theorist | **Adopted as default** |
| Optimization to fold into single column | Analyst, Reviewer | Available as opt-in |

**Resolution:** Per-interaction is more modular, debuggable, and allows independent verification. The extra column cost (one per interaction) is acceptable. A `fold_accumulated_columns()` helper can merge them for production VMs that need to minimize column count.

---

## 5. Committee Recommendation

### Scope Clarification

The `Packing` enum (Word2L, Word4L, DWordHL, etc.) from lambda_vm is **out of scope**. It is VM-specific syntactic sugar for combining columns using powers of 2. The library only provides core LogUp protocol machinery: fingerprinting, bus interactions, constraints, multi-table shared challenges. Users pass column values directly and are responsible for the soundness of whatever combining they do externally.

### Architecture

```
crates/provers/stark/src/lookup/
  mod.rs              -- Re-exports, module documentation, AirWithLogUp trait
  bus.rs              -- BusInteraction, BusValue, Multiplicity, sender/receiver
  fingerprint.rs      -- Fingerprint computation, alpha-combination
  constraints.rs      -- LookupTermConstraint, LookupAccumulatedConstraint
  trace_builder.rs    -- Auxiliary trace construction helpers
  debug.rs            -- BusDebugTracker (behind cfg(debug_assertions))
  types.rs            -- BusPublicInputs, BusConfig
```

### Phased Implementation Plan

#### Phase 1: Core Types and Single-Table LogUp (Priority: Highest)

**Goal:** A working LogUp module that can replace the hand-rolled example.

**Files to create:**
- `lookup/mod.rs` -- `AirWithLogUp` trait, re-exports
- `lookup/bus.rs` -- `BusInteraction`, `BusValue`, `Multiplicity`
- `lookup/fingerprint.rs` -- fingerprint computation with alpha powers
- `lookup/constraints.rs` -- `LookupTermConstraint` (degree 2), `LookupAccumulatedConstraint` (degree 1)
- `lookup/trace_builder.rs` -- `build_logup_term_column()`, `build_accumulated_column()`
- `lookup/types.rs` -- `BusPublicInputs`, `BusConfig`

**Files to modify:**
- `prover.rs` -- extend `ProvingError` with `BusImbalance`, `ZeroFingerprint`
- `traits.rs` -- no changes needed (additive trait extension)
- `mod.rs` -- add `pub mod lookup;`

**Notes on Packing:** No `Packing` enum. `BusValue` provides column references and optional linear combinations. Users compute any column-combining (Word2L, etc.) in their trace generation and pass the result as a direct column value. Soundness of combining is the user's responsibility.

**Tests:**
- Port lambda_vm's single-table test cases
- Rewrite `read_only_memory_logup.rs` as validation (keep original for regression)
- Cross-validate: same input must produce same proof output

**Milestone:** Single-table LogUp proof verifies correctly with automated bus constraints.

#### Phase 2: Multi-Table Prover Restructuring (Priority: Critical for buses)

**Goal:** Phased `multi_prove_with_buses()` with shared LogUp challenges.

**Prerequisite:** Write a concrete transcript protocol specification documenting exact ordering of operations:

```
Protocol: Multi-Table LogUp Proving
1. For each table i:
   a. Commit main trace -> append Merkle root to transcript
   b. If preprocessed: append hardcoded commitment to transcript
2. Sample shared LogUp challenges:
   a. z = transcript.sample_field_element()
   b. alpha = transcript.sample_field_element()
3. For each table i:
   a. Build auxiliary trace using (z, alpha)
   b. Commit auxiliary trace -> append Merkle root to transcript
4. For each table i:
   a. Run Rounds 2-4 of STARK protocol independently
5. Bus balance check:
   a. Sum all final_accumulated values across all tables
   b. Assert sum == 0
```

**Files to modify:**
- `prover.rs` -- decompose `prove()` into `commit_main_trace()`, `build_and_commit_aux()`, `prove_remaining_rounds()`
- `multi_table_prover.rs` -- new `multi_prove_with_buses()` implementing the phased protocol
- `multi_table_verifier.rs` -- new `multi_verify_with_buses()` mirroring the protocol

**Tests:**
- Port lambda_vm's multi-table example (CPU + ADD + MUL tables)
- Fixed-seed deterministic tests that compare transcript states
- Existing `multi_prove` tests must still pass

**Milestone:** Multi-table proof with cross-table buses verifies correctly.

#### Phase 3: Debug Support (Priority: High for usability)

**Goal:** Diagnostic tooling for bus imbalance.

**Files to create:**
- `lookup/debug.rs` -- `BusDebugTracker` with orphan sender/receiver detection, multiplicity mismatch analysis

**Feature flag:** `debug-checks` (mirrors lambda_vm)

**Milestone:** Debug output identifies which table/row/bus causes an imbalance.

#### Phase 4: Preprocessed Tables (Priority: Medium)

**Goal:** Support for deterministic lookup tables (range checks, bitwise operations).

**Files to modify:**
- `lookup/mod.rs` -- extend `AirWithLogUp` with `precomputed_commitment()`
- `prover.rs` -- handle preprocessed commitments in multi-table flow
- `verifier.rs` -- verify preprocessed commitment matches hardcoded value

**Milestone:** A bitwise lookup table with hardcoded commitment proves and verifies.

#### Phase 5: Migration and Polish (Priority: Medium)

**Goal:** Replace the hand-rolled example; optimize.

**Tasks:**
- Rewrite `examples/read_only_memory_logup.rs` using `AirWithLogUp` trait
- Add batch fingerprint inversion (Montgomery's trick)
- Cache alpha powers
- Add module-level documentation with Haboeck 2022 reference
- Performance benchmarks (6 metrics from analyst's plan)

**Milestone:** All existing tests pass. New API is documented with examples.

---

## 6. Deferred Work (Future Proposals)

1. **LogUp-GKR:** Eliminate auxiliary columns by reducing fractional sum to sumcheck. Requires bridging `IsStarkTranscript` and `DefaultTranscript`, converting trace polynomials to multilinear evaluations, and modifying the composition polynomial. Estimated multi-month effort.

2. **Packing utilities:** `Packing` enum (Word2L, Word4L, DWordHL, QuadHL, etc.) is VM-specific syntactic sugar. Excluded from this effort entirely. VM projects that need it can build it on top of the `BusValue` abstraction.

3. **Plonk LogUp integration:** LogUp can replace Plookup with better asymptotics, but requires custom Plonk gates.

---

## 7. Critical Soundness Requirements

These MUST be enforced in the implementation:

1. **Extension field for challenges:** On fields with `p < 2^64`, challenges `(z, alpha)` must be sampled from an extension field with `|F_ext| >= 2^{100}`.

2. **Bus ID uniqueness:** Enforce at construction time that no two buses share the same ID.

3. **Shared challenges across tables:** For multi-table buses, all tables MUST use the same `(z, alpha)` from a shared transcript state after all main traces are committed.

4. **Zero fingerprint handling:** Use `Result` (not panic) for `fingerprint.inv()`. Return `ProvingError::ZeroFingerprint { row }` and allow the caller to retry with fresh randomness.

5. **Column value soundness is the user's responsibility.** The library does not validate or constrain how users compute the values they pass as `BusValue`. If users combine columns externally (e.g., byte packing with powers of 2), they must ensure the combining is injective over their field.

---

## 8. Dissenting Opinions

**Theorist dissent on LogUp-GKR deferral:** The theorist believes LogUp-GKR should remain in scope as Phase 2, arguing that the existing sumcheck infrastructure is close to usable and the auxiliary column savings are significant (10-20 columns eliminated for a typical VM). The reviewer counters that the transcript incompatibility between sumcheck and STARK provers makes this a multi-month effort that should not block the core LogUp work. **The committee sides with the reviewer: defer LogUp-GKR.**

**Packing discussion rendered moot:** The user clarified that `Packing` is VM-specific sugar and out of scope. The library provides `BusValue` for column references; users handle any combining externally. No dissent on this point.

---

## 9. Action Plan

| Priority | Task | Assigned | Depends On |
|----------|------|----------|------------|
| P0 | Write multi-table transcript protocol specification | Research Planner | -- |
| P0 | Implement `lookup/` module core types (Phase 1) | Experimentalist | -- |
| P0 | Implement packing validation with field-size checks | Experimentalist | Core types |
| P1 | Implement `LookupTermConstraint` + `LookupAccumulatedConstraint` | Experimentalist | Core types |
| P1 | Implement `build_logup_auxiliary_trace()` helper | Experimentalist | Constraints |
| P1 | Port single-table test suite from lambda_vm | Experimentalist | Trace builder |
| P2 | Decompose `prove()` into phases | Experimentalist | Phase 1 complete |
| P2 | Implement `multi_prove_with_buses()` | Experimentalist | Phase decomposition |
| P2 | Implement `multi_verify_with_buses()` | Experimentalist | multi_prove |
| P3 | Implement `BusDebugTracker` | Writer | Phase 2 complete |
| P3 | Implement preprocessed table support | Experimentalist | Phase 2 complete |
| P4 | Rewrite `read_only_memory_logup.rs` with new API | Writer | Phase 3 |
| P4 | Performance benchmarks | Analyst | Phase 3 |
| P4 | Module documentation | Writer | All phases |

---

## References

- Haboeck, U. (2022). "Multivariate lookups based on logarithmic derivatives." IACR ePrint 2022/1530.
- Haboeck, U. (2023). "Improving logarithmic derivative lookups using GKR." IACR ePrint 2023/1284.
- Polygon Miden VM LogUp Documentation. https://0xpolygonmiden.github.io/miden-vm/design/lookups/logup.html
- LambdaClass Blog. "LogUp Lookup Argument and its Implementation using Lambdaworks." https://blog.lambdaclass.com/logup-lookup-argument-and-its-implementation-using-lambdaworks-for-continuous-read-only-memory/
