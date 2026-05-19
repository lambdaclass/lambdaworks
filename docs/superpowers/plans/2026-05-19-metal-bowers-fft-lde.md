# Metal Bowers FFT + LDE → Merkle Commit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an end-to-end Metal GPU pipeline that takes a 2^20 × 64 Goldilocks (or Fp3) coefficient matrix, performs blowup-2 LDE via a fused Bowers-G NTT, and writes Merkle-leaf hashes ready for the existing Keccak Merkle tree builder. Target: ≥ 2× speedup over the current Metal commit path at 2^20 × 64.

**Architecture:** Three new Metal kernels per field (`bowers_lde_stage0`, `bowers_lde_middle`, `bowers_lde_tail`) sharing a templated DIF butterfly. Coset multiply is folded into stage 0. Output is in bitrev order within each column; the Merkle layer absorbs that convention via a new `LeafOrder` flag. Opt-in behind a `bowers-fft` cargo feature; existing Metal FFT untouched.

**Tech Stack:** Rust, Metal shading language, `lambdaworks-gpu` Metal abstractions, Criterion. Spec: `docs/superpowers/specs/2026-05-19-metal-bowers-fft-lde-design.md`.

---

## File Structure

**Create:**
- `crates/math/src/fft/gpu/metal/bowers/mod.rs` — module entry, public re-exports
- `crates/math/src/fft/gpu/metal/bowers/twiddles.rs` — CPU-side bitrev twiddles + coset powers, cached
- `crates/math/src/fft/gpu/metal/bowers/dispatcher.rs` — `metal_bowers_lde`, `metal_bowers_lde_fp3`, fused-tail K computation
- `crates/math/src/gpu/metal/shaders/fft/bowers_butterfly.h.metal` — templated DIF butterfly
- `crates/math/src/gpu/metal/shaders/fft/bowers_lde.h.metal` — templated kernel bodies (stage0, middle, tail)
- `crates/math/src/gpu/metal/shaders/fft/bowers_lde_goldilocks.metal` — Goldilocks kernel entry points
- `crates/math/src/gpu/metal/shaders/fft/bowers_lde_goldilocks_fp3.metal` — Fp3 kernel entry points
- `crates/math/src/fft/cpu/ntt_bowers_goldilocks.rs` — promote the untracked CPU reference into the tree
- `crates/math/src/fft/cpu/ntt_bowers_fp3.rs` — Fp3 CPU Bowers reference (for golden-vector tests)
- `crates/math/src/fft/gpu/metal/bowers/tests.rs` — unit tests (golden, coset, determinism, multicol)
- `crates/math/benches/criterion_metal_bowers_lde.rs` — Criterion bench at target shape
- `crates/provers/stark-gpu/src/metal/commit_bowers.rs` — `commit_matrix_bowers` glue
- `crates/provers/stark-gpu/tests/commit_bowers_roundtrip.rs` — end-to-end commit+verify

**Modify:**
- `crates/math/src/fft/gpu/metal/mod.rs` — add `pub mod bowers;`
- `crates/math/src/fft/cpu/mod.rs` — add `pub mod ntt_bowers_goldilocks; pub mod ntt_bowers_fp3;`
- `crates/math/Cargo.toml` — bench entry for `criterion_metal_bowers_lde`
- `crates/provers/stark-gpu/Cargo.toml` — add `bowers-fft` feature
- `crates/provers/stark-gpu/src/metal/mod.rs` — gate `pub mod commit_bowers;` behind `bowers-fft`
- `crates/provers/stark-gpu/src/commit.rs` — add `LeafOrder { Natural, Bitrev }` to the commitment config

---

## Conventions used in every task

- `cargo test -p lambdaworks-math --lib --features metal -- bowers` — runs all Bowers unit tests
- `cargo test -p lambdaworks-stark-gpu --features "metal bowers-fft"` — runs stark-gpu Bowers tests
- `cargo clippy -p lambdaworks-math --features metal -- -D warnings`
- `cargo fmt -p lambdaworks-math -p lambdaworks-stark-gpu -- --check`
- Commit messages: `feat(metal-bowers): <thing>` or `test(metal-bowers): <thing>`
- After every code change: clippy + fmt + relevant tests. Each task ends with a commit.

---

## Task 1 — Promote CPU Bowers reference into the tree

**Files:** Create `crates/math/src/fft/cpu/ntt_bowers_goldilocks.rs` (already exists untracked); modify `crates/math/src/fft/cpu/mod.rs`.

- [ ] Step 1: `ls crates/math/src/fft/cpu/ntt_bowers_goldilocks.rs` — verify file exists.
- [ ] Step 2: Append `pub mod ntt_bowers_goldilocks;` to `crates/math/src/fft/cpu/mod.rs`.
- [ ] Step 3: `cargo build -p lambdaworks-math --lib` — expect success.
- [ ] Step 4: Append a smoke test:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    const G: u128 = 7;
    #[test]
    fn bowers_ntt_smoke_log4() {
        let n = 16usize;
        let tw = compute_bowers_twiddles(n, G);
        let mut data: Vec<u64> = (0..n as u64).collect();
        let original = data.clone();
        ntt_bowers(&mut data, &tw);
        assert_ne!(data, original);
        assert!(data.iter().all(|&x| x < P));
    }
}
```

- [ ] Step 5: `cargo test -p lambdaworks-math --lib ntt_bowers_goldilocks::tests` — expect PASS.
- [ ] Step 6: Commit `feat(metal-bowers): land CPU Goldilocks Bowers NTT reference`.

---

## Task 2 — Fp3 CPU Bowers reference

**Files:** Create `crates/math/src/fft/cpu/ntt_bowers_fp3.rs`; modify `crates/math/src/fft/cpu/mod.rs`.

- [ ] Step 1: Create the file with `ntt_bowers_fp3(data: &mut [Fp3], twiddles: &[Fp])` — bit-reverse input, then for `log_half in 0..log_n` run DIF butterflies with `w = twiddles[num_blocks + b]`, `wbb = bb * w` (Fp3 × Fp via `IsSubFieldOf`), `data[a] = a + wbb`, `data[b] = a - wbb`. Plus a `fp3_bowers_smoke_log4` test.
- [ ] Step 2: Add `pub mod ntt_bowers_fp3;` to `crates/math/src/fft/cpu/mod.rs`.
- [ ] Step 3: `cargo test -p lambdaworks-math --lib ntt_bowers_fp3`. If trait-bound errors about `Fp3 * Fp`, copy imports from `crates/math/src/fft/cpu/ops.rs`. Iterate until green.
- [ ] Step 4: Commit `feat(metal-bowers): add Fp3 CPU Bowers NTT reference`.

---

## Task 3 — Bowers module scaffolding

**Files:** Create `crates/math/src/fft/gpu/metal/bowers/{mod,twiddles,dispatcher,tests}.rs`; modify `crates/math/src/fft/gpu/metal/mod.rs`.

- [ ] Step 1: Create `mod.rs`:

```rust
//! Bowers-G Metal NTT + fused LDE pipeline.
//! Design: docs/superpowers/specs/2026-05-19-metal-bowers-fft-lde-design.md
pub mod dispatcher;
pub mod twiddles;
#[cfg(test)]
mod tests;
pub use dispatcher::{metal_bowers_lde, metal_bowers_lde_fp3};
```

- [ ] Step 2: Create `twiddles.rs` and `tests.rs` each as a single-line `//! Placeholder.` stub. Create `dispatcher.rs` with both public functions as `unimplemented!()` stubs returning `Result<(), MetalError>`, matching the signatures used in Task 7.
- [ ] Step 3: Append `pub mod bowers;` to `crates/math/src/fft/gpu/metal/mod.rs`.
- [ ] Step 4: `cargo build -p lambdaworks-math --features metal` — expect success.
- [ ] Step 5: Commit `feat(metal-bowers): scaffold bowers module`.

---

## Task 4 — Bitrev twiddles + coset powers + cache

**Files:** Modify `crates/math/src/fft/gpu/metal/bowers/twiddles.rs`.

- [ ] Step 1: Replace `twiddles.rs` with the implementation: `bowers_twiddles_goldilocks(log_n) -> Vec<Fp>` (computes `[ω^0..ω^(n/2-1)]` then bit-reverses), `coset_powers_goldilocks(log_n, g) -> Vec<Fp>` (computes `[g^0..g^(n-1)]`), thread-safe caches via `once_cell::Lazy<Mutex<HashMap<_, _>>>` keyed by `log_n` for twiddles and `(log_n, *g.value())` for coset. Use `Goldilocks64Field::get_primitive_root_of_unity(log_n as u64).unwrap()` for ω.
- [ ] Step 2: Add three tests:
  - `bowers_twiddles_match_cpu_ref_log10`: element-wise vs `compute_bowers_twiddles(1024, 7)`.
  - `coset_powers_basic`: `cp[0]==1`, `cp[1]==g`, `cp[2]==g*g` for `g=Fp::from(3)`, `log_n=4`.
  - `twiddle_cache_returns_equal_vectors`: two calls return equal vecs.
- [ ] Step 3: `cargo add once_cell -p lambdaworks-math` if not already a dep.
- [ ] Step 4: `cargo test -p lambdaworks-math --lib --features metal bowers::twiddles`, then clippy, fmt.
- [ ] Step 5: Commit `feat(metal-bowers): bitrev twiddles + coset powers with cache`.

---

## Task 5 — Templated DIF butterfly shader header

**Files:** Create `crates/math/src/gpu/metal/shaders/fft/bowers_butterfly.h.metal`.

- [ ] Step 1: Write the header with two templates:

```cpp
#pragma once
#include <metal_stdlib>

template<typename Fp, typename TwFp>
inline void bowers_dif_butterfly(thread Fp& a, thread Fp& b, TwFp w) {
    Fp sum  = a + b;
    Fp diff = (a - b) * w;
    a = sum;
    b = diff;
}

inline void bowers_butterfly_indices(
    uint32_t thread_pos, uint32_t stage, uint32_t n,
    thread uint32_t& a_idx, thread uint32_t& b_idx, thread uint32_t& tw_idx
) {
    uint32_t half       = n >> (stage + 1);
    uint32_t num_blocks = 1u << stage;
    uint32_t block      = thread_pos / half;
    uint32_t pos_in_blk = thread_pos & (half - 1);
    a_idx  = block * (half << 1) + pos_in_blk;
    b_idx  = a_idx + half;
    tw_idx = num_blocks + block;
}
```

- [ ] Step 2: `cargo build -p lambdaworks-math --features metal` — expect success.
- [ ] Step 3: Commit `feat(metal-bowers): templated DIF butterfly shader header`.

---

## Task 6 — Stage0 / middle / tail kernel bodies

**Files:** Create `crates/math/src/gpu/metal/shaders/fft/bowers_lde.h.metal`.

- [ ] Step 1: Write three templated bodies. `bowers_lde_stage0_body<Fp, TwFp>` computes indices for `stage=0`, loads `a = data[a_idx] * coset_powers[a_idx]` and `b = data[b_idx] * coset_powers[b_idx]`, calls `bowers_dif_butterfly` with `w = twiddles_bitrev[tw_idx]`, writes back. `bowers_lde_middle_body<Fp, TwFp>` does the same minus coset for arbitrary `stage`. `bowers_lde_tail_body<Fp, TwFp>` loads BLOCK=(1<<num_stages) elements into `threadgroup Fp* shared_data`, runs `num_stages` butterfly stages locally (computing `global_block = (tg_id << k) + local_block` for the twiddle at `global_stage = start_stage + k`), writes back. Use `threadgroup_barrier(metal::mem_flags::mem_threadgroup)` between stages and around load/store.
- [ ] Step 2: `cargo build -p lambdaworks-math --features metal`.
- [ ] Step 3: Commit `feat(metal-bowers): stage0/middle/tail kernel bodies`.

---

## Task 7 — Goldilocks kernels + Rust dispatcher

**Files:** Create `crates/math/src/gpu/metal/shaders/fft/bowers_lde_goldilocks.metal`; modify `crates/math/src/fft/gpu/metal/bowers/dispatcher.rs`.

- [ ] Step 1: `grep -rn "typedef.*[Gg]oldilocks\|using.*[Gg]oldilocks" crates/math/src/gpu/metal/shaders/` to find the existing `Fp` typedef and header path.
- [ ] Step 2: Write the Goldilocks kernel file with three `[[kernel]]` entry points (`bowers_lde_stage0_goldilocks`, `bowers_lde_middle_goldilocks`, `bowers_lde_tail_goldilocks`) each instantiating the corresponding `bowers_lde_*_body<Fp, Fp>`. Buffers: `device Fp* data [[buffer(0)]]`, `constant Fp* twiddles_bitrev [[buffer(1)]]`, `constant Fp* coset_powers [[buffer(2)]]` (stage0 only), `constant uint32_t& n` and `constant uint32_t& stage`/`start_stage`+`num_stages`. Tail kernel takes `threadgroup Fp* shared_data [[threadgroup(0)]]` plus `threadgroup_position_in_grid`, `thread_position_in_threadgroup`, `threads_per_threadgroup`.
- [ ] Step 3: `sed -n '120,260p' crates/math/src/fft/gpu/metal/ops.rs` — mirror the `MetalState` helper method names (`setup_pipeline`, `command_buffer`, `set_threadgroup_memory_length`, `alloc_buffer*`).
- [ ] Step 4: Rewrite `dispatcher.rs`. Define a private `dispatch_bowers_lde_generic(... elem_bytes, stage0_name, middle_name, tail_name ...)` that:
  - computes `k = fused_tail_k(log_n, elem_bytes as usize)` where `fused_tail_k` = `((32*1024)/sizeof_fp).trailing_zeros().min(log_n.saturating_sub(1))`
  - returns `Err(MetalError::InvalidFftSize)` if `log_n < k + 1`
  - `state.blit_buffer(cols, out)?` to seed in-place work on `out`
  - for `col in 0..num_cols`: dispatch `stage0` (`MTLSize::new(n/2, 1, 1)` threads, offset `col * n * elem_bytes`)
  - for `stage in 1..(log_n - k)` then for each `col`: dispatch `middle` similarly
  - for each `col`: dispatch `tail` as thread-groups (`num_blocks` groups of `min(256, block/2)` threads, `set_threadgroup_memory_length(0, block * elem_bytes)`)
  - `state.wait_until_completed(); Ok(())`
  - then `metal_bowers_lde` calls it with `elem_bytes=8` and the Goldilocks kernel names; `metal_bowers_lde_fp3` stays `unimplemented!("Task 12")`.
- [ ] Step 5: `cargo build -p lambdaworks-math --features metal`; reconcile `MetalState` helper names against the actual API in `ops.rs`.
- [ ] Step 6: Commit `feat(metal-bowers): Goldilocks kernels + Rust dispatcher`.

---

## Task 8 — Golden-vector tests: Metal vs CPU Bowers

**Files:** Modify `crates/math/src/fft/gpu/metal/bowers/{dispatcher,tests}.rs`.

- [ ] Step 1: Add `#[cfg(test)] pub(crate) fn metal_bowers_fft_no_coset(...)` in `dispatcher.rs` that allocates a `vec![1u64; n]` coset buffer and calls `metal_bowers_lde`.
- [ ] Step 2: Replace `tests.rs` with a `run_metal_bowers_vs_cpu(log_n)` helper that: builds `cached_bowers_twiddles_goldilocks(log_n)`, uploads them, builds input `(0..n).map(|i| (i.wrapping_mul(0x9E37_79B9_7F4A_7C15)) % P)`, calls `metal_bowers_fft_no_coset`, downloads `gpu_out`, computes the CPU `ntt_bowers` reference, asserts `gpu_out[i] == cpu_out[bitrev(i)]` for all i (GPU outputs bitrev order, CPU outputs natural). Add tests `metal_bowers_matches_cpu_log{4,10,16,18,20}` each calling the helper.
- [ ] Step 3: `cargo test -p lambdaworks-math --lib --features metal metal_bowers_matches_cpu -- --nocapture`. Debug order on FAIL: bitrev twiddle index off-by-one → DIT vs DIF mix-up → natural-order input convention.
- [ ] Step 4: Commit `test(metal-bowers): golden vectors vs CPU Bowers reference`.

---

## Task 9 — Coset LDE correctness vs `evaluate_offset_fft`

**Files:** Modify `crates/math/src/fft/gpu/metal/bowers/tests.rs`.

- [ ] Step 1: Append `metal_bowers_lde_matches_cpu_evaluate_offset_log10`: build `coeffs: Vec<Fp> = (0..n).map(Fp::from)`, fetch cached bitrev twiddles and cached coset powers for `Fp::from(3u64)`, call `metal_bowers_lde`, download, then assert `gpu_out[i] == evaluate_offset_fft(&poly, 1, coset)[bitrev(i)]`.
- [ ] Step 2: `cargo test -p lambdaworks-math --lib --features metal metal_bowers_lde_matches_cpu`.
- [ ] Step 3: Commit `test(metal-bowers): coset LDE matches evaluate_offset_fft`.

---

## Task 10 — Determinism test

**Files:** Modify `crates/math/src/fft/gpu/metal/bowers/tests.rs`.

- [ ] Step 1: Append `metal_bowers_lde_deterministic_across_states`: run the same input twice with two freshly constructed `MetalState`s and `assert_eq!(outs[0], outs[1])`.
- [ ] Step 2: `cargo test ... metal_bowers_lde_deterministic` then commit `test(metal-bowers): determinism across MetalStates`.

---

## Task 11 — Multi-column dispatch correctness

**Files:** Modify `crates/math/src/fft/gpu/metal/bowers/tests.rs`.

- [ ] Step 1: Append `metal_bowers_lde_multicol_log10_64cols`: build 64 distinct CPU references via `evaluate_offset_fft`, pack inputs col-major into a single `Vec<u64>` of length `n*64`, call `metal_bowers_lde` with `num_cols=64`, download, verify `gpu_out[c*n + i] == cpu_outs[c][bitrev(i)]` for all `c, i`.
- [ ] Step 2: `cargo test ... metal_bowers_lde_multicol` then commit `test(metal-bowers): 64-column dispatch correctness`.

---

## Task 12 — Fp3 kernel + dispatcher + golden test

**Files:** Create `crates/math/src/gpu/metal/shaders/fft/bowers_lde_goldilocks_fp3.metal`; modify `dispatcher.rs` and `tests.rs`.

- [ ] Step 1: `grep -rn "Fp3\|GoldilocksFp3\|fp3" crates/math/src/gpu/metal/shaders/` to find Fp3 typedef.
- [ ] Step 2: Write the Fp3 kernel file with three entry points instantiating `bowers_lde_*_body<Fp3, Fp>`. Note: `twiddles_bitrev` stays base-field `Fp`; `coset_powers` is `Fp3`. Threadgroup buffer in tail is `threadgroup Fp3*`.
- [ ] Step 3: Replace the `metal_bowers_lde_fp3` stub in `dispatcher.rs` with a call to `dispatch_bowers_lde_generic` using `elem_bytes=24` and Fp3 kernel names.
- [ ] Step 4: Append Fp3 test `metal_bowers_fp3_matches_cpu_log10` to `tests.rs`. Pack Fp3 inputs as `[u64;3]` per element, upload, dispatch with the existing base-field bitrev twiddles, download, compare each limb against `ntt_bowers_fp3` CPU reference permuted by bitrev. Adjust `FieldElement<Fp3>::value()` accessor to the actual layout.
- [ ] Step 5: `cargo test ... metal_bowers_fp3` then commit `feat(metal-bowers): Fp3 kernel + dispatcher + golden test`.

---

## Task 13 — `LeafOrder` flag in commitment config

**Files:** the commit-config file in `crates/provers/stark-gpu/`.

- [ ] Step 1: `git grep -nE "struct .*Commit.*Config|pub struct .*Commitment" crates/provers/stark-gpu/` — find the file.
- [ ] Step 2: Add to that file:

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum LeafOrder {
    #[default] Natural,
    Bitrev,
}
```

And append `pub leaf_order: LeafOrder,` to `CommitmentConfig` (preserve all existing fields; `Default` ensures `..Default::default()` callers still get `Natural`).

- [ ] Step 3: Add near the verifier:

```rust
pub fn map_leaf_index(eval_idx: usize, log_n: u32, order: LeafOrder) -> usize {
    match order {
        LeafOrder::Natural => eval_idx,
        LeafOrder::Bitrev  => ((eval_idx as u32).reverse_bits() as usize) >> (32 - log_n),
    }
}
```

Use it everywhere a query index becomes a leaf index. `Natural` callers unchanged.

- [ ] Step 4: `cargo build -p lambdaworks-stark-gpu --features metal`.
- [ ] Step 5: Commit `feat(stark-gpu): LeafOrder enum + query-index mapping`.

---

## Task 14 — `commit_matrix_bowers` glue + feature flag

**Files:** Modify `crates/provers/stark-gpu/Cargo.toml` and `src/metal/mod.rs`; create `src/metal/commit_bowers.rs`.

- [ ] Step 1: Append `bowers-fft = ["metal"]` under `[features]` in `Cargo.toml`.
- [ ] Step 2: Append `#[cfg(feature = "bowers-fft")] pub mod commit_bowers;` to `src/metal/mod.rs`.
- [ ] Step 3: `git grep -nE "transpose_and_hash|merkle_build|keccak.*leaves" crates/provers/stark-gpu/src/` to find the existing helpers.
- [ ] Step 4: Write `commit_bowers.rs` exposing `commit_matrix_bowers(cols_buf, log_n, num_cols, coset_offset, state) -> Result<BowersCommitment, MetalError>` that: caches twiddles+coset, calls `metal_bowers_lde` into a fresh `lde_buf`, calls the existing fused transpose+Keccak leaf kernel, calls the existing Merkle tree builder, returns a `BowersCommitment { root, config: CommitmentConfig { leaf_order: LeafOrder::Bitrev, ..Default::default() }, log_n, num_cols }`.
- [ ] Step 5: `cargo build -p lambdaworks-stark-gpu --features bowers-fft`.
- [ ] Step 6: Commit `feat(stark-gpu): commit_matrix_bowers glue + bowers-fft feature`.

---

## Task 15 — End-to-end commit + verify round-trip test

**Files:** Create `crates/provers/stark-gpu/tests/commit_bowers_roundtrip.rs`.

- [ ] Step 1: `git grep -nE "fn open_leaf|fn verify_leaf|fn open_merkle|fn verify_merkle" crates/provers/stark-gpu/src/` — find the open/verify API.
- [ ] Step 2: Write the test (gated `#![cfg(feature = "bowers-fft")]`): build a 1024 × 64 input, call `commit_matrix_bowers`, assert `commit.config.leaf_order == LeafOrder::Bitrev`, then call `open_leaf(&commit, map_leaf_index(42, log_n, commit.config.leaf_order), &state)` and `verify_leaf(&commit.root, leaf_idx, &proof)`.
- [ ] Step 3: `cargo test -p lambdaworks-stark-gpu --features bowers-fft --test commit_bowers_roundtrip`.
- [ ] Step 4: Commit `test(stark-gpu): bowers commit + verify round-trip`.

---

## Task 16 — Criterion bench at target shape

**Files:** Create `crates/math/benches/criterion_metal_bowers_lde.rs`; modify `crates/math/Cargo.toml`.

- [ ] Step 1: Write a `#[cfg(feature = "metal")]` Criterion bench iterating over `[(18,64),(20,64),(20,256)]`. Inside the closure call only `metal_bowers_lde(...)` (twiddles, coset, and buffers set up once outside). `group.throughput(Throughput::Elements((n*num_cols) as u64))`.
- [ ] Step 2: Append `[[bench]] name = "criterion_metal_bowers_lde" harness = false required-features = ["metal"]` to `crates/math/Cargo.toml`.
- [ ] Step 3: `cargo bench -p lambdaworks-math --features metal --bench criterion_metal_bowers_lde`. Record numbers. Re-run the existing Metal commit bench on the same machine for the baseline.
- [ ] Step 4: Commit `bench(metal-bowers): criterion bench at target shape`.

---

## Task 17 — Final sweep + spec status

- [ ] Step 1: `cargo clippy -p lambdaworks-math --features metal -- -D warnings`; same for stark-gpu with `bowers-fft`; `cargo fmt -p lambdaworks-math -p lambdaworks-stark-gpu -- --check`.
- [ ] Step 2: `cargo test -p lambdaworks-math --features metal -- fft` and `... -- circle` — confirm no regressions (we didn't touch existing shaders).
- [ ] Step 3: Edit the spec: change `**Status:** Draft (awaiting review)` to `**Status:** Implemented` (or `Implemented — target met` if ≥ 2×). Append a `## Results` section with baseline ms / Bowers ms / speedup for both Goldilocks and Fp3 at the target shape.
- [ ] Step 4: Commit `docs(metal-bowers): mark spec implemented + record bench results`.

---

## Self-review

- [x] Every spec section is covered: LDE pipeline (Tasks 5–7), Fp3 (Task 12), `LeafOrder` + verifier (Tasks 13–15), correctness tests (Tasks 8–11, 12, 15), benchmarks (Task 16), error handling (Task 7 dispatcher), no-regression (Task 17).
- [x] No "TBD"/placeholders.
- [x] Names consistent across tasks: `metal_bowers_lde`, `metal_bowers_lde_fp3`, `metal_bowers_fft_no_coset`, `cached_bowers_twiddles_goldilocks`, `cached_coset_powers_goldilocks`, `dispatch_bowers_lde_generic`, `fused_tail_k`, `LeafOrder`, `CommitmentConfig`, `map_leaf_index`, `commit_matrix_bowers`, `BowersCommitment`, `bowers_lde_{stage0,middle,tail}_goldilocks{,_fp3}`.
- [x] Each task ends with a commit.
- [x] Tests added before or alongside implementation (Tasks 1, 2, 4, 8–11, 12, 15).
