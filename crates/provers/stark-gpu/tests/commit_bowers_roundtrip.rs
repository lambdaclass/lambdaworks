//! End-to-end round-trip test for `commit_matrix_bowers`.
//!
//! Verifies the GPU Bowers-G LDE + Keccak Merkle commit produces the same
//! root and proof paths as the trusted CPU path:
//!   per-column `evaluate_offset_fft` -> bit-reversed rows -> `cpu_batch_commit`.

#![cfg(all(test, target_os = "macos", feature = "bowers-fft"))]

use lambdaworks_gpu::metal::abstractions::state::MetalState;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_stark_gpu::metal::commit_bowers::commit_matrix_bowers;
use lambdaworks_stark_gpu::metal::merkle::{cpu_batch_commit, GpuKeccakMerkleState};
use stark_platinum_prover::trace::columns2rows_bit_reversed;

type FpE = FieldElement<Goldilocks64Field>;

fn run_roundtrip(log_n: u32, num_cols: usize, coset: u64) {
    let n = 1usize << log_n;
    let g = FpE::from(coset);

    let coeff_cols: Vec<Vec<FpE>> = (0..num_cols)
        .map(|c| {
            (0..n as u64)
                .map(|i| FpE::from((i + c as u64 * 7919).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
                .collect()
        })
        .collect();

    // CPU reference: coset LDE per column, bit-reversed-row Merkle commit.
    let eval_cols: Vec<Vec<FpE>> = coeff_cols
        .iter()
        .map(|col| {
            let poly = Polynomial::new(col);
            Polynomial::evaluate_offset_fft::<Goldilocks64Field>(&poly, 1, None, &g)
                .expect("evaluate_offset_fft")
        })
        .collect();
    let cpu_rows = columns2rows_bit_reversed(&eval_cols);
    let (cpu_tree, cpu_root) = cpu_batch_commit(&cpu_rows).expect("cpu_batch_commit");

    // GPU path.
    let lde_state = MetalState::new(None).expect("metal state");
    let keccak_state = GpuKeccakMerkleState::new().expect("keccak state");
    let (gpu_tree, gpu_root) =
        commit_matrix_bowers(&coeff_cols, g, &lde_state, &keccak_state).expect("commit");

    assert_eq!(gpu_root, cpu_root, "Merkle root mismatch (log_n={log_n})");

    for &pos in &[0usize, 1, n / 3, n / 2, n - 1] {
        let cpu_proof = cpu_tree.get_proof_by_pos(pos);
        let gpu_proof = gpu_tree.get_proof_by_pos(pos);
        assert_eq!(cpu_proof.is_some(), gpu_proof.is_some(), "proof at {pos}");
        if let (Some(cp), Some(gp)) = (cpu_proof, gpu_proof) {
            assert_eq!(cp.merkle_path, gp.merkle_path, "path mismatch at {pos}");
        }
    }
}

#[test]
fn bowers_commit_roundtrip_log10_64cols() {
    run_roundtrip(10, 64, 7);
}

#[test]
fn bowers_commit_roundtrip_log8_3cols() {
    run_roundtrip(8, 3, 3);
}
