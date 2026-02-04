//! External comparison benchmarks: Lambdaworks vs Plonky3 (Poseidon)
//!
//! TODO: Currently blocked - field mismatch:
//! - Lambdaworks has Poseidon for Stark252
//! - Plonky3 has Poseidon2 for Goldilocks, BabyBear, Mersenne31
//!
//! Options to enable comparison:
//! 1. Add Poseidon parameters for Goldilocks/BabyBear to Lambdaworks
//! 2. Add Stark252 support to Plonky3
//! 3. Compare on Bn254 (Plonky3 has it, would need to add to LW)

use criterion::Criterion;

pub fn bench_lambdaworks(_c: &mut Criterion) {
    // TODO: Implement when field overlap exists
}

pub fn bench_plonky3(_c: &mut Criterion) {
    // TODO: Implement when field overlap exists
}
