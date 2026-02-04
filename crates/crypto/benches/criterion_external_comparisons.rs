//! External comparisons benchmarks: Lambdaworks vs Plonky3
//!
//! Main entry point for all external library comparison benchmarks.
//!
//! Run all: cargo bench --bench criterion_external_comparisons
//! Run specific: cargo bench --bench criterion_external_comparisons -- "Poseidon"

mod external_comparisons;

use criterion::{criterion_group, criterion_main, Criterion};

use external_comparisons::hash::poseidon_comparison;
use external_comparisons::merkle::merkle_comparison;

// ============================================
// HASH BENCHMARKS (LW vs Plonky3)
// ============================================

criterion_group!(
    name = poseidon_hash;
    config = Criterion::default().sample_size(10);
    targets = poseidon_comparison::bench_lambdaworks, poseidon_comparison::bench_plonky3
);

// ============================================
// MERKLE TREE BENCHMARKS (LW vs Plonky3)
// ============================================

criterion_group!(
    name = merkle_tree;
    config = Criterion::default().sample_size(10);
    targets = merkle_comparison::bench_lambdaworks, merkle_comparison::bench_plonky3
);

// ============================================
// MAIN
// ============================================

criterion_main!(poseidon_hash, merkle_tree);
