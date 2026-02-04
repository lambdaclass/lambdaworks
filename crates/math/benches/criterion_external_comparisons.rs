//! External comparisons benchmarks: Lambdaworks vs Arkworks vs Plonky3
//!
//! Main entry point for all external library comparison benchmarks.
//!
//! Run all: cargo bench --bench criterion_external_comparisons
//! Run specific: cargo bench --bench criterion_external_comparisons -- "Goldilocks"

mod external_comparisons;

use criterion::{criterion_group, criterion_main, Criterion};

use external_comparisons::curves::{
    bls12_381_curve_comparison, bls12_381_msm_comparison, bls12_381_pairing_comparison,
    bn254_curve_comparison, bn254_msm_comparison, bn254_pairing_comparison,
};
use external_comparisons::fft::{baby_bear_fft_comparison, goldilocks_fft_comparison};
use external_comparisons::fields::{
    baby_bear_comparison, bls12_381_field_comparison, bn254_field_comparison,
    goldilocks_comparison, mersenne31_comparison,
};

// ============================================
// FIELD BENCHMARKS (LW vs Plonky3)
// ============================================

criterion_group!(
    name = goldilocks_field;
    config = Criterion::default().sample_size(10);
    targets = goldilocks_comparison::bench_lambdaworks, goldilocks_comparison::bench_plonky3
);

criterion_group!(
    name = baby_bear_field;
    config = Criterion::default().sample_size(10);
    targets = baby_bear_comparison::bench_lambdaworks, baby_bear_comparison::bench_plonky3
);

criterion_group!(
    name = mersenne31_field;
    config = Criterion::default().sample_size(10);
    targets = mersenne31_comparison::bench_lambdaworks, mersenne31_comparison::bench_plonky3
);

// ============================================
// FIELD BENCHMARKS (LW vs Arkworks)
// ============================================

criterion_group!(
    name = bn254_field;
    config = Criterion::default().sample_size(10);
    targets =
        bn254_field_comparison::bench_bn254_fr_lambdaworks,
        bn254_field_comparison::bench_bn254_fr_arkworks,
        bn254_field_comparison::bench_bn254_fq_lambdaworks,
        bn254_field_comparison::bench_bn254_fq_arkworks
);

criterion_group!(
    name = bls12_381_field;
    config = Criterion::default().sample_size(10);
    targets =
        bls12_381_field_comparison::bench_bls12_381_fr_lambdaworks,
        bls12_381_field_comparison::bench_bls12_381_fr_arkworks,
        bls12_381_field_comparison::bench_bls12_381_fq_lambdaworks,
        bls12_381_field_comparison::bench_bls12_381_fq_arkworks
);

// ============================================
// CURVE BENCHMARKS (LW vs Arkworks)
// ============================================

criterion_group!(
    name = bn254_curve;
    config = Criterion::default().sample_size(10);
    targets =
        bn254_curve_comparison::bench_lambdaworks,
        bn254_curve_comparison::bench_arkworks
);

criterion_group!(
    name = bls12_381_curve;
    config = Criterion::default().sample_size(10);
    targets =
        bls12_381_curve_comparison::bench_lambdaworks,
        bls12_381_curve_comparison::bench_arkworks
);

// ============================================
// MSM BENCHMARKS (LW vs Arkworks)
// ============================================

criterion_group!(
    name = bn254_msm;
    config = Criterion::default().sample_size(10);
    targets =
        bn254_msm_comparison::bench_lambdaworks,
        bn254_msm_comparison::bench_arkworks
);

criterion_group!(
    name = bls12_381_msm;
    config = Criterion::default().sample_size(10);
    targets =
        bls12_381_msm_comparison::bench_lambdaworks,
        bls12_381_msm_comparison::bench_arkworks
);

// ============================================
// FFT BENCHMARKS (LW vs Plonky3)
// ============================================

criterion_group!(
    name = goldilocks_fft;
    config = Criterion::default().sample_size(10);
    targets = goldilocks_fft_comparison::bench_lambdaworks, goldilocks_fft_comparison::bench_plonky3
);

criterion_group!(
    name = baby_bear_fft;
    config = Criterion::default().sample_size(10);
    targets = baby_bear_fft_comparison::bench_lambdaworks, baby_bear_fft_comparison::bench_plonky3
);

// ============================================
// PAIRING BENCHMARKS (LW vs Arkworks)
// ============================================

criterion_group!(
    name = bn254_pairing;
    config = Criterion::default().sample_size(10);
    targets =
        bn254_pairing_comparison::bench_lambdaworks,
        bn254_pairing_comparison::bench_arkworks
);

criterion_group!(
    name = bls12_381_pairing;
    config = Criterion::default().sample_size(10);
    targets =
        bls12_381_pairing_comparison::bench_lambdaworks,
        bls12_381_pairing_comparison::bench_arkworks
);

// ============================================
// MAIN
// ============================================

criterion_main!(
    // Fields (LW vs Plonky3)
    goldilocks_field,
    baby_bear_field,
    mersenne31_field,
    // Fields (LW vs Arkworks)
    bn254_field,
    bls12_381_field,
    // FFT (LW vs Plonky3)
    goldilocks_fft,
    baby_bear_fft,
    // Curves (LW vs Arkworks)
    bn254_curve,
    bls12_381_curve,
    // MSM (LW vs Arkworks)
    bn254_msm,
    bls12_381_msm,
    // Pairings (LW vs Arkworks)
    bn254_pairing,
    bls12_381_pairing
);
