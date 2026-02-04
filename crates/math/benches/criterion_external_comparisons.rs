//! External comparisons benchmarks: Lambdaworks vs Arkworks vs Plonky3
//!
//! Main entry point for all external library comparison benchmarks.
//!
//! Run all: cargo bench --bench criterion_external_comparisons
//! Run specific: cargo bench --bench criterion_external_comparisons -- "Goldilocks"

mod external_comparisons;

use criterion::{criterion_group, criterion_main, Criterion};

use external_comparisons::curves::{
    bls12_381_curve_comparison, bls12_381_g2_comparison, bls12_381_msm_comparison,
    bls12_381_pairing_comparison, bn254_curve_comparison, bn254_g2_comparison,
    bn254_msm_comparison, bn254_pairing_comparison, point_serialization_comparison,
    subgroup_check_comparison,
};
use external_comparisons::fft::{
    baby_bear_fft_comparison, batch_fft_comparison, coset_fft_comparison,
    goldilocks_fft_comparison, ifft_comparison,
};
use external_comparisons::fields::{
    baby_bear_comparison, babybear_fp4_comparison, batch_inversion_comparison,
    bls12_381_field_comparison, bn254_field_comparison, extension_field_comparison,
    goldilocks_comparison, mersenne31_comparison,
};
use external_comparisons::polynomials::polynomial_comparison;

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

criterion_group!(
    name = babybear_fp4_field;
    config = Criterion::default().sample_size(10);
    targets = babybear_fp4_comparison::bench_lambdaworks, babybear_fp4_comparison::bench_plonky3
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
// EXTENSION FIELD BENCHMARKS (LW vs Arkworks)
// ============================================

criterion_group!(
    name = bn254_extension_fields;
    config = Criterion::default().sample_size(10);
    targets =
        extension_field_comparison::bench_bn254_fp2_lambdaworks,
        extension_field_comparison::bench_bn254_fp2_arkworks,
        extension_field_comparison::bench_bn254_fp6_lambdaworks,
        extension_field_comparison::bench_bn254_fp6_arkworks,
        extension_field_comparison::bench_bn254_fp12_lambdaworks,
        extension_field_comparison::bench_bn254_fp12_arkworks
);

criterion_group!(
    name = bls12_381_extension_fields;
    config = Criterion::default().sample_size(10);
    targets =
        extension_field_comparison::bench_bls12_381_fp2_lambdaworks,
        extension_field_comparison::bench_bls12_381_fp2_arkworks,
        extension_field_comparison::bench_bls12_381_fp6_lambdaworks,
        extension_field_comparison::bench_bls12_381_fp6_arkworks,
        extension_field_comparison::bench_bls12_381_fp12_lambdaworks,
        extension_field_comparison::bench_bls12_381_fp12_arkworks
);

// ============================================
// BATCH INVERSION BENCHMARKS (LW vs Arkworks)
// ============================================

criterion_group!(
    name = bn254_batch_inv;
    config = Criterion::default().sample_size(10);
    targets =
        batch_inversion_comparison::bench_bn254_fr_lambdaworks,
        batch_inversion_comparison::bench_bn254_fr_arkworks,
        batch_inversion_comparison::bench_bn254_fq_lambdaworks,
        batch_inversion_comparison::bench_bn254_fq_arkworks
);

criterion_group!(
    name = bls12_381_batch_inv;
    config = Criterion::default().sample_size(10);
    targets =
        batch_inversion_comparison::bench_bls12_381_fr_lambdaworks,
        batch_inversion_comparison::bench_bls12_381_fr_arkworks,
        batch_inversion_comparison::bench_bls12_381_fq_lambdaworks,
        batch_inversion_comparison::bench_bls12_381_fq_arkworks
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
// G2 CURVE BENCHMARKS (LW vs Arkworks)
// ============================================

criterion_group!(
    name = bn254_g2;
    config = Criterion::default().sample_size(10);
    targets =
        bn254_g2_comparison::bench_lambdaworks,
        bn254_g2_comparison::bench_arkworks
);

criterion_group!(
    name = bls12_381_g2;
    config = Criterion::default().sample_size(10);
    targets =
        bls12_381_g2_comparison::bench_lambdaworks,
        bls12_381_g2_comparison::bench_arkworks
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
// COSET FFT / LDE BENCHMARKS (LW vs Plonky3)
// ============================================

criterion_group!(
    name = goldilocks_coset_fft;
    config = Criterion::default().sample_size(10);
    targets = coset_fft_comparison::bench_goldilocks_lambdaworks, coset_fft_comparison::bench_goldilocks_plonky3
);

criterion_group!(
    name = babybear_coset_fft;
    config = Criterion::default().sample_size(10);
    targets = coset_fft_comparison::bench_babybear_lambdaworks, coset_fft_comparison::bench_babybear_plonky3
);

// ============================================
// BATCH FFT BENCHMARKS (LW vs Plonky3)
// ============================================

criterion_group!(
    name = goldilocks_batch_fft;
    config = Criterion::default().sample_size(10);
    targets = batch_fft_comparison::bench_goldilocks_lambdaworks, batch_fft_comparison::bench_goldilocks_plonky3
);

criterion_group!(
    name = babybear_batch_fft;
    config = Criterion::default().sample_size(10);
    targets = batch_fft_comparison::bench_babybear_lambdaworks, batch_fft_comparison::bench_babybear_plonky3
);

// ============================================
// IFFT / INTERPOLATION BENCHMARKS (LW vs Plonky3)
// ============================================

criterion_group!(
    name = goldilocks_ifft;
    config = Criterion::default().sample_size(10);
    targets = ifft_comparison::bench_goldilocks_lambdaworks, ifft_comparison::bench_goldilocks_plonky3
);

criterion_group!(
    name = babybear_ifft;
    config = Criterion::default().sample_size(10);
    targets = ifft_comparison::bench_babybear_lambdaworks, ifft_comparison::bench_babybear_plonky3
);

criterion_group!(
    name = goldilocks_batch_ifft;
    config = Criterion::default().sample_size(10);
    targets = ifft_comparison::bench_goldilocks_batch_ifft_lambdaworks, ifft_comparison::bench_goldilocks_batch_ifft_plonky3
);

criterion_group!(
    name = babybear_batch_ifft;
    config = Criterion::default().sample_size(10);
    targets = ifft_comparison::bench_babybear_batch_ifft_lambdaworks, ifft_comparison::bench_babybear_batch_ifft_plonky3
);

// ============================================
// POLYNOMIAL BENCHMARKS (LW vs Plonky3)
// ============================================

criterion_group!(
    name = polynomial_ops;
    config = Criterion::default().sample_size(10);
    targets = polynomial_comparison::bench_lambdaworks, polynomial_comparison::bench_plonky3
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
// POINT SERIALIZATION BENCHMARKS (LW vs Arkworks)
// ============================================

criterion_group!(
    name = bn254_serialization;
    config = Criterion::default().sample_size(10);
    targets =
        point_serialization_comparison::bench_bn254_g1_lambdaworks,
        point_serialization_comparison::bench_bn254_g1_arkworks,
        point_serialization_comparison::bench_bn254_g2_lambdaworks,
        point_serialization_comparison::bench_bn254_g2_arkworks
);

criterion_group!(
    name = bls12_381_serialization;
    config = Criterion::default().sample_size(10);
    targets =
        point_serialization_comparison::bench_bls12_381_g1_lambdaworks,
        point_serialization_comparison::bench_bls12_381_g1_arkworks,
        point_serialization_comparison::bench_bls12_381_g2_lambdaworks,
        point_serialization_comparison::bench_bls12_381_g2_arkworks
);

// ============================================
// SUBGROUP CHECK BENCHMARKS (LW vs Arkworks)
// ============================================

criterion_group!(
    name = bn254_subgroup;
    config = Criterion::default().sample_size(10);
    targets =
        subgroup_check_comparison::bench_bn254_g1_lambdaworks,
        subgroup_check_comparison::bench_bn254_g1_arkworks,
        subgroup_check_comparison::bench_bn254_g2_lambdaworks,
        subgroup_check_comparison::bench_bn254_g2_arkworks
);

criterion_group!(
    name = bls12_381_subgroup;
    config = Criterion::default().sample_size(10);
    targets =
        subgroup_check_comparison::bench_bls12_381_g1_lambdaworks,
        subgroup_check_comparison::bench_bls12_381_g1_arkworks,
        subgroup_check_comparison::bench_bls12_381_g2_lambdaworks,
        subgroup_check_comparison::bench_bls12_381_g2_arkworks
);

// ============================================
// MAIN
// ============================================

criterion_main!(
    // Fields (LW vs Plonky3)
    goldilocks_field,
    baby_bear_field,
    mersenne31_field,
    // Extension Fields (LW vs Plonky3)
    babybear_fp4_field,
    // Fields (LW vs Arkworks)
    bn254_field,
    bls12_381_field,
    // Extension Fields (LW vs Arkworks)
    bn254_extension_fields,
    bls12_381_extension_fields,
    // Batch Inversion (LW vs Arkworks)
    bn254_batch_inv,
    bls12_381_batch_inv,
    // FFT (LW vs Plonky3)
    goldilocks_fft,
    baby_bear_fft,
    // Coset FFT / LDE (LW vs Plonky3)
    goldilocks_coset_fft,
    babybear_coset_fft,
    // Batch FFT (LW vs Plonky3)
    goldilocks_batch_fft,
    babybear_batch_fft,
    // IFFT / Interpolation (LW vs Plonky3)
    goldilocks_ifft,
    babybear_ifft,
    goldilocks_batch_ifft,
    babybear_batch_ifft,
    // Polynomials (LW vs Plonky3)
    polynomial_ops,
    // Curves G1 (LW vs Arkworks)
    bn254_curve,
    bls12_381_curve,
    // Curves G2 (LW vs Arkworks)
    bn254_g2,
    bls12_381_g2,
    // MSM (LW vs Arkworks)
    bn254_msm,
    bls12_381_msm,
    // Pairings (LW vs Arkworks)
    bn254_pairing,
    bls12_381_pairing,
    // Point Serialization (LW vs Arkworks)
    bn254_serialization,
    bls12_381_serialization,
    // Subgroup Checks (LW vs Arkworks)
    bn254_subgroup,
    bls12_381_subgroup
);
