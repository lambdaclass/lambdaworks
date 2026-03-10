//! Range check using univariate LogUp-GKR with FRI commitments.
//!
//! Same range check as `range_check.rs`, but using the univariate IOP (Phase 2)
//! with FRI polynomial commitment scheme instead of raw multilinear GKR.
//!
//! Proves that a set of values are all in the range [0, 2^n) by treating
//! the range as a ROM table and the values as accesses.
//!
//! Run with: cargo run -p lambdaworks-gkr-logup --example univariate_range_check

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;

use lambdaworks_gkr_logup::fri::pcs::FriCommitmentScheme;
use lambdaworks_gkr_logup::fri::types::FriConfig;
use lambdaworks_gkr_logup::univariate::domain::CyclicDomain;
use lambdaworks_gkr_logup::univariate::iop::{prove_with_pcs, verify_with_pcs};
use lambdaworks_gkr_logup::univariate::lagrange::UnivariateLagrange;
use lambdaworks_gkr_logup::univariate_layer::UnivariateLayer;
use lambdaworks_gkr_logup::verifier::Gate;

type F = Degree4BabyBearExtensionField;
type FE = FieldElement<F>;

/// Builds a range check and returns whether it passed.
fn range_check(values: &[u64], n_bits: u32, label: &str) -> bool {
    let range_size = 1u64 << n_bits;
    let z = FE::from(1_000_003u64);

    println!("  Values:    {:?}", values);
    println!("  Range:     [0, {})", range_size);

    // --- Access side: each value v_i contributes 1/(z - v_i) ---
    let access_dens: Vec<FE> = values.iter().map(|&v| z - FE::from(v)).collect();
    let access_log_size = values.len().ilog2() as usize;
    let access_domain = CyclicDomain::new(access_log_size).unwrap();
    let access_uni = UnivariateLagrange::new(access_dens, access_domain).unwrap();
    let access_layer = UnivariateLayer::LogUpSingles {
        denominators: access_uni,
        denominator_commitment: None,
    };

    // --- Table side: range table 0..2^n with multiplicities ---
    let multiplicities: Vec<FE> = (0..range_size)
        .map(|j| FE::from(values.iter().filter(|&&v| v == j).count() as u64))
        .collect();
    let table_dens: Vec<FE> = (0..range_size).map(|j| z - FE::from(j)).collect();
    let table_domain = CyclicDomain::new(n_bits as usize).unwrap();
    let table_num = UnivariateLagrange::new(multiplicities, table_domain.clone()).unwrap();
    let table_den = UnivariateLagrange::new(table_dens, table_domain).unwrap();
    let table_layer = UnivariateLayer::LogUpMultiplicities {
        numerators: table_num,
        denominators: table_den,
        numerator_commitment: None,
        denominator_commitment: None,
    };

    println!(
        "  Access layer: {} elements ({} vars), Table layer: {} elements ({} vars)",
        values.len(),
        access_log_size,
        range_size,
        n_bits,
    );

    let pcs = FriCommitmentScheme::new(FriConfig::default());

    // --- Prove access side ---
    let mut prover_transcript = DefaultTranscript::<F>::new(b"rc_access");
    let (access_proof, _) =
        prove_with_pcs::<F, _, FriCommitmentScheme>(&mut prover_transcript, access_layer, &pcs)
            .unwrap();

    // --- Prove table side ---
    let mut prover_transcript = DefaultTranscript::<F>::new(b"rc_table");
    let (table_proof, _) =
        prove_with_pcs::<F, _, FriCommitmentScheme>(&mut prover_transcript, table_layer, &pcs)
            .unwrap();

    // --- Verify access side ---
    let mut verifier_transcript = DefaultTranscript::<F>::new(b"rc_access");
    if let Err(e) = verify_with_pcs::<F, _, FriCommitmentScheme>(
        Gate::LogUp,
        &access_proof,
        &mut verifier_transcript,
        &pcs,
    ) {
        println!("  Access GKR verification: FAILED ({e})");
        return false;
    }

    // --- Verify table side ---
    let mut verifier_transcript = DefaultTranscript::<F>::new(b"rc_table");
    if let Err(e) = verify_with_pcs::<F, _, FriCommitmentScheme>(
        Gate::LogUp,
        &table_proof,
        &mut verifier_transcript,
        &pcs,
    ) {
        println!("  Table GKR verification: FAILED ({e})");
        return false;
    }
    println!("  GKR verification: PASSED");

    // --- ROM check: fractions must match ---
    let access_out = &access_proof.gkr_proof.output_claims;
    let table_out = &table_proof.gkr_proof.output_claims;
    let lhs = access_out[0] * table_out[1];
    let rhs = table_out[0] * access_out[1];
    let passed = lhs == rhs;

    if passed {
        println!("  Range check {}: PASSED", label);
    } else {
        println!("  Range check {}: FAILED (value out of range!)", label);
    }

    passed
}

fn main() {
    println!("=== Range Check (Univariate IOP + FRI) ===\n");

    // --- Test 1: all values in [0, 8) ---
    println!("[Test 1] 8 values, all in range [0, 8)");
    let valid = range_check(&[0, 1, 2, 3, 4, 5, 6, 7], 3, "#1");
    assert!(valid);
    println!();

    // --- Test 2: repeated values, all in [0, 8) ---
    println!("[Test 2] 8 values with repeats, all in range [0, 8)");
    let valid = range_check(&[0, 0, 3, 3, 7, 7, 1, 5], 3, "#2");
    assert!(valid);
    println!();

    // --- Test 3: values in [0, 16), different sizes ---
    println!("[Test 3] 8 values in range [0, 16) (access layer smaller than table)");
    let valid = range_check(&[0, 5, 10, 15, 3, 8, 12, 1], 4, "#3");
    assert!(valid);
    println!();

    // --- Test 4: value out of range ---
    println!("[Test 4] One value (9) is out of range [0, 8)");
    let valid = range_check(&[0, 1, 2, 3, 4, 5, 6, 9], 3, "#4");
    assert!(!valid);
    println!();

    // --- Test 5: larger range check ---
    println!("[Test 5] 32 random-ish values in range [0, 64)");
    let values: Vec<u64> = (0..32).map(|i| (i * 7 + 3) % 64).collect();
    let valid = range_check(&values, 6, "#5");
    assert!(valid);
    println!();

    println!("All tests passed!");
}
