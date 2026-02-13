//! Range check using batch LogUp-GKR.
//!
//! Proves that a set of values are all in the range [0, 2^n) by treating
//! the range as a ROM table and the values as accesses.
//!
//! The LogUp identity is:
//!   ∑ 1/(z - v_i) = ∑ m_j/(z - j)   for j in 0..2^n
//!
//! where v_i are the values to check and m_j is how many times j appears.
//!
//! Run with: cargo run -p lambdaworks-gkr-logup --example range_check

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

use lambdaworks_gkr_logup::mle::Mle;
use lambdaworks_gkr_logup::{prove_batch, verify_batch, Gate, Layer};

const MODULUS: u64 = 2013265921;
type F = U64PrimeField<MODULUS>;
type FE = FieldElement<F>;

/// Builds a range check and returns whether it passed.
fn range_check(values: &[u64], n_bits: u32, label: &str) -> bool {
    let range_size = 1u64 << n_bits; // 2^n
    let z = FE::from(1_000_003); // evaluation point (must not collide with 0..range_size)

    println!("  Values:    {:?}", values);
    println!("  Range:     [0, {})", range_size);

    // --- Access side: each value v_i contributes 1/(z - v_i) ---
    let access_dens: Vec<FE> = values.iter().map(|&v| z - FE::from(v)).collect();
    let access_layer = Layer::LogUpSingles {
        denominators: Mle::new(access_dens),
    };

    // --- Table side: range table 0..2^n with multiplicities ---
    let multiplicities: Vec<FE> = (0..range_size)
        .map(|j| FE::from(values.iter().filter(|&&v| v == j).count() as u64))
        .collect();
    let table_dens: Vec<FE> = (0..range_size).map(|j| z - FE::from(j)).collect();
    let table_layer = Layer::LogUpMultiplicities {
        numerators: Mle::new(multiplicities),
        denominators: Mle::new(table_dens),
    };

    println!(
        "  Access layer: {} elements ({} vars), Table layer: {} elements ({} vars)",
        values.len(),
        access_layer.n_variables(),
        range_size,
        table_layer.n_variables(),
    );

    // --- Batch prove ---
    let mut prover_ch = DefaultTranscript::<F>::new(&[]);
    let (proof, _) = prove_batch(&mut prover_ch, vec![access_layer, table_layer]);

    // --- Batch verify ---
    let mut verifier_ch = DefaultTranscript::<F>::new(&[]);
    let gkr_result = verify_batch(&[Gate::LogUp, Gate::LogUp], &proof, &mut verifier_ch);

    if let Err(e) = &gkr_result {
        println!("  GKR verification: FAILED ({})", e);
        return false;
    }
    println!("  GKR verification: PASSED");

    // --- ROM check: fractions must match ---
    let access_out = &proof.output_claims_by_instance[0]; // [num, den]
    let table_out = &proof.output_claims_by_instance[1]; // [num, den]
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
    println!("=== Range Check with LogUp-GKR ===\n");

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
