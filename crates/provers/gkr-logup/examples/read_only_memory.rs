//! Read-only memory check using batch LogUp-GKR.
//!
//! Demonstrates how to prove that a set of memory accesses are all valid
//! reads from a fixed table, using the LogUp argument with GKR.
//!
//! Run with: cargo run -p lambdaworks-gkr-logup --example read_only_memory

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

use lambdaworks_math::polynomial::DenseMultilinearPolynomial;

use lambdaworks_gkr_logup::{prove_batch, verify_batch, Gate, Layer};

const MODULUS: u64 = 2013265921;
type F = U64PrimeField<MODULUS>;
type FE = FieldElement<F>;

fn main() {
    println!("=== Read-Only Memory Check with LogUp-GKR ===\n");

    // -------------------------------------------------------
    // Setup: define the ROM table and memory accesses
    // -------------------------------------------------------
    let table_values: Vec<u64> = vec![10, 20, 30, 40];
    let accesses: Vec<u64> = vec![20, 10, 20, 30, 10, 20, 40, 30];

    // Count how many times each table entry is accessed (multiplicities)
    let multiplicities: Vec<u64> = table_values
        .iter()
        .map(|t| accesses.iter().filter(|&&a| a == *t).count() as u64)
        .collect();

    println!("ROM table:       {:?}", table_values);
    println!("Accesses:        {:?}", accesses);
    println!("Multiplicities:  {:?}", multiplicities);
    println!();

    // -------------------------------------------------------
    // Build the LogUp argument
    // -------------------------------------------------------
    // Choose a random evaluation point z.
    // In production this comes from Fiat-Shamir; here we pick a fixed one.
    let z = FE::from(100);

    // Access side: each access a_i contributes 1/(z - a_i)
    // This is LogUpSingles (all numerators = 1).
    let access_denominators: Vec<FE> = accesses.iter().map(|&a| z - FE::from(a)).collect();
    let access_layer = Layer::LogUpSingles {
        denominators: DenseMultilinearPolynomial::new(access_denominators),
    };

    // Table side: each entry t_j contributes m_j/(z - t_j)
    // This is LogUpMultiplicities.
    let table_denominators: Vec<FE> = table_values.iter().map(|&t| z - FE::from(t)).collect();
    let table_multiplicities: Vec<FE> = multiplicities.iter().map(|&m| FE::from(m)).collect();
    let table_layer = Layer::LogUpMultiplicities {
        numerators: DenseMultilinearPolynomial::new(table_multiplicities),
        denominators: DenseMultilinearPolynomial::new(table_denominators),
    };

    println!(
        "Access layer: {} elements ({} variables)",
        accesses.len(),
        access_layer.n_variables()
    );
    println!(
        "Table layer:  {} elements ({} variables)",
        table_values.len(),
        table_layer.n_variables()
    );
    println!();

    // -------------------------------------------------------
    // Prove: batch both instances into a single GKR proof
    // -------------------------------------------------------
    println!("--- Proving ---");
    let mut prover_channel = DefaultTranscript::<F>::new(&[]);
    let (proof, _artifact) = prove_batch(&mut prover_channel, vec![access_layer, table_layer]);

    println!(
        "Batch proof: {} sumcheck layers, {} instances",
        proof.sumcheck_proofs.len(),
        proof.output_claims_by_instance.len()
    );
    println!();

    // -------------------------------------------------------
    // Verify: batch verification
    // -------------------------------------------------------
    println!("--- Verifying ---");
    let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
    let result = verify_batch(&[Gate::LogUp, Gate::LogUp], &proof, &mut verifier_channel);

    match &result {
        Ok(artifact) => {
            println!("GKR verification: PASSED");
            println!("  OOD point length: {}", artifact.ood_point.len());
            println!(
                "  Variables by instance: {:?}",
                artifact.n_variables_by_instance
            );
        }
        Err(e) => {
            println!("GKR verification: FAILED ({})", e);
            return;
        }
    }
    println!();

    // -------------------------------------------------------
    // ROM check: compare output fractions
    // -------------------------------------------------------
    // If accesses are valid: sum 1/(z-a_i) == sum m_j/(z-t_j)
    // Both sides produce a fraction [numerator, denominator].
    // Cross-multiply to check equality without division.
    println!("--- ROM Consistency Check ---");
    let access_output = &proof.output_claims_by_instance[0]; // [num, den]
    let table_output = &proof.output_claims_by_instance[1]; // [num, den]

    let lhs = access_output[0] * table_output[1]; // access_num * table_den
    let rhs = table_output[0] * access_output[1]; // table_num * access_den

    if lhs == rhs {
        println!("ROM check: PASSED (all accesses are in the table)");
    } else {
        println!("ROM check: FAILED (some access is not in the table)");
    }
    println!();

    // -------------------------------------------------------
    // Negative example: invalid access
    // -------------------------------------------------------
    println!("=== Negative Test: Invalid Access ===\n");

    let bad_accesses: Vec<u64> = vec![20, 10, 20, 30, 10, 20, 50, 30]; // 50 is NOT in table
    println!("Accesses (with invalid 50): {:?}", bad_accesses);

    let bad_access_dens: Vec<FE> = bad_accesses.iter().map(|&a| z - FE::from(a)).collect();
    let bad_access_layer = Layer::LogUpSingles {
        denominators: DenseMultilinearPolynomial::new(bad_access_dens),
    };

    // Table side stays the same (same multiplicities as before for the "expected" pattern)
    let table_dens2: Vec<FE> = table_values.iter().map(|&t| z - FE::from(t)).collect();
    let table_mults2: Vec<FE> = multiplicities.iter().map(|&m| FE::from(m)).collect();
    let table_layer2 = Layer::LogUpMultiplicities {
        numerators: DenseMultilinearPolynomial::new(table_mults2),
        denominators: DenseMultilinearPolynomial::new(table_dens2),
    };

    let mut p_ch = DefaultTranscript::<F>::new(&[]);
    let (bad_proof, _) = prove_batch(&mut p_ch, vec![bad_access_layer, table_layer2]);

    let mut v_ch = DefaultTranscript::<F>::new(&[]);
    let bad_result = verify_batch(&[Gate::LogUp, Gate::LogUp], &bad_proof, &mut v_ch);

    match &bad_result {
        Ok(_) => println!("GKR verification: PASSED (each tree is internally consistent)"),
        Err(e) => println!("GKR verification: FAILED ({})", e),
    }

    let bad_access_out = &bad_proof.output_claims_by_instance[0];
    let bad_table_out = &bad_proof.output_claims_by_instance[1];
    let bad_lhs = bad_access_out[0] * bad_table_out[1];
    let bad_rhs = bad_table_out[0] * bad_access_out[1];

    if bad_lhs == bad_rhs {
        println!("ROM check: PASSED");
    } else {
        println!("ROM check: FAILED (invalid access detected!)");
    }
}
