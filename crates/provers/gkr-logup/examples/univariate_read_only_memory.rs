//! Read-only memory check using univariate LogUp-GKR with FRI commitments.
//!
//! Same ROM check as `read_only_memory.rs`, but using the univariate IOP (Phase 2)
//! with FRI polynomial commitment scheme instead of raw multilinear GKR.
//!
//! Flow:
//! 1. Build access and table sides as univariate layers on cyclic domains
//! 2. Commit each column via FRI (Merkle roots)
//! 3. Run GKR + univariate sumcheck for each side
//! 4. Verify both proofs, compare output fractions
//!
//! Run with: cargo run -p lambdaworks-gkr-logup --example univariate_read_only_memory

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

fn main() {
    println!("=== Read-Only Memory Check (Univariate IOP + FRI) ===\n");

    // -------------------------------------------------------
    // Setup: define the ROM table and memory accesses
    // -------------------------------------------------------
    let table_values: Vec<u64> = vec![10, 20, 30, 40];
    let accesses: Vec<u64> = vec![20, 10, 20, 30, 10, 20, 40, 30];

    let multiplicities: Vec<u64> = table_values
        .iter()
        .map(|t| accesses.iter().filter(|&&a| a == *t).count() as u64)
        .collect();

    println!("ROM table:       {:?}", table_values);
    println!("Accesses:        {:?}", accesses);
    println!("Multiplicities:  {:?}", multiplicities);
    println!();

    // -------------------------------------------------------
    // Build univariate layers on cyclic domains
    // -------------------------------------------------------
    let z = FE::from(100u64);

    // Access side: LogUpSingles on a domain of size 8
    let access_dens: Vec<FE> = accesses.iter().map(|&a| z - FE::from(a)).collect();
    let access_domain = CyclicDomain::new(3).unwrap(); // log2(8) = 3
    let access_uni = UnivariateLagrange::new(access_dens, access_domain).unwrap();
    let access_layer = UnivariateLayer::LogUpSingles {
        denominators: access_uni,
        denominator_commitment: None,
    };

    // Table side: LogUpMultiplicities on a domain of size 4
    let table_dens: Vec<FE> = table_values.iter().map(|&t| z - FE::from(t)).collect();
    let table_mults: Vec<FE> = multiplicities.iter().map(|&m| FE::from(m)).collect();
    let table_domain = CyclicDomain::new(2).unwrap(); // log2(4) = 2
    let table_num = UnivariateLagrange::new(table_mults, table_domain.clone()).unwrap();
    let table_den = UnivariateLagrange::new(table_dens, table_domain).unwrap();
    let table_layer = UnivariateLayer::LogUpMultiplicities {
        numerators: table_num,
        denominators: table_den,
        numerator_commitment: None,
        denominator_commitment: None,
    };

    println!(
        "Access layer: {} elements (cyclic domain, {} variables)",
        accesses.len(),
        accesses.len().ilog2()
    );
    println!(
        "Table layer:  {} elements (cyclic domain, {} variables)",
        table_values.len(),
        table_values.len().ilog2()
    );
    println!();

    // -------------------------------------------------------
    // Prove both sides with FRI commitment scheme
    // -------------------------------------------------------
    let pcs = FriCommitmentScheme::new(FriConfig::default());

    println!("--- Proving access side ---");
    let mut prover_transcript = DefaultTranscript::<F>::new(b"rom_access");
    let (access_proof, _) =
        prove_with_pcs::<F, _, FriCommitmentScheme>(&mut prover_transcript, access_layer, &pcs)
            .unwrap();
    println!(
        "  GKR layers: {}, opened values: {}",
        access_proof.gkr_proof.sumcheck_proofs.len(),
        access_proof.opened_values.len()
    );

    println!("--- Proving table side ---");
    let mut prover_transcript = DefaultTranscript::<F>::new(b"rom_table");
    let (table_proof, _) =
        prove_with_pcs::<F, _, FriCommitmentScheme>(&mut prover_transcript, table_layer, &pcs)
            .unwrap();
    println!(
        "  GKR layers: {}, opened values: {}",
        table_proof.gkr_proof.sumcheck_proofs.len(),
        table_proof.opened_values.len()
    );
    println!();

    // -------------------------------------------------------
    // Verify both sides
    // -------------------------------------------------------
    println!("--- Verifying access side ---");
    let mut verifier_transcript = DefaultTranscript::<F>::new(b"rom_access");
    verify_with_pcs::<F, _, FriCommitmentScheme>(
        Gate::LogUp,
        &access_proof,
        &mut verifier_transcript,
        &pcs,
    )
    .unwrap();
    println!("  Verification: PASSED");

    println!("--- Verifying table side ---");
    let mut verifier_transcript = DefaultTranscript::<F>::new(b"rom_table");
    verify_with_pcs::<F, _, FriCommitmentScheme>(
        Gate::LogUp,
        &table_proof,
        &mut verifier_transcript,
        &pcs,
    )
    .unwrap();
    println!("  Verification: PASSED");
    println!();

    // -------------------------------------------------------
    // ROM check: compare output fractions
    // -------------------------------------------------------
    // Each GKR proof produces output_claims = [numerator, denominator]
    // for the accumulated fraction at the root of the binary tree.
    //
    // If accesses are valid: sum 1/(z-a_i) == sum m_j/(z-t_j)
    // Cross-multiply to check equality without division.
    println!("--- ROM Consistency Check ---");
    let access_output = &access_proof.gkr_proof.output_claims; // [num, den]
    let table_output = &table_proof.gkr_proof.output_claims; // [num, den]

    let lhs = access_output[0] * table_output[1];
    let rhs = table_output[0] * access_output[1];

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

    let bad_dens: Vec<FE> = bad_accesses.iter().map(|&a| z - FE::from(a)).collect();
    let bad_domain = CyclicDomain::new(3).unwrap();
    let bad_uni = UnivariateLagrange::new(bad_dens, bad_domain).unwrap();
    let bad_layer = UnivariateLayer::LogUpSingles {
        denominators: bad_uni,
        denominator_commitment: None,
    };

    let mut prover_transcript = DefaultTranscript::<F>::new(b"rom_bad");
    let (bad_proof, _) =
        prove_with_pcs::<F, _, FriCommitmentScheme>(&mut prover_transcript, bad_layer, &pcs)
            .unwrap();

    let mut verifier_transcript = DefaultTranscript::<F>::new(b"rom_bad");
    verify_with_pcs::<F, _, FriCommitmentScheme>(
        Gate::LogUp,
        &bad_proof,
        &mut verifier_transcript,
        &pcs,
    )
    .unwrap();
    println!("GKR verification: PASSED (internally consistent)");

    let bad_output = &bad_proof.gkr_proof.output_claims;
    let bad_lhs = bad_output[0] * table_output[1];
    let bad_rhs = table_output[0] * bad_output[1];

    if bad_lhs == bad_rhs {
        println!("ROM check: PASSED");
    } else {
        println!("ROM check: FAILED (invalid access detected!)");
    }
}
