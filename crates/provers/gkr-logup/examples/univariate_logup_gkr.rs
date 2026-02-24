//! End-to-end example of the univariate LogUp-GKR IOP (Section 5 of ePrint 2023/1284).
//!
//! Demonstrates:
//! 1. LogUp Singles: ROM lookup with univariate commitments
//! 2. LogUp Multiplicities: table + multiplicities with univariate commitments
//! 3. Grand Product: simple product argument with univariate commitments

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;

use lambdaworks_gkr_logup::univariate::domain::CyclicDomain;
use lambdaworks_gkr_logup::univariate::iop::{prove_univariate, verify_univariate};
use lambdaworks_gkr_logup::univariate::lagrange::UnivariateLagrange;
use lambdaworks_gkr_logup::univariate_layer::UnivariateLayer;
use lambdaworks_gkr_logup::verifier::Gate;

type F = Degree4BabyBearExtensionField;
type FE = FieldElement<F>;

fn main() {
    println!("=== Univariate LogUp-GKR IOP (Section 5) ===\n");

    test_grand_product();
    test_logup_singles();
    test_logup_multiplicities();

    println!("\n=== All examples passed! ===");
}

fn test_grand_product() {
    println!("Example 1: Grand Product (univariate commitment)");

    let values: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
    let domain = CyclicDomain::new(3).unwrap();
    let uni = UnivariateLagrange::new(values, domain).unwrap();

    let layer = UnivariateLayer::GrandProduct {
        values: uni,
        commitment: None,
    };

    let mut prover_transcript = DefaultTranscript::<F>::new(b"grand_product_example");
    let (proof, result) = prove_univariate(&mut prover_transcript, layer).unwrap();

    println!(
        "  Proof generated: {} GKR layers",
        proof.gkr_proof.sumcheck_proofs.len()
    );
    println!("  OOD point dimension: {}", result.ood_point.len());
    println!("  Claims to verify: {}", result.claims_to_verify.len());

    let mut verifier_transcript = DefaultTranscript::<F>::new(b"grand_product_example");
    verify_univariate(Gate::GrandProduct, &proof, &mut verifier_transcript).unwrap();

    println!("  ✓ Grand Product verified!\n");
}

fn test_logup_singles() {
    println!("Example 2: LogUp Singles (ROM lookup)");

    // Simulate a ROM lookup: 8 accesses to a table
    let z = FE::from(100u64);
    let accesses: Vec<u64> = vec![20, 10, 20, 30, 10, 20, 40, 30];
    let dens: Vec<FE> = accesses.iter().map(|&a| z.clone() - FE::from(a)).collect();

    let domain = CyclicDomain::new(3).unwrap();
    let uni = UnivariateLagrange::new(dens, domain).unwrap();

    let layer = UnivariateLayer::LogUpSingles {
        denominators: uni,
        denominator_commitment: None,
    };

    let mut prover_transcript = DefaultTranscript::<F>::new(b"logup_singles_example");
    let (proof, result) = prove_univariate(&mut prover_transcript, layer).unwrap();

    println!(
        "  Proof generated: {} GKR layers",
        proof.gkr_proof.sumcheck_proofs.len()
    );
    println!("  OOD point dimension: {}", result.ood_point.len());
    println!("  Lagrange column size: {}", proof.lagrange_column.len());

    let mut verifier_transcript = DefaultTranscript::<F>::new(b"logup_singles_example");
    verify_univariate(Gate::LogUp, &proof, &mut verifier_transcript).unwrap();

    println!("  ✓ LogUp Singles verified!\n");
}

fn test_logup_multiplicities() {
    println!("Example 3: LogUp Multiplicities (table + multiplicities)");

    let z = FE::from(1000u64);
    let table: Vec<u64> = vec![3, 5, 7, 9, 11, 13, 15, 17];

    let table_dens: Vec<FE> = table.iter().map(|&t| z.clone() - FE::from(t)).collect();
    let multiplicities: Vec<FE> = table.iter().map(|_| FE::one()).collect();

    let domain = CyclicDomain::new(3).unwrap();
    let num = UnivariateLagrange::new(multiplicities, domain.clone()).unwrap();
    let den = UnivariateLagrange::new(table_dens, domain).unwrap();

    let layer = UnivariateLayer::LogUpMultiplicities {
        numerators: num,
        denominators: den,
        numerator_commitment: None,
        denominator_commitment: None,
    };

    let mut prover_transcript = DefaultTranscript::<F>::new(b"logup_mult_example");
    let (proof, result) = prove_univariate(&mut prover_transcript, layer).unwrap();

    println!(
        "  Proof generated: {} GKR layers",
        proof.gkr_proof.sumcheck_proofs.len()
    );
    println!("  Committed columns: {}", proof.committed_columns.len());
    println!("  OOD point dimension: {}", result.ood_point.len());

    let mut verifier_transcript = DefaultTranscript::<F>::new(b"logup_mult_example");
    verify_univariate(Gate::LogUp, &proof, &mut verifier_transcript).unwrap();

    println!("  ✓ LogUp Multiplicities verified!\n");
}
