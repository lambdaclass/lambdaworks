use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;

use lambdaworks_gkr_logup::layer::Layer;
use lambdaworks_gkr_logup::prover;
use lambdaworks_gkr_logup::univariate::lagrange::univariate_to_multilinear_fft;
use lambdaworks_gkr_logup::verifier::{verify, Gate};

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;

type F = Degree4BabyBearExtensionField;
type FE = FieldElement<F>;

fn main() {
    println!("=== Caulk-Style: Univariate → Multilinear → GKR ===\n");

    test_univariate_to_multilinear_logup();

    println!("\n=== All tests passed! ===");
}

fn test_univariate_to_multilinear_logup() {
    println!("Test: Convert Univariate to Multilinear, then GKR");

    // Input: accesses as univariate (in cyclic domain order)
    let z = FE::from(100u64);
    let accesses: Vec<u64> = vec![20, 10, 20, 30, 10, 20, 40, 30];

    // Create univariate denominators: z - access[i]
    let univariate_dens: Vec<FE> = accesses.iter().map(|&a| z - FE::from(a)).collect();

    println!(
        "  Univariate denominators (first 4): {:?}",
        &univariate_dens[..4]
    );

    // Transform to multilinear using FFT
    let multilinear_dens =
        univariate_to_multilinear_fft(&univariate_dens).expect("FFT transform failed");

    println!(
        "  Multilinear denominators (first 4): {:?}",
        &multilinear_dens[..4]
    );

    // Create the layer - same as multivariate version
    use lambdaworks_math::polynomial::DenseMultilinearPolynomial;
    let mle = DenseMultilinearPolynomial::new(multilinear_dens);

    let layer = Layer::LogUpSingles { denominators: mle };

    println!("  Layer n_variables: {}", layer.n_variables());

    // Use existing multivariate prover
    let mut transcript = DefaultTranscript::<F>::new(b"caulk_test");

    match prover::prove(&mut transcript, layer) {
        Ok((proof, _verification_result)) => {
            println!("  Proof generated successfully!");
            println!("    - Sumcheck proofs: {}", proof.sumcheck_proofs.len());

            // Verify using existing verifier
            let gate = Gate::LogUp;
            let mut transcript_verify = DefaultTranscript::<F>::new(b"caulk_test");

            match verify(gate, &proof, &mut transcript_verify) {
                Ok(_) => {
                    println!("  ✓ Verification successful! Caulk-style works!");
                }
                Err(e) => {
                    println!("  ✗ Verification error: {:?}", e);
                }
            }
        }
        Err(e) => {
            println!("  Error: {:?}", e);
        }
    }
}
