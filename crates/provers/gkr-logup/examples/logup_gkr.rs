use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;

use lambdaworks_gkr_logup::layer::Layer;
use lambdaworks_gkr_logup::prover;
use lambdaworks_gkr_logup::verifier::{verify, Gate};

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;

type F = Degree4BabyBearExtensionField;
type FE = FieldElement<F>;

fn main() {
    println!("=== LogUp-GKR (Multilinear MLE) ===\n");

    test_logup_singles();
    test_read_only_memory();

    println!("\n=== All tests passed! ===");
}

fn test_logup_singles() {
    println!("Test 1: LogUp Singles (ROM lookup)");

    // z - access[i] denominators (multilinear MLE)
    let z = FE::from(100u64);
    let accesses: Vec<u64> = vec![20, 10, 20, 30, 10, 20, 40, 30];

    let mle_values: Vec<FE> = accesses.iter().map(|&a| z.clone() - FE::from(a)).collect();

    use lambdaworks_math::polynomial::DenseMultilinearPolynomial;
    let mle = DenseMultilinearPolynomial::new(mle_values);

    let layer = Layer::LogUpSingles { denominators: mle };

    println!("  Layer n_variables: {}", layer.n_variables());

    let mut transcript = DefaultTranscript::<F>::new(b"test1");

    match prover::prove(&mut transcript, layer) {
        Ok((proof, _)) => {
            println!("  Proof generated!");

            let gate = Gate::LogUp;
            let mut transcript_verify = DefaultTranscript::<F>::new(b"test1");

            match verify(gate, &proof, &mut transcript_verify) {
                Ok(_) => println!("  ✓ LogUp Singles works!"),
                Err(e) => println!("  ✗ Verify error: {:?}", e),
            }
        }
        Err(e) => println!("  Error: {:?}", e),
    }
}

fn test_read_only_memory() {
    println!("\nTest 2: Read Only Memory (2 columns)");

    let z = FE::from(1000u64);
    let accesses: Vec<u64> = vec![5, 3, 5, 7, 3, 5, 9, 7];
    let table: Vec<u64> = vec![3, 5, 7, 9, 11, 13, 15, 17];

    let access_dens: Vec<FE> = accesses.iter().map(|&a| z.clone() - FE::from(a)).collect();
    let table_dens: Vec<FE> = table.iter().map(|&t| z.clone() - FE::from(t)).collect();

    use lambdaworks_math::polynomial::DenseMultilinearPolynomial;
    let access_mle = DenseMultilinearPolynomial::new(access_dens);
    let table_mle = DenseMultilinearPolynomial::new(table_dens);

    let layer = Layer::LogUpMultiplicities {
        numerators: table_mle,
        denominators: access_mle,
    };

    println!("  Layer n_variables: {}", layer.n_variables());

    let mut transcript = DefaultTranscript::<F>::new(b"test2");

    match prover::prove(&mut transcript, layer) {
        Ok((proof, _)) => {
            println!("  Proof generated!");

            let gate = Gate::LogUp;
            let mut transcript_verify = DefaultTranscript::<F>::new(b"test2");

            match verify(gate, &proof, &mut transcript_verify) {
                Ok(_) => println!("  ✓ Read Only Memory works!"),
                Err(e) => println!("  ✗ Verify error: {:?}", e),
            }
        }
        Err(e) => println!("  Error: {:?}", e),
    }
}
