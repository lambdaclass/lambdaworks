//! Standalone FRI (Fast Reed-Solomon IOP of Proximity) protocol.
//!
//! Provides a low-degree test for univariate polynomials over FFT-friendly fields,
//! using Merkle tree commitments (Keccak256) for each folding layer.

pub mod fold;
pub mod types;

pub(crate) mod commit;
pub(crate) mod query;
pub mod verify;

pub mod pcs;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::{AsBytes, ByteConversion};

use types::{FriConfig, FriError, FriProof};

/// Run the full FRI prover: commit + query.
///
/// Takes a polynomial in coefficient form and produces a FRI proof.
pub fn fri_prove<F, T>(
    poly: &Polynomial<FieldElement<F>>,
    config: &FriConfig,
    transcript: &mut T,
) -> Result<FriProof<F>, FriError>
where
    F: IsFFTField,
    F::BaseType: Send + Sync,
    FieldElement<F>: AsBytes + ByteConversion + Clone + Send + Sync,
    T: IsTranscript<F>,
{
    // Commit phase
    let commit_result = commit::fri_commit(poly, config, transcript)?;

    // Sample query indices from transcript
    let first_domain_size = if commit_result.layers.is_empty() {
        // Constant polynomial — no layers
        0
    } else {
        commit_result.layers[0].domain_size
    };

    let query_indices: Vec<usize> = if first_domain_size == 0 {
        // No layers to query
        vec![0; config.num_queries]
    } else {
        (0..config.num_queries)
            .map(|_| transcript.sample_u64((first_domain_size / 2) as u64) as usize)
            .collect()
    };

    // Query phase
    let query_rounds = if commit_result.layers.is_empty() {
        vec![vec![]; config.num_queries]
    } else {
        query::fri_query_all(
            &query_indices,
            &commit_result.layers,
            &commit_result.merkle_trees,
        )?
    };

    let layer_merkle_roots = commit_result.layers.iter().map(|l| l.merkle_root).collect();

    Ok(FriProof {
        layer_merkle_roots,
        query_rounds,
        final_value: commit_result.final_value,
    })
}

pub use verify::fri_verify;

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;

    type F = Degree4BabyBearExtensionField;
    type FE = FieldElement<F>;

    #[test]
    fn fri_roundtrip_degree_7() {
        let coeffs: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let poly = Polynomial::new(&coeffs);
        let config = FriConfig {
            log_blowup: 1,
            num_queries: 10,
        };

        let mut prover_transcript = DefaultTranscript::<F>::new(b"fri_test");
        let proof = fri_prove(&poly, &config, &mut prover_transcript).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"fri_test");
        fri_verify(&proof, 8, &config, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn fri_roundtrip_degree_15() {
        let coeffs: Vec<FE> = (1..=16).map(|i| FE::from(i as u64)).collect();
        let poly = Polynomial::new(&coeffs);
        let config = FriConfig {
            log_blowup: 2,
            num_queries: 15,
        };

        let mut prover_transcript = DefaultTranscript::<F>::new(b"fri16");
        let proof = fri_prove(&poly, &config, &mut prover_transcript).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"fri16");
        fri_verify(&proof, 16, &config, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn fri_reject_tampered_final_value() {
        let coeffs: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let poly = Polynomial::new(&coeffs);
        let config = FriConfig {
            log_blowup: 1,
            num_queries: 10,
        };

        let mut prover_transcript = DefaultTranscript::<F>::new(b"tamper");
        let mut proof = fri_prove(&poly, &config, &mut prover_transcript).unwrap();

        // Tamper with the final value
        proof.final_value = proof.final_value.clone() + FE::one();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"tamper");
        let result = fri_verify(&proof, 8, &config, &mut verifier_transcript);
        assert!(result.is_err());
    }

    #[test]
    fn fri_constant_polynomial() {
        let poly = Polynomial::new(&[FE::from(42u64)]);
        let config = FriConfig {
            log_blowup: 1,
            num_queries: 5,
        };

        let mut prover_transcript = DefaultTranscript::<F>::new(b"const");
        let proof = fri_prove(&poly, &config, &mut prover_transcript).unwrap();

        assert!(proof.layer_merkle_roots.is_empty());
        assert_eq!(proof.final_value, FE::from(42u64));

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"const");
        fri_verify(&proof, 1, &config, &mut verifier_transcript).unwrap();
    }
}
