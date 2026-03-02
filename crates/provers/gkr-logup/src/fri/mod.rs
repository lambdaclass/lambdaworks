//! Standalone FRI (Fast Reed-Solomon IOP of Proximity) protocol.
//!
//! Provides a low-degree test for univariate polynomials over FFT-friendly fields,
//! using Merkle tree commitments (Keccak256) for each folding layer.

pub mod fold;
pub mod types;

pub mod commit;
pub mod query;
pub mod verify;

pub mod pcs;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::{AsBytes, ByteConversion};

use commit::FriCommitResult;
use types::{FriConfig, FriError, FriProof};

/// Run the FRI commit phase and sample query indices from the transcript.
///
/// Returns the commit result (layers, Merkle trees, final value) and the
/// sampled query indices. This allows callers to extract additional
/// decommitments (e.g. original polynomial evaluations) at the same
/// query positions before assembling the proof.
pub fn fri_commit_and_sample<F, T>(
    poly: &Polynomial<FieldElement<F>>,
    config: &FriConfig,
    transcript: &mut T,
) -> Result<(FriCommitResult<F>, Vec<usize>), FriError>
where
    F: IsFFTField,
    F::BaseType: Send + Sync,
    FieldElement<F>: AsBytes + ByteConversion + Clone + Send + Sync,
    T: IsTranscript<F>,
{
    let commit_result = commit::fri_commit(poly, config, transcript)?;

    let first_domain_size = if commit_result.layers.is_empty() {
        0
    } else {
        commit_result.layers[0].domain_size
    };

    let mut query_indices: Vec<usize> = if first_domain_size == 0 {
        vec![0; config.num_queries]
    } else {
        (0..config.num_queries)
            .map(|_| transcript.sample_u64((first_domain_size / 2) as u64) as usize)
            .collect()
    };

    // Deduplicate: duplicate queries don't add security and waste prover/verifier work.
    // Both sides sample the same indices from the transcript, so both get the same
    // unique set after dedup.
    // NOTE: With domain_size/2 possible indices, the expected number of unique queries
    // from `num_queries` samples is `(domain_size/2) * (1 - (1 - 2/domain_size)^num_queries)`.
    // For small domains (e.g. degree 8 with 2x blowup → 8 indices, 30 queries),
    // dedup significantly reduces the effective query count. This is acceptable
    // because small domains have few FRI layers and the final value check dominates.
    query_indices.sort_unstable();
    query_indices.dedup();

    Ok((commit_result, query_indices))
}

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
    let (commit_result, query_indices) = fri_commit_and_sample(poly, config, transcript)?;

    // Query phase
    let query_rounds = if commit_result.layers.is_empty() {
        vec![vec![]; query_indices.len()]
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
        proof.final_value += FE::one();

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
