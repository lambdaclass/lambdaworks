extern crate alloc;
use alloc::vec::Vec;

use lambdaworks_math::circle::domain::CircleDomain;
use lambdaworks_math::circle::fold::{fold_pair, natural_to_butterfly};
use lambdaworks_math::circle::traits::IsCircleFriField;
use lambdaworks_math::circle::twiddles::{get_twiddles, TwiddlesConfig};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::traits::AsBytes;

use crate::fiat_shamir::is_transcript::IsTranscript;
use crate::merkle_tree::backends::types::Keccak256Backend;

use super::errors::CircleFriError;
use super::query::CircleFriDecommitment;

/// All data the verifier needs from the prover.
#[derive(Debug, Clone)]
pub struct CircleFriProof<F: IsCircleFriField> {
    /// Merkle roots for each committed layer.
    pub layer_merkle_roots: Vec<[u8; 32]>,
    /// The final constant value.
    pub final_value: FieldElement<F>,
    /// Decommitments for each query.
    pub decommitments: Vec<CircleFriDecommitment<F>>,
}

/// Verifies a Circle FRI proof.
///
/// The verifier derives query indices from the Fiat-Shamir transcript
/// (after absorbing all Merkle roots and the final value), ensuring
/// the prover cannot choose favorable query positions.
///
/// # Arguments
/// * `proof`       - The Circle FRI proof to verify
/// * `domain`      - The original evaluation domain
/// * `num_queries` - Number of queries to sample from the transcript
/// * `transcript`  - Fiat-Shamir transcript (must be seeded identically to the prover)
pub fn circle_fri_verify<F: IsCircleFriField>(
    proof: &CircleFriProof<F>,
    domain: &CircleDomain<F>,
    num_queries: usize,
    transcript: &mut impl IsTranscript<F>,
) -> Result<bool, CircleFriError>
where
    FieldElement<F>: AsBytes,
{
    let num_layers = proof.layer_merkle_roots.len();
    let domain_size = domain.size();

    // Precompute inverse twiddles (same as the prover)
    let inv_twiddles = get_twiddles(domain.coset.clone(), TwiddlesConfig::Interpolation);

    if num_layers != inv_twiddles.len() {
        return Err(CircleFriError::InconsistentProof(
            "proof layer count does not match expected fold depth",
        ));
    }

    // Precompute inv(2) once for all fold_pair calls
    let inv_two = FieldElement::<F>::from(2u64).inv().unwrap();

    // Reconstruct challenges from transcript (same sequence as prover)
    let challenges: Vec<FieldElement<F>> = proof
        .layer_merkle_roots
        .iter()
        .map(|root| {
            transcript.append_bytes(root);
            transcript.sample_field_element()
        })
        .collect();
    transcript.append_field_element(&proof.final_value);

    // Derive query indices from transcript (same as the prover)
    let query_indices: Vec<usize> = (0..num_queries)
        .map(|_| transcript.sample_u64(domain_size as u64) as usize)
        .collect();

    if proof.decommitments.len() != num_queries {
        return Err(CircleFriError::InconsistentProof(
            "number of decommitments does not match num_queries",
        ));
    }

    // Verify each query's decommitment chain
    for (&nat_idx, decommitment) in query_indices.iter().zip(&proof.decommitments) {
        let mut idx = natural_to_butterfly(nat_idx, domain_size);

        if decommitment.layer_sibling_evals.len() != num_layers
            || decommitment.layer_auth_paths_own.len() != num_layers
            || decommitment.layer_auth_paths_sibling.len() != num_layers
        {
            return Err(CircleFriError::InconsistentProof(
                "decommitment layer count does not match number of committed layers",
            ));
        }

        // The initial evaluation at the query point (provided by the prover,
        // verified against the Merkle commitment below)
        let mut current_eval = decommitment.eval_at_query.clone();

        let mut layer_size = domain_size;

        for layer_idx in 0..num_layers {
            let half = layer_size / 2;
            let pair_idx = idx % half;
            let is_first_half = idx < half;

            let sibling_eval = decommitment.layer_sibling_evals[layer_idx].clone();
            let sibling_idx = if is_first_half {
                idx + half
            } else {
                idx - half
            };

            // Verify Merkle proof for own evaluation
            if !decommitment.layer_auth_paths_own[layer_idx].verify::<Keccak256Backend<F>>(
                &proof.layer_merkle_roots[layer_idx],
                idx,
                &current_eval,
            ) {
                return Err(CircleFriError::VerificationFailed(
                    layer_idx,
                    "Merkle proof failed for own evaluation",
                ));
            }

            // Verify Merkle proof for sibling evaluation
            if !decommitment.layer_auth_paths_sibling[layer_idx].verify::<Keccak256Backend<F>>(
                &proof.layer_merkle_roots[layer_idx],
                sibling_idx,
                &sibling_eval,
            ) {
                return Err(CircleFriError::VerificationFailed(
                    layer_idx,
                    "Merkle proof failed for sibling evaluation",
                ));
            }

            // Compute fold: f_hi is the first-half element, f_lo is the second-half element
            let (f_hi, f_lo) = if is_first_half {
                (&current_eval, &sibling_eval)
            } else {
                (&sibling_eval, &current_eval)
            };

            current_eval = fold_pair(
                f_hi,
                f_lo,
                &inv_twiddles[layer_idx][pair_idx],
                &challenges[layer_idx],
                &inv_two,
            );

            // Move to next layer
            idx = pair_idx;
            layer_size = half;
        }

        // After all folds, current_eval should match the final constant
        if current_eval != proof.final_value {
            return Err(CircleFriError::FinalValueMismatch);
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circle_fri::commit::circle_fri_commit;
    use crate::circle_fri::query::circle_fri_query;
    use crate::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::circle::polynomial::evaluate_cfft;
    use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;

    type FE = FieldElement<Mersenne31Field>;

    fn run_circle_fri_protocol(log_domain_size: u32, num_queries: usize) {
        let n = 1usize << log_domain_size;
        let coeffs: Vec<FE> = (1..=n).map(|i| FE::from(i as u64)).collect();
        let evals = evaluate_cfft(coeffs);
        let domain = CircleDomain::new_standard(log_domain_size);

        // Prover: commit phase
        let seed = [0x13, 0x37];
        let mut prover_transcript = DefaultTranscript::<Mersenne31Field>::new(&seed);
        let commitment = circle_fri_commit(&evals, &domain, &mut prover_transcript)
            .expect("commit should succeed");

        // Prover: sample query indices from transcript (must happen in same order as verifier)
        let query_indices: Vec<usize> = (0..num_queries)
            .map(|_| prover_transcript.sample_u64(n as u64) as usize)
            .collect();

        // Prover: query phase
        let decommitments =
            circle_fri_query(&commitment, &query_indices, n).expect("query should succeed");

        // Build proof (query_indices are NOT included — verifier derives them)
        let proof = CircleFriProof {
            layer_merkle_roots: commitment
                .layers
                .iter()
                .map(|l| l.merkle_tree.root)
                .collect(),
            final_value: commitment.final_value,
            decommitments,
        };

        // Verifier: reconstruct transcript and verify
        let mut verifier_transcript = DefaultTranscript::<Mersenne31Field>::new(&seed);
        let result = circle_fri_verify(&proof, &domain, num_queries, &mut verifier_transcript);
        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
        assert!(result.expect("already checked is_ok"));
    }

    #[test]
    fn circle_fri_end_to_end_8_points() {
        run_circle_fri_protocol(3, 2);
    }

    #[test]
    fn circle_fri_end_to_end_16_points() {
        run_circle_fri_protocol(4, 3);
    }

    #[test]
    fn circle_fri_end_to_end_64_points() {
        run_circle_fri_protocol(6, 5);
    }

    #[test]
    fn circle_fri_end_to_end_1024_points() {
        run_circle_fri_protocol(10, 10);
    }

    #[test]
    fn circle_fri_rejects_tampered_final_value() {
        let n = 16usize;
        let num_queries = 3;
        let coeffs: Vec<FE> = (1..=n).map(|i| FE::from(i as u64)).collect();
        let evals = evaluate_cfft(coeffs);
        let domain = CircleDomain::new_standard(4);

        let seed = [0xAB, 0xCD];
        let mut prover_transcript = DefaultTranscript::<Mersenne31Field>::new(&seed);
        let commitment = circle_fri_commit(&evals, &domain, &mut prover_transcript)
            .expect("commit should succeed");

        let query_indices: Vec<usize> = (0..num_queries)
            .map(|_| prover_transcript.sample_u64(n as u64) as usize)
            .collect();
        let decommitments =
            circle_fri_query(&commitment, &query_indices, n).expect("query should succeed");

        // Tamper with the final value
        let proof = CircleFriProof {
            layer_merkle_roots: commitment
                .layers
                .iter()
                .map(|l| l.merkle_tree.root)
                .collect(),
            final_value: commitment.final_value + FE::from(1u64),
            decommitments,
        };

        let mut verifier_transcript = DefaultTranscript::<Mersenne31Field>::new(&seed);
        let result = circle_fri_verify(&proof, &domain, num_queries, &mut verifier_transcript);
        assert!(result.is_err());
    }

    #[test]
    fn circle_fri_rejects_tampered_evaluation() {
        let n = 8usize;
        let num_queries = 1;
        let coeffs: Vec<FE> = (1..=n).map(|i| FE::from(i as u64)).collect();
        let evals = evaluate_cfft(coeffs);
        let domain = CircleDomain::new_standard(3);

        let seed = [0x00];
        let mut prover_transcript = DefaultTranscript::<Mersenne31Field>::new(&seed);
        let commitment = circle_fri_commit(&evals, &domain, &mut prover_transcript)
            .expect("commit should succeed");

        let query_indices: Vec<usize> = (0..num_queries)
            .map(|_| prover_transcript.sample_u64(n as u64) as usize)
            .collect();
        let mut decommitments =
            circle_fri_query(&commitment, &query_indices, n).expect("query should succeed");

        // Tamper with the claimed evaluation
        decommitments[0].eval_at_query += FE::from(1u64);

        let proof = CircleFriProof {
            layer_merkle_roots: commitment
                .layers
                .iter()
                .map(|l| l.merkle_tree.root)
                .collect(),
            final_value: commitment.final_value,
            decommitments,
        };

        let mut verifier_transcript = DefaultTranscript::<Mersenne31Field>::new(&seed);
        let result = circle_fri_verify(&proof, &domain, num_queries, &mut verifier_transcript);
        // Should fail - either Merkle proof fails or final value mismatch
        assert!(result.is_err());
    }
}
