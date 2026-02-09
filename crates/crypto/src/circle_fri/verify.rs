extern crate alloc;
use alloc::vec::Vec;

use lambdaworks_math::circle::domain::CircleDomain;
use lambdaworks_math::circle::fold::{fold_pair, natural_to_butterfly};
use lambdaworks_math::circle::twiddles::{get_twiddles, TwiddlesConfig};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;

use crate::fiat_shamir::is_transcript::IsTranscript;
use crate::merkle_tree::backends::types::Keccak256Backend;

use super::errors::CircleFriError;
use super::query::CircleFriDecommitment;

type FE = FieldElement<Mersenne31Field>;
type FriBackend = Keccak256Backend<Mersenne31Field>;

/// All data the verifier needs from the prover.
#[derive(Debug, Clone)]
pub struct CircleFriProof {
    /// Merkle roots for each committed layer.
    pub layer_merkle_roots: Vec<[u8; 32]>,
    /// The final constant value.
    pub final_value: FE,
    /// Decommitments for each query.
    pub decommitments: Vec<CircleFriDecommitment>,
    /// The queried indices in natural order.
    pub query_indices: Vec<usize>,
    /// The evaluation at each queried index in the original domain.
    pub query_evaluations: Vec<FE>,
}

/// Verifies a Circle FRI proof.
///
/// # Arguments
/// * `proof`      - The Circle FRI proof to verify
/// * `domain`     - The original evaluation domain
/// * `transcript` - Fiat-Shamir transcript (must be seeded identically to the prover)
pub fn circle_fri_verify(
    proof: &CircleFriProof,
    domain: &CircleDomain,
    transcript: &mut impl IsTranscript<Mersenne31Field>,
) -> Result<bool, CircleFriError> {
    let num_layers = proof.layer_merkle_roots.len();

    // Precompute inverse twiddles (same as the prover)
    let inv_twiddles = get_twiddles(domain.coset.clone(), TwiddlesConfig::Interpolation);

    // Reconstruct challenges from transcript (same sequence as prover)
    let mut challenges = Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        transcript.append_bytes(&proof.layer_merkle_roots[layer_idx]);
        let challenge: FE = transcript.sample_field_element();
        challenges.push(challenge);
    }
    transcript.append_field_element(&proof.final_value);

    let domain_size = domain.size();

    // Verify each query's decommitment chain
    for (q_pos, decommitment) in proof.query_indices.iter().zip(&proof.decommitments) {
        let nat_idx = *q_pos;
        let mut idx = natural_to_butterfly(nat_idx, domain_size);

        // The initial evaluation at the query point
        let q_eval_pos = proof
            .query_indices
            .iter()
            .position(|&x| x == nat_idx)
            .unwrap();
        let mut current_eval = proof.query_evaluations[q_eval_pos];

        let mut layer_size = domain_size;

        for layer_idx in 0..num_layers {
            let half = layer_size / 2;
            let pair_idx = idx % half;
            let is_first_half = idx < half;

            let sibling_eval = decommitment.layer_sibling_evals[layer_idx];
            let sibling_idx = if is_first_half {
                idx + half
            } else {
                idx - half
            };

            // Verify Merkle proof for own evaluation
            if !decommitment.layer_auth_paths_own[layer_idx].verify::<FriBackend>(
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
            if !decommitment.layer_auth_paths_sibling[layer_idx].verify::<FriBackend>(
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

    fn run_circle_fri_protocol(log_domain_size: u32, num_queries: usize) {
        let n = 1usize << log_domain_size;
        let coeffs: Vec<FE> = (1..=n).map(|i| FE::from(i as u64)).collect();
        let evals = evaluate_cfft(coeffs);
        let domain = CircleDomain::new_standard(log_domain_size);

        // Prover: commit phase
        let seed = [0x13, 0x37];
        let mut prover_transcript = DefaultTranscript::<Mersenne31Field>::new(&seed);
        let commitment = circle_fri_commit(&evals, &domain, &mut prover_transcript).unwrap();

        // Prover: sample query indices from transcript
        let query_indices: Vec<usize> = (0..num_queries)
            .map(|_| prover_transcript.sample_u64(n as u64) as usize)
            .collect();
        let query_evaluations: Vec<FE> = query_indices.iter().map(|&i| evals[i]).collect();

        // Prover: query phase
        let decommitments = circle_fri_query(&commitment, &query_indices, n).unwrap();

        // Build proof
        let proof = CircleFriProof {
            layer_merkle_roots: commitment
                .layers
                .iter()
                .map(|l| l.merkle_tree.root)
                .collect(),
            final_value: commitment.final_value,
            decommitments,
            query_indices: query_indices.clone(),
            query_evaluations,
        };

        // Verifier: reconstruct transcript and verify
        let mut verifier_transcript = DefaultTranscript::<Mersenne31Field>::new(&seed);
        let result = circle_fri_verify(&proof, &domain, &mut verifier_transcript);
        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
        assert!(result.unwrap());
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
        let coeffs: Vec<FE> = (1..=n).map(|i| FE::from(i as u64)).collect();
        let evals = evaluate_cfft(coeffs);
        let domain = CircleDomain::new_standard(4);

        let seed = [0xAB, 0xCD];
        let mut prover_transcript = DefaultTranscript::<Mersenne31Field>::new(&seed);
        let commitment = circle_fri_commit(&evals, &domain, &mut prover_transcript).unwrap();

        let query_indices = vec![0, 3];
        let query_evaluations: Vec<FE> = query_indices.iter().map(|&i| evals[i]).collect();
        let decommitments = circle_fri_query(&commitment, &query_indices, n).unwrap();

        // Tamper with the final value
        let proof = CircleFriProof {
            layer_merkle_roots: commitment
                .layers
                .iter()
                .map(|l| l.merkle_tree.root)
                .collect(),
            final_value: commitment.final_value + FE::from(1u64),
            decommitments,
            query_indices,
            query_evaluations,
        };

        let mut verifier_transcript = DefaultTranscript::<Mersenne31Field>::new(&seed);
        let result = circle_fri_verify(&proof, &domain, &mut verifier_transcript);
        assert!(result.is_err());
    }

    #[test]
    fn circle_fri_rejects_tampered_evaluation() {
        let n = 8usize;
        let coeffs: Vec<FE> = (1..=n).map(|i| FE::from(i as u64)).collect();
        let evals = evaluate_cfft(coeffs);
        let domain = CircleDomain::new_standard(3);

        let seed = [0x00];
        let mut prover_transcript = DefaultTranscript::<Mersenne31Field>::new(&seed);
        let commitment = circle_fri_commit(&evals, &domain, &mut prover_transcript).unwrap();

        let query_indices = vec![1];
        let mut query_evaluations: Vec<FE> = query_indices.iter().map(|&i| evals[i]).collect();
        let decommitments = circle_fri_query(&commitment, &query_indices, n).unwrap();

        // Tamper with the claimed evaluation
        query_evaluations[0] = query_evaluations[0] + FE::from(1u64);

        let proof = CircleFriProof {
            layer_merkle_roots: commitment
                .layers
                .iter()
                .map(|l| l.merkle_tree.root)
                .collect(),
            final_value: commitment.final_value,
            decommitments,
            query_indices,
            query_evaluations,
        };

        let mut verifier_transcript = DefaultTranscript::<Mersenne31Field>::new(&seed);
        let result = circle_fri_verify(&proof, &domain, &mut verifier_transcript);
        // Should fail - either Merkle proof fails or final value mismatch
        assert!(result.is_err());
    }
}
