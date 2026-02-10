extern crate alloc;
use alloc::vec::Vec;

use lambdaworks_math::circle::fold::natural_to_butterfly;
use lambdaworks_math::circle::traits::IsCircleFriField;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::traits::AsBytes;

use crate::merkle_tree::proof::Proof;

use super::commit::CircleFriCommitment;
use super::errors::CircleFriError;

/// Decommitment for a single query across all FRI layers.
#[derive(Debug, Clone)]
pub struct CircleFriDecommitment<F: IsCircleFriField> {
    /// For each layer: the evaluation at the query's fold-partner position.
    pub layer_sibling_evals: Vec<FieldElement<F>>,
    /// For each layer: Merkle authentication paths for both the query position
    /// and its fold-partner position.
    pub layer_auth_paths_own: Vec<Proof<[u8; 32]>>,
    pub layer_auth_paths_sibling: Vec<Proof<[u8; 32]>>,
}

/// Produces decommitments for the given query indices.
///
/// # Arguments
/// * `commitment` - The Circle FRI commitment from the commit phase
/// * `query_indices` - Indices in the **natural-order** evaluation domain
/// * `domain_size` - Size of the original evaluation domain
///
/// # Returns
/// One `CircleFriDecommitment` per query index.
pub fn circle_fri_query<F: IsCircleFriField>(
    commitment: &CircleFriCommitment<F>,
    query_indices: &[usize],
    domain_size: usize,
) -> Result<Vec<CircleFriDecommitment<F>>, CircleFriError>
where
    FieldElement<F>: AsBytes,
{
    for &idx in query_indices {
        if idx >= domain_size {
            return Err(CircleFriError::QueryIndexOutOfBounds {
                index: idx,
                domain_size,
            });
        }
    }

    let mut decommitments = Vec::with_capacity(query_indices.len());

    for &nat_idx in query_indices {
        let mut sibling_evals = Vec::new();
        let mut auth_paths_own = Vec::new();
        let mut auth_paths_sibling = Vec::new();

        // Convert natural-order query index to butterfly-order index
        let mut idx = natural_to_butterfly(nat_idx, domain_size);

        for layer in &commitment.layers {
            let layer_size = layer.evaluations.len();
            let half = layer_size / 2;

            // Determine the fold pair
            let pair_idx = idx % half;
            let sibling_idx = if idx < half { idx + half } else { idx - half };

            // Extract sibling evaluation
            sibling_evals.push(layer.evaluations[sibling_idx].clone());

            // Merkle proofs for both positions
            let proof_own = layer
                .merkle_tree
                .get_proof_by_pos(idx)
                .ok_or(CircleFriError::MerkleProofFailed(idx))?;
            let proof_sibling = layer
                .merkle_tree
                .get_proof_by_pos(sibling_idx)
                .ok_or(CircleFriError::MerkleProofFailed(sibling_idx))?;

            auth_paths_own.push(proof_own);
            auth_paths_sibling.push(proof_sibling);

            // Next layer index: the fold result goes to pair_idx
            idx = pair_idx;
        }

        decommitments.push(CircleFriDecommitment {
            layer_sibling_evals: sibling_evals,
            layer_auth_paths_own: auth_paths_own,
            layer_auth_paths_sibling: auth_paths_sibling,
        });
    }

    Ok(decommitments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circle_fri::commit::circle_fri_commit;
    use crate::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::circle::domain::CircleDomain;
    use lambdaworks_math::circle::polynomial::evaluate_cfft;
    use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;

    type FE = FieldElement<Mersenne31Field>;

    #[test]
    fn query_produces_correct_number_of_decommitments() {
        let coeffs: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let evals = evaluate_cfft(coeffs);
        let domain = CircleDomain::new_standard(3);
        let mut transcript = DefaultTranscript::<Mersenne31Field>::new(&[]);
        let commitment = circle_fri_commit(&evals, &domain, &mut transcript)
            .expect("commit should succeed for valid inputs");

        let queries = vec![0, 3, 5];
        let decommitments = circle_fri_query(&commitment, &queries, domain.size())
            .expect("query should succeed for valid inputs");
        assert_eq!(decommitments.len(), 3);

        for d in &decommitments {
            assert_eq!(d.layer_sibling_evals.len(), commitment.layers.len());
            assert_eq!(d.layer_auth_paths_own.len(), commitment.layers.len());
            assert_eq!(d.layer_auth_paths_sibling.len(), commitment.layers.len());
        }
    }
}
