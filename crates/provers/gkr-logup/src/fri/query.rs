//! FRI query phase: generate decommitments for sampled indices.

use lambdaworks_crypto::merkle_tree::backends::types::Keccak256Backend;
use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::traits::AsBytes;

use super::types::{FriError, FriLayerData, FriQueryRound};

/// Generate query decommitments for one query index across all FRI layers.
///
/// At each layer the query index is reduced: `index = index % (domain_size / 2)`.
/// For each layer we provide the evaluation at `index` and its symmetric partner
/// `index + domain_size/2`, along with Merkle authentication paths.
pub(crate) fn fri_query_single<F>(
    mut index: usize,
    layers: &[FriLayerData<F>],
    merkle_trees: &[MerkleTree<Keccak256Backend<F>>],
) -> Result<Vec<FriQueryRound<F>>, FriError>
where
    F: IsFFTField,
    FieldElement<F>: AsBytes + Clone + Send + Sync,
    F::BaseType: Send + Sync,
{
    let mut rounds = Vec::with_capacity(layers.len());

    for (i, (layer, tree)) in layers.iter().zip(merkle_trees.iter()).enumerate() {
        let half = layer.domain_size / 2;
        // Normalize index to the lower half
        let idx = index % half;
        let idx_sym = idx + half;

        let eval = layer.evaluations[idx].clone();
        let eval_sym = layer.evaluations[idx_sym].clone();

        let auth_path = tree
            .get_proof_by_pos(idx)
            .ok_or_else(|| FriError::MerkleError(format!("no proof at index {idx}, layer {i}")))?;

        let auth_path_sym = tree.get_proof_by_pos(idx_sym).ok_or_else(|| {
            FriError::MerkleError(format!("no proof at symmetric index {idx_sym}, layer {i}"))
        })?;

        rounds.push(FriQueryRound {
            eval,
            eval_sym,
            auth_path,
            auth_path_sym,
        });

        // For the next layer, the domain halves
        index = idx;
    }

    Ok(rounds)
}

/// Generate query decommitments for all query indices.
pub(crate) fn fri_query_all<F>(
    query_indices: &[usize],
    layers: &[FriLayerData<F>],
    merkle_trees: &[MerkleTree<Keccak256Backend<F>>],
) -> Result<Vec<Vec<FriQueryRound<F>>>, FriError>
where
    F: IsFFTField,
    FieldElement<F>: AsBytes + Clone + Send + Sync,
    F::BaseType: Send + Sync,
{
    query_indices
        .iter()
        .map(|&idx| fri_query_single(idx, layers, merkle_trees))
        .collect()
}
