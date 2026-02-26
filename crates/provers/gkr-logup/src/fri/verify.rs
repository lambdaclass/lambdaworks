//! FRI verification: replay transcript, check fold consistency + Merkle proofs.

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_crypto::merkle_tree::backends::types::Keccak256Backend;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::traits::{AsBytes, ByteConversion};

use super::fold::fold_eval;
use super::types::{FriConfig, FriError, FriProof};

/// Verify a FRI proof.
///
/// 1. Replay transcript to reconstruct folding challenges (betas).
/// 2. Sample query indices from transcript.
/// 3. For each query at each layer: verify Merkle proofs, check fold consistency.
/// 4. Check the final folded value matches the claimed constant.
///
/// Returns the sampled query indices on success, so callers can use them to
/// verify consistency with original polynomial commitments.
pub fn fri_verify<F, T>(
    proof: &FriProof<F>,
    poly_degree_bound: usize,
    config: &FriConfig,
    transcript: &mut T,
) -> Result<Vec<usize>, FriError>
where
    F: IsFFTField,
    F::BaseType: Send + Sync,
    FieldElement<F>: AsBytes + ByteConversion + Clone + Send + Sync,
    T: IsTranscript<F>,
{
    let num_layers = proof.layer_merkle_roots.len();
    let blowup = config.blowup_factor();

    // Replay commit phase on transcript to recover betas
    let mut betas = Vec::with_capacity(num_layers);
    for root in &proof.layer_merkle_roots {
        transcript.append_bytes(root);
        let beta: FieldElement<F> = transcript.sample_field_element();
        betas.push(beta);
    }

    // Append final value
    transcript.append_field_element(&proof.final_value);

    // Sample query indices
    let first_domain_size = poly_degree_bound.next_power_of_two() * blowup;
    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| transcript.sample_u64((first_domain_size / 2) as u64) as usize)
        .collect();

    if proof.query_rounds.len() != config.num_queries {
        return Err(FriError::InvalidConfig(format!(
            "expected {} query rounds, got {}",
            config.num_queries,
            proof.query_rounds.len()
        )));
    }

    // Compute domain sizes for each layer
    let mut domain_sizes = Vec::with_capacity(num_layers);
    let mut ds = first_domain_size;
    for _ in 0..num_layers {
        domain_sizes.push(ds);
        ds /= 2;
    }

    // Compute the primitive root of unity for the first domain
    let first_root = F::get_primitive_root_of_unity(first_domain_size.trailing_zeros() as u64)
        .map_err(|_| FriError::FftError("no root of unity for domain".into()))?;

    // For each query
    for (q, query_rounds) in proof.query_rounds.iter().enumerate() {
        if query_rounds.len() != num_layers {
            return Err(FriError::InvalidConfig(format!(
                "query {q}: expected {num_layers} rounds, got {}",
                query_rounds.len()
            )));
        }

        let mut index = query_indices[q];

        for (layer, round) in query_rounds.iter().enumerate() {
            let ds = domain_sizes[layer];
            let half = ds / 2;
            let idx = index % half;
            let idx_sym = idx + half;

            // Verify Merkle proofs
            let root = &proof.layer_merkle_roots[layer];
            if !round
                .auth_path
                .verify::<Keccak256Backend<F>>(root, idx, &round.eval)
            {
                return Err(FriError::MerkleProofFailed { query: q, layer });
            }
            if !round
                .auth_path_sym
                .verify::<Keccak256Backend<F>>(root, idx_sym, &round.eval_sym)
            {
                return Err(FriError::MerkleProofFailed { query: q, layer });
            }

            // The domain point at position idx in the LDE domain.
            // Domain is roots of unity of size ds: omega_ds^idx.
            let omega_ds = first_root.pow(first_domain_size / ds);
            let x = omega_ds.pow(idx);

            // Fold consistency check: the folded evaluation should match
            // the evaluation at the corresponding index in the next layer.
            let folded = fold_eval(&round.eval, &round.eval_sym, &betas[layer], &x)?;

            if layer + 1 < num_layers {
                // The folded polynomial lives on a domain of size ds/2.
                // The folded value is at position idx in that domain.
                // The next layer normalizes: next_idx = idx % next_half.
                // If idx < next_half: the folded value matches next_round.eval
                // If idx >= next_half: the folded value matches next_round.eval_sym
                let next_ds = domain_sizes[layer + 1];
                let next_half = next_ds / 2;
                let next_round = &query_rounds[layer + 1];

                let next_eval = if idx < next_half {
                    &next_round.eval
                } else {
                    &next_round.eval_sym
                };

                if folded != *next_eval {
                    return Err(FriError::FoldConsistencyFailed { query: q, layer });
                }
            } else {
                // Last layer: folded should equal the final constant
                if folded != proof.final_value {
                    return Err(FriError::FinalValueMismatch);
                }
            }

            index = idx;
        }
    }

    Ok(query_indices)
}
