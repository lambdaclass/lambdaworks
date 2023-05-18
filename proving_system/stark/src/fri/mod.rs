pub mod fri_commitment;
pub mod fri_decommit;
mod fri_functions;
use crate::air::AIR;
use crate::fri::fri_commitment::FriLayer;
use crate::{transcript_to_field, transcript_to_usize, Domain};
use lambdaworks_crypto::hash::sha3::Sha3Hasher;

pub use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
pub use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_math::field::traits::{IsFFTField, IsField};
use lambdaworks_math::traits::ByteConversion;
pub use lambdaworks_math::{
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
    polynomial::Polynomial,
};

use self::fri_decommit::FriDecommitment;
use self::fri_functions::{fold_polynomial, next_domain};

pub type FriMerkleTree<F> = MerkleTree<F>;
pub(crate) const HASHER: Sha3Hasher = Sha3Hasher::new();

pub fn fri_commit_phase<F: IsField, T: Transcript>(
    number_layers: usize,
    p_0: Polynomial<FieldElement<F>>,
    domain_0: &[FieldElement<F>],
    transcript: &mut T,
) -> (FieldElement<F>, Vec<FriLayer<F>>)
where
    FieldElement<F>: ByteConversion,
{
    let mut fri_layer_list = Vec::with_capacity(number_layers);
    let mut current_layer = FriLayer::new(p_0, domain_0);
    fri_layer_list.push(current_layer.clone());

    // >>>> Send commitment: [p₀]
    transcript.append(&current_layer.merkle_tree.root.to_bytes_be());

    for _ in 1..number_layers {
        // <<<< Receive challenge 𝜁ₖ₋₁
        let zeta = transcript_to_field(transcript);

        // Compute layer polynomial and domain
        let next_poly = fold_polynomial(&current_layer.poly, &zeta);
        let next_domain = next_domain(&current_layer.domain);
        current_layer = FriLayer::new(next_poly, &next_domain);
        fri_layer_list.push(current_layer.clone());

        // >>>> Send commitment: [pₖ]
        transcript.append(&current_layer.merkle_tree.root.to_bytes_be());
    }

    // <<<< Receive challenge: 𝜁ₙ₋₁
    let zeta = transcript_to_field(transcript);

    let last_poly = fold_polynomial(&current_layer.poly, &zeta);

    let last_value = last_poly
        .coefficients()
        .get(0)
        .unwrap_or(&FieldElement::zero())
        .clone();

    // >>>> Send value: pₙ
    transcript.append(&last_value.to_bytes_be());

    (last_value, fri_layer_list)
}

pub fn fri_query_phase<F: IsFFTField, A: AIR<Field = F>, T: Transcript>(
    air: &A,
    domain: &Domain<F>,
    fri_layers: &Vec<FriLayer<F>>,
    transcript: &mut T,
) -> (Vec<FriDecommitment<F>>, usize)
where
    FieldElement<F>: ByteConversion,
{
    if let Some(first_layer) = fri_layers.get(0) {
        let number_of_queries = air.context().options.fri_number_of_queries;
        let mut iotas: Vec<usize> = Vec::with_capacity(number_of_queries);
        let query_list = (0..number_of_queries)
            .map(|_| {
                // <<<< Receive challenge 𝜄ₛ (iota_s)
                let iota_s = transcript_to_usize(transcript) % 2_usize.pow(domain.lde_root_order);

                let first_layer_evaluation = first_layer.evaluation[iota_s].clone();
                let first_layer_auth_path =
                    first_layer.merkle_tree.get_proof_by_pos(iota_s).unwrap();

                let mut layers_auth_paths_sym = vec![];
                let mut layers_evaluations_sym = vec![];

                for layer in fri_layers {
                    // symmetric element
                    let index_sym = (iota_s + layer.domain.len() / 2) % layer.domain.len();
                    let evaluation_sym = layer.evaluation[index_sym].clone();
                    let auth_path_sym = layer.merkle_tree.get_proof_by_pos(index_sym).unwrap();
                    layers_auth_paths_sym.push(auth_path_sym);
                    layers_evaluations_sym.push(evaluation_sym);
                }
                iotas.push(iota_s);

                FriDecommitment {
                    layers_auth_paths_sym,
                    layers_evaluations_sym,
                    first_layer_evaluation,
                    first_layer_auth_path,
                }
            })
            .collect();

        (query_list, iotas[0])
    } else {
        (vec![], 0)
    }
}
