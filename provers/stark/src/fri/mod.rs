pub mod fri_commitment;
pub mod fri_decommit;
mod fri_functions;

use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use lambdaworks_math::fft::polynomial::FFTPoly;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::traits::Serializable;
pub use lambdaworks_math::{
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
    polynomial::Polynomial,
};

use crate::config::{BatchedMerkleTree, BatchedMerkleTreeBackend};
use crate::transcript::IsStarkTranscript;

use self::fri_commitment::FriLayer;
use self::fri_decommit::FriDecommitment;
use self::fri_functions::fold_polynomial;

pub fn commit_phase<F: IsFFTField>(
    number_layers: usize,
    p_0: Polynomial<FieldElement<F>>,
    transcript: &mut impl IsStarkTranscript<F>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> (
    FieldElement<F>,
    Vec<FriLayer<F, BatchedMerkleTreeBackend<F>>>,
)
where
    FieldElement<F>: Serializable,
{
    let mut domain_size = domain_size;

    let mut fri_layer_list = Vec::with_capacity(number_layers);
    let mut current_layer: FriLayer<F, BatchedMerkleTreeBackend<F>>;
    let mut current_poly = p_0;

    let mut coset_offset = coset_offset.clone();

    for _ in 1..number_layers {
        // <<<< Receive challenge 𝜁ₖ₋₁
        let zeta = transcript.sample_field_element();
        coset_offset = coset_offset.square();
        domain_size /= 2;

        // Compute layer polynomial and domain
        current_poly = fold_polynomial(&current_poly, &zeta) * FieldElement::from(2);
        current_layer = new_fri_layer(&current_poly, &coset_offset, domain_size);
        let new_data = &current_layer.merkle_tree.root;
        fri_layer_list.push(current_layer.clone()); // TODO: remove this clone

        // >>>> Send commitment: [pₖ]
        transcript.append_bytes(new_data);
    }

    // <<<< Receive challenge: 𝜁ₙ₋₁
    let zeta = transcript.sample_field_element();

    let last_poly = fold_polynomial(&current_poly, &zeta) * FieldElement::from(2);

    let last_value = last_poly
        .coefficients()
        .get(0)
        .unwrap_or(&FieldElement::zero())
        .clone();

    // >>>> Send value: pₙ
    transcript.append_field_element(&last_value);

    (last_value, fri_layer_list)
}

pub fn query_phase<F: IsFFTField>(
    fri_layers: &Vec<FriLayer<F, BatchedMerkleTreeBackend<F>>>,
    iotas: &[usize],
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: Serializable,
{
    if !fri_layers.is_empty() {
        let query_list = iotas
            .iter()
            .map(|iota_s| {
                let mut layers_evaluations_sym = Vec::new();
                let mut layers_auth_paths_sym = Vec::new();

                let mut index = *iota_s;
                for layer in fri_layers {
                    // symmetric element
                    let evaluation_sym = layer.evaluation[index ^ 1].clone();
                    let auth_path_sym = layer.merkle_tree.get_proof_by_pos(index >> 1).unwrap();
                    layers_evaluations_sym.push(evaluation_sym);
                    layers_auth_paths_sym.push(auth_path_sym);

                    index >>= 1;
                }

                FriDecommitment {
                    layers_auth_paths: layers_auth_paths_sym,
                    layers_evaluations_sym,
                    query_index: *iota_s as u64,
                }
            })
            .collect();

        query_list
    } else {
        vec![]
    }
}

pub fn new_fri_layer<F: IsFFTField>(
    poly: &Polynomial<FieldElement<F>>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> crate::fri::fri_commitment::FriLayer<F, BatchedMerkleTreeBackend<F>>
where
    F: IsFFTField,
    FieldElement<F>: Serializable,
{
    let mut evaluation = poly
        .evaluate_offset_fft(1, Some(domain_size), coset_offset)
        .unwrap(); // TODO: return error

    in_place_bit_reverse_permute(&mut evaluation);

    let mut to_commit = Vec::new();
    for chunk in evaluation.chunks(2) {
        to_commit.push(vec![chunk[0].clone(), chunk[1].clone()]);
    }

    let merkle_tree = BatchedMerkleTree::build(&to_commit);

    FriLayer::new(&evaluation, merkle_tree, coset_offset.clone(), domain_size)
}
