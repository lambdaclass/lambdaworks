pub mod air;
pub mod fri;

use air::polynomials::get_cp_and_tp;
use fri::fri_decommit::fri_decommit_layers;
use lambdaworks_crypto::{fiat_shamir::transcript::Transcript, merkle_tree::Proof};
use lambdaworks_math::polynomial::Polynomial;
use winterfell::{
    crypto::hashers::Blake3_256,
    math::{fields::f128::BaseElement, StarkField},
    prover::constraints::CompositionPoly,
    Air, AuxTraceRandElements, Serializable, Trace, TraceTable,
};

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::{
    field::fields::u384_prime_field::{IsMontgomeryConfiguration, MontgomeryBackendPrimeField},
    unsigned_integer::element::U384,
};

type U384PrimeField = MontgomeryBackendPrimeField<crate::air::polynomials::MontgomeryConfig>;
type U384FieldElement = FieldElement<U384PrimeField>;
const MODULUS_MINUS_1: U384 =
    U384::from("800000000000011000000000000000000000000000000000000000000000000");

pub fn generate_vec_roots(
    subgroup_size: u64,
    coset_factor: u64,
) -> (Vec<U384FieldElement>, U384FieldElement) {
    let MODULUS_MINUS_1_FIELD: U384FieldElement = U384FieldElement::new(MODULUS_MINUS_1);
    let subgroup_size_u384: U384FieldElement = subgroup_size.into();
    let generator_field: U384FieldElement = 3.into();
    let coset_factor_u384: U384FieldElement = coset_factor.into();

    let exp = (MODULUS_MINUS_1_FIELD) / subgroup_size_u384;
    let exp_384 = *exp.value();

    let generator_of_subgroup = generator_field.pow(exp_384);

    let mut numbers = Vec::new();

    for exp in 0..subgroup_size {
        let ret = generator_of_subgroup.pow(exp) * &coset_factor_u384;
        numbers.push(ret.clone());
    }

    (numbers, generator_of_subgroup)
}

#[derive(Debug)]
pub struct StarkQueryProof {
    // TODO: fill this when we know what a proof entails
    pub trace_lde_poly_root: U384FieldElement,
    pub trace_lde_poly_evaluations: Vec<U384FieldElement>,
    /// Merkle paths for the trace polynomial evaluations
    pub trace_lde_poly_inclusion_proofs: Vec<Proof<U384PrimeField, DefaultHasher>>,

    composition_poly_root: U384FieldElement,
}

pub type StarkProof = Vec<StarkQueryProof>;

pub use lambdaworks_crypto::merkle_tree::{DefaultHasher, MerkleTree};
pub type FriMerkleTree = MerkleTree<U384PrimeField, DefaultHasher>;

pub fn prove<A>(
    air: A,
    trace: TraceTable<A::BaseField>,
    pub_inputs: A::PublicInputs,
) -> StarkQueryProof
where
    A: Air<BaseField = BaseElement>,
{
    let mut transcript = Transcript::new();
    // * Generate composition polynomials using Winterfell
    let (mut composition_poly, mut trace_poly) = get_cp_and_tp(air, trace, pub_inputs).unwrap();

    // * Generate Coset
    let (roots_of_unity, primitive_root) = crate::generate_vec_roots(1024, 1);

    // * Do Reed-Solomon on the trace and composition polynomials using some blowup factor
    let composition_poly_lde = composition_poly.evaluate_slice(roots_of_unity.as_slice());
    let trace_poly_lde = trace_poly.evaluate_slice(roots_of_unity.as_slice());

    // * Commit to both polynomials using a Merkle Tree
    let composition_poly_lde_merkle_tree = FriMerkleTree::build(composition_poly_lde.as_slice());
    let trace_poly_lde_merkle_tree = FriMerkleTree::build(&trace_poly_lde.as_slice());

    // * Do FRI on the composition polynomials
    let lde_fri_commitment = crate::fri::fri(&mut composition_poly, &roots_of_unity);

    // * Sample q_1, ..., q_m using Fiat-Shamir
    // let q_1 = transcript.challenge();
    // @@@@@@@@@@@@@@@@@@@@@@
    let q_1: usize = 2;

    // * For every q_i, do FRI decommitment
    let decommitment = fri_decommit_layers(&lde_fri_commitment, q_1);

    // * For every trace polynomial t_i, provide the evaluations on every q_i, q_i * g, q_i * g^2
    // TODO: Check the evaluation points corresponding to our program (it's not fibonacci).
    let evaluation_points = vec![
        U384FieldElement::from(q_1 as u64),
        U384FieldElement::from(q_1 as u64) * primitive_root.clone(),
        U384FieldElement::from(q_1 as u64) * primitive_root.pow(2_usize),
    ];

    let trace_lde_poly_evaluations = trace_poly.evaluate_slice(&evaluation_points);

    let mut merkle_paths = vec![];

    merkle_paths.push(
        trace_poly_lde_merkle_tree
            .get_proof_by_pos(q_1, &evaluation_points[0])
            .unwrap(),
    );
    merkle_paths.push(
        trace_poly_lde_merkle_tree
            .get_proof_by_pos(q_1 + 1, &evaluation_points[1])
            .unwrap(),
    );
    merkle_paths.push(
        trace_poly_lde_merkle_tree
            .get_proof_by_pos(q_1 + 2, &evaluation_points[2])
            .unwrap(),
    );

    let trace_root = trace_poly_lde_merkle_tree.root.borrow().clone().hash;
    let composition_poly_root = composition_poly_lde_merkle_tree.root.borrow().clone().hash;

    StarkQueryProof {
        trace_lde_poly_root: trace_root,
        trace_lde_poly_evaluations,
        trace_lde_poly_inclusion_proofs: merkle_paths,

        composition_poly_root,
    }
}

/*
- Trace merkle tree root
- Composition poly merkle tree root
- Fri Layers merkle tree roots
- Trace Polys merkle paths
- Composition poly merkle paths
- Fri layer polys evaluations (in the q_i's and -q_i's)
*/

/*
    - Check merkle paths for trace poly lde evaluations
*/
pub fn verify(proof: StarkQueryProof) -> bool {
    let trace_poly_root = proof.trace_lde_poly_root;

    // TODO: Use Fiat Shamir
    let q_1: u64 = 2;
    let (_roots_of_unity, primitive_root) = crate::generate_vec_roots(1024, 1);
    // TODO: This is hardcoded, it should not be.
    let evaluation_points = vec![
        U384FieldElement::from(q_1),
        U384FieldElement::from(q_1) * primitive_root.clone(),
        U384FieldElement::from(q_1) * primitive_root.pow(2_usize),
    ];

    for (merkle_proof, evaluation_point) in proof
        .trace_lde_poly_inclusion_proofs
        .iter()
        .zip(evaluation_points)
    {
        if merkle_proof.value != evaluation_point || !merkle_proof.verify(trace_poly_root.clone()) {
            return false;
        }
    }

    return true;
}

#[cfg(test)]
mod tests {
    use super::prove;
    use winterfell::{FieldExtension, ProofOptions};

    #[test]
    fn test_prove() {}
}
