pub mod air;
pub mod fri;

use std::primitive;

use air::polynomials::get_cp_and_tp;
use fri::fri_decommit::{fri_decommit_layers, FriDecommitment};
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_math::polynomial::{self, Polynomial};
use thiserror::__private::AsDynError;
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

// DEFINITION OF THE USED FIELD
#[derive(Clone, Debug)]
pub struct MontgomeryConfig;
impl IsMontgomeryConfiguration for MontgomeryConfig {
    const MODULUS: U384 =
        // hex 17
        U384::from("11");
}

pub type PrimeField = MontgomeryBackendPrimeField<MontgomeryConfig>;
pub type FE = FieldElement<PrimeField>;

const MODULUS_MINUS_1: U384 = U384::sub(
    &MontgomeryConfig::MODULUS,
    &U384::from("1"),
).0;

/// Subgroup generator to generate the roots of unity 
const FIELD_SUBGROUP_GENERATOR: u64 = 3;

// DEFINITION OF CONSTANTS

const ORDER_OF_ROOTS_OF_UNITY_TRACE: u64 = 4;
const ORDER_OF_ROOTS_OF_UNITY_FOR_LDE: u64 = 16;

// DEFINITION OF FUNCTIONS

pub fn generate_primitive_root(subgroup_size: u64) -> FE {
    let modulus_minus_1_field: FE = FE::new(MODULUS_MINUS_1);
    let subgroup_size: FE = subgroup_size.into();
    let generator_field: FE = FIELD_SUBGROUP_GENERATOR.into();
    let exp = (&modulus_minus_1_field) / &subgroup_size;
    generator_field.pow(*exp.value())
}

/// This functions takes a roots of unity and a coset factor
/// If coset_factor is 1, it's just expanding the roots of unity 
/// w ^ 0, w ^ 1, w ^ 2 .... w ^ n-1
/// If coset_factor is h
/// h * w ^ 0, h * w ^ 1 .... h * w ^ n-1
// doesn't need to return the primitive root w ^ 1
pub fn generate_roots_of_unity_coset(
    coset_factor: u64,
    primitive_root: &FE,
) -> Vec<FE> {

    let coset_factor: FE = coset_factor.into();

    let mut numbers = vec![coset_factor.clone()];
    let mut exp: u64 = 1;
    let mut next_root = primitive_root.pow(exp) * &coset_factor;
    while next_root != coset_factor {
        numbers.push(next_root);
        exp += 1;
        next_root = primitive_root.pow(exp) * &coset_factor;
    }
    numbers
}

#[derive(Debug, Clone)]
pub struct StarkQueryProof {
    pub trace_lde_poly_root: FE,
    pub trace_lde_poly_evaluations: Vec<FE>,
    /// Merkle paths for the trace polynomial evaluations
    pub trace_lde_poly_inclusion_proofs: Vec<Proof<PrimeField, DefaultHasher>>,
    pub composition_poly_lde_evaluations: Vec<FE>,
    pub fri_layers_merkle_roots: Vec<FE>,
    pub fri_decommitment: FriDecommitment,
}

pub type StarkProof = Vec<StarkQueryProof>;

pub use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
pub use lambdaworks_crypto::merkle_tree::DefaultHasher;
pub type FriMerkleTree = MerkleTree<PrimeField, DefaultHasher>;

pub fn fibonacci_trace(initial_values: [FE; 2]) -> Vec<FE> {
    let mut ret: Vec<FE> = vec![];

    ret.push(initial_values[0].clone());
    ret.push(initial_values[1].clone());

    for i in 2..(ORDER_OF_ROOTS_OF_UNITY_TRACE as usize) {
        ret.push(ret[i - 1].clone() + ret[i - 2].clone());
    }

    ret
}

pub fn prove(
    // air: A,
    // trace: TraceTable<A::BaseField>,
    // pub_inputs: A::PublicInputs,
    pub_inputs: [FE; 2],
) -> StarkQueryProof
// where
//     A: Air<BaseField = BaseElement>,
{
    let transcript = &mut Transcript::new();
    // * Generate composition polynomials using Winterfell
    // let (mut composition_poly, mut trace_poly) = get_cp_and_tp(air, trace, pub_inputs).unwrap();

    // * Generate Coset

    let trace_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_TRACE);
    let trace_roots_of_unity =
        generate_roots_of_unity_coset(1, &trace_primitive_root);

    let lde_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_FOR_LDE);
    let lde_roots_of_unity = generate_roots_of_unity_coset(1, &lde_primitive_root);

    let trace = fibonacci_trace(pub_inputs);

    let trace_poly = Polynomial::interpolate(&trace_roots_of_unity, &trace);

    let mut composition_poly = get_composition_poly(trace_poly.clone(), &trace_primitive_root);

    // * Do Reed-Solomon on the trace and composition polynomials using some blowup factor
    let trace_poly_lde = trace_poly.evaluate_slice(lde_roots_of_unity.as_slice());

    // * Commit to both polynomials using a Merkle Tree
    let trace_poly_lde_merkle_tree = FriMerkleTree::build(&trace_poly_lde.as_slice());

    // * Do FRI on the composition polynomials
    let lde_fri_commitment =
        crate::fri::fri(&mut composition_poly, &lde_roots_of_unity, transcript);

    // * Sample q_1, ..., q_m using Fiat-Shamir
    // let q_1 = transcript.challenge();
    // @@@@@@@@@@@@@@@@@@@@@@
    let q_1: usize = 4;

    // * For every q_i, do FRI decommitment
    let fri_decommitment = fri_decommit_layers(&lde_fri_commitment, q_1);

    /*
        IMPORTANT NOTE:
        When we commit to the trace polynomial, let's call it f, we commit to an LDE of it.
        On the other hand, the fibonacci constraint (and in general, any constraint) related to f applies
        only using non-LDE roots of unity.
        In this case, the constraint is f(w^2 x) - f(w x) - f(x), where w is a 2^n root of unity.
        But for the commitment we use g, a 2^{nb} root of unity (b is the blowup factor).
        When we sample a value x to evaluate the trace polynomial on, it has to be a 2^{nb} root of unity,
        so with fiat-shamir we sample a random index in that range.
        When we provide evaluations, we provide them for x*(w^2), x*w and x.
    */

    let evaluation_points = vec![
        lde_primitive_root.pow(q_1),
        lde_primitive_root.pow(q_1) * &trace_primitive_root,
        lde_primitive_root.pow(q_1) * (&trace_primitive_root * &trace_primitive_root),
    ];

    let trace_lde_poly_evaluations = trace_poly.evaluate_slice(&evaluation_points);
    let composition_poly_lde_evaluation = composition_poly.evaluate(&evaluation_points[0]);

    let mut merkle_paths = vec![];

    merkle_paths.push(
        trace_poly_lde_merkle_tree
            .get_proof_by_pos(q_1, trace_lde_poly_evaluations[0].clone())
            .unwrap(),
    );
    merkle_paths.push(
        trace_poly_lde_merkle_tree
            .get_proof_by_pos(
                q_1 + (ORDER_OF_ROOTS_OF_UNITY_FOR_LDE / ORDER_OF_ROOTS_OF_UNITY_TRACE) as usize,
                trace_lde_poly_evaluations[1].clone(),
            )
            .unwrap(),
    );
    merkle_paths.push(
        trace_poly_lde_merkle_tree
            .get_proof_by_pos(
                q_1 + (ORDER_OF_ROOTS_OF_UNITY_FOR_LDE / ORDER_OF_ROOTS_OF_UNITY_TRACE) as usize
                    * 2,
                trace_lde_poly_evaluations[2].clone(),
            )
            .unwrap(),
    );

    let trace_root = trace_poly_lde_merkle_tree.root.clone();

    let fri_layers_merkle_roots: Vec<FE> = lde_fri_commitment
        .iter()
        .map(|fri_commitment| fri_commitment.merkle_tree.root.clone())
        .collect();

    StarkQueryProof {
        trace_lde_poly_root: trace_root,
        trace_lde_poly_evaluations,
        trace_lde_poly_inclusion_proofs: merkle_paths,
        composition_poly_lde_evaluations: vec![composition_poly_lde_evaluation],
        fri_layers_merkle_roots: fri_layers_merkle_roots,
        fri_decommitment: fri_decommitment,
    }
}

fn get_composition_poly(
    trace_poly: Polynomial<FE>,
    root_of_unity: &FE,
) -> Polynomial<FE> {
    let w_squared_x = Polynomial::new(&vec![
        FE::zero(),
        root_of_unity * root_of_unity,
    ]);
    let w_x = Polynomial::new(&vec![FE::zero(), root_of_unity.clone()]);

    polynomial::compose(&trace_poly, &w_squared_x)
        - polynomial::compose(&trace_poly, &w_x)
        - trace_poly
}

pub fn verify(proof: StarkQueryProof) -> bool {
    let trace_poly_root = proof.trace_lde_poly_root;

    let mut lde_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_FOR_LDE);

    let evaluations = proof.trace_lde_poly_evaluations;

    // TODO: These could be multiple evaluations depending on how many q_i are sampled with Fiat Shamir
    let composition_poly_lde_evaluation = proof.composition_poly_lde_evaluations[0].clone();

    if composition_poly_lde_evaluation != &evaluations[2] - &evaluations[1] - &evaluations[0] {
        return false;
    }

    for merkle_proof in proof.trace_lde_poly_inclusion_proofs {
        if !merkle_proof.verify(trace_poly_root.clone()) {
            return false;
        }
    }

    // FRI VERIFYING BEGINS HERE
    let decommitment_index: u64 = 4;

    // For each (merkle_root, merkle_auth_path)
    // With the auth path containining the element that the
    // path proves it's existance
    for (
        layer_number,
        (fri_layer_merkle_root, (fri_layer_auth_path, fri_layer_auth_path_symmetric)),
    ) in proof
        .fri_layers_merkle_roots
        .iter()
        .zip(proof.fri_decommitment.layer_merkle_paths.iter())
        .enumerate()
        // Since we always derive the current layer from the previous layer
        // We start with the second one, skipping the first, so previous is layer is the first one
        .skip(1)
    {
        if !fri_layer_auth_path.verify(fri_layer_merkle_root.clone()) {
            return false;
        }

        if !fri_layer_auth_path_symmetric.verify(fri_layer_merkle_root.clone()) {
            return false;
        }

        // TODO: use Fiat Shamir
        let beta: u64 = 4;

        let (previous_auth_path, previous_auth_path_symmetric) =
            proof
                .fri_decommitment
                .layer_merkle_paths
                .get(layer_number - 1)
                // TODO: Check at the start of the FRI operation
                // if layer_merkle_paths has the right amount of elements
                .unwrap();

        // evaluation point = w ^ i in the Stark literature 
        let evaluation_point = lde_primitive_root.pow(decommitment_index);
    
        // v is the calculated element for the 
        // co linearity check
        let two = &FE::new(U384::from("2"));
        let beta = FE::new(U384::from_u64(beta));
        let v = 
            (&previous_auth_path.value + &previous_auth_path_symmetric.value) 
                / two
            + 
            beta * (&previous_auth_path.value - &previous_auth_path_symmetric.value)
                / (two * evaluation_point);

        lde_primitive_root = lde_primitive_root.pow(2_usize);

        if v != fri_layer_auth_path.value {
            return false;
        }
    }

    // For each fri layer merkle proof check:
    // That each merkle path verifies

    // Sample beta with fiat shamir
    // Compute v = [P_i(z_i) + P_i(-z_i)] / 2 + beta * [P_i(z_i) - P_i(-z_i)] / (2 * z_i)
    // Where P_i is the folded polynomial of the i-th fiat shamir round
    // z_i is obtained from the first z (that was derived through fiat-shamir) through a known calculation
    // The calculation is, given the index, index % length_of_evaluation_domain

    // Check that v = P_{i+1}(z_i)

    return true;
}

// TODOS after basic fibonacci works:
// - Add Fiat Shamir
// - Add Zerofiers
// - Check last evaluation point
// - Instead of returning a bool, make an error type encoding each possible failure in the verifying pipeline so failures give more info.
// - Unhardcode polynomials, use Winterfell AIR
// - Coset evaluation

#[cfg(test)]
mod tests {
    use crate::{verify, FE};

    use super::prove;
    use lambdaworks_math::{field::element::FieldElement, unsigned_integer::element::U384};
    use winterfell::{FieldExtension, ProofOptions};

    #[test]
    fn test_prove() {
        let result = prove([
            FE::new(U384::from("1")),
            FE::new(U384::from("1")),
        ]);
        assert!(verify(result));
    }
}
