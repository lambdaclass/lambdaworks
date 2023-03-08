pub mod constraints;
pub mod fri;

use constraints::boundary::{BoundaryConstraint, BoundaryConstraints};
use std::ops::{Div, Mul};

use fri::fri_decommit::{fri_decommit_layers, FriDecommitment};
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_math::polynomial::{self, Polynomial};

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::{
    field::fields::montgomery_backed_prime_fields::{IsModulus, U384PrimeField},
    unsigned_integer::element::U384,
};

// DEFINITION OF THE USED FIELD
#[derive(Clone, Debug)]
pub struct Modulus;
impl IsModulus<U384> for Modulus {
    const MODULUS: U384 =
        // hex 17
        U384::from("800000000000011000000000000000000000000000000000000000000000001");
}

pub type PrimeField = U384PrimeField<Modulus>;
pub type FE = FieldElement<PrimeField>;

const MODULUS_MINUS_1: U384 = U384::sub(&Modulus::MODULUS, &U384::from("1")).0;

/// Subgroup generator to generate the roots of unity
const FIELD_SUBGROUP_GENERATOR: u64 = 3;

// DEFINITION OF CONSTANTS

const ORDER_OF_ROOTS_OF_UNITY_TRACE: u64 = 32;
const ORDER_OF_ROOTS_OF_UNITY_FOR_LDE: u64 = 1024;

// We are using 3 as the offset as it's our field's generator.
const COSET_OFFSET: u64 = 3;

// DEFINITION OF FUNCTIONS

pub fn generate_primitive_root(subgroup_size: u64) -> FE {
    let modulus_minus_1_field: FE = FE::new(MODULUS_MINUS_1);
    let subgroup_size: FE = subgroup_size.into();
    let generator_field: FE = FIELD_SUBGROUP_GENERATOR.into();
    let exp = (&modulus_minus_1_field) / &subgroup_size;
    generator_field.pow(exp.representative())
}

/// This functions takes a roots of unity and a coset factor
/// If coset_factor is 1, it's just expanding the roots of unity
/// w ^ 0, w ^ 1, w ^ 2 .... w ^ n-1
/// If coset_factor is h
/// h * w ^ 0, h * w ^ 1 .... h * w ^ n-1
pub fn generate_roots_of_unity_coset(coset_factor: u64, primitive_root: &FE) -> Vec<FE> {
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

pub fn prove(trace: &[FE]) -> StarkQueryProof {
    let transcript = &mut Transcript::new();

    // * Generate Coset
    let trace_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_TRACE);
    let trace_roots_of_unity = generate_roots_of_unity_coset(1, &trace_primitive_root);

    let lde_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_FOR_LDE);
    let lde_roots_of_unity_coset = generate_roots_of_unity_coset(COSET_OFFSET, &lde_primitive_root);

    let trace_poly = Polynomial::interpolate(&trace_roots_of_unity, trace);

    // * Do Reed-Solomon on the trace and composition polynomials using some blowup factor
    let trace_poly_lde = trace_poly.evaluate_slice(lde_roots_of_unity_coset.as_slice());

    // * Commit to both polynomials using a Merkle Tree
    let trace_poly_lde_merkle_tree = FriMerkleTree::build(trace_poly_lde.as_slice());

    // * Sample q_1, ..., q_m using Fiat-Shamir
    // let q_1 = transcript.challenge();
    // @@@@@@@@@@@@@@@@@@@@@@
    let q_1: usize = 4;
    let alpha_bc = FE::from(2);
    let alpha_t = FE::from(3);

    // START EVALUATION POINTS BLOCK
    // This depends on the AIR
    // It's related to the non FRI verification

    let offset = FE::from(COSET_OFFSET);

    // These are evaluations over the trace polynomial
    let evaluation_points = vec![
        &offset * lde_primitive_root.pow(q_1),
        &offset * lde_primitive_root.pow(q_1) * &trace_primitive_root,
        &offset * lde_primitive_root.pow(q_1) * (&trace_primitive_root * &trace_primitive_root),
    ];
    let trace_lde_poly_evaluations = trace_poly.evaluate_slice(&evaluation_points);
    let merkle_paths = vec![
        trace_poly_lde_merkle_tree.get_proof_by_pos(q_1).unwrap(),
        trace_poly_lde_merkle_tree
            .get_proof_by_pos(
                q_1 + (ORDER_OF_ROOTS_OF_UNITY_FOR_LDE / ORDER_OF_ROOTS_OF_UNITY_TRACE) as usize,
            )
            .unwrap(),
        trace_poly_lde_merkle_tree
            .get_proof_by_pos(
                q_1 + (ORDER_OF_ROOTS_OF_UNITY_FOR_LDE / ORDER_OF_ROOTS_OF_UNITY_TRACE) as usize
                    * 2,
            )
            .unwrap(),
    ];

    // These are evaluations over the composition polynomial
    let mut composition_poly =
        compute_composition_poly(trace_poly, &trace_primitive_root, &[alpha_t, alpha_bc]);
    let composition_poly_lde_evaluation = composition_poly.evaluate(&evaluation_points[0]);

    // This is needed to check  the element is in the root
    let trace_root = trace_poly_lde_merkle_tree.root;

    // END EVALUATION BLOCK

    // * Do FRI on the composition polynomials
    let lde_fri_commitment =
        crate::fri::fri(&mut composition_poly, &lde_roots_of_unity_coset, transcript);

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

    let fri_layers_merkle_roots: Vec<FE> = lde_fri_commitment
        .iter()
        .map(|fri_commitment| fri_commitment.merkle_tree.root.clone())
        .collect();

    StarkQueryProof {
        trace_lde_poly_root: trace_root,
        trace_lde_poly_evaluations,
        trace_lde_poly_inclusion_proofs: merkle_paths,
        composition_poly_lde_evaluations: vec![composition_poly_lde_evaluation],
        fri_layers_merkle_roots,
        fri_decommitment,
    }
}

fn compute_composition_poly(
    trace_poly: Polynomial<FE>,
    primitive_root: &FE,
    random_coeffs: &[FE; 2],
) -> Polynomial<FE> {
    let w_squared_x = Polynomial::new(&[FE::zero(), primitive_root * primitive_root]);
    let w_x = Polynomial::new(&[FE::zero(), primitive_root.clone()]);

    // Hard-coded fibonacci transition constraints
    let transition_poly = polynomial::compose(&trace_poly, &w_squared_x)
        - polynomial::compose(&trace_poly, &w_x)
        - trace_poly.clone();
    let zerofier = compute_zerofier(primitive_root, ORDER_OF_ROOTS_OF_UNITY_TRACE as usize);

    let transition_quotient = transition_poly.div(zerofier);

    // Hard-coded fibonacci boundary constraints
    let a0_constraint = BoundaryConstraint::new_simple(0, FE::from(1));
    let a1_constraint = BoundaryConstraint::new_simple(1, FE::from(1));
    let boundary_constraints =
        BoundaryConstraints::from_constraints(vec![a0_constraint, a1_constraint]);

    let boundary_quotient =
        compute_boundary_quotient(&boundary_constraints, 0, primitive_root, &trace_poly);

    transition_quotient.mul(random_coeffs[0].clone())
        + boundary_quotient.mul(random_coeffs[1].clone())
}

fn compute_zerofier(primitive_root: &FE, root_order: usize) -> Polynomial<FE> {
    let roots_of_unity_vanishing_polynomial =
        Polynomial::new_monomial(FE::one(), root_order) - Polynomial::new(&[FE::one()]);
    let exceptions_to_vanishing_polynomial =
        Polynomial::new(&[-primitive_root.pow(root_order - 2), FE::one()])
            * Polynomial::new(&[-primitive_root.pow(root_order - 1), FE::one()]);

    roots_of_unity_vanishing_polynomial.div(exceptions_to_vanishing_polynomial)
}

fn compute_boundary_quotient(
    constraints: &BoundaryConstraints<FE>,
    col: usize,
    primitive_root: &FE,
    trace_poly: &Polynomial<FE>,
) -> Polynomial<FE> {
    let domain = constraints.generate_roots_of_unity(primitive_root);
    let values = constraints.values(col);
    let zerofier = constraints.compute_zerofier(primitive_root);

    let poly = Polynomial::interpolate(&domain, &values);

    (trace_poly.clone() - poly).div(zerofier)
}

pub fn verify(proof: &StarkQueryProof) -> bool {
    let transcript = &mut Transcript::new();

    let trace_poly_root = &proof.trace_lde_poly_root;
    let trace_evaluations = &proof.trace_lde_poly_evaluations;

    // TODO: These could be multiple evaluations depending on how many q_i are sampled with Fiat Shamir
    let composition_polynomial_evaluation_from_prover = &proof.composition_poly_lde_evaluations[0];

    // TODO: Fiat-Shamir
    let q_1: usize = 4;
    let alpha_bc = FE::from(2);
    let alpha_t = FE::from(3);

    let trace_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_TRACE);
    let lde_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_FOR_LDE);
    let zerofier = compute_zerofier(
        &trace_primitive_root,
        ORDER_OF_ROOTS_OF_UNITY_TRACE as usize,
    );

    let offset = FE::from(COSET_OFFSET);
    let evaluation_point = &lde_primitive_root.pow(q_1) * &offset;

    // TODO: This is done to get the boundary zerofier - It should not be made like this
    let a0_constraint = BoundaryConstraint::new_simple(0, FE::from(1));
    let a1_constraint = BoundaryConstraint::new_simple(1, FE::from(1));
    let boundary_constraints =
        BoundaryConstraints::from_constraints(vec![a0_constraint, a1_constraint]);
    let boundary_zerofier = boundary_constraints.compute_zerofier(&trace_primitive_root);

    let composition_polynomial_evaluation_from_trace = ((&trace_evaluations[2]
        - &trace_evaluations[1]
        - &trace_evaluations[0])
        / zerofier.evaluate(&evaluation_point))
        * alpha_t
        + ((&trace_evaluations[0] - FE::from(1)) / boundary_zerofier.evaluate(&evaluation_point))
            * alpha_bc;

    if *composition_polynomial_evaluation_from_prover
        != composition_polynomial_evaluation_from_trace
    {
        return false;
    }

    let q_1: usize = 4;

    let trace_evaluation_point_indexes = vec![
        q_1,
        q_1 + (ORDER_OF_ROOTS_OF_UNITY_FOR_LDE / ORDER_OF_ROOTS_OF_UNITY_TRACE) as usize,
        q_1 + (ORDER_OF_ROOTS_OF_UNITY_FOR_LDE / ORDER_OF_ROOTS_OF_UNITY_TRACE) as usize * 2,
    ];

    for (merkle_proof, (index, value)) in proof
        .trace_lde_poly_inclusion_proofs
        .iter()
        .zip(trace_evaluation_point_indexes.iter().zip(trace_evaluations))
    {
        if !merkle_proof.verify(trace_poly_root, *index, value) {
            return false;
        }
    }

    fri_verify(
        &proof.fri_layers_merkle_roots,
        &proof.fri_decommitment,
        transcript,
    )
}

/// Performs FRI verification for some decommitment
pub fn fri_verify(
    fri_layers_merkle_roots: &[FE],
    fri_decommitment: &FriDecommitment,
    _transcript: &mut Transcript,
) -> bool {
    // For each fri layer merkle proof check:
    // That each merkle path verifies

    // Sample beta with fiat shamir
    // Compute v = [P_i(z_i) + P_i(-z_i)] / 2 + beta * [P_i(z_i) - P_i(-z_i)] / (2 * z_i)
    // Where P_i is the folded polynomial of the i-th fiat shamir round
    // z_i is obtained from the first z (that was derived through fiat-shamir) through a known calculation
    // The calculation is, given the index, index % length_of_evaluation_domain

    // Check that v = P_{i+1}(z_i)

    let decommitment_index: u64 = 4;

    let mut lde_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_FOR_LDE);
    let mut offset = FE::from(COSET_OFFSET);

    // For each (merkle_root, merkle_auth_path) / fold
    // With the auth path containining the element that the
    // path proves it's existance
    for (
        layer_number,
        (
            fri_layer_merkle_root,
            (
                (fri_layer_auth_path, fri_layer_auth_path_symmetric),
                (auth_path_evaluation, auth_path_evaluation_symmetric),
            ),
        ),
    ) in fri_layers_merkle_roots
        .iter()
        .zip(
            fri_decommitment
                .layer_merkle_paths
                .iter()
                .zip(fri_decommitment.layer_evaluations.iter()),
        )
        .enumerate()
        // Since we always derive the current layer from the previous layer
        // We start with the second one, skipping the first, so previous is layer is the first one
        .skip(1)
    {
        // This is the current layer's evaluation domain length. We need it to know what the decommitment index for the current
        // layer is, so we can check the merkle paths at the right index.
        let current_layer_domain_length = ORDER_OF_ROOTS_OF_UNITY_FOR_LDE as usize >> layer_number;

        let layer_evaluation_index: usize =
            decommitment_index as usize % current_layer_domain_length;
        if !fri_layer_auth_path.verify(
            fri_layer_merkle_root,
            layer_evaluation_index,
            auth_path_evaluation,
        ) {
            return false;
        }

        let layer_evaluation_index_symmetric: usize = (decommitment_index as usize
            + current_layer_domain_length)
            % current_layer_domain_length;

        if !fri_layer_auth_path_symmetric.verify(
            fri_layer_merkle_root,
            layer_evaluation_index_symmetric,
            auth_path_evaluation_symmetric,
        ) {
            return false;
        }

        // TODO: use Fiat Shamir
        let beta: u64 = 4;

        let (previous_auth_path_evaluation, previous_path_evaluation_symmetric) = fri_decommitment
            .layer_evaluations
            .get(layer_number - 1)
            // TODO: Check at the start of the FRI operation
            // if layer_merkle_paths has the right amount of elements
            .unwrap();

        // evaluation point = offset * w ^ i in the Stark literature
        let evaluation_point = &offset * lde_primitive_root.pow(decommitment_index);

        // v is the calculated element for the
        // co linearity check
        let two = &FE::new(U384::from("2"));
        let beta = FE::new(U384::from_u64(beta));
        let v = (previous_auth_path_evaluation + previous_path_evaluation_symmetric) / two
            + &beta * (previous_auth_path_evaluation - previous_path_evaluation_symmetric)
                / (two * evaluation_point);

        lde_primitive_root = lde_primitive_root.pow(2_usize);
        offset = offset.pow(2_usize);

        if v != *auth_path_evaluation {
            return false;
        }

        // On the last iteration, also check the provided last evaluation point.
        if layer_number == fri_layers_merkle_roots.len() - 1 {
            let last_evaluation_point = &offset * lde_primitive_root.pow(decommitment_index);

            let last_v = (auth_path_evaluation + auth_path_evaluation_symmetric) / two
                + &beta * (auth_path_evaluation - auth_path_evaluation_symmetric)
                    / (two * &last_evaluation_point);

            if last_v != fri_decommitment.last_layer_evaluation {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compute_zerofier,
        constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
        generate_primitive_root, verify, FE,
    };

    use super::prove;
    use lambdaworks_math::unsigned_integer::element::U384;

    #[test]
    fn test_prove() {
        let trace = fibonacci_trace([FE::new(U384::from("1")), FE::new(U384::from("1"))]);
        let result = prove(&trace);
        assert!(verify(&result));
    }

    #[test]
    fn should_fail_verify_if_evaluations_are_not_in_merkle_tree() {
        let trace = fibonacci_trace([FE::new(U384::from("1")), FE::new(U384::from("1"))]);
        let mut bad_proof = prove(&trace);

        bad_proof.composition_poly_lde_evaluations[0] = FE::new(U384::from("5"));
        bad_proof.trace_lde_poly_evaluations[0] = FE::new(U384::from("0"));
        bad_proof.trace_lde_poly_evaluations[1] = FE::new(U384::from("4"));
        bad_proof.trace_lde_poly_evaluations[2] = FE::new(U384::from("9"));

        assert!(!verify(&bad_proof));
    }

    #[test]
    fn should_fail_verify_if_point_returned_is_one_of_different_index() {
        let trace = fibonacci_trace([FE::new(U384::from("1")), FE::new(U384::from("1"))]);
        let proof = bad_index_prover(&trace);
        assert!(!verify(&proof));
    }

    #[test]
    fn should_fail_if_trace_is_only_one_number_for_fibonacci() {
        let trace = bad_trace([FE::new(U384::from("1")), FE::new(U384::from("1"))]);
        let bad_proof = prove(&trace);
        assert!(!verify(&bad_proof));
    }
    pub fn bad_index_prover(trace: &[FE]) -> StarkQueryProof {
        let transcript = &mut Transcript::new();

        // * Generate Coset
        let trace_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_TRACE);
        let trace_roots_of_unity = generate_roots_of_unity_coset(1, &trace_primitive_root);

        let lde_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_FOR_LDE);
        let lde_roots_of_unity = generate_roots_of_unity_coset(1, &lde_primitive_root);

        let trace_poly = Polynomial::interpolate(&trace_roots_of_unity, trace);

        // * Do Reed-Solomon on the trace and composition polynomials using some blowup factor
        let trace_poly_lde = trace_poly.evaluate_slice(lde_roots_of_unity.as_slice());

        // * Commit to both polynomials using a Merkle Tree
        let trace_poly_lde_merkle_tree = FriMerkleTree::build(trace_poly_lde.as_slice());

        // MALICIOUS MOVE: Change queried point
        let q_1: usize = 1;

        // START EVALUATION POINTS BLOCK
        // This depends on the AIR
        // It's related to the non FRI verification

        // These are evaluations over the trace polynomial
        let evaluation_points = vec![
            lde_primitive_root.pow(q_1),
            lde_primitive_root.pow(q_1) * &trace_primitive_root,
            lde_primitive_root.pow(q_1) * (&trace_primitive_root * &trace_primitive_root),
        ];

        let trace_lde_poly_evaluations = trace_poly.evaluate_slice(&evaluation_points);

        let merkle_paths = vec![
            trace_poly_lde_merkle_tree.get_proof_by_pos(q_1).unwrap(),
            trace_poly_lde_merkle_tree
                .get_proof_by_pos(
                    q_1 + (ORDER_OF_ROOTS_OF_UNITY_FOR_LDE / ORDER_OF_ROOTS_OF_UNITY_TRACE)
                        as usize,
                )
                .unwrap(),
            trace_poly_lde_merkle_tree
                .get_proof_by_pos(
                    q_1 + (ORDER_OF_ROOTS_OF_UNITY_FOR_LDE / ORDER_OF_ROOTS_OF_UNITY_TRACE)
                        as usize
                        * 2,
                )
                .unwrap(),
        ];

        let alpha_bc = FE::from(2);
        let alpha_t = FE::from(3);

        // These are evaluations over the composition polynomial
        let mut composition_poly =
            compute_composition_poly(trace_poly, &trace_primitive_root, &[alpha_t, alpha_bc]);
        let composition_poly_lde_evaluation = composition_poly.evaluate(&evaluation_points[0]);

        // This is needed to check  the element is in the root
        let trace_root = trace_poly_lde_merkle_tree.root;

        // END EVALUATION BLOCK

        // Enough lies, time to respect the randomness here

        let q_1 = 4;
        // * Do FRI on the composition polynomials
        let lde_fri_commitment =
            crate::fri::fri(&mut composition_poly, &lde_roots_of_unity, transcript);

        // * For every q_i, do FRI decommitment
        let fri_decommitment = fri_decommit_layers(&lde_fri_commitment, q_1);

        let fri_layers_merkle_roots: Vec<FE> = lde_fri_commitment
            .iter()
            .map(|fri_commitment| fri_commitment.merkle_tree.root.clone())
            .collect();

        StarkQueryProof {
            trace_lde_poly_root: trace_root,
            trace_lde_poly_evaluations,
            trace_lde_poly_inclusion_proofs: merkle_paths,
            composition_poly_lde_evaluations: vec![composition_poly_lde_evaluation],
            fri_layers_merkle_roots,
            fri_decommitment,
        }
    }

    pub fn bad_trace(initial_values: [FE; 2]) -> Vec<FE> {
        let mut ret: Vec<FE> = vec![];

        ret.push(initial_values[0].clone());
        ret.push(initial_values[1].clone());

        for i in 2..(ORDER_OF_ROOTS_OF_UNITY_TRACE) {
            ret.push(FE::new(U384::from_u64(i)));
        }

        ret
    }

    #[test]
    fn test_wrong_boundary_constraints_does_not_verify() {
        // The first public input is set to 2, this should not verify because our constraints are hard-coded
        // to assert this first element is 1.
        let trace = fibonacci_trace([FE::new(U384::from("2")), FE::new(U384::from("3"))]);
        let result = prove(&trace);
        assert!(!verify(&result));
    }

    #[test]
    fn zerofier_is_the_correct_one() {
        let primitive_root = generate_primitive_root(8);
        let zerofier = compute_zerofier(&primitive_root, 8);

        for i in 0_usize..6_usize {
            assert_eq!(zerofier.evaluate(&primitive_root.pow(i)), FE::zero());
        }

        assert_ne!(zerofier.evaluate(&primitive_root.pow(6_usize)), FE::zero());
        assert_ne!(zerofier.evaluate(&primitive_root.pow(7_usize)), FE::zero());
    }

    #[test]
    fn test_get_boundary_quotient() {
        // Build boundary constraints
        let a0 = BoundaryConstraint::new_simple(0, FE::new(U384::from("1")));
        let a1 = BoundaryConstraint::new_simple(1, FE::new(U384::from("1")));
        let result = BoundaryConstraint::new_simple(7, FE::new(U384::from("15")));

        let boundary_constraints = BoundaryConstraints::from_constraints(vec![a0, a1, result]);

        // Build trace polynomial
        let pub_inputs = [FE::new(U384::from("1")), FE::new(U384::from("1"))];
        let trace = test_utils::fibonacci_trace(pub_inputs, 8);
        let trace_primitive_root = generate_primitive_root(8);
        let trace_roots_of_unity = generate_roots_of_unity_coset(1, &trace_primitive_root);
        let trace_poly = Polynomial::interpolate(&trace_roots_of_unity, &trace);

        // Build boundary polynomial
        let domain = boundary_constraints.generate_roots_of_unity(&trace_primitive_root);
        let values = boundary_constraints.values(0);
        let boundary_poly = Polynomial::interpolate(&domain, &values);
        let zerofier = boundary_constraints.compute_zerofier(&trace_primitive_root);

        // Test get_boundary_quotient
        let boundary_quotient =
            compute_boundary_quotient(&boundary_constraints, 0, &trace_primitive_root, &trace_poly);

        assert_eq!(
            boundary_quotient,
            (trace_poly - boundary_poly).div(zerofier)
        );
    }

    #[test]
    fn test_transcript_to_field() {
        use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
        let transcript = &mut Transcript::new();

        let vec_bytes: Vec<u8> = vec![2, 3, 5, 7, 8];
        transcript.append(&vec_bytes);

        let ret_value = transcript.challenge();
        println!("{ret_value:?}");

        let ret_value_8: [u8; 8] = [
            ret_value[0],
            ret_value[1],
            ret_value[2],
            ret_value[3],
            ret_value[4],
            ret_value[5],
            ret_value[6],
            ret_value[7],
        ];
        let ret_value_u64 = u64::from_be_bytes(ret_value_8);

        use super::*;
        let f = FE::from(ret_value_u64);
        println!("{f:?}");
    }
}

#[cfg(test)]
mod test_utils {
    use super::*;

    pub(crate) fn fibonacci_trace(initial_values: [FE; 2], iters: usize) -> Vec<FE> {
        let mut ret: Vec<FE> = vec![];

        ret.push(initial_values[0].clone());
        ret.push(initial_values[1].clone());

        for i in 2..iters {
            ret.push(ret[i - 1].clone() + ret[i - 2].clone());
        }

        ret
    }
}
