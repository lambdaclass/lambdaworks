pub mod air;
pub mod fri;

use air::constraints::boundary::{BoundaryConstraint, BoundaryConstraints};
use air::constraints::evaluator::ConstraintEvaluator;
use air::AIR;
use std::ops::{Div, Mul};

use fri::fri_decommit::{fri_decommit_layers, FriDecommitment};
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::polynomial::{self, Polynomial};

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::{
    field::fields::montgomery_backed_prime_fields::{IsMontgomeryConfiguration, U384PrimeField},
    unsigned_integer::element::U384,
};

// DEFINITION OF THE USED FIELD
#[derive(Clone, Debug)]
pub struct MontgomeryConfig;
impl IsMontgomeryConfiguration<6> for MontgomeryConfig {
    const MODULUS: U384 =
        // hex 17
        U384::from("800000000000011000000000000000000000000000000000000000000000001");
}

pub type PrimeField = U384PrimeField<MontgomeryConfig>;
pub type FE = FieldElement<PrimeField>;

const MODULUS_MINUS_1: U384 = U384::sub(&MontgomeryConfig::MODULUS, &U384::from("1")).0;

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
    pub trace_ood_evaluations: Vec<FE>,
    pub composition_poly_evaluations: Vec<FE>,
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

pub fn prove<F: IsField, A: AIR<F>>(trace: &[FE], air: A) -> StarkQueryProof {
    let transcript = &mut Transcript::new();

    // * Generate Coset
    let trace_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_TRACE);
    let trace_roots_of_unity = generate_roots_of_unity_coset(1, &trace_primitive_root);

    let lde_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_FOR_LDE);
    let lde_roots_of_unity_coset = generate_roots_of_unity_coset(COSET_OFFSET, &lde_primitive_root);

    let trace_poly = Polynomial::interpolate(&trace_roots_of_unity, trace);
    let lde_trace = trace_poly.evaluate_slice(&lde_roots_of_unity_coset);

    // // In the literature, these are H_1 and H_2, which satisfy
    // // H(X) = H_1(X^2) + X * H_2(X^2)
    // let (composition_poly_even, composition_poly_odd) =
    //     compute_composition_polys(&trace_poly, &trace_primitive_root);

    // TODO: Fiat-Shamir
    // z is the Out of domain evaluation point used in Deep FRI. It needs to be a point outside
    // of both the roots of unity and its corresponding coset used for the lde commitment.
    let z = FE::from(3);
    let z_squared = &z * &z;

    // **********

    // Get coefficients from Fiat-Shamir

    // Create evaluation table
    let evaluator = ConstraintEvaluator::new(&air, trace_poly, trace_primitive_root);
    let constraint_evaluations = evaluator.evaluate(&lde_trace, &lde_roots_of_unity_coset);

    // Get composition poly
    let composition_poly = constraint_evaluations.into_poly();

    // **********

    // Evaluate H_1 and H_2 in z^2.
    let composition_poly_evaluations = vec![
        composition_poly_even.evaluate(&z_squared),
        composition_poly_odd.evaluate(&z_squared),
    ];

    // The points z, (w * z), and (w^2 * z) needed by the verifier for the evaluation
    // consistency check.
    let trace_evaluation_points = vec![
        z.clone(),
        z.clone() * &trace_primitive_root,
        z.clone() * (&trace_primitive_root * &trace_primitive_root),
    ];

    let trace_ood_evaluations = trace_poly.evaluate_slice(&trace_evaluation_points);

    // END EVALUATION BLOCK

    // Compute DEEP composition polynomial so we can commit to it using FRI.
    let mut deep_composition_poly = compute_deep_composition_poly(
        &trace_poly,
        &composition_poly_even,
        &composition_poly_odd,
        &z,
        &trace_primitive_root,
    );

    // * Do FRI on the composition polynomials
    let lde_fri_commitment = crate::fri::fri(
        &mut deep_composition_poly,
        &lde_roots_of_unity_coset,
        transcript,
    );

    // TODO: Fiat-Shamir
    let fri_decommitment_index: usize = 4;

    // * For every q_i, do FRI decommitment
    let fri_decommitment = fri_decommit_layers(&lde_fri_commitment, fri_decommitment_index);

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
        trace_ood_evaluations,
        composition_poly_evaluations,
        fri_layers_merkle_roots,
        fri_decommitment,
    }
}

/// Given the trace polynomial and a primitive root, returns the deep FRI composition polynomial
/// `H`, split into its even and odd parts `H_1` and `H_2`. In the fibonacci example, `H` is the following
/// `H(X)` = `C_1(X) (alpha_1 * X^(D - D_1) + beta_1) + C_2(X) (alpha_2 X^(D - D_2) + beta_2)`, where `C_1`
/// is the boundary constraint polynomial and `C_2` is the transition constraint polynomial and the alphas
/// and betas are provided by the verifier (a.k.a. sampled using Fiat-Shamir).
fn compute_composition_polys(
    trace_poly: &Polynomial<FE>,
    primitive_root: &FE,
) -> (Polynomial<FE>, Polynomial<FE>) {
    let transition_quotient = compute_transition_quotient(primitive_root, trace_poly);

    // Hard-coded fibonacci boundary constraints
    let a0_constraint = BoundaryConstraint::new_simple(0, FE::from(1));
    let a1_constraint = BoundaryConstraint::new_simple(1, FE::from(1));
    let boundary_constraints =
        BoundaryConstraints::from_constraints(vec![a0_constraint, a1_constraint]);

    let boundary_quotient =
        compute_boundary_quotient(&boundary_constraints, 0, primitive_root, trace_poly);

    // TODO: Fiat-Shamir
    let alpha_1 = FE::one();
    let alpha_2 = FE::one();
    let beta_1 = FE::one();
    let beta_2 = FE::one();

    let maximum_degree = ORDER_OF_ROOTS_OF_UNITY_TRACE as usize;

    let d_1 = boundary_quotient.degree();
    let d_2 = transition_quotient.degree();

    let constraint_composition_poly = boundary_quotient.mul(
        Polynomial::new_monomial(alpha_1, maximum_degree - d_1)
            + Polynomial::new_monomial(beta_1, 0),
    ) + transition_quotient.mul(
        Polynomial::new_monomial(alpha_2, maximum_degree - d_2)
            + Polynomial::new_monomial(beta_2, 0),
    );

    constraint_composition_poly.even_odd_decomposition()
}

/// Returns the transition quotient polynomial that encodes what valid computation means.
/// In the fibonacci example, this is the polynomial `t(w^2 * X) - t(w * X) - t(X)` divided by
/// its zerofier (we call it quotient because of the division by the zerofier).
fn compute_transition_quotient(primitive_root: &FE, trace_poly: &Polynomial<FE>) -> Polynomial<FE> {
    let w_squared_x = Polynomial::new(&[FE::zero(), primitive_root * primitive_root]);
    let w_x = Polynomial::new(&[FE::zero(), primitive_root.clone()]);

    // Hard-coded fibonacci transition constraints
    let transition_poly = polynomial::compose(trace_poly, &w_squared_x)
        - polynomial::compose(trace_poly, &w_x)
        - trace_poly.clone();
    let zerofier = compute_zerofier(primitive_root, ORDER_OF_ROOTS_OF_UNITY_TRACE as usize);

    transition_poly.div(zerofier)
}

/// Returns the evaluation of the transition quotient polynomial on the provided ood evaluation point
/// required by DEEP FRI. This function is used by the verifier to check consistency between the trace
/// and the composition polynomial.
fn compute_transition_quotient_ood_evaluation(
    primitive_root: &FE,
    trace_poly_ood_evaluations: &[FE],
    ood_evaluation_point: &FE,
) -> FE {
    let zerofier = compute_zerofier(primitive_root, ORDER_OF_ROOTS_OF_UNITY_TRACE as usize);

    (&trace_poly_ood_evaluations[2]
        - &trace_poly_ood_evaluations[1]
        - &trace_poly_ood_evaluations[0])
        / zerofier.evaluate(ood_evaluation_point)
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

/// Returns the evaluation of the boundary constraint quotient polynomial on the provided ood evaluation point
/// required by DEEP FRI. This function is used by the verifier to check consistency between the trace
/// and the composition polynomial.
fn compute_boundary_quotient_ood_evaluation(
    constraints: &BoundaryConstraints<FE>,
    col: usize,
    primitive_root: &FE,
    trace_poly_ood_evaluation: &FE,
    ood_evaluation_point: &FE,
) -> FE {
    let domain = constraints.generate_roots_of_unity(primitive_root);
    let values = constraints.values(col);
    let zerofier = constraints.compute_zerofier(primitive_root);

    let poly = Polynomial::interpolate(&domain, &values);

    (trace_poly_ood_evaluation - poly.evaluate(ood_evaluation_point))
        / zerofier.evaluate(ood_evaluation_point)
}

/// Returns the DEEP composition polynomial that the prover then commits to using
/// FRI. This polynomial is a linear combination of the trace polynomial and the
/// composition polynomial, with coefficients sampled by the verifier (i.e. using Fiat-Shamir).
fn compute_deep_composition_poly(
    trace_poly: &Polynomial<FE>,
    even_composition_poly: &Polynomial<FE>,
    odd_composition_poly: &Polynomial<FE>,
    ood_evaluation_point: &FE,
    primitive_root: &FE,
) -> Polynomial<FE> {
    // TODO: Fiat-Shamir
    let gamma_1 = FE::one();
    let gamma_2 = FE::one();
    let gamma_3 = FE::one();
    let gamma_4 = FE::one();

    let first_term = (trace_poly.clone()
        - Polynomial::new_monomial(trace_poly.evaluate(ood_evaluation_point), 0))
        / (Polynomial::new_monomial(FE::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point.clone(), 0));
    let second_term = (trace_poly.clone()
        - Polynomial::new_monomial(
            trace_poly.evaluate(&(ood_evaluation_point * primitive_root)),
            0,
        ))
        / (Polynomial::new_monomial(FE::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point * primitive_root, 0));

    // Evaluate in X^2
    let even_composition_poly = polynomial::compose(
        even_composition_poly,
        &Polynomial::new_monomial(FE::one(), 2),
    );
    let odd_composition_poly = polynomial::compose(
        odd_composition_poly,
        &Polynomial::new_monomial(FE::one(), 2),
    );

    let third_term = (even_composition_poly.clone()
        - Polynomial::new_monomial(
            even_composition_poly.evaluate(&ood_evaluation_point.clone()),
            0,
        ))
        / (Polynomial::new_monomial(FE::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point * ood_evaluation_point, 0));
    let fourth_term = (odd_composition_poly.clone()
        - Polynomial::new_monomial(odd_composition_poly.evaluate(ood_evaluation_point), 0))
        / (Polynomial::new_monomial(FE::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point * ood_evaluation_point, 0));

    first_term * gamma_1 + second_term * gamma_2 + third_term * gamma_3 + fourth_term * gamma_4
}

pub fn verify(proof: &StarkQueryProof) -> bool {
    let transcript = &mut Transcript::new();

    // BEGIN TRACE <-> Composition poly consistency evaluation check

    let trace_poly_ood_evaluations = &proof.trace_ood_evaluations;
    let composition_poly_evaluations = &proof.composition_poly_evaluations;

    let trace_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_TRACE);

    // Instantiate boundary constraints
    // Compute the ood evaluation of the boundary constraints polynomial given the trace ood evaluation
    // This is C_1(z)
    let a0_constraint = BoundaryConstraint::new_simple(0, FE::from(1));
    let a1_constraint = BoundaryConstraint::new_simple(1, FE::from(1));
    let boundary_constraints =
        BoundaryConstraints::from_constraints(vec![a0_constraint, a1_constraint]);

    // TODO: Fiat-Shamir
    let z = FE::from(3);

    // C_1(z)
    let boundary_quotient_ood_evaluation = compute_boundary_quotient_ood_evaluation(
        &boundary_constraints,
        0,
        &trace_primitive_root,
        &trace_poly_ood_evaluations[0],
        &z,
    );

    // C_2(z)
    let transition_poly_ood_evaluation = compute_transition_quotient_ood_evaluation(
        &trace_primitive_root,
        trace_poly_ood_evaluations,
        &z,
    );

    let maximum_degree = ORDER_OF_ROOTS_OF_UNITY_TRACE as usize;

    let d_1 = ((ORDER_OF_ROOTS_OF_UNITY_TRACE - 1) - 2) as usize;
    // This is information that should come from the trace, we are hardcoding it in this case though.
    let d_2: usize = 1;

    // TODO: Fiat-Shamir
    let alpha_1 = FE::one();
    let alpha_2 = FE::one();
    let beta_1 = FE::one();
    let beta_2 = FE::one();

    let constraint_composition_poly_evaluation = boundary_quotient_ood_evaluation
        * (alpha_1 * z.pow(maximum_degree - d_1) + beta_1)
        + transition_poly_ood_evaluation * (alpha_2 * z.pow(maximum_degree - d_2) + beta_2);

    let constraint_composition_poly_claimed_evaluation =
        &composition_poly_evaluations[0] + &z * &composition_poly_evaluations[1];

    if constraint_composition_poly_claimed_evaluation != constraint_composition_poly_evaluation {
        return false;
    }

    // END TRACE <-> Composition poly consistency evaluation check

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

    // TODO: Fiat-Shamir
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
    use lambdaworks_math::unsigned_integer::element::{UnsignedInteger, U384};

    #[test]
    fn test_prove() {
        let trace = fibonacci_trace([FE::new(U384::from("1")), FE::new(U384::from("1"))]);
        let result = prove(&trace);
        assert!(verify(&result));
    }

    #[test]
    fn should_fail_if_ood_evaluations_pass_but_correspond_to_a_different_sampled_value() {
        let trace = fibonacci_trace([FE::new(U384::from("1")), FE::new(U384::from("1"))]);
        let mut bad_proof = prove(&trace);

        // These evaluations correspond to setting the ood evaluation point (a.k.a `z`) to 4.

        let first_trace_ood_evaluation_value = UnsignedInteger {
            limbs: [
                0,
                0,
                387583242588599888,
                2716095501519382283,
                2995400032997838439,
                15837643492768946428,
            ],
        };
        let first_trace_ood_evaluation = FE::new(first_trace_ood_evaluation_value);

        let second_trace_ood_evaluation_value = UnsignedInteger {
            limbs: [
                0,
                0,
                344316390940107663,
                838782849685282584,
                1825672079522363921,
                5527532739549145309,
            ],
        };
        let second_trace_ood_evaluation = FE::new(second_trace_ood_evaluation_value);

        let third_trace_ood_evaluation_value = UnsignedInteger {
            limbs: [
                0,
                0,
                323190822739600133,
                6686103559212955167,
                476839516384840355,
                12374770482546122681,
            ],
        };
        let third_trace_ood_evaluation = FE::new(third_trace_ood_evaluation_value);

        let trace_ood_evaluations = vec![
            first_trace_ood_evaluation,
            second_trace_ood_evaluation,
            third_trace_ood_evaluation,
        ];

        bad_proof.trace_ood_evaluations = trace_ood_evaluations;

        let first_composition_poly_evaluation_value = UnsignedInteger {
            limbs: [
                0,
                0,
                367599086662620033,
                10438144981158232368,
                10895062570311699404,
                8480136268066526442,
            ],
        };
        let first_composition_poly_evaluation = FE::new(first_composition_poly_evaluation_value);

        let second_composition_poly_evaluation_value = UnsignedInteger {
            limbs: [
                0,
                0,
                85561094729361526,
                10157379032212998235,
                12887891473753049008,
                14545924983273114449,
            ],
        };
        let second_composition_poly_evaluation = FE::new(second_composition_poly_evaluation_value);

        let composition_poly_evaluations = vec![
            first_composition_poly_evaluation,
            second_composition_poly_evaluation,
        ];

        bad_proof.composition_poly_evaluations = composition_poly_evaluations;

        assert!(!verify(&bad_proof));
    }

    #[test]
    fn should_fail_if_composition_poly_ood_evaluations_are_wrong() {
        let trace = fibonacci_trace([FE::new(U384::from("1")), FE::new(U384::from("1"))]);
        let mut bad_proof = prove(&trace);

        bad_proof.composition_poly_evaluations[0] = FE::new(U384::from("1238913"));
        bad_proof.composition_poly_evaluations[1] = FE::new(U384::from("129312"));

        assert!(!verify(&bad_proof));
    }

    #[test]
    fn should_not_verify_if_the_fri_decommitment_index_is_the_wrong_one() {
        let trace = fibonacci_trace([FE::new(U384::from("1")), FE::new(U384::from("1"))]);
        let proof = bad_index_prover(&trace);
        assert!(!verify(&proof));
    }

    #[test]
    fn should_fail_if_the_ood_evaluation_point_is_the_wrong_one() {
        let trace = fibonacci_trace([FE::new(U384::from("1")), FE::new(U384::from("1"))]);
        let proof = bad_ood_evaluation_point_prover(&trace);
        assert!(!verify(&proof));
    }

    #[test]
    fn should_fail_if_trace_is_only_one_number_for_fibonacci() {
        let trace = bad_trace([FE::new(U384::from("1")), FE::new(U384::from("1"))]);
        let bad_proof = prove(&trace);
        assert!(!verify(&bad_proof));
    }

    pub fn bad_ood_evaluation_point_prover(trace: &[FE]) -> StarkQueryProof {
        let transcript = &mut Transcript::new();

        // * Generate Coset
        let trace_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_TRACE);
        let trace_roots_of_unity = generate_roots_of_unity_coset(1, &trace_primitive_root);

        let lde_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_FOR_LDE);
        let lde_roots_of_unity_coset =
            generate_roots_of_unity_coset(COSET_OFFSET, &lde_primitive_root);

        let trace_poly = Polynomial::interpolate(&trace_roots_of_unity, trace);

        // In the literature, these are H_1 and H_2, which satisfy
        // H(X) = H_1(X^2) + X * H_2(X^2)
        let (composition_poly_even, composition_poly_odd) =
            compute_composition_polys(&trace_poly, &trace_primitive_root);

        // MALICIOUS MOVE: wrong ood evaluation point
        let z = FE::from(4);
        let z_squared = &z * &z;

        // Evaluate H_1 and H_2 in z^2.
        let composition_poly_evaluations = vec![
            composition_poly_even.evaluate(&z_squared),
            composition_poly_odd.evaluate(&z_squared),
        ];

        // The points z, (w * z), and (w^2 * z) needed by the verifier for the evaluation
        // consistency check.
        let trace_evaluation_points = vec![
            z.clone(),
            z.clone() * &trace_primitive_root,
            z.clone() * (&trace_primitive_root * &trace_primitive_root),
        ];

        let trace_ood_evaluations = trace_poly.evaluate_slice(&trace_evaluation_points);

        // END EVALUATION BLOCK

        // Compute DEEP composition polynomial so we can commit to it using FRI.
        let mut deep_composition_poly = compute_deep_composition_poly(
            &trace_poly,
            &composition_poly_even,
            &composition_poly_odd,
            &z,
            &trace_primitive_root,
        );

        // * Do FRI on the composition polynomials
        let lde_fri_commitment = crate::fri::fri(
            &mut deep_composition_poly,
            &lde_roots_of_unity_coset,
            transcript,
        );

        // TODO: Fiat-Shamir
        let fri_decommitment_index: usize = 4;

        // * For every q_i, do FRI decommitment
        let fri_decommitment = fri_decommit_layers(&lde_fri_commitment, fri_decommitment_index);

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
            trace_ood_evaluations,
            composition_poly_evaluations,
            fri_layers_merkle_roots,
            fri_decommitment,
        }
    }

    pub fn bad_index_prover(trace: &[FE]) -> StarkQueryProof {
        let transcript = &mut Transcript::new();

        // * Generate Coset
        let trace_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_TRACE);
        let trace_roots_of_unity = generate_roots_of_unity_coset(1, &trace_primitive_root);

        let lde_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_FOR_LDE);
        let lde_roots_of_unity_coset =
            generate_roots_of_unity_coset(COSET_OFFSET, &lde_primitive_root);

        let trace_poly = Polynomial::interpolate(&trace_roots_of_unity, trace);

        // In the literature, these are H_1 and H_2, which satisfy
        // H(X) = H_1(X^2) + X * H_2(X^2)
        let (composition_poly_even, composition_poly_odd) =
            compute_composition_polys(&trace_poly, &trace_primitive_root);

        // TODO: Fiat-Shamir
        // z is the Out of domain evaluation point used in Deep FRI. It needs to be a point outside
        // of both the roots of unity and its corresponding coset used for the lde commitment.
        let z = FE::from(3);
        let z_squared = &z * &z;

        // Evaluate H_1 and H_2 in z^2.
        let composition_poly_evaluations = vec![
            composition_poly_even.evaluate(&z_squared),
            composition_poly_odd.evaluate(&z_squared),
        ];

        // The points z, (w * z), and (w^2 * z) needed by the verifier for the evaluation
        // consistency check.
        let trace_evaluation_points = vec![
            z.clone(),
            z.clone() * &trace_primitive_root,
            z.clone() * (&trace_primitive_root * &trace_primitive_root),
        ];

        let trace_ood_evaluations = trace_poly.evaluate_slice(&trace_evaluation_points);

        // END EVALUATION BLOCK

        // Compute DEEP composition polynomial so we can commit to it using FRI.
        let mut deep_composition_poly = compute_deep_composition_poly(
            &trace_poly,
            &composition_poly_even,
            &composition_poly_odd,
            &z,
            &trace_primitive_root,
        );

        // * Do FRI on the composition polynomials
        let lde_fri_commitment = crate::fri::fri(
            &mut deep_composition_poly,
            &lde_roots_of_unity_coset,
            transcript,
        );

        // MALICIOUS MOVE: wrong index
        let fri_decommitment_index: usize = 1;

        // * For every q_i, do FRI decommitment
        let fri_decommitment = fri_decommit_layers(&lde_fri_commitment, fri_decommitment_index);

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
            trace_ood_evaluations,
            composition_poly_evaluations,
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
