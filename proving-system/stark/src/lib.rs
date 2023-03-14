pub mod constraints;
pub mod fri;
pub mod prover;
pub mod utils;
pub mod verifier;

use fri::fri_decommit::FriDecommitment;
use lambdaworks_math::field::element::FieldElement;
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

pub const ORDER_OF_ROOTS_OF_UNITY_TRACE: u64 = 32;
pub const ORDER_OF_ROOTS_OF_UNITY_FOR_LDE: u64 = 1024;

// We are using 3 as the offset as it's our field's generator.
pub const COSET_OFFSET: u64 = 3;

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

#[cfg(test)]
mod tests {
    use super::prover::{
        compute_boundary_quotient, compute_composition_polys, compute_deep_composition_poly, prove,
    };
    use super::utils::{compute_zerofier, fibonacci_trace};
    use super::verifier::verify;
    use super::*;
    use crate::fri::fri_decommit::fri_decommit_layers;
    use crate::{
        constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
        generate_primitive_root, FE,
    };
    use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
    use lambdaworks_math::polynomial::Polynomial;
    use lambdaworks_math::unsigned_integer::element::{UnsignedInteger, U384};
    use std::ops::Div;

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
