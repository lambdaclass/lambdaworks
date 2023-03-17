pub mod constraints;
pub mod fri;
pub mod prover;
pub mod utils;
pub mod verifier;

use fri::fri_decommit::FriDecommitment;
use lambdaworks_crypto::{fiat_shamir::transcript::Transcript, merkle_tree::proof::Proof};
use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::u256_two_adic_prime_field::U256MontgomeryTwoAdicPrimeField,
};

pub struct ProofConfig {
    pub count_queries: usize,
    pub blowup_factor: usize,
}

pub type PrimeField = U256MontgomeryTwoAdicPrimeField;
pub type FE = FieldElement<PrimeField>;

// DEFINITION OF CONSTANTS

pub const ORDER_OF_ROOTS_OF_UNITY_TRACE: usize = 32;
pub const ORDER_OF_ROOTS_OF_UNITY_FOR_LDE: usize = 1024;

// We are using 3 as the offset as it's our field's generator.
pub const COSET_OFFSET: u64 = 3;

#[derive(Debug, Clone)]
pub struct StarkQueryProof {
    pub trace_lde_poly_evaluations: Vec<FE>,
    /// Merkle paths for the trace polynomial evaluations
    pub trace_lde_poly_inclusion_proofs: Vec<Proof<PrimeField, DefaultHasher>>,
    pub composition_poly_lde_evaluations: Vec<FE>,
    pub fri_decommitment: FriDecommitment,
}

pub struct StarkProof {
    pub trace_lde_poly_root: FE,
    pub fri_layers_merkle_roots: Vec<FE>,
    pub query_list: Vec<StarkQueryProof>,
}

pub use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
pub use lambdaworks_crypto::merkle_tree::DefaultHasher;
pub type FriMerkleTree = MerkleTree<PrimeField, DefaultHasher>;

#[cfg(test)]
mod tests {
    use super::prover::{compute_composition_poly, prove};
    use super::utils::{
        compute_boundary_quotient, compute_zerofier, fibonacci_trace, generate_roots_of_unity_coset,
    };
    use super::verifier::verify;
    use super::*;
    use crate::{
        constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
        fri::{fri_decommit::fri_decommit_layers, Polynomial},
        FE,
    };
    use lambdaworks_math::{field::traits::IsTwoAdicField, unsigned_integer::element::U256};
    use std::ops::Div;

    #[test]
    fn test_prove() {
        let proof_config = super::ProofConfig {
            count_queries: 30,
            blowup_factor: 4,
        };

        let trace = fibonacci_trace([FE::new(U256::from("1")), FE::new(U256::from("1"))]);
        let result = prove(&trace, &proof_config);
        assert!(verify(&result));
    }

    #[test]
    fn should_fail_verify_if_evaluations_are_not_in_merkle_tree() {
        let trace = fibonacci_trace([FE::new(U256::from("1")), FE::new(U256::from("1"))]);
        let proof_config = super::ProofConfig {
            count_queries: 30,
            blowup_factor: 4,
        };
        let mut bad_proof = prove(&trace, &proof_config);

        bad_proof.query_list[0].composition_poly_lde_evaluations[0] = FE::new(U256::from("5"));
        bad_proof.query_list[0].trace_lde_poly_evaluations[0] = FE::new(U256::from("0"));
        bad_proof.query_list[0].trace_lde_poly_evaluations[1] = FE::new(U256::from("4"));
        bad_proof.query_list[0].trace_lde_poly_evaluations[2] = FE::new(U256::from("9"));

        assert!(!verify(&bad_proof));
    }

    #[test]
    fn should_fail_verify_if_point_returned_is_one_of_different_index() {
        let trace = fibonacci_trace([FE::new(U256::from("1")), FE::new(U256::from("1"))]);
        let proof = bad_index_prover(&trace);
        assert!(!verify(&proof));
    }

    #[test]
    fn should_fail_if_trace_is_only_one_number_for_fibonacci() {
        let trace = bad_trace([FE::new(U256::from("1")), FE::new(U256::from("1"))]);
        let proof_config = super::ProofConfig {
            count_queries: 30,
            blowup_factor: 4,
        };
        let bad_proof = prove(&trace, &proof_config);
        assert!(!verify(&bad_proof));
    }
    pub fn bad_index_prover(trace: &[FE]) -> StarkProof {
        let transcript = &mut Transcript::new();

        // * Generate Coset
        let trace_primitive_root = PrimeField::get_primitive_root_of_unity(
            ORDER_OF_ROOTS_OF_UNITY_TRACE.trailing_zeros() as u64,
        )
        .unwrap();
        let trace_roots_of_unity = generate_roots_of_unity_coset(1, &trace_primitive_root);

        let lde_primitive_root = PrimeField::get_primitive_root_of_unity(
            ORDER_OF_ROOTS_OF_UNITY_FOR_LDE.trailing_zeros() as u64,
        )
        .unwrap();
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
                    q_1 + (ORDER_OF_ROOTS_OF_UNITY_FOR_LDE / ORDER_OF_ROOTS_OF_UNITY_TRACE),
                )
                .unwrap(),
            trace_poly_lde_merkle_tree
                .get_proof_by_pos(
                    q_1 + (ORDER_OF_ROOTS_OF_UNITY_FOR_LDE / ORDER_OF_ROOTS_OF_UNITY_TRACE) * 2,
                )
                .unwrap(),
        ];

        let alpha_bc = FE::from(2);
        let alpha_t = FE::from(3);

        // These are evaluations over the composition polynomial
        let mut composition_poly =
            compute_composition_poly(&trace_poly, &trace_primitive_root, &[alpha_t, alpha_bc]);
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

        StarkProof {
            trace_lde_poly_root: trace_root,
            fri_layers_merkle_roots,
            query_list: vec![StarkQueryProof {
                trace_lde_poly_evaluations,
                trace_lde_poly_inclusion_proofs: merkle_paths,
                composition_poly_lde_evaluations: vec![composition_poly_lde_evaluation],
                fri_decommitment,
            }],
        }
    }

    pub fn bad_trace(initial_values: [FE; 2]) -> Vec<FE> {
        let mut ret: Vec<FE> = vec![];

        ret.push(initial_values[0].clone());
        ret.push(initial_values[1].clone());

        for i in 2..(ORDER_OF_ROOTS_OF_UNITY_TRACE) {
            ret.push(FE::new(U256::from_u64(i as u64)));
        }

        ret
    }

    #[test]
    fn test_wrong_boundary_constraints_does_not_verify() {
        // The first public input is set to 2, this should not verify because our constraints are hard-coded
        // to assert this first element is 1.
        let proof_config = super::ProofConfig {
            count_queries: 30,
            blowup_factor: 4,
        };

        let trace = fibonacci_trace([FE::new(U256::from("2")), FE::new(U256::from("3"))]);
        let result = prove(&trace, &proof_config);
        assert!(!verify(&result));
    }

    #[test]
    fn zerofier_is_the_correct_one() {
        let primitive_root = PrimeField::get_primitive_root_of_unity(3).unwrap();
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
        let a0 = BoundaryConstraint::new_simple(0, FE::new(U256::from("1")));
        let a1 = BoundaryConstraint::new_simple(1, FE::new(U256::from("1")));
        let result = BoundaryConstraint::new_simple(7, FE::new(U256::from("15")));

        let boundary_constraints = BoundaryConstraints::from_constraints(vec![a0, a1, result]);

        // Build trace polynomial
        let pub_inputs = [FE::new(U256::from("1")), FE::new(U256::from("1"))];
        let trace = test_utils::fibonacci_trace(pub_inputs, 8);
        let trace_primitive_root = PrimeField::get_primitive_root_of_unity(3).unwrap();
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

        let f = super::transcript_to_field(transcript);
        println!("{f:?}");
    }
}

// TODO: change this to use more bits
pub fn transcript_to_field(transcript: &mut Transcript) -> FE {
    let ret_value = transcript.challenge();
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
    FE::from(ret_value_u64)
}

pub fn transcript_to_usize(transcript: &mut Transcript) -> usize {
    const CANT_BYTES_USIZE: usize = (usize::BITS / 8) as usize;
    let ret_value = transcript.challenge();
    usize::from_be_bytes(
        ret_value
            .into_iter()
            .take(CANT_BYTES_USIZE)
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap(),
    )
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
