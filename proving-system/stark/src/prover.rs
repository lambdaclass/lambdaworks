use super::constraints::boundary::{BoundaryConstraint, BoundaryConstraints};
use super::fri::fri_decommit::fri_decommit_layers;
use super::utils::{compute_boundary_quotient, compute_zerofier, generate_roots_of_unity_coset};
use super::{
    transcript_to_field, transcript_to_usize, FriMerkleTree, PrimeField, ProofConfig, StarkProof,
    StarkQueryProof, COSET_OFFSET, FE, ORDER_OF_ROOTS_OF_UNITY_FOR_LDE,
    ORDER_OF_ROOTS_OF_UNITY_TRACE,
};
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::{
    field::traits::IsTwoAdicField,
    polynomial::{self, Polynomial},
};
use std::ops::{Div, Mul};

// FIXME remove unwrap() calls and return errors
pub fn prove(trace: &[FE], proof_config: &ProofConfig) -> StarkProof {
    let transcript = &mut Transcript::new();
    let mut query_list = Vec::<StarkQueryProof>::new();

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
    let lde_roots_of_unity_coset = generate_roots_of_unity_coset(COSET_OFFSET, &lde_primitive_root);

    let trace_poly = Polynomial::interpolate(&trace_roots_of_unity, trace);

    // * Do Reed-Solomon on the trace and composition polynomials using some blowup factor
    let trace_poly_lde = trace_poly.evaluate_slice(lde_roots_of_unity_coset.as_slice());

    // * Commit to both polynomials using a Merkle Tree
    let trace_poly_lde_merkle_tree = FriMerkleTree::build(trace_poly_lde.as_slice());

    let alpha_bc = transcript_to_field(transcript);
    let alpha_t = transcript_to_field(transcript);

    // START EVALUATION POINTS BLOCK
    // This depends on the AIR
    // It's related to the non FRI verification

    let offset = FE::from(COSET_OFFSET);

    // This is needed to check  the element is in the root
    let trace_lde_poly_root = trace_poly_lde_merkle_tree.root.clone();

    // END EVALUATION BLOCK

    // These are evaluations over the composition polynomial
    let mut composition_poly =
        compute_composition_poly(&trace_poly, &trace_primitive_root, &[alpha_t, alpha_bc]);

    // * Do FRI on the composition polynomials
    let lde_fri_commitment =
        crate::fri::fri(&mut composition_poly, &lde_roots_of_unity_coset, transcript);

    let fri_layers_merkle_roots: Vec<FE> = lde_fri_commitment
        .iter()
        .map(|fri_commitment| fri_commitment.merkle_tree.root.clone())
        .collect();

    for _i in 0..proof_config.count_queries {
        // These are evaluations over the trace polynomial
        // TODO @@@ this should be refactored
        // * Sample q_1, ..., q_m using Fiat-Shamir
        let q_i: usize = transcript_to_usize(transcript) % ORDER_OF_ROOTS_OF_UNITY_FOR_LDE;
        transcript.append(&q_i.to_be_bytes());

        let evaluation_points = vec![
            &offset * lde_primitive_root.pow(q_i),
            &offset * lde_primitive_root.pow(q_i) * &trace_primitive_root,
            &offset * lde_primitive_root.pow(q_i) * (&trace_primitive_root * &trace_primitive_root),
        ];
        let trace_lde_poly_evaluations = trace_poly.evaluate_slice(&evaluation_points);
        let merkle_paths = vec![
            trace_poly_lde_merkle_tree.get_proof_by_pos(q_i).unwrap(),
            trace_poly_lde_merkle_tree
                .get_proof_by_pos(
                    (q_i + (ORDER_OF_ROOTS_OF_UNITY_FOR_LDE / ORDER_OF_ROOTS_OF_UNITY_TRACE))
                        % ORDER_OF_ROOTS_OF_UNITY_FOR_LDE,
                )
                .unwrap(),
            trace_poly_lde_merkle_tree
                .get_proof_by_pos(
                    (q_i + (ORDER_OF_ROOTS_OF_UNITY_FOR_LDE / ORDER_OF_ROOTS_OF_UNITY_TRACE) * 2)
                        % ORDER_OF_ROOTS_OF_UNITY_FOR_LDE,
                )
                .unwrap(),
        ];

        let composition_poly_lde_evaluation = composition_poly.evaluate(&evaluation_points[0]);

        // * For every q_i, do FRI decommitment
        let fri_decommitment = fri_decommit_layers(&lde_fri_commitment, q_i);

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

        query_list.push(StarkQueryProof {
            trace_lde_poly_evaluations,
            trace_lde_poly_inclusion_proofs: merkle_paths,
            composition_poly_lde_evaluations: vec![composition_poly_lde_evaluation],
            fri_decommitment,
        });
    }

    StarkProof {
        trace_lde_poly_root,
        fri_layers_merkle_roots,
        query_list,
    }
}

pub(crate) fn compute_composition_poly(
    trace_poly: &Polynomial<FE>,
    primitive_root: &FE,
    random_coeffs: &[FE; 2],
) -> Polynomial<FE> {
    let w_squared_x = Polynomial::new(&[FE::zero(), primitive_root * primitive_root]);
    let w_x = Polynomial::new(&[FE::zero(), primitive_root.clone()]);

    // Hard-coded fibonacci transition constraints
    let transition_poly = polynomial::compose(trace_poly, &w_squared_x)
        - polynomial::compose(trace_poly, &w_x)
        - trace_poly.clone();
    let zerofier = compute_zerofier(primitive_root, ORDER_OF_ROOTS_OF_UNITY_TRACE);

    let transition_quotient = transition_poly.div(zerofier);

    // Hard-coded fibonacci boundary constraints
    let a0_constraint = BoundaryConstraint::new_simple(0, FE::from(1));
    let a1_constraint = BoundaryConstraint::new_simple(1, FE::from(1));
    let boundary_constraints =
        BoundaryConstraints::from_constraints(vec![a0_constraint, a1_constraint]);

    let boundary_quotient =
        compute_boundary_quotient(&boundary_constraints, 0, primitive_root, trace_poly);

    transition_quotient.mul(random_coeffs[0].clone())
        + boundary_quotient.mul(random_coeffs[1].clone())
}
