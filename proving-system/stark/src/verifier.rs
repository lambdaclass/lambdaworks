use super::constraints::boundary::{BoundaryConstraint, BoundaryConstraints};
use super::fri::fri_decommit::FriDecommitment;
use super::utils::{
    compute_boundary_quotient_ood_evaluation, compute_transition_quotient_ood_evaluation,
};
use super::{
    generate_primitive_root, StarkQueryProof, COSET_OFFSET, FE, ORDER_OF_ROOTS_OF_UNITY_FOR_LDE,
    ORDER_OF_ROOTS_OF_UNITY_TRACE,
};
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::unsigned_integer::element::U384;

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
