use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsField, IsTwoAdicField},
    },
    polynomial::{self, Polynomial},
    traits::ByteConversion,
};

use super::{
    air::{constraints::evaluator::ConstraintEvaluator, frame::Frame, trace::TraceTable, AIR},
    fri::{fri, fri_decommit::fri_decommit_layers},
    StarkQueryProof, COSET_OFFSET,
};

// FIXME remove unwrap() calls and return errors
pub fn prove<F: IsField + IsTwoAdicField, A: AIR + AIR<Field = F>>(
    trace: &[FieldElement<F>],
    air: &A,
) -> StarkQueryProof<F>
where
    FieldElement<F>: ByteConversion,
{
    let transcript = &mut Transcript::new();
    // let mut query_list = Vec::<StarkQueryProof>::new();

    let root_order = air.context().trace_length.trailing_zeros();
    // * Generate Coset
    let trace_primitive_root = F::get_primitive_root_of_unity(root_order as u64).unwrap();

    let trace_roots_of_unity = F::get_powers_of_primitive_root_coset(
        root_order as u64,
        air.context().trace_length,
        &FieldElement::<F>::one(),
    )
    .unwrap();

    let lde_root_order =
        (air.context().trace_length * air.options().blowup_factor as usize).trailing_zeros();
    let lde_roots_of_unity_coset = F::get_powers_of_primitive_root_coset(
        lde_root_order as u64,
        air.context().trace_length * air.options().blowup_factor as usize,
        &FieldElement::<F>::from(COSET_OFFSET),
    )
    .unwrap();

    let trace_poly = Polynomial::interpolate(&trace_roots_of_unity, trace);
    let lde_trace = trace_poly.evaluate_slice(&lde_roots_of_unity_coset);

    // TODO: Fiat-Shamir
    // z is the Out of domain evaluation point used in Deep FRI. It needs to be a point outside
    // of both the roots of unity and its corresponding coset used for the lde commitment.
    let z = FieldElement::from(2);
    let z_squared = &z * &z;

    let lde_trace = TraceTable::new(lde_trace, 1);

    // Create evaluation table
    let evaluator = ConstraintEvaluator::new(air, &trace_poly, &trace_primitive_root);

    // TODO: Fiat-Shamir
    let alpha = FieldElement::one();
    let beta = FieldElement::one();

    let alpha_and_beta_transition_coefficients = vec![(alpha.clone(), beta.clone())];
    let constraint_evaluations = evaluator.evaluate(
        &lde_trace,
        &lde_roots_of_unity_coset,
        &alpha_and_beta_transition_coefficients,
        (&alpha, &beta),
    );

    // Get composition poly
    let composition_poly =
        constraint_evaluations.compute_composition_poly(&lde_roots_of_unity_coset);

    let (composition_poly_even, composition_poly_odd) = composition_poly.even_odd_decomposition();
    // Evaluate H_1 and H_2 in z^2.
    let composition_poly_evaluations = vec![
        composition_poly_even.evaluate(&z_squared),
        composition_poly_odd.evaluate(&z_squared),
    ];

    let trace_ood_frame_evaluations = Frame::<F>::construct_ood_frame(
        &[trace_poly.clone()],
        &z,
        &air.context().transition_offsets,
        &trace_primitive_root,
    );

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
    let lde_fri_commitment = fri(
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

    let fri_layers_merkle_roots: Vec<FieldElement<F>> = lde_fri_commitment
        .iter()
        .map(|fri_commitment| fri_commitment.merkle_tree.root.clone())
        .collect();

    let ret = StarkQueryProof {
        trace_ood_frame_evaluations,
        composition_poly_evaluations,
        fri_layers_merkle_roots,
        fri_decommitment,
    };

    ret
}

/// Returns the DEEP composition polynomial that the prover then commits to using
/// FRI. This polynomial is a linear combination of the trace polynomial and the
/// composition polynomial, with coefficients sampled by the verifier (i.e. using Fiat-Shamir).
fn compute_deep_composition_poly<F: IsField>(
    trace_poly: &Polynomial<FieldElement<F>>,
    even_composition_poly: &Polynomial<FieldElement<F>>,
    odd_composition_poly: &Polynomial<FieldElement<F>>,
    ood_evaluation_point: &FieldElement<F>,
    primitive_root: &FieldElement<F>,
) -> Polynomial<FieldElement<F>> {
    // TODO: Fiat-Shamir
    let gamma_1 = FieldElement::one();
    let gamma_2 = FieldElement::one();
    let gamma_3 = FieldElement::one();
    let gamma_4 = FieldElement::one();

    let first_term = (trace_poly.clone()
        - Polynomial::new_monomial(trace_poly.evaluate(ood_evaluation_point), 0))
        / (Polynomial::new_monomial(FieldElement::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point.clone(), 0));
    let second_term = (trace_poly.clone()
        - Polynomial::new_monomial(
            trace_poly.evaluate(&(ood_evaluation_point * primitive_root)),
            0,
        ))
        / (Polynomial::new_monomial(FieldElement::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point * primitive_root, 0));

    // Evaluate in X^2
    let even_composition_poly = polynomial::compose(
        even_composition_poly,
        &Polynomial::new_monomial(FieldElement::one(), 2),
    );
    let odd_composition_poly = polynomial::compose(
        odd_composition_poly,
        &Polynomial::new_monomial(FieldElement::one(), 2),
    );

    let third_term = (even_composition_poly.clone()
        - Polynomial::new_monomial(
            even_composition_poly.evaluate(&ood_evaluation_point.clone()),
            0,
        ))
        / (Polynomial::new_monomial(FieldElement::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point * ood_evaluation_point, 0));
    let fourth_term = (odd_composition_poly.clone()
        - Polynomial::new_monomial(odd_composition_poly.evaluate(ood_evaluation_point), 0))
        / (Polynomial::new_monomial(FieldElement::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point * ood_evaluation_point, 0));

    first_term * gamma_1 + second_term * gamma_2 + third_term * gamma_3 + fourth_term * gamma_4
}
