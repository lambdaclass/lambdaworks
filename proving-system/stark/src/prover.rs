use super::constraints::boundary::{BoundaryConstraint, BoundaryConstraints};
use super::fri::fri_decommit::fri_decommit_layers;
use super::utils::compute_zerofier;
use super::{
    generate_primitive_root, generate_roots_of_unity_coset, StarkQueryProof, COSET_OFFSET, FE,
    ORDER_OF_ROOTS_OF_UNITY_FOR_LDE, ORDER_OF_ROOTS_OF_UNITY_TRACE,
};
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::polynomial::{self, Polynomial};
use std::ops::{Div, Mul};

pub fn prove(trace: &[FE]) -> StarkQueryProof {
    let transcript = &mut Transcript::new();

    // * Generate Coset
    let trace_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_TRACE);
    let trace_roots_of_unity = generate_roots_of_unity_coset(1, &trace_primitive_root);

    let lde_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_FOR_LDE);
    let lde_roots_of_unity_coset = generate_roots_of_unity_coset(COSET_OFFSET, &lde_primitive_root);

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
pub(crate) fn compute_composition_polys(
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

pub(crate) fn compute_boundary_quotient(
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

/// Returns the DEEP composition polynomial that the prover then commits to using
/// FRI. This polynomial is a linear combination of the trace polynomial and the
/// composition polynomial, with coefficients sampled by the verifier (i.e. using Fiat-Shamir).
pub(crate) fn compute_deep_composition_poly(
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
    let gamma_5 = FE::one();

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

    let third_term = (trace_poly.clone()
        - Polynomial::new_monomial(
            trace_poly.evaluate(&(ood_evaluation_point * primitive_root * primitive_root)),
            0,
        ))
        / (Polynomial::new_monomial(FE::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point * primitive_root * primitive_root, 0));

    // Evaluate in X^2
    let even_composition_poly = polynomial::compose(
        even_composition_poly,
        &Polynomial::new_monomial(FE::one(), 2),
    );
    let odd_composition_poly = polynomial::compose(
        odd_composition_poly,
        &Polynomial::new_monomial(FE::one(), 2),
    );

    let fourth_term = (even_composition_poly.clone()
        - Polynomial::new_monomial(
            even_composition_poly.evaluate(&ood_evaluation_point.clone()),
            0,
        ))
        / (Polynomial::new_monomial(FE::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point * ood_evaluation_point, 0));
    let fifth_term = (odd_composition_poly.clone()
        - Polynomial::new_monomial(odd_composition_poly.evaluate(ood_evaluation_point), 0))
        / (Polynomial::new_monomial(FE::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point * ood_evaluation_point, 0));

    first_term * gamma_1
        + second_term * gamma_2
        + third_term * gamma_3
        + fourth_term * gamma_4
        + fifth_term * gamma_5
}
