use super::{
    air::{constraints::evaluator::ConstraintEvaluator, frame::Frame, trace::TraceTable, AIR},
    fri::{fri_commit_phase, fri_decommit::fri_decommit_layers},
    sample_z_ood,
};
use crate::{
    batch_sample_challenges,
    fri::{fri_commitment::FriLayer, HASHER},
    proof::{DeepConsistencyCheck, StarkProof, FriQuery},
    transcript_to_field, transcript_to_usize, Domain,
};
#[cfg(not(feature = "test_fiat_shamir"))]
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::{fiat_shamir::transcript::Transcript, merkle_tree::merkle::MerkleTree};

#[cfg(feature = "test_fiat_shamir")]
use lambdaworks_crypto::fiat_shamir::test_transcript::TestTranscript;

use lambdaworks_fft::errors::FFTError;
use lambdaworks_fft::polynomial::FFTPoly;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
    traits::ByteConversion,
};

struct Round1<F: IsFFTField> {
    trace_polys: Vec<Polynomial<FieldElement<F>>>,
    lde_trace: TraceTable<F>,
    lde_trace_merkle_trees: Vec<MerkleTree<F>>,
    lde_trace_merkle_roots: Vec<FieldElement<F>>,
}

struct Round2<F: IsFFTField> {
    composition_poly_even: Polynomial<FieldElement<F>>,
    composition_poly_odd: Polynomial<FieldElement<F>>,
    // Merkle trees of H_1 and H_2 at the LDE Domain
    composition_poly_merkle_trees: Vec<MerkleTree<F>>,
    // Commitments of H_1, and H_2
    composition_poly_roots: Vec<FieldElement<F>>,
}

struct Round3<F: IsFFTField> {
    trace_ood_frame_evaluations: Frame<F>,
    composition_poly_ood_evaluations: [FieldElement<F>; 2],
}

struct Round4<F: IsFFTField> {
    fri_last_value: FieldElement<F>,
    fri_layers_merkle_roots: Vec<FieldElement<F>>,
    deep_consistency_check: DeepConsistencyCheck<F>,
    query_list: Vec<FriQuery<F>>,
}

#[cfg(feature = "test_fiat_shamir")]
fn round_0_transcript_initialization() -> TestTranscript {
    TestTranscript::new()
}

#[cfg(not(feature = "test_fiat_shamir"))]
fn round_0_transcript_initialization() -> DefaultTranscript {
    // TODO: add strong fiat shamir
    DefaultTranscript::new()
}

fn batch_commit<F>(vectors: Vec<Vec<FieldElement<F>>>) -> (Vec<MerkleTree<F>>, Vec<FieldElement<F>>)
where
    F: IsFFTField,
    FieldElement<F>: ByteConversion,
{
    let trees = vectors
        .iter()
        .map(|col| MerkleTree::build(col, Box::new(HASHER)))
        .collect::<Vec<MerkleTree<F>>>();

    let roots = trees.iter().map(|tree| tree.root.clone()).collect();
    (trees, roots)
}

fn evaluate_polynomial_on_lde_domain<F, A>(
    p: &Polynomial<FieldElement<F>>,
    air: &A,
) -> Result<Vec<FieldElement<F>>, FFTError>
where
    F: IsFFTField,
    Polynomial<FieldElement<F>>: FFTPoly<F>,
    A: AIR<Field = F>,
{
    // Evaluate those polynomials t_j on the large domain D_LDE.
    p.evaluate_offset_fft(
        &FieldElement::<F>::from(air.options().coset_offset),
        air.options().blowup_factor as usize,
    )
}

fn commit_original_trace<F, A>(trace: &TraceTable<F>, air: &A) -> Round1<F>
where
    F: IsFFTField,
    A: AIR<Field = F>,
    FieldElement<F>: ByteConversion,
{
    // The trace M_ij is part of the input. Interpolate the polynomials t_j
    // corresponding to the first part of the RAP.
    let trace_polys = trace.compute_trace_polys();

    // Evaluate those polynomials t_j on the large domain D_LDE.
    let lde_trace_evaluations = trace_polys
        .iter()
        .map(|poly| evaluate_polynomial_on_lde_domain(poly, air))
        .collect::<Result<Vec<Vec<FieldElement<F>>>, FFTError>>()
        .unwrap();

    let lde_trace = TraceTable::new_from_cols(&lde_trace_evaluations);

    // Compute commitments [t_j].
    let (lde_trace_merkle_trees, lde_trace_merkle_roots) = batch_commit(lde_trace.cols());

    Round1 {
        trace_polys,
        lde_trace,
        lde_trace_merkle_trees,
        lde_trace_merkle_roots,
    }
}

fn commit_extended_trace() {
    // TODO
}

fn round_1_randomized_air_with_preprocessing<F: IsFFTField, A: AIR<Field = F>>(
    trace: &TraceTable<F>,
    air: &A,
) -> Round1<F>
where
    FieldElement<F>: ByteConversion,
{
    let round_1_result = commit_original_trace(trace, air);
    commit_extended_trace();
    round_1_result
}

fn round_2_compute_composition_polynomial<F, A>(
    air: &A,
    domain: &Domain<F>,
    round_1_result: &Round1<F>,
    transition_coeffs: &[(FieldElement<F>, FieldElement<F>)],
    boundary_coeffs: &[(FieldElement<F>, FieldElement<F>)],
) -> Round2<F>
where
    F: IsFFTField,
    A: AIR<Field = F>,
    FieldElement<F>: ByteConversion,
{
    // Create evaluation table
    let evaluator = ConstraintEvaluator::new(
        air,
        &round_1_result.trace_polys,
        &domain.trace_primitive_root,
    );

    let constraint_evaluations = evaluator.evaluate(
        &round_1_result.lde_trace,
        &domain.lde_roots_of_unity_coset,
        transition_coeffs,
        boundary_coeffs,
    );

    // Get the composition poly H
    let composition_poly =
        constraint_evaluations.compute_composition_poly(&domain.lde_roots_of_unity_coset);

    let (composition_poly_even, composition_poly_odd) = composition_poly.even_odd_decomposition();

    let lde_composition_poly_even_evaluations =
        evaluate_polynomial_on_lde_domain(&composition_poly_even, air).unwrap();
    let lde_composition_poly_odd_evaluations =
        evaluate_polynomial_on_lde_domain(&composition_poly_odd, air).unwrap();

    let (composition_poly_merkle_trees, composition_poly_roots) = batch_commit(vec![
        lde_composition_poly_even_evaluations,
        lde_composition_poly_odd_evaluations,
    ]);

    Round2 {
        composition_poly_even,
        composition_poly_odd,
        composition_poly_merkle_trees,
        composition_poly_roots,
    }
}

fn round_3_evaluate_polynomials_in_out_of_domain_element<F: IsFFTField, A: AIR<Field = F>>(
    air: &A,
    domain: &Domain<F>,
    round_1_result: &Round1<F>,
    round_2_result: &Round2<F>,
    z: &FieldElement<F>,
) -> Round3<F>
where
    FieldElement<F>: ByteConversion,
{
    let z_squared = z * z;

    // Evaluate H_1 and H_2 in z^2.
    let composition_poly_ood_evaluations = [
        round_2_result.composition_poly_even.evaluate(&z_squared),
        round_2_result.composition_poly_odd.evaluate(&z_squared),
    ];

    // Returns the Out of Domain Frame for the given trace polynomials, out of domain evaluation point (called `z` in the literature),
    // frame offsets given by the AIR and primitive root used for interpolating the trace polynomials.
    // An out of domain frame is nothing more than the evaluation of the trace polynomials in the points required by the
    // verifier to check the consistency between the trace and the composition polynomial.
    //
    // In the fibonacci example, the ood frame is simply the evaluations `[t(z), t(z * g), t(z * g^2)]`, where `t` is the trace
    // polynomial and `g` is the primitive root of unity used when interpolating `t`.
    let ood_trace_evaluations = Frame::get_trace_evaluations(
        &round_1_result.trace_polys,
        z,
        &air.context().transition_offsets,
        &domain.trace_primitive_root,
    );

    let trace_ood_frame_data = ood_trace_evaluations.into_iter().flatten().collect();
    let trace_ood_frame_evaluations =
        Frame::new(trace_ood_frame_data, round_1_result.trace_polys.len());

    Round3 {
        trace_ood_frame_evaluations,
        composition_poly_ood_evaluations,
    }
}

#[allow(clippy::too_many_arguments)]
fn fri_query_phase<F: IsFFTField, A: AIR<Field = F>, T: Transcript>(
    air: &A,
    domain: &Domain<F>,
    round_1_result: &Round1<F>,
    round_2_result: &Round2<F>,
    fri_layers: &Vec<FriLayer<F>>,
    fri_layers_merkle_roots: &[FieldElement<F>],
    transcript: &mut T,
) -> (Vec<FriQuery<F>>, DeepConsistencyCheck<F>)
where
    FieldElement<F>: ByteConversion,
{
    let q_0 = transcript_to_usize(transcript) % 2_usize.pow(domain.lde_root_order);

    let query_list = (0..air.context().options.fri_number_of_queries)
        .map(|i| {
            let q_i = if i > 0 {
                // * Sample q_1, ..., q_m using Fiat-Shamir
                transcript_to_usize(transcript) % 2_usize.pow(domain.lde_root_order)
            } else {
                q_0
            };

            // * For every q_i, do FRI decommitment
            let fri_decommitment = fri_decommit_layers(fri_layers, q_i);
            FriQuery {
                fri_layers_merkle_roots: fri_layers_merkle_roots.to_vec(),
                fri_decommitment,
            }
        })
        .collect();

    // Query
    let deep_consistency_check = build_deep_consistency_check(
        domain,
        round_1_result,
        q_0,
        &round_2_result.composition_poly_even,
        &round_2_result.composition_poly_odd,
    );

    (query_list, deep_consistency_check)
}

fn round_4_compute_and_run_fri_on_the_deep_composition_polynomial<
    F: IsFFTField,
    A: AIR<Field = F>,
    T: Transcript,
>(
    air: &A,
    domain: &Domain<F>,
    round_1_result: &Round1<F>,
    round_2_result: &Round2<F>,
    z: &FieldElement<F>,
    transcript: &mut T,
) -> Round4<F>
where
    FieldElement<F>: ByteConversion,
{
    let trace_poly_coeffients = batch_sample_challenges::<F, T>(
        air.context().transition_offsets.len() * air.context().trace_columns,
        transcript,
    );

    let composition_poly_coeffients = [
        transcript_to_field(transcript),
        transcript_to_field(transcript),
    ];

    // Compute DEEP composition polynomial so we can commit to it using FRI.
    let deep_composition_poly = compute_deep_composition_poly(
        air,
        &round_1_result.trace_polys,
        &round_2_result.composition_poly_even,
        &round_2_result.composition_poly_odd,
        z,
        &domain.trace_primitive_root,
        &composition_poly_coeffients,
        &trace_poly_coeffients,
    );

    // * Do FRI on the deep composition polynomial
    let (fri_last_value, fri_layers) = fri_commit_phase(
        domain.root_order as usize,
        deep_composition_poly,
        &domain.lde_roots_of_unity_coset,
        transcript,
    );

    let fri_layers_merkle_roots: Vec<_> = fri_layers
        .iter()
        .map(|layer| layer.merkle_tree.root.clone())
        .collect();


    let (query_list, deep_consistency_check) = fri_query_phase(
        air,
        domain,
        round_1_result,
        round_2_result,
        &fri_layers,
        &fri_layers_merkle_roots,
        transcript,
    );
    Round4 {
        fri_last_value,
        fri_layers_merkle_roots,
        deep_consistency_check,
        query_list,
    }
}

/// Returns the DEEP composition polynomial that the prover then commits to using
/// FRI. This polynomial is a linear combination of the trace polynomial and the
/// composition polynomial, with coefficients sampled by the verifier (i.e. using Fiat-Shamir).
fn compute_deep_composition_poly<A: AIR, F: IsFFTField>(
    air: &A,
    trace_polys: &[Polynomial<FieldElement<F>>],
    even_composition_poly: &Polynomial<FieldElement<F>>,
    odd_composition_poly: &Polynomial<FieldElement<F>>,
    ood_evaluation_point: &FieldElement<F>,
    primitive_root: &FieldElement<F>,
    composition_poly_coeffients: &[FieldElement<F>; 2],
    trace_poly_coeffients: &[FieldElement<F>],
) -> Polynomial<FieldElement<F>> {
    let transition_offsets = air.context().transition_offsets;

    // Get trace evaluations needed for the trace terms of the deep composition polynomial
    let trace_evaluations = Frame::get_trace_evaluations(
        trace_polys,
        ood_evaluation_point,
        &transition_offsets,
        primitive_root,
    );

    // Compute all the trace terms of the deep composition polynomial. There will be one
    // term for every trace polynomial and every trace evaluation.
    let mut trace_terms = Polynomial::zero();
    for (i, trace_poly) in trace_polys.iter().enumerate() {
        for (j, (trace_evaluation, offset)) in trace_evaluations
            .iter()
            .zip(&transition_offsets)
            .enumerate()
        {
            let eval = trace_evaluation[i].clone();
            let shifted_root_of_unity = ood_evaluation_point * primitive_root.pow(*offset);
            let poly = (trace_poly - eval)
                / Polynomial::new(&[-shifted_root_of_unity, FieldElement::one()]);

            trace_terms =
                trace_terms + poly * &trace_poly_coeffients[i * trace_evaluations.len() + j];
        }
    }

    let ood_point_squared = ood_evaluation_point * ood_evaluation_point;

    let even_composition_poly_term = (even_composition_poly.clone()
        - Polynomial::new_monomial(even_composition_poly.evaluate(&ood_point_squared), 0))
        / (Polynomial::new_monomial(FieldElement::one(), 1)
            - Polynomial::new_monomial(ood_point_squared.clone(), 0));

    let odd_composition_poly_term = (odd_composition_poly.clone()
        - Polynomial::new_monomial(odd_composition_poly.evaluate(&ood_point_squared), 0))
        / (Polynomial::new_monomial(FieldElement::one(), 1)
            - Polynomial::new_monomial(ood_point_squared.clone(), 0));

    trace_terms
        + even_composition_poly_term * &composition_poly_coeffients[0]
        + odd_composition_poly_term * &composition_poly_coeffients[1]
}

fn build_deep_consistency_check<F: IsFFTField>(
    domain: &Domain<F>,
    round_1_result: &Round1<F>,
    index_to_verify: usize,
    composition_poly_even: &Polynomial<FieldElement<F>>,
    composition_poly_odd: &Polynomial<FieldElement<F>>,
) -> DeepConsistencyCheck<F>
where
    FieldElement<F>: ByteConversion,
{
    let index = index_to_verify % domain.lde_roots_of_unity_coset.len();
    let lde_trace_merkle_proofs = round_1_result
        .lde_trace_merkle_trees
        .iter()
        .map(|tree| tree.get_proof_by_pos(index).unwrap())
        .collect();

    let d_evaluation_point = &domain.lde_roots_of_unity_coset[index];
    let lde_trace_evaluations = round_1_result.lde_trace.get_row(index).to_vec();

    let composition_poly_evaluations = vec![
        composition_poly_even.evaluate(d_evaluation_point),
        composition_poly_odd.evaluate(d_evaluation_point),
    ];

    DeepConsistencyCheck {
        lde_trace_merkle_roots: round_1_result.lde_trace_merkle_roots.clone(),
        lde_trace_merkle_proofs,
        lde_trace_evaluations,
        composition_poly_evaluations,
    }
}

// FIXME remove unwrap() calls and return errors
pub fn prove<F: IsFFTField, A: AIR<Field = F>>(trace: &TraceTable<F>, air: &A) -> StarkProof<F>
where
    FieldElement<F>: ByteConversion,
{
    #[cfg(debug_assertions)]
    trace.validate(air);

    let domain = Domain::new(air);

    let mut transcript = round_0_transcript_initialization();

    // Round 1
    let round_1_result = round_1_randomized_air_with_preprocessing(trace, air);

    for root in round_1_result.lde_trace_merkle_roots.iter() {
        transcript.append(&root.to_bytes_be());
    }

    // Round 2
    // These are the challenges alpha^B_j and beta^B_j
    let boundary_coeffs_alphas =
        batch_sample_challenges(round_1_result.trace_polys.len(), &mut transcript);
    let boundary_coeffs_betas =
        batch_sample_challenges(round_1_result.trace_polys.len(), &mut transcript);
    let boundary_coeffs: Vec<_> = boundary_coeffs_alphas
        .into_iter()
        .zip(boundary_coeffs_betas)
        .collect();

    // These are the challenges alpha^T_j and beta^T_j
    let transition_coeffs_alphas =
        batch_sample_challenges(air.context().num_transition_constraints, &mut transcript);
    let transition_coeffs_betas =
        batch_sample_challenges(air.context().num_transition_constraints, &mut transcript);
    let transition_coeffs: Vec<_> = transition_coeffs_alphas
        .into_iter()
        .zip(transition_coeffs_betas)
        .collect();
    let round_2_result = round_2_compute_composition_polynomial(
        air,
        &domain,
        &round_1_result,
        &transition_coeffs,
        &boundary_coeffs,
    );

    transcript.append(&round_2_result.composition_poly_roots[0].to_bytes_be());
    transcript.append(&round_2_result.composition_poly_roots[1].to_bytes_be());

    // Round 3
    let z = sample_z_ood(
        &domain.lde_roots_of_unity_coset,
        &domain.trace_roots_of_unity,
        &mut transcript,
    );

    let round_3_result = round_3_evaluate_polynomials_in_out_of_domain_element(
        air,
        &domain,
        &round_1_result,
        &round_2_result,
        &z,
    );

    // H_1(z^2)
    transcript.append(&round_3_result.composition_poly_ood_evaluations[0].to_bytes_be());
    // H_2(z^2)
    transcript.append(&round_3_result.composition_poly_ood_evaluations[1].to_bytes_be());
    // These are the values t_j(z)
    for element in round_3_result.trace_ood_frame_evaluations.get_row(0).iter() {
        transcript.append(&element.to_bytes_be());
    }
    // These are the values t_j(gz)
    for element in round_3_result.trace_ood_frame_evaluations.get_row(1).iter() {
        transcript.append(&element.to_bytes_be());
    }

    // Round 4
    let round_4_result = round_4_compute_and_run_fri_on_the_deep_composition_polynomial(
        air,
        &domain,
        &round_1_result,
        &round_2_result,
        &z,
        &mut transcript,
    );

    StarkProof {
        lde_trace_merkle_roots: round_1_result.lde_trace_merkle_roots,
        composition_poly_roots: round_2_result.composition_poly_roots,
        fri_layers_merkle_roots: round_4_result.fri_layers_merkle_roots,
        fri_last_value: round_4_result.fri_last_value,
        trace_ood_frame_evaluations: round_3_result.trace_ood_frame_evaluations,
        composition_poly_ood_evaluations: round_3_result.composition_poly_ood_evaluations,
        deep_consistency_check: round_4_result.deep_consistency_check,
        query_list: round_4_result.query_list,
    }
}
