use super::{
    air::{constraints::evaluator::ConstraintEvaluator, frame::Frame, trace::TraceTable, AIR},
    fri::{fri, fri_commitment::FriCommitmentVec, fri_decommit::fri_decommit_layers},
    sample_z_ood,
};
use crate::{
    fri::HASHER,
    proof::{DeepConsistencyCheck, StarkProof, StarkQueryProof},
    transcript_to_field, transcript_to_usize, Domain,
};
#[cfg(not(feature = "test_fiat_shamir"))]
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::{fiat_shamir::transcript::Transcript, merkle_tree::merkle::MerkleTree};

#[cfg(feature = "test_fiat_shamir")]
use lambdaworks_crypto::fiat_shamir::test_transcript::TestTranscript;

use lambdaworks_fft::errors::FFTError;
use lambdaworks_fft::polynomial::evaluate_offset_fft_with_len;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsTwoAdicField},
    polynomial::Polynomial,
    traits::ByteConversion,
};
use log::info;

struct Round1<F: IsTwoAdicField> {
    trace_polys: Vec<Polynomial<FieldElement<F>>>,
    lde_trace: TraceTable<F>,
    lde_trace_merkle_trees: Vec<MerkleTree<F>>,
    lde_trace_merkle_roots: Vec<FieldElement<F>>,
}

struct Round2<F: IsTwoAdicField> {
    composition_poly_even: Polynomial<FieldElement<F>>,
    composition_poly_odd: Polynomial<FieldElement<F>>,
}

struct Round3<F: IsTwoAdicField> {
    trace_ood_frame_evaluations: Frame<F>,
    composition_poly_ood_evaluations: Vec<FieldElement<F>>,
}

struct Round4<F: IsTwoAdicField> {
    fri_layers_merkle_roots: Vec<FieldElement<F>>,
    deep_consistency_check: DeepConsistencyCheck<F>,
    query_list: Vec<StarkQueryProof<F>>,
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

fn commit_original_trace<F: IsTwoAdicField, A: AIR<Field = F>>(
    trace: &TraceTable<F>,
    air: &A,
) -> Round1<F>
where
    FieldElement<F>: ByteConversion,
{
    // The trace M_ij is part of the input. Interpolate the polynomials t_j
    // corresponding to the first part of the RAP.
    let trace_polys = trace.compute_trace_polys();

    // Evaluate those polynomials t_j on the large domain D_LDE.
    let lde_trace_evaluations = trace_polys
        .iter()
        .map(|poly| {
            // FIXME: This function should be changed when `evaluate_offset_fft`
            // handles the case for the zero polynomial.
            evaluate_offset_fft_with_len(
                poly,
                trace.n_rows(),
                &FieldElement::<F>::from(air.options().coset_offset),
                air.options().blowup_factor as usize,
            )
        })
        .collect::<Result<Vec<Vec<FieldElement<F>>>, FFTError>>()
        .unwrap();

    let lde_trace = TraceTable::new_from_cols(&lde_trace_evaluations);

    // Compute commitments [t_j].
    let lde_trace_merkle_trees = lde_trace
        .cols()
        .iter()
        .map(|col| MerkleTree::build(col, Box::new(HASHER)))
        .collect::<Vec<MerkleTree<F>>>();

    let lde_trace_merkle_roots = lde_trace_merkle_trees
        .iter()
        .map(|tree| tree.root.clone())
        .collect();

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

fn round_1_randomized_air_with_preprocessing<F: IsTwoAdicField, A: AIR<Field = F>>(
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

fn round_2_compute_composition_polynomial<F: IsTwoAdicField, A: AIR<Field = F>>(
    air: &A,
    domain: &Domain<F>,
    round_1_result: &Round1<F>,
    transition_coeffs: &[(FieldElement<F>, FieldElement<F>)],
    boundary_coeffs: &[(FieldElement<F>, FieldElement<F>)],
) -> Round2<F> {
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

    Round2 {
        composition_poly_even,
        composition_poly_odd,
    }
}

fn round_3_evaluate_polynomials_in_out_of_domain_element<F: IsTwoAdicField, A: AIR<Field = F>>(
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
    let composition_poly_ood_evaluations = vec![
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

fn fri_commit_phase<F: IsTwoAdicField, A: AIR<Field = F>, T: Transcript>(
    air: &A,
    domain: &Domain<F>,
    round_1_result: &Round1<F>,
    round_2_result: &Round2<F>,
    z: &FieldElement<F>,
    transcript: &mut T,
) -> (FriCommitmentVec<F>, Vec<FieldElement<F>>)
where
    FieldElement<F>: ByteConversion,
{
    // Compute DEEP composition polynomial so we can commit to it using FRI.
    let mut deep_composition_poly = compute_deep_composition_poly(
        air,
        &round_1_result.trace_polys,
        &round_2_result.composition_poly_even,
        &round_2_result.composition_poly_odd,
        z,
        &domain.trace_primitive_root,
        transcript,
    );

    // * Do FRI on the composition polynomials
    let lde_fri_commitment = fri(
        &mut deep_composition_poly,
        &domain.lde_roots_of_unity_coset,
        transcript,
    );

    let fri_layers_merkle_roots: Vec<_> = lde_fri_commitment
        .iter()
        .map(|fri_commitment| fri_commitment.merkle_tree.root.clone())
        .collect();

    (lde_fri_commitment, fri_layers_merkle_roots)
}

#[allow(clippy::too_many_arguments)]
fn fri_query_phase<F: IsTwoAdicField, A: AIR<Field = F>, T: Transcript>(
    air: &A,
    domain: &Domain<F>,
    round_1_result: &Round1<F>,
    round_2_result: &Round2<F>,
    q_0: usize,
    lde_fri_commitment: &FriCommitmentVec<F>,
    fri_layers_merkle_roots: &[FieldElement<F>],
    transcript: &mut T,
) -> (Vec<StarkQueryProof<F>>, DeepConsistencyCheck<F>)
where
    FieldElement<F>: ByteConversion,
{
    // Query
    let deep_consistency_check = build_deep_consistency_check(
        domain,
        round_1_result,
        q_0,
        &round_2_result.composition_poly_even,
        &round_2_result.composition_poly_odd,
    );

    let query_list = (0..air.context().options.fri_number_of_queries)
        .map(|i| {
            let q_i = if i > 0 {
                // * Sample q_1, ..., q_m using Fiat-Shamir
                let q = transcript_to_usize(transcript) % 2_usize.pow(domain.lde_root_order);
                transcript.append(&q.to_be_bytes());
                q
            } else {
                q_0
            };

            // * For every q_i, do FRI decommitment
            let fri_decommitment = fri_decommit_layers(lde_fri_commitment, q_i);
            StarkQueryProof {
                fri_layers_merkle_roots: fri_layers_merkle_roots.to_vec(),
                fri_decommitment,
            }
        })
        .collect();

    (query_list, deep_consistency_check)
}

fn round_4_compute_and_run_fri_on_the_deep_composition_polynomial<
    F: IsTwoAdicField,
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
    let (lde_fri_commitment, fri_layers_merkle_roots) =
        fri_commit_phase(air, domain, round_1_result, round_2_result, z, transcript);

    let q_0 = transcript_to_usize(transcript) % 2_usize.pow(domain.lde_root_order);
    transcript.append(&q_0.to_be_bytes());

    let (query_list, deep_consistency_check) = fri_query_phase(
        air,
        domain,
        round_1_result,
        round_2_result,
        q_0,
        &lde_fri_commitment,
        &fri_layers_merkle_roots,
        transcript,
    );
    Round4 {
        fri_layers_merkle_roots,
        deep_consistency_check,
        query_list,
    }
}

/// Returns the DEEP composition polynomial that the prover then commits to using
/// FRI. This polynomial is a linear combination of the trace polynomial and the
/// composition polynomial, with coefficients sampled by the verifier (i.e. using Fiat-Shamir).
fn compute_deep_composition_poly<A: AIR, F: IsTwoAdicField, T: Transcript>(
    air: &A,
    trace_polys: &[Polynomial<FieldElement<F>>],
    even_composition_poly: &Polynomial<FieldElement<F>>,
    odd_composition_poly: &Polynomial<FieldElement<F>>,
    ood_evaluation_point: &FieldElement<F>,
    primitive_root: &FieldElement<F>,
    transcript: &mut T,
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
        for (trace_evaluation, offset) in trace_evaluations.iter().zip(&transition_offsets) {
            let eval = trace_evaluation[i].clone();
            let root_of_unity = ood_evaluation_point * primitive_root.pow(*offset);
            let poly = (trace_poly.clone() - Polynomial::new_monomial(eval, 0))
                / (Polynomial::new_monomial(FieldElement::<F>::one(), 1)
                    - Polynomial::new_monomial(root_of_unity, 0));
            let coeff = transcript_to_field::<F, T>(transcript);

            trace_terms = trace_terms + poly * coeff;
        }
    }

    // Get coefficients for even and odd terms of the composition polynomial H(x)
    let gamma_even = transcript_to_field::<F, T>(transcript);
    let gamma_odd = transcript_to_field::<F, T>(transcript);

    let ood_point_squared = ood_evaluation_point * ood_evaluation_point;

    let even_composition_poly_term = (even_composition_poly.clone()
        - Polynomial::new_monomial(even_composition_poly.evaluate(&ood_point_squared), 0))
        / (Polynomial::new_monomial(FieldElement::one(), 1)
            - Polynomial::new_monomial(ood_point_squared.clone(), 0));

    let odd_composition_poly_term = (odd_composition_poly.clone()
        - Polynomial::new_monomial(odd_composition_poly.evaluate(&ood_point_squared), 0))
        / (Polynomial::new_monomial(FieldElement::one(), 1)
            - Polynomial::new_monomial(ood_point_squared.clone(), 0));

    trace_terms + even_composition_poly_term * gamma_even + odd_composition_poly_term * gamma_odd
}

fn build_deep_consistency_check<F: IsTwoAdicField>(
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
pub fn prove<F: IsTwoAdicField, A: AIR<Field = F>>(trace: &TraceTable<F>, air: &A) -> StarkProof<F>
where
    FieldElement<F>: ByteConversion,
{
    info!("Starting proof generation...");

    #[cfg(debug_assertions)]
    trace.validate(air);

    let domain = Domain::new(air);

    let transcript = &mut round_0_transcript_initialization();

    // Fiat-Shamir
    // z is the Out of domain evaluation point used in Deep FRI. It needs to be a point outside
    // of both the roots of unity and its corresponding coset used for the lde commitment.
    // TODO: This has to be sampled after round 2 according to the protocol
    let z = sample_z_ood(
        &domain.lde_roots_of_unity_coset,
        &domain.trace_roots_of_unity,
        transcript,
    );

    let round_1_result = round_1_randomized_air_with_preprocessing(trace, air);

    // Sample challenges for round 2
    // These are the challenges alpha^B_j and beta^B_j
    let boundary_coeffs: Vec<(FieldElement<F>, FieldElement<F>)> =
        (0..round_1_result.trace_polys.len())
            .map(|_| {
                (
                    transcript_to_field(transcript),
                    transcript_to_field(transcript),
                )
            })
            .collect();

    // These are the challenges alpha^T_j and beta^T_j
    let transition_coeffs: Vec<(FieldElement<F>, FieldElement<F>)> =
        (0..air.context().num_transition_constraints)
            .map(|_| {
                (
                    transcript_to_field(transcript),
                    transcript_to_field(transcript),
                )
            })
            .collect();

    let round_2_result = round_2_compute_composition_polynomial(
        air,
        &domain,
        &round_1_result,
        &transition_coeffs,
        &boundary_coeffs,
    );

    let round_3_result = round_3_evaluate_polynomials_in_out_of_domain_element(
        air,
        &domain,
        &round_1_result,
        &round_2_result,
        &z,
    );

    let round_4_result = round_4_compute_and_run_fri_on_the_deep_composition_polynomial(
        air,
        &domain,
        &round_1_result,
        &round_2_result,
        &z,
        transcript,
    );

    info!("End proof generation");

    StarkProof {
        fri_layers_merkle_roots: round_4_result.fri_layers_merkle_roots,
        trace_ood_frame_evaluations: round_3_result.trace_ood_frame_evaluations,
        composition_poly_ood_evaluations: round_3_result.composition_poly_ood_evaluations,
        deep_consistency_check: round_4_result.deep_consistency_check,
        query_list: round_4_result.query_list,
    }
}
