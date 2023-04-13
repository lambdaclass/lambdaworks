use super::{
    air::{constraints::evaluator::ConstraintEvaluator, frame::Frame, trace::TraceTable, AIR},
    fri::{fri, fri_decommit::fri_decommit_layers},
    sample_z_ood,
};
use crate::{
    proof::{DeepConsistencyCheck, StarkProof, StarkQueryProof},
    transcript_to_field, transcript_to_usize,
};
use lambdaworks_crypto::{
    fiat_shamir::transcript::Transcript,
    hash::traits::IsCryptoHash,
    merkle_tree::{merkle::MerkleTree, proof::Proof},
};
use lambdaworks_math::{
    fft::errors::FFTError,
    field::{element::FieldElement, traits::IsTwoAdicField},
    polynomial::Polynomial,
    traits::ByteConversion,
};

struct DeepConsistencyCheckArgs<'a, F: IsTwoAdicField, A: AIR<Field = F>> {
    pub air: &'a A,
    pub transcript: &'a mut Transcript,
    pub trace: &'a TraceTable<F>,
    pub lde_trace_evaluations: &'a [Vec<FieldElement<F>>],
    pub composition_poly_even: &'a Polynomial<FieldElement<F>>,
    pub composition_poly_odd: &'a Polynomial<FieldElement<F>>,
    pub deep_composition_poly: &'a Polynomial<FieldElement<F>>,
    pub lde_roots_of_unity_coset: &'a [FieldElement<F>],
}

// FIXME remove unwrap() calls and return errors
pub fn prove<F: IsTwoAdicField, A: AIR<Field = F>, H: IsCryptoHash<F> + Clone>(
    trace: &TraceTable<F>,
    air: &A,
) -> StarkProof<F, H>
where
    FieldElement<F>: ByteConversion,
{
    let transcript = &mut Transcript::new();

    let blowup_factor = air.options().blowup_factor as usize;
    let coset_offset = FieldElement::<F>::from(air.options().coset_offset);

    let root_order = air.context().trace_length.trailing_zeros();
    // * Generate Coset
    let trace_primitive_root = F::get_primitive_root_of_unity(root_order as u64).unwrap();
    let trace_roots_of_unity = F::get_powers_of_primitive_root_coset(
        root_order as u64,
        air.context().trace_length,
        &FieldElement::<F>::one(),
    )
    .unwrap();

    let lde_root_order = (air.context().trace_length * blowup_factor).trailing_zeros();
    let lde_roots_of_unity_coset = F::get_powers_of_primitive_root_coset(
        lde_root_order as u64,
        air.context().trace_length * blowup_factor,
        &coset_offset,
    )
    .unwrap();

    let trace_polys = trace.compute_trace_polys();
    let lde_trace_evaluations = trace_polys
        .iter()
        .map(|poly| {
            poly.evaluate_offset_fft(
                &FieldElement::<F>::from(air.options().coset_offset),
                air.options().blowup_factor as usize,
            )
        })
        .collect::<Result<Vec<Vec<FieldElement<F>>>, FFTError>>()
        .unwrap();

    let lde_trace = TraceTable::new_from_cols(&lde_trace_evaluations);

    // Fiat-Shamir
    // z is the Out of domain evaluation point used in Deep FRI. It needs to be a point outside
    // of both the roots of unity and its corresponding coset used for the lde commitment.
    let z = sample_z_ood(&lde_roots_of_unity_coset, &trace_roots_of_unity, transcript);

    let z_squared = &z * &z;

    // Create evaluation table
    let evaluator = ConstraintEvaluator::new(air, &trace_polys, &trace_primitive_root);

    let boundary_coeffs: Vec<(FieldElement<F>, FieldElement<F>)> = (0..trace_polys.len())
        .map(|_| {
            (
                transcript_to_field(transcript),
                transcript_to_field(transcript),
            )
        })
        .collect();

    let transition_coeffs: Vec<(FieldElement<F>, FieldElement<F>)> =
        (0..air.context().num_transition_constraints)
            .map(|_| {
                (
                    transcript_to_field(transcript),
                    transcript_to_field(transcript),
                )
            })
            .collect();

    let constraint_evaluations = evaluator.evaluate(
        &lde_trace,
        &lde_roots_of_unity_coset,
        &transition_coeffs,
        &boundary_coeffs,
    );

    // Get the composition poly H
    let composition_poly =
        constraint_evaluations.compute_composition_poly(&lde_roots_of_unity_coset);

    let (composition_poly_even, composition_poly_odd) = composition_poly.even_odd_decomposition();
    // Evaluate H_1 and H_2 in z^2.
    let composition_poly_ood_evaluations = vec![
        composition_poly_even.evaluate(&z_squared),
        composition_poly_odd.evaluate(&z_squared),
    ];

    // Returns the Out of Domain Frame for the given trace polynomials, out of domain evaluation point (called `z` in the literature),
    // frame offsets given by the AIR and primitive root used for interpolating the trace polynomials.
    // An out of domain frame is nothing more than the evaluation of the trace polynomials in the points required by the
    // verifier to check the consistency between the trace and the composition polynomial.
    //
    // In the fibonacci example, the ood frame is simply the evaluations `[t(z), t(z * g), t(z * g^2)]`, where `t` is the trace
    // polynomial and `g` is the primitive root of unity used when interpolating `t`.
    let ood_trace_evaluations = Frame::get_trace_evaluations(
        &trace_polys,
        &z,
        &air.context().transition_offsets,
        &trace_primitive_root,
    );

    let trace_ood_frame_data = ood_trace_evaluations.into_iter().flatten().collect();
    let trace_ood_frame_evaluations = Frame::new(trace_ood_frame_data, trace_polys.len());

    // END EVALUATION BLOCK

    // Compute DEEP composition polynomial so we can commit to it using FRI.
    let mut deep_composition_poly = compute_deep_composition_poly(
        air,
        transcript,
        &trace_polys,
        &composition_poly_even,
        &composition_poly_odd,
        &z,
        &trace_primitive_root,
    );

    let deep_consistency_check_args = &mut DeepConsistencyCheckArgs {
        air,
        transcript,
        trace: &lde_trace,
        lde_trace_evaluations: &lde_trace_evaluations,
        composition_poly_even: &composition_poly_even,
        composition_poly_odd: &composition_poly_odd,
        deep_composition_poly: &deep_composition_poly,
        lde_roots_of_unity_coset: &lde_roots_of_unity_coset,
    };

    let deep_consistency_check = build_deep_consistency_check(deep_consistency_check_args);

    // * Do FRI on the composition polynomials
    let lde_fri_commitment = fri(
        &mut deep_composition_poly,
        &lde_roots_of_unity_coset,
        transcript,
    );

    let fri_layers_merkle_roots: Vec<_> = lde_fri_commitment
        .iter()
        .map(|fri_commitment| fri_commitment.merkle_tree.root.clone())
        .collect();

    let query_list = (0..air.context().options.fri_number_of_queries)
        .map(|_| {
            // * Sample q_1, ..., q_m using Fiat-Shamir
            let q_i = transcript_to_usize(transcript) % 2_usize.pow(lde_root_order);
            transcript.append(&q_i.to_be_bytes());

            // * For every q_i, do FRI decommitment
            let fri_decommitment = fri_decommit_layers(&lde_fri_commitment, q_i);
            StarkQueryProof {
                fri_layers_merkle_roots: fri_layers_merkle_roots.clone(),
                fri_decommitment,
            }
        })
        .collect();

    StarkProof {
        fri_layers_merkle_roots,
        trace_ood_frame_evaluations,
        deep_consistency_check,
        composition_poly_ood_evaluations,
        query_list,
    }
}

/// Returns the DEEP composition polynomial that the prover then commits to using
/// FRI. This polynomial is a linear combination of the trace polynomial and the
/// composition polynomial, with coefficients sampled by the verifier (i.e. using Fiat-Shamir).
fn compute_deep_composition_poly<F: IsTwoAdicField, A: AIR<Field = F>>(
    air: &A,
    transcript: &mut Transcript,
    trace_polys: &[Polynomial<FieldElement<F>>],
    even_composition_poly: &Polynomial<FieldElement<F>>,
    odd_composition_poly: &Polynomial<FieldElement<F>>,
    ood_evaluation_point: &FieldElement<F>,
    primitive_root: &FieldElement<F>,
) -> Polynomial<FieldElement<F>> {
    let transition_offsets = air.context().transition_offsets;

    // Get the number of trace terms the DEEP composition poly will have.
    // One coefficient will be sampled for each of them.
    let trace_term_coeffs = (0..trace_polys.len())
        .map(|_| {
            (0..transition_offsets.len())
                .map(|_| transcript_to_field::<F>(transcript))
                .collect()
        })
        .collect::<Vec<Vec<FieldElement<F>>>>();

    // Get coefficients for even and odd terms of the composition polynomial H(x)
    let gamma_even = transcript_to_field::<F>(transcript);
    let gamma_odd = transcript_to_field::<F>(transcript);

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
    for ((i, trace_poly), coeff_row) in trace_polys.iter().enumerate().zip(trace_term_coeffs) {
        for ((trace_evaluation, offset), coeff) in trace_evaluations
            .iter()
            .zip(&transition_offsets)
            .zip(coeff_row)
        {
            let eval = trace_evaluation[i].clone();
            let root_of_unity = ood_evaluation_point * primitive_root.pow(*offset);
            let poly = (trace_poly.clone() - Polynomial::new_monomial(eval, 0))
                / (Polynomial::new_monomial(FieldElement::<F>::one(), 1)
                    - Polynomial::new_monomial(root_of_unity, 0));

            trace_terms = trace_terms + poly * coeff.clone();
        }
    }

    let ood_evaluation_point_squared = ood_evaluation_point * ood_evaluation_point;

    let even_composition_poly_term = (even_composition_poly.clone()
        - Polynomial::new_monomial(
            even_composition_poly.evaluate(&ood_evaluation_point_squared),
            0,
        ))
        / (Polynomial::new_monomial(FieldElement::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point_squared.clone(), 0));

    let odd_composition_poly_term = (odd_composition_poly.clone()
        - Polynomial::new_monomial(
            odd_composition_poly.evaluate(&ood_evaluation_point_squared),
            0,
        ))
        / (Polynomial::new_monomial(FieldElement::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point_squared.clone(), 0));

    trace_terms + even_composition_poly_term * gamma_even + odd_composition_poly_term * gamma_odd
}

fn build_deep_consistency_check<
    F: IsTwoAdicField,
    A: AIR<Field = F>,
    H: IsCryptoHash<F> + Clone,
>(
    args: &mut DeepConsistencyCheckArgs<F, A>,
) -> DeepConsistencyCheck<F, H> {
    let consistency_check_idx = transcript_to_usize(args.transcript)
        % (args.air.context().trace_length * args.air.options().blowup_factor as usize);
    let consistency_check_x = &args.lde_roots_of_unity_coset[consistency_check_idx];

    let lde_trace_frame = Frame::read_from_trace(
        args.trace,
        consistency_check_idx,
        args.air.options().blowup_factor,
        &args.air.context().transition_offsets,
    );

    let lde_trace_merkle_trees = args
        .lde_trace_evaluations
        .iter()
        .map(|evaluation| MerkleTree::build(evaluation))
        .collect::<Vec<MerkleTree<_, H>>>();

    let lde_trace_merkle_roots = lde_trace_merkle_trees
        .iter()
        .map(|merkle_tree| merkle_tree.root.clone())
        .collect::<Vec<FieldElement<F>>>();

    let lde_trace_merkle_proofs = (0..lde_trace_frame.num_rows())
        .map(|i| lde_trace_frame.get_row(i))
        .map(|row| {
            row.iter()
                .zip(&lde_trace_merkle_trees)
                .map(|(evaluation, merkle_tree)| merkle_tree.get_proof(evaluation).unwrap())
                .collect()
        })
        .collect::<Vec<Vec<Proof<F, H>>>>();

    let composition_poly_evaluations = vec![
        args.composition_poly_even.evaluate(consistency_check_x),
        args.composition_poly_odd.evaluate(consistency_check_x),
    ];

    let deep_poly_evaluation = args.deep_composition_poly.evaluate(consistency_check_x);

    DeepConsistencyCheck {
        lde_trace_merkle_roots,
        lde_trace_merkle_proofs,
        lde_trace_frame,
        composition_poly_evaluations,
        deep_poly_evaluation,
    }
}
