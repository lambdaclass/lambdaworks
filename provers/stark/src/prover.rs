use std::marker::PhantomData;
#[cfg(feature = "instruments")]
use std::time::Instant;

use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_crypto::merkle_tree::traits::IsMerkleTreeBackend;
use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use lambdaworks_math::fft::{errors::FFTError, polynomial::FFTPoly};
use lambdaworks_math::traits::Serializable;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};
use log::info;

#[cfg(feature = "parallel")]
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::config::FriMerkleTreeBackend;
#[cfg(debug_assertions)]
use crate::debug::validate_trace;
use crate::fri::fri_commitment::FriLayer;
use crate::fri::{Fri, IsFri};
use crate::transcript::IsStarkTranscript;

use super::config::{BatchedMerkleTree, Commitment};
use super::constraints::evaluator::ConstraintEvaluator;
use super::domain::Domain;
use super::frame::Frame;
use super::fri::fri_decommit::FriDecommitment;
use super::grinding::generate_nonce_with_grinding;
use super::proof::options::ProofOptions;
use super::proof::stark::{DeepPolynomialOpenings, StarkProof};
use super::trace::TraceTable;
use super::traits::AIR;

#[derive(Debug)]
pub enum ProvingError {
    WrongParameter(String),
}

pub struct Round1<F, A>
where
    F: IsFFTField,
    A: AIR<Field = F>,
    FieldElement<F>: Serializable,
{
    pub(crate) trace_polys: Vec<Polynomial<FieldElement<F>>>,
    pub(crate) lde_trace: TraceTable<F>,
    pub(crate) lde_trace_merkle_trees: Vec<BatchedMerkleTree<F>>,
    pub(crate) lde_trace_merkle_roots: Vec<Commitment>,
    pub(crate) rap_challenges: A::RAPChallenges,
}

pub struct Round2<F>
where
    F: IsFFTField,
    FieldElement<F>: Serializable,
{
    pub(crate) composition_poly_parts: Vec<Polynomial<FieldElement<F>>>,
    pub(crate) lde_composition_poly_evaluations: Vec<Vec<FieldElement<F>>>,
    pub(crate) composition_poly_merkle_tree: BatchedMerkleTree<F>,
    pub(crate) composition_poly_root: Commitment,
}

pub struct Round3<F: IsFFTField> {
    trace_ood_evaluations: Vec<Vec<FieldElement<F>>>,
    composition_poly_parts_ood_evaluation: Vec<FieldElement<F>>,
}

pub struct Round4<F: IsFFTField> {
    fri_last_value: FieldElement<F>,
    fri_layers_merkle_roots: Vec<Commitment>,
    deep_poly_openings: Vec<DeepPolynomialOpenings<F>>,
    deep_poly_openings_sym: Vec<DeepPolynomialOpenings<F>>,
    query_list: Vec<FriDecommitment<F>>,
    nonce: u64,
}

pub trait IsStarkProver {
    type Field: IsFFTField;
    type MerkleTreeBackend: IsMerkleTreeBackend<Node = [u8; 32]> + Clone;

    fn fri_commit_phase(
        number_layers: usize,
        p_0: Polynomial<FieldElement<Self::Field>>,
        transcript: &mut impl IsStarkTranscript<Self::Field>,
        coset_offset: &FieldElement<Self::Field>,
        domain_size: usize,
    ) -> (
        FieldElement<Self::Field>,
        Vec<FriLayer<Self::Field, Self::MerkleTreeBackend>>,
    )
    where
        FieldElement<Self::Field>: Serializable;

    fn fri_query_phase(
        fri_layers: &Vec<FriLayer<Self::Field, Self::MerkleTreeBackend>>,
        iotas: &[usize],
    ) -> Vec<FriDecommitment<Self::Field>>
    where
        FieldElement<Self::Field>: Serializable;

    fn batch_commit(
        vectors: &[Vec<FieldElement<Self::Field>>],
    ) -> (BatchedMerkleTree<Self::Field>, Commitment)
    where
        FieldElement<Self::Field>: Serializable,
    {
        let tree = BatchedMerkleTree::<Self::Field>::build(vectors);
        let commitment = tree.root;
        (tree, commitment)
    }

    fn evaluate_polynomial_on_lde_domain(
        p: &Polynomial<FieldElement<Self::Field>>,
        blowup_factor: usize,
        domain_size: usize,
        offset: &FieldElement<Self::Field>,
    ) -> Result<Vec<FieldElement<Self::Field>>, FFTError>
    where
        Polynomial<FieldElement<Self::Field>>: FFTPoly<Self::Field>,
    {
        let evaluations = p.evaluate_offset_fft(blowup_factor, Some(domain_size), offset)?;
        let step = evaluations.len() / (domain_size * blowup_factor);
        match step {
            1 => Ok(evaluations),
            _ => Ok(evaluations.into_iter().step_by(step).collect()),
        }
    }

    fn apply_permutation(vector: &mut Vec<FieldElement<Self::Field>>, permutation: &[usize]) {
        assert_eq!(
            vector.len(),
            permutation.len(),
            "Vector and permutation must have the same length"
        );

        let mut temp = Vec::with_capacity(vector.len());
        for &index in permutation {
            temp.push(vector[index].clone());
        }

        vector.clear();
        vector.extend(temp);
    }

    #[allow(clippy::type_complexity)]
    fn interpolate_and_commit(
        trace: &TraceTable<Self::Field>,
        domain: &Domain<Self::Field>,
        transcript: &mut impl IsStarkTranscript<Self::Field>,
    ) -> (
        Vec<Polynomial<FieldElement<Self::Field>>>,
        Vec<Vec<FieldElement<Self::Field>>>,
        BatchedMerkleTree<Self::Field>,
        Commitment,
    )
    where
        FieldElement<Self::Field>: Serializable + Send + Sync,
    {
        let trace_polys = trace.compute_trace_polys();

        // Evaluate those polynomials t_j on the large domain D_LDE.
        let lde_trace_evaluations = Self::compute_lde_trace_evaluations(&trace_polys, domain);

        // Compute commitments [t_j].
        let lde_trace = TraceTable::new_from_cols(&lde_trace_evaluations);
        let (lde_trace_merkle_tree, lde_trace_merkle_root) = Self::batch_commit(&lde_trace.rows());

        // >>>> Send commitments: [tⱼ]
        transcript.append_bytes(&lde_trace_merkle_root);

        (
            trace_polys,
            lde_trace_evaluations,
            lde_trace_merkle_tree,
            lde_trace_merkle_root,
        )
    }

    fn compute_lde_trace_evaluations(
        trace_polys: &[Polynomial<FieldElement<Self::Field>>],
        domain: &Domain<Self::Field>,
    ) -> Vec<Vec<FieldElement<Self::Field>>>
    where
        FieldElement<Self::Field>: Send + Sync,
    {
        #[cfg(not(feature = "parallel"))]
        let trace_polys_iter = trace_polys.iter();
        #[cfg(feature = "parallel")]
        let trace_polys_iter = trace_polys.par_iter();

        trace_polys_iter
            .map(|poly| {
                Self::evaluate_polynomial_on_lde_domain(
                    poly,
                    domain.blowup_factor,
                    domain.interpolation_domain_size,
                    &domain.coset_offset,
                )
            })
            .collect::<Result<Vec<Vec<FieldElement<Self::Field>>>, FFTError>>()
            .unwrap()
    }

    fn round_1_randomized_air_with_preprocessing<A: AIR<Field = Self::Field>>(
        air: &A,
        main_trace: &TraceTable<Self::Field>,
        domain: &Domain<Self::Field>,
        transcript: &mut impl IsStarkTranscript<Self::Field>,
    ) -> Result<Round1<Self::Field, A>, ProvingError>
    where
        FieldElement<Self::Field>: Serializable + Send + Sync,
    {
        let (mut trace_polys, mut evaluations, main_merkle_tree, main_merkle_root) =
            Self::interpolate_and_commit(main_trace, domain, transcript);

        let rap_challenges = air.build_rap_challenges(transcript);

        let aux_trace = air.build_auxiliary_trace(main_trace, &rap_challenges);

        let mut lde_trace_merkle_trees = vec![main_merkle_tree];
        let mut lde_trace_merkle_roots = vec![main_merkle_root];
        if !aux_trace.is_empty() {
            // Check that this is valid for interpolation
            let (aux_trace_polys, aux_trace_polys_evaluations, aux_merkle_tree, aux_merkle_root) =
                Self::interpolate_and_commit(&aux_trace, domain, transcript);
            trace_polys.extend_from_slice(&aux_trace_polys);
            evaluations.extend_from_slice(&aux_trace_polys_evaluations);
            lde_trace_merkle_trees.push(aux_merkle_tree);
            lde_trace_merkle_roots.push(aux_merkle_root);
        }

        let lde_trace = TraceTable::new_from_cols(&evaluations);

        Ok(Round1 {
            trace_polys,
            lde_trace,
            lde_trace_merkle_roots,
            lde_trace_merkle_trees,
            rap_challenges,
        })
    }

    fn commit_composition_polynomial(
        lde_composition_poly_parts_evaluations: &[Vec<FieldElement<Self::Field>>],
    ) -> (BatchedMerkleTree<Self::Field>, Commitment)
    where
        FieldElement<Self::Field>: Serializable,
    {
        // TODO: Remove clones
        let number_of_parts = lde_composition_poly_parts_evaluations.len();
        let mut lde_composition_poly_evaluations = Vec::new();
        for i in 0..lde_composition_poly_parts_evaluations[0].len() {
            let mut row = Vec::new();
            for j in 0..number_of_parts {
                row.push(lde_composition_poly_parts_evaluations[j][i].clone());
            }
            lde_composition_poly_evaluations.push(row);
        }

        Self::batch_commit(&lde_composition_poly_evaluations)
    }

    fn round_2_compute_composition_polynomial<A>(
        air: &A,
        domain: &Domain<Self::Field>,
        round_1_result: &Round1<Self::Field, A>,
        transition_coefficients: &[FieldElement<Self::Field>],
        boundary_coefficients: &[FieldElement<Self::Field>],
    ) -> Round2<Self::Field>
    where
        A: AIR<Field = Self::Field> + Send + Sync,
        A::RAPChallenges: Send + Sync,
        FieldElement<Self::Field>: Serializable + Send + Sync,
    {
        // Create evaluation table
        let evaluator = ConstraintEvaluator::new(air, &round_1_result.rap_challenges);

        let constraint_evaluations = evaluator.evaluate(
            &round_1_result.lde_trace,
            domain,
            transition_coefficients,
            boundary_coefficients,
            &round_1_result.rap_challenges,
        );

        // Get the composition poly H
        let composition_poly =
            constraint_evaluations.compute_composition_poly(&domain.coset_offset);

        let number_of_parts = air.composition_poly_degree_bound() / air.trace_length();
        let composition_poly_parts = composition_poly.break_in_parts(number_of_parts);


        let lde_composition_poly_parts_evaluations: Vec<_> = composition_poly_parts
            .iter()
            .map(|part| {
                Self::evaluate_polynomial_on_lde_domain(
                    part,
                    domain.blowup_factor,
                    domain.interpolation_domain_size,
                    &domain.coset_offset,
                )
                .unwrap()
            })
            .collect();

        let (composition_poly_merkle_tree, composition_poly_root) =
            Self::commit_composition_polynomial(&lde_composition_poly_parts_evaluations);

        Round2 {
            lde_composition_poly_evaluations: lde_composition_poly_parts_evaluations,
            composition_poly_parts,
            composition_poly_merkle_tree,
            composition_poly_root,
        }
    }

    fn round_3_evaluate_polynomials_in_out_of_domain_element<A: AIR<Field = Self::Field>>(
        air: &A,
        domain: &Domain<Self::Field>,
        round_1_result: &Round1<Self::Field, A>,
        round_2_result: &Round2<Self::Field>,
        z: &FieldElement<Self::Field>,
    ) -> Round3<Self::Field>
    where
        FieldElement<Self::Field>: Serializable,
    {
        let z_power = z.pow(round_2_result.composition_poly_parts.len());

        // Evaluate H_i in z^N for all i, where N is the number of parts the composition poly was
        // broken into.
        let composition_poly_parts_ood_evaluation: Vec<_> = round_2_result
            .composition_poly_parts
            .iter()
            .map(|part| part.evaluate(&z_power))
            .collect();

        // Returns the Out of Domain Frame for the given trace polynomials, out of domain evaluation point (called `z` in the literature),
        // frame offsets given by the AIR and primitive root used for interpolating the trace polynomials.
        // An out of domain frame is nothing more than the evaluation of the trace polynomials in the points required by the
        // verifier to check the consistency between the trace and the composition polynomial.
        //
        // In the fibonacci example, the ood frame is simply the evaluations `[t(z), t(z * g), t(z * g^2)]`, where `t` is the trace
        // polynomial and `g` is the primitive root of unity used when interpolating `t`.
        let trace_ood_evaluations = Frame::get_trace_evaluations(
            &round_1_result.trace_polys,
            z,
            &air.context().transition_offsets,
            &domain.trace_primitive_root,
        );

        Round3 {
            trace_ood_evaluations,
            composition_poly_parts_ood_evaluation,
        }
    }

    fn round_4_compute_and_run_fri_on_the_deep_composition_polynomial<A: AIR<Field = Self::Field>>(
        air: &A,
        domain: &Domain<Self::Field>,
        round_1_result: &Round1<Self::Field, A>,
        round_2_result: &Round2<Self::Field>,
        round_3_result: &Round3<Self::Field>,
        z: &FieldElement<Self::Field>,
        transcript: &mut impl IsStarkTranscript<Self::Field>,
    ) -> Round4<Self::Field>
    where
        FieldElement<Self::Field>: Serializable + Send + Sync,
    {
        let coset_offset_u64 = air.context().proof_options.coset_offset;
        let coset_offset = FieldElement::<Self::Field>::from(coset_offset_u64);

        let gamma = transcript.sample_field_element();
        let n_terms_composition_poly = round_2_result.lde_composition_poly_evaluations.len();
        let n_terms_trace = air.context().transition_offsets.len() * air.context().trace_columns;

        // <<<< Receive challenges: 𝛾, 𝛾'
        let mut deep_composition_coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                .take(n_terms_composition_poly + n_terms_trace)
                .collect();

        let trace_poly_coeffients: Vec<_> = deep_composition_coefficients
            .drain(..n_terms_trace)
            .collect();

        // <<<< Receive challenges: 𝛾ⱼ, 𝛾ⱼ'
        let gammas = deep_composition_coefficients;

        // Compute p₀ (deep composition polynomial)
        let deep_composition_poly = Self::compute_deep_composition_poly(
            air,
            &round_1_result.trace_polys,
            round_2_result,
            round_3_result,
            z,
            &domain.trace_primitive_root,
            &gammas,
            &trace_poly_coeffients,
        );

        let domain_size = domain.lde_roots_of_unity_coset.len();

        // FRI commit and query phases
        let (fri_last_value, fri_layers) = Self::fri_commit_phase(
            domain.root_order as usize,
            deep_composition_poly,
            transcript,
            &coset_offset,
            domain_size,
        );

        // grinding: generate nonce and append it to the transcript
        let security_bits = air.context().proof_options.grinding_factor;
        let mut nonce = 0;
        if security_bits > 0 {
            let transcript_challenge = transcript.state();
            nonce = generate_nonce_with_grinding(&transcript_challenge, security_bits)
                .expect("nonce not found");
            transcript.append_bytes(&nonce.to_be_bytes());
        }

        let number_of_queries = air.options().fri_number_of_queries;
        let iotas = Self::sample_query_indexes(number_of_queries, &domain, transcript);
        let query_list = Self::fri_query_phase(&fri_layers, &iotas);

        let fri_layers_merkle_roots: Vec<_> = fri_layers
            .iter()
            .map(|layer| layer.merkle_tree.root)
            .collect();

        let (deep_poly_openings, deep_poly_openings_sym) =
            Self::open_deep_composition_poly(domain, round_1_result, round_2_result, &iotas);

        Round4 {
            fri_last_value,
            fri_layers_merkle_roots,
            deep_poly_openings,
            deep_poly_openings_sym,
            query_list,
            nonce,
        }
    }

    fn sample_query_indexes(
        number_of_queries: usize,
        domain: &Domain<Self::Field>,
        transcript: &mut impl IsStarkTranscript<Self::Field>,
    ) -> Vec<usize> {
        (0..number_of_queries)
            .map(|_| (transcript.sample_u64(domain.lde_roots_of_unity_coset.len() as u64)) as usize)
            .collect::<Vec<usize>>()
    }

    /// Returns the DEEP composition polynomial that the prover then commits to using
    /// FRI. This polynomial is a linear combination of the trace polynomial and the
    /// composition polynomial, with coefficients sampled by the verifier (i.e. using Fiat-Shamir).
    #[allow(clippy::too_many_arguments)]
    fn compute_deep_composition_poly<A>(
        air: &A,
        trace_polys: &[Polynomial<FieldElement<Self::Field>>],
        round_2_result: &Round2<Self::Field>,
        round_3_result: &Round3<Self::Field>,
        z: &FieldElement<Self::Field>,
        primitive_root: &FieldElement<Self::Field>,
        composition_poly_gammas: &[FieldElement<Self::Field>],
        trace_terms_gammas: &[FieldElement<Self::Field>],
    ) -> Polynomial<FieldElement<Self::Field>>
    where
        A: AIR<Field = Self::Field>,
        FieldElement<Self::Field>: Serializable + Send + Sync,
    {
        let domain = Domain::new(air);
        let z_power = z.pow(round_2_result.composition_poly_parts.len());

        // ∑ᵢ 𝛾ᵢ ( Hᵢ − Hᵢ(z^N) ) / ( X − z^N )
        let mut h_terms = Polynomial::zero();
        for (i, part) in round_2_result.composition_poly_parts.iter().enumerate() {
            // h_i_eval is the evaluation of the i-th part of the composition polynomial at z^N,
            // where N is the number of parts of the composition polynomial.
            let h_i_eval = &round_3_result.composition_poly_parts_ood_evaluation[i];
            let h_i_term = &composition_poly_gammas[i] * (part - h_i_eval);
            h_terms = h_terms + h_i_term;
        }
        assert_eq!(h_terms.evaluate(&z_power), FieldElement::zero());
        h_terms.ruffini_division_inplace(&z_power);

        // Get trace evaluations needed for the trace terms of the deep composition polynomial
        let transition_offsets = &air.context().transition_offsets;
        let trace_frame_evaluations = &round_3_result.trace_ood_evaluations;

        // Compute the sum of all the trace terms of the deep composition polynomial.
        // There is one term for every trace polynomial and for every row in the frame.
        // ∑ ⱼₖ [ 𝛾ₖ ( tⱼ − tⱼ(z) ) / ( X − zgᵏ )]

        // @@@ this could be const
        let trace_frame_length = trace_frame_evaluations.len();

        #[cfg(feature = "parallel")]
        let trace_terms = trace_polys
            .par_iter()
            .enumerate()
            .fold(
                || Polynomial::zero(),
                |trace_terms, (i, t_j)| {
                    compute_trace_term(
                        &trace_terms,
                        (i, t_j),
                        trace_frame_length,
                        trace_terms_gammas,
                        trace_frame_evaluations,
                        transition_offsets,
                        (z, primitive_root),
                    )
                },
            )
            .reduce(|| Polynomial::zero(), |a, b| a + b);

        #[cfg(not(feature = "parallel"))]
        let trace_terms =
            trace_polys
                .iter()
                .enumerate()
                .fold(Polynomial::zero(), |trace_terms, (i, t_j)| {
                    Self::compute_trace_term(
                        &trace_terms,
                        (i, t_j),
                        trace_frame_length,
                        trace_terms_gammas,
                        trace_frame_evaluations,
                        transition_offsets,
                        (z, primitive_root),
                    )
                });

        h_terms + trace_terms
    }

    fn compute_trace_term(
        trace_terms: &Polynomial<FieldElement<Self::Field>>,
        (i, t_j): (usize, &Polynomial<FieldElement<Self::Field>>),
        trace_frame_length: usize,
        trace_terms_gammas: &[FieldElement<Self::Field>],
        trace_frame_evaluations: &[Vec<FieldElement<Self::Field>>],
        transition_offsets: &[usize],
        (z, primitive_root): (&FieldElement<Self::Field>, &FieldElement<Self::Field>),
    ) -> Polynomial<FieldElement<Self::Field>>
    where
        FieldElement<Self::Field>: Serializable + Send + Sync,
    {
        let i_times_trace_frame_evaluation = i * trace_frame_length;
        let iter_trace_gammas = trace_terms_gammas
            .iter()
            .skip(i_times_trace_frame_evaluation);
        let trace_int = trace_frame_evaluations
            .iter()
            .zip(transition_offsets)
            .zip(iter_trace_gammas)
            .fold(
                Polynomial::zero(),
                |trace_agg, ((eval, offset), trace_gamma)| {
                    // @@@ we can avoid this clone
                    let t_j_z = &eval[i];
                    // @@@ this can be pre-computed
                    let z_shifted = z * primitive_root.pow(*offset);
                    let mut poly = t_j - t_j_z;
                    poly.ruffini_division_inplace(&z_shifted);
                    trace_agg + poly * trace_gamma
                },
            );

        trace_terms + trace_int
    }

    fn open_composition_poly(
        _domain: &Domain<Self::Field>,
        composition_poly_merkle_tree: &BatchedMerkleTree<Self::Field>,
        lde_composition_poly_evaluations: &[Vec<FieldElement<Self::Field>>],
        index: usize,
    ) -> (Proof<Commitment>, Vec<FieldElement<Self::Field>>)
    where
        FieldElement<Self::Field>: Serializable,
    {
        let proof = composition_poly_merkle_tree
            .get_proof_by_pos(index)
            .unwrap();

        // Hi openings
        let lde_composition_poly_parts_evaluation: Vec<_> = lde_composition_poly_evaluations
            .iter()
            .map(|part| part[index].clone())
            .collect();

        (proof, lde_composition_poly_parts_evaluation)
    }

    fn open_trace_polys(
        _domain: &Domain<Self::Field>,
        lde_trace_merkle_trees: &Vec<BatchedMerkleTree<Self::Field>>,
        lde_trace: &TraceTable<Self::Field>,
        index: usize,
    ) -> (Vec<Proof<Commitment>>, Vec<FieldElement<Self::Field>>)
    where
        FieldElement<Self::Field>: Serializable,
    {
        let lde_trace_evaluations = lde_trace.get_row(index).to_vec();

        // Trace polynomials openings
        #[cfg(feature = "parallel")]
        let merkle_trees_iter = lde_trace_merkle_trees.par_iter();
        #[cfg(not(feature = "parallel"))]
        let merkle_trees_iter = lde_trace_merkle_trees.iter();

        let lde_trace_merkle_proofs: Vec<Proof<[u8; 32]>> = merkle_trees_iter
            .map(|tree| tree.get_proof_by_pos(index).unwrap())
            .collect();

        (lde_trace_merkle_proofs, lde_trace_evaluations)
    }

    /// Open the deep composition polynomial on a list of indexes
    /// and their symmetric elements.
    fn open_deep_composition_poly<A: AIR<Field = Self::Field>>(
        domain: &Domain<Self::Field>,
        round_1_result: &Round1<Self::Field, A>,
        round_2_result: &Round2<Self::Field>,
        indexes_to_open: &[usize], // list of iotas
    ) -> (
        Vec<DeepPolynomialOpenings<Self::Field>>,
        Vec<DeepPolynomialOpenings<Self::Field>>,
    )
    where
        FieldElement<Self::Field>: Serializable,
    {
        let indexes_symmetric: Vec<_> = indexes_to_open
            .iter()
            .map(|iota| iota + domain.lde_roots_of_unity_coset.len() / 2)
            .collect();

        let all_indexes = vec![indexes_symmetric, indexes_to_open.to_vec()];
        let mut openings: Vec<_> =
            all_indexes
                .iter()
                .map(|indexes| {
                    indexes.iter().map(|index_to_open| {
                let index = index_to_open % domain.lde_roots_of_unity_coset.len();

                let (lde_composition_poly_proof, lde_composition_poly_parts_evaluation) =
                    Self::open_composition_poly(
                        domain,
                        &round_2_result.composition_poly_merkle_tree,
                        &round_2_result.lde_composition_poly_evaluations,
                        index,
                    );

                let (lde_trace_merkle_proofs, lde_trace_evaluations) = Self::open_trace_polys(
                    domain,
                    &round_1_result.lde_trace_merkle_trees,
                    &round_1_result.lde_trace,
                    index,
                );

                DeepPolynomialOpenings {
                    lde_composition_poly_proof,
                    lde_composition_poly_parts_evaluation,
                    lde_trace_merkle_proofs,
                    lde_trace_evaluations,
                }
            }).collect()
                })
                .collect();
        (openings.pop().unwrap(), openings.pop().unwrap())
    }

    // FIXME remove unwrap() calls and return errors
    fn prove<A>(
        main_trace: &TraceTable<Self::Field>,
        pub_inputs: &A::PublicInputs,
        proof_options: &ProofOptions,
        mut transcript: impl IsStarkTranscript<Self::Field>,
    ) -> Result<StarkProof<Self::Field>, ProvingError>
    where
        A: AIR<Field = Self::Field> + Send + Sync,
        A::RAPChallenges: Send + Sync,
        FieldElement<Self::Field>: Serializable + Send + Sync,
    {
        info!("Started proof generation...");
        #[cfg(feature = "instruments")]
        println!("- Started round 0: Air Initialization");
        #[cfg(feature = "instruments")]
        let timer0 = Instant::now();

        let air = A::new(main_trace.n_rows(), pub_inputs, proof_options);
        let domain = Domain::new(&air);

        #[cfg(feature = "instruments")]
        let elapsed0 = timer0.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed0);

        // ===================================
        // ==========|   Round 1   |==========
        // ===================================

        #[cfg(feature = "instruments")]
        println!("- Started round 1: RAP");
        #[cfg(feature = "instruments")]
        let timer1 = Instant::now();

        let round_1_result = Self::round_1_randomized_air_with_preprocessing::<A>(
            &air,
            main_trace,
            &domain,
            &mut transcript,
        )?;

        #[cfg(debug_assertions)]
        validate_trace(
            &air,
            &round_1_result.trace_polys,
            &domain,
            &round_1_result.rap_challenges,
        );

        #[cfg(feature = "instruments")]
        let elapsed1 = timer1.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed1);

        // ===================================
        // ==========|   Round 2   |==========
        // ===================================

        #[cfg(feature = "instruments")]
        println!("- Started round 2: Compute composition polynomial");
        #[cfg(feature = "instruments")]
        let timer2 = Instant::now();

        // <<<< Receive challenge: 𝛽
        let beta = transcript.sample_field_element();
        let num_boundary_constraints = air
            .boundary_constraints(&round_1_result.rap_challenges)
            .constraints
            .len();

        let num_transition_constraints = air.context().num_transition_constraints;

        let mut coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &beta))
                .take(num_boundary_constraints + num_transition_constraints)
                .collect();

        let transition_coefficients: Vec<_> =
            coefficients.drain(..num_transition_constraints).collect();
        let boundary_coefficients = coefficients;

        let round_2_result = Self::round_2_compute_composition_polynomial(
            &air,
            &domain,
            &round_1_result,
            &transition_coefficients,
            &boundary_coefficients,
        );

        // >>>> Send commitments: [H₁], [H₂]
        transcript.append_bytes(&round_2_result.composition_poly_root);

        #[cfg(feature = "instruments")]
        let elapsed2 = timer2.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed2);

        // ===================================
        // ==========|   Round 3   |==========
        // ===================================

        #[cfg(feature = "instruments")]
        println!("- Started round 3: Evaluate polynomial in out of domain elements");
        #[cfg(feature = "instruments")]
        let timer3 = Instant::now();

        // <<<< Receive challenge: z
        let z = transcript.sample_z_ood(
            &domain.lde_roots_of_unity_coset,
            &domain.trace_roots_of_unity,
        );

        let round_3_result = Self::round_3_evaluate_polynomials_in_out_of_domain_element(
            &air,
            &domain,
            &round_1_result,
            &round_2_result,
            &z,
        );

        // >>>> Send values: tⱼ(zgᵏ)
        for i in 0..round_3_result.trace_ood_evaluations[0].len() {
            for j in 0..round_3_result.trace_ood_evaluations.len() {
                transcript.append_field_element(&round_3_result.trace_ood_evaluations[j][i]);
            }
        }

        // >>>> Send values: Hᵢ(z^N)
        for element in round_3_result.composition_poly_parts_ood_evaluation.iter() {
            transcript.append_field_element(element);
        }

        #[cfg(feature = "instruments")]
        let elapsed3 = timer3.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed3);

        // ===================================
        // ==========|   Round 4   |==========
        // ===================================

        #[cfg(feature = "instruments")]
        println!("- Started round 4: FRI");
        #[cfg(feature = "instruments")]
        let timer4 = Instant::now();

        // Part of this round is running FRI, which is an interactive
        // protocol on its own. Therefore we pass it the transcript
        // to simulate the interactions with the verifier.
        let round_4_result = Self::round_4_compute_and_run_fri_on_the_deep_composition_polynomial(
            &air,
            &domain,
            &round_1_result,
            &round_2_result,
            &round_3_result,
            &z,
            &mut transcript,
        );

        #[cfg(feature = "instruments")]
        let elapsed4 = timer4.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed4);

        #[cfg(feature = "instruments")]
        {
            let total_time = elapsed1 + elapsed2 + elapsed3 + elapsed4;
            println!(
                " Fraction of proving time per round: {:.4} {:.4} {:.4} {:.4} {:.4}",
                elapsed0.as_nanos() as f64 / total_time.as_nanos() as f64,
                elapsed1.as_nanos() as f64 / total_time.as_nanos() as f64,
                elapsed2.as_nanos() as f64 / total_time.as_nanos() as f64,
                elapsed3.as_nanos() as f64 / total_time.as_nanos() as f64,
                elapsed4.as_nanos() as f64 / total_time.as_nanos() as f64
            );
        }

        info!("End proof generation");

        let trace_ood_frame_evaluations = Frame::new(
            round_3_result
                .trace_ood_evaluations
                .into_iter()
                .flatten()
                .collect(),
            round_1_result.trace_polys.len(),
        );

        Ok(StarkProof {
            // [tⱼ]
            lde_trace_merkle_roots: round_1_result.lde_trace_merkle_roots,
            // tⱼ(zgᵏ)
            trace_ood_frame_evaluations,
            // [H₁] and [H₂]
            composition_poly_root: round_2_result.composition_poly_root,
            // Hᵢ(z^N)
            composition_poly_parts_ood_evaluation: round_3_result
                .composition_poly_parts_ood_evaluation,
            // [pₖ]
            fri_layers_merkle_roots: round_4_result.fri_layers_merkle_roots,
            // pₙ
            fri_last_value: round_4_result.fri_last_value,
            // Open(p₀(D₀), 𝜐ₛ), Open(pₖ(Dₖ), −𝜐ₛ^(2ᵏ))
            query_list: round_4_result.query_list,
            // Open(H₁(D_LDE, 𝜐₀), Open(H₂(D_LDE, 𝜐₀), Open(tⱼ(D_LDE), 𝜐₀)
            deep_poly_openings: round_4_result.deep_poly_openings,
            // Open(H₁(D_LDE, 𝜐₀), Open(H₂(D_LDE, 𝜐₀), Open(tⱼ(D_LDE), 𝜐₀)
            deep_poly_openings_sym: round_4_result.deep_poly_openings_sym,
            // nonce obtained from grinding
            nonce: round_4_result.nonce,

            trace_length: air.trace_length(),
        })
    }
}

pub struct Prover<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F> IsStarkProver for Prover<F>
where
    F: IsFFTField,
    FieldElement<F>: Serializable,
{
    type Field = F;
    type MerkleTreeBackend = FriMerkleTreeBackend<F>;

    fn fri_commit_phase(
        number_layers: usize,
        p_0: Polynomial<FieldElement<F>>,
        transcript: &mut impl IsStarkTranscript<F>,
        coset_offset: &FieldElement<F>,
        domain_size: usize,
    ) -> (FieldElement<F>, Vec<FriLayer<F, Self::MerkleTreeBackend>>)
    where
        F: IsFFTField,
        FieldElement<F>: Serializable,
    {
        Fri::fri_commit_phase(number_layers, p_0, transcript, &coset_offset, domain_size)
    }

    fn fri_query_phase(
        fri_layers: &Vec<FriLayer<Self::Field, Self::MerkleTreeBackend>>,
        iotas: &[usize],
    ) -> Vec<FriDecommitment<Self::Field>>
    where
        FieldElement<Self::Field>: Serializable,
    {
        Fri::fri_query_phase(fri_layers, iotas)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        examples::simple_fibonacci::{self, FibonacciPublicInputs},
        proof::options::ProofOptions,
        Felt252,
    };

    use super::*;
    use lambdaworks_math::{
        field::{
            element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
            traits::IsFFTField,
        },
        polynomial::Polynomial,
    };

    #[test]
    fn test_domain_constructor() {
        let pub_inputs = FibonacciPublicInputs {
            a0: Felt252::one(),
            a1: Felt252::one(),
        };
        let trace = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 8);
        let trace_length = trace.n_rows();
        let coset_offset = 3;
        let blowup_factor: usize = 2;
        let grinding_factor = 20;

        let proof_options = ProofOptions {
            blowup_factor: blowup_factor as u8,
            fri_number_of_queries: 1,
            coset_offset,
            grinding_factor,
        };

        let domain = Domain::new(&simple_fibonacci::FibonacciAIR::new(
            trace_length,
            &pub_inputs,
            &proof_options,
        ));
        assert_eq!(domain.blowup_factor, 2);
        assert_eq!(domain.interpolation_domain_size, trace_length);
        assert_eq!(domain.root_order, trace_length.trailing_zeros());
        assert_eq!(
            domain.lde_root_order,
            (trace_length * blowup_factor).trailing_zeros()
        );
        assert_eq!(domain.coset_offset, FieldElement::from(coset_offset));

        let primitive_root = Stark252PrimeField::get_primitive_root_of_unity(
            (trace_length * blowup_factor).trailing_zeros() as u64,
        )
        .unwrap();

        assert_eq!(
            domain.trace_primitive_root,
            primitive_root.pow(blowup_factor)
        );
        for i in 0..(trace_length * blowup_factor) {
            assert_eq!(
                domain.lde_roots_of_unity_coset[i],
                FieldElement::from(coset_offset) * primitive_root.pow(i)
            );
        }
    }

    #[test]
    fn test_evaluate_polynomial_on_lde_domain_on_trace_polys() {
        let trace = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 8);
        let trace_length = trace.n_rows();
        let trace_polys = trace.compute_trace_polys();
        let coset_offset = Felt252::from(3);
        let blowup_factor: usize = 2;
        let domain_size = 8;

        let primitive_root = Stark252PrimeField::get_primitive_root_of_unity(
            (trace_length * blowup_factor).trailing_zeros() as u64,
        )
        .unwrap();

        for poly in trace_polys.iter() {
            let lde_evaluation = Prover::evaluate_polynomial_on_lde_domain(
                poly,
                blowup_factor,
                domain_size,
                &coset_offset,
            )
            .unwrap();
            assert_eq!(lde_evaluation.len(), trace_length * blowup_factor);
            for (i, evaluation) in lde_evaluation.iter().enumerate() {
                assert_eq!(
                    *evaluation,
                    poly.evaluate(&(coset_offset * primitive_root.pow(i)))
                );
            }
        }
    }

    #[test]
    fn test_evaluate_polynomial_on_lde_domain_edge_case() {
        let poly = Polynomial::new_monomial(Felt252::one(), 8);
        let blowup_factor: usize = 4;
        let domain_size: usize = 8;
        let offset = Felt252::from(3);
        let evaluations =
            Prover::evaluate_polynomial_on_lde_domain(&poly, blowup_factor, domain_size, &offset)
                .unwrap();
        assert_eq!(evaluations.len(), domain_size * blowup_factor);

        let primitive_root: Felt252 = Stark252PrimeField::get_primitive_root_of_unity(
            (domain_size * blowup_factor).trailing_zeros() as u64,
        )
        .unwrap();
        for (i, eval) in evaluations.iter().enumerate() {
            assert_eq!(*eval, poly.evaluate(&(offset * primitive_root.pow(i))));
        }
    }
}
