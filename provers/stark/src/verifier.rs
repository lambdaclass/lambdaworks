#[cfg(feature = "instruments")]
use std::time::Instant;

use lambdaworks_crypto::merkle_tree::proof::Proof;
//use itertools::multizip;
#[cfg(not(feature = "test_fiat_shamir"))]
use log::error;

use crate::{
    config::Commitment, proof::stark::DeepPolynomialOpening, transcript::IsStarkTranscript,
};
use lambdaworks_math::{
    fft::cpu::bit_reversing::reverse_index,
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        traits::IsFFTField,
    },
    traits::Serializable,
};

use super::{
    config::BatchedMerkleTreeBackend,
    domain::Domain,
    fri::fri_decommit::FriDecommitment,
    grinding,
    proof::{options::ProofOptions, stark::StarkProof},
    traits::AIR,
};

pub struct Verifier {}

impl IsStarkVerifier for Verifier {
    type Field = Stark252PrimeField;
}

pub struct Challenges<F, A>
where
    F: IsFFTField,
    A: AIR<Field = F>,
{
    pub z: FieldElement<F>,
    pub boundary_coeffs: Vec<FieldElement<F>>,
    pub transition_coeffs: Vec<FieldElement<F>>,
    pub trace_term_coeffs: Vec<Vec<FieldElement<F>>>,
    pub gammas: Vec<FieldElement<F>>,
    pub zetas: Vec<FieldElement<F>>,
    pub iotas: Vec<usize>,
    pub rap_challenges: A::RAPChallenges,
    pub grinding_seed: [u8; 32],
}

pub type DeepPolynomialEvaluations<F> = (Vec<FieldElement<F>>, Vec<FieldElement<F>>);

pub trait IsStarkVerifier {
    type Field: IsFFTField;

    fn sample_query_indexes(
        number_of_queries: usize,
        domain: &Domain<Self::Field>,
        transcript: &mut impl IsStarkTranscript<Self::Field>,
    ) -> Vec<usize> {
        let domain_size = domain.lde_roots_of_unity_coset.len() as u64;
        (0..number_of_queries)
            .map(|_| (transcript.sample_u64(domain_size >> 1)) as usize)
            .collect::<Vec<usize>>()
    }

    fn step_1_replay_rounds_and_recover_challenges<A>(
        air: &A,
        proof: &StarkProof<Self::Field>,
        domain: &Domain<Self::Field>,
        transcript: &mut impl IsStarkTranscript<Self::Field>,
    ) -> Challenges<Self::Field, A>
    where
        FieldElement<Self::Field>: Serializable,
        A: AIR<Field = Self::Field>,
    {
        // ===================================
        // ==========|   Round 1   |==========
        // ===================================

        // <<<< Receive commitments:[tⱼ]
        transcript.append_bytes(&proof.lde_trace_merkle_roots[0]);

        let rap_challenges = air.build_rap_challenges(transcript);

        if let Some(root) = proof.lde_trace_merkle_roots.get(1) {
            transcript.append_bytes(root);
        }

        // ===================================
        // ==========|   Round 2   |==========
        // ===================================

        // <<<< Receive challenge: 𝛽
        let beta = transcript.sample_field_element();
        let num_boundary_constraints = air.boundary_constraints(&rap_challenges).constraints.len();

        let num_transition_constraints = air.context().num_transition_constraints;

        let mut coefficients: Vec<_> = (0..num_boundary_constraints + num_transition_constraints)
            .map(|i| beta.pow(i))
            .collect();

        let transition_coeffs: Vec<_> = coefficients.drain(..num_transition_constraints).collect();
        let boundary_coeffs = coefficients;

        // <<<< Receive commitments: [H₁], [H₂]
        transcript.append_bytes(&proof.composition_poly_root);

        // ===================================
        // ==========|   Round 3   |==========
        // ===================================

        // >>>> Send challenge: z
        let z = transcript.sample_z_ood(
            &domain.lde_roots_of_unity_coset,
            &domain.trace_roots_of_unity,
        );

        // <<<< Receive values: tⱼ(zgᵏ)
        for i in 0..proof.trace_ood_evaluations.width {
            for j in 0..proof.trace_ood_evaluations.height {
                transcript.append_field_element(&proof.trace_ood_evaluations.get_row(j)[i]);
            }
        }
        // <<<< Receive value: Hᵢ(z^N)
        for element in proof.composition_poly_parts_ood_evaluation.iter() {
            transcript.append_field_element(element);
        }

        // ===================================
        // ==========|   Round 4   |==========
        // ===================================

        let n_terms_composition_poly = proof.composition_poly_parts_ood_evaluation.len();
        let n_terms_trace = air.context().transition_offsets.len() * air.context().trace_columns;
        let gamma = transcript.sample_field_element();

        // <<<< Receive challenges: 𝛾, 𝛾'
        let mut deep_composition_coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                .take(n_terms_composition_poly + n_terms_trace)
                .collect();

        let trace_term_coeffs: Vec<_> = deep_composition_coefficients
            .drain(..n_terms_trace)
            .collect::<Vec<_>>()
            .chunks(air.context().transition_offsets.len())
            .map(|chunk| chunk.to_vec())
            .collect();

        // <<<< Receive challenges: 𝛾ⱼ, 𝛾ⱼ'
        let gammas = deep_composition_coefficients;

        // FRI commit phase
        let merkle_roots = &proof.fri_layers_merkle_roots;
        let mut zetas = merkle_roots
            .iter()
            .map(|root| {
                // >>>> Send challenge 𝜁ₖ
                let element = transcript.sample_field_element();
                // <<<< Receive commitment: [pₖ] (the first one is [p₀])
                transcript.append_bytes(root);
                element
            })
            .collect::<Vec<FieldElement<Self::Field>>>();

        // >>>> Send challenge 𝜁ₙ₋₁
        zetas.push(transcript.sample_field_element());

        // <<<< Receive value: pₙ
        transcript.append_field_element(&proof.fri_last_value);

        // Receive grinding value
        let security_bits = air.context().proof_options.grinding_factor;
        let mut grinding_seed = [0u8; 32];
        if security_bits > 0 {
            if let Some(nonce_value) = proof.nonce {
                grinding_seed = transcript.state();
                transcript.append_bytes(&nonce_value.to_be_bytes());
            }
        }

        // FRI query phase
        // <<<< Send challenges 𝜄ₛ (iota_s)
        let number_of_queries = air.options().fri_number_of_queries;
        let iotas = Self::sample_query_indexes(number_of_queries, domain, transcript);

        Challenges {
            z,
            boundary_coeffs,
            transition_coeffs,
            trace_term_coeffs,
            gammas,
            zetas,
            iotas,
            rap_challenges,
            grinding_seed,
        }
    }

    fn step_2_verify_claimed_composition_polynomial<A>(
        air: &A,
        proof: &StarkProof<Self::Field>,
        domain: &Domain<Self::Field>,
        challenges: &Challenges<Self::Field, A>,
    ) -> bool
    where
        A: AIR<Field = Self::Field>,
    {
        let boundary_constraints = air.boundary_constraints(&challenges.rap_challenges);

        let trace_length = air.trace_length();
        let number_of_b_constraints = boundary_constraints.constraints.len();

        #[allow(clippy::type_complexity)]
        let (boundary_c_i_evaluations_num, mut boundary_c_i_evaluations_den): (
            Vec<FieldElement<Self::Field>>,
            Vec<FieldElement<Self::Field>>,
        ) = (0..number_of_b_constraints)
            .map(|index| {
                let step = boundary_constraints.constraints[index].step;
                let point = &domain.trace_primitive_root.pow(step as u64);
                let trace_idx = boundary_constraints.constraints[index].col;
                let trace_evaluation = &proof.trace_ood_evaluations.get_row(0)[trace_idx];
                let boundary_zerofier_challenges_z_den = &challenges.z - point;

                let boundary_quotient_ood_evaluation_num =
                    trace_evaluation - &boundary_constraints.constraints[index].value;

                (
                    boundary_quotient_ood_evaluation_num,
                    boundary_zerofier_challenges_z_den,
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .unzip();

        FieldElement::inplace_batch_inverse(&mut boundary_c_i_evaluations_den).unwrap();

        let boundary_quotient_ood_evaluation: FieldElement<Self::Field> =
            boundary_c_i_evaluations_num
                .iter()
                .zip(&boundary_c_i_evaluations_den)
                .zip(&challenges.boundary_coeffs)
                .map(|((num, den), beta)| num * den * beta)
                .fold(FieldElement::<Self::Field>::zero(), |acc, x| acc + x);

        let periodic_values = air
            .get_periodic_column_polynomials()
            .iter()
            .map(|poly| poly.evaluate(&challenges.z))
            .collect::<Vec<FieldElement<Self::Field>>>();

        let transition_ood_frame_evaluations = air.compute_transition(
            &(proof.trace_ood_evaluations).into_frame(A::STEP_SIZE),
            &periodic_values,
            &challenges.rap_challenges,
        );

        let denominator = (&challenges.z.pow(trace_length) - FieldElement::<Self::Field>::one())
            .inv()
            .unwrap();

        let exemption = air
            .transition_exemptions_verifier(
                domain.trace_roots_of_unity.iter().last().expect("has last"),
            )
            .iter()
            .map(|poly| poly.evaluate(&challenges.z))
            .collect::<Vec<FieldElement<Self::Field>>>();

        let unity = &FieldElement::one();
        let transition_c_i_evaluations_sum = transition_ood_frame_evaluations
            .iter()
            .zip(&air.context().transition_exemptions)
            .zip(&challenges.transition_coeffs)
            .fold(FieldElement::zero(), |acc, ((eval, except), beta)| {
                let except = except
                    .checked_sub(1)
                    .map(|i| &exemption[i])
                    .unwrap_or(unity);
                acc + &denominator * eval * beta * except
            });

        let composition_poly_ood_evaluation =
            &boundary_quotient_ood_evaluation + transition_c_i_evaluations_sum;

        let composition_poly_claimed_ood_evaluation = proof
            .composition_poly_parts_ood_evaluation
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, coeff| {
                acc * &challenges.z + coeff
            });

        composition_poly_claimed_ood_evaluation == composition_poly_ood_evaluation
    }

    fn step_3_verify_fri<A>(
        proof: &StarkProof<Self::Field>,
        domain: &Domain<Self::Field>,
        challenges: &Challenges<Self::Field, A>,
    ) -> bool
    where
        FieldElement<Self::Field>: Serializable,
        A: AIR<Field = Self::Field>,
    {
        let (deep_poly_evaluations, deep_poly_evaluations_sym) =
            Self::reconstruct_deep_composition_poly_evaluations_for_all_queries(
                challenges, domain, proof,
            );

        // verify FRI
        let mut evaluation_point_inverse = challenges
            .iotas
            .iter()
            .map(|iota| Self::query_challenge_to_evaluation_point(*iota, domain))
            .collect::<Vec<FieldElement<Self::Field>>>();
        FieldElement::inplace_batch_inverse(&mut evaluation_point_inverse).unwrap();
        proof
            .query_list
            .iter()
            .zip(&challenges.iotas)
            .zip(evaluation_point_inverse)
            .enumerate()
            .fold(true, |mut result, (i, ((proof_s, iota_s), eval))| {
                // this is done in constant time
                result &= Self::verify_query_and_sym_openings(
                    proof,
                    &challenges.zetas,
                    *iota_s,
                    proof_s,
                    eval,
                    &deep_poly_evaluations[i],
                    &deep_poly_evaluations_sym[i],
                );
                result
            })
    }

    fn query_challenge_to_evaluation_point(
        iota: usize,
        domain: &Domain<Self::Field>,
    ) -> FieldElement<Self::Field> {
        domain.lde_roots_of_unity_coset
            [reverse_index(iota * 2, domain.lde_roots_of_unity_coset.len() as u64)]
        .clone()
    }

    fn query_challenge_to_evaluation_point_sym(
        iota: usize,
        domain: &Domain<Self::Field>,
    ) -> FieldElement<Self::Field> {
        domain.lde_roots_of_unity_coset
            [reverse_index(iota * 2 + 1, domain.lde_roots_of_unity_coset.len() as u64)]
        .clone()
    }

    fn verify_opening(
        proof: &Proof<Commitment>,
        root: &Commitment,
        index: usize,
        value: &[FieldElement<Self::Field>],
    ) -> bool
    where
        FieldElement<Self::Field>: Serializable,
    {
        proof.verify::<BatchedMerkleTreeBackend<Self::Field>>(root, index, &value.to_owned())
    }

    /// Verify opening Open(tⱼ(D_LDE), 𝜐) and Open(tⱼ(D_LDE), -𝜐) for all trace polynomials tⱼ,
    /// where 𝜐 and -𝜐 are the elements corresponding to the index challenge `iota`.
    fn verify_trace_openings(
        num_main_columns: usize,
        proof: &StarkProof<Self::Field>,
        deep_poly_openings: &DeepPolynomialOpening<Self::Field>,
        deep_poly_openings_sym: &DeepPolynomialOpening<Self::Field>,
        iota: usize,
    ) -> bool
    where
        FieldElement<Self::Field>: Serializable,
    {
        let lde_trace_evaluations = vec![
            deep_poly_openings.lde_trace_evaluations[..num_main_columns].to_vec(),
            deep_poly_openings.lde_trace_evaluations[num_main_columns..].to_vec(),
        ];

        let index = iota * 2;
        let openings_are_valid = proof
            .lde_trace_merkle_roots
            .iter()
            .zip(&deep_poly_openings.lde_trace_merkle_proofs)
            .zip(lde_trace_evaluations)
            .fold(true, |acc, ((merkle_root, merkle_proof), evaluation)| {
                acc & Self::verify_opening(merkle_proof, merkle_root, index, &evaluation)
            });

        let lde_trace_evaluations_sym = vec![
            deep_poly_openings_sym.lde_trace_evaluations[..num_main_columns].to_vec(),
            deep_poly_openings_sym.lde_trace_evaluations[num_main_columns..].to_vec(),
        ];

        let index_sym = iota * 2 + 1;
        let openings_sym_are_valid = proof
            .lde_trace_merkle_roots
            .iter()
            .zip(&deep_poly_openings_sym.lde_trace_merkle_proofs)
            .zip(lde_trace_evaluations_sym)
            .fold(true, |acc, ((merkle_root, merkle_proof), evaluation)| {
                acc & Self::verify_opening(merkle_proof, merkle_root, index_sym, &evaluation)
            });
        openings_are_valid & openings_sym_are_valid
    }

    /// Verify opening Open(Hᵢ(D_LDE), 𝜐) and Open(Hᵢ(D_LDE), -𝜐) for all parts Hᵢof the composition
    /// polynomial, where 𝜐 and -𝜐 are the elements corresponding to the index challenge `iota`.
    fn verify_composition_poly_opening(
        deep_poly_openings: &DeepPolynomialOpening<Self::Field>,
        deep_poly_openings_sym: &DeepPolynomialOpening<Self::Field>,
        composition_poly_merkle_root: &Commitment,
        iota: &usize,
    ) -> bool
    where
        FieldElement<Self::Field>: Serializable,
    {
        let mut value = deep_poly_openings
            .lde_composition_poly_parts_evaluation
            .clone();
        value.extend_from_slice(&deep_poly_openings_sym.lde_composition_poly_parts_evaluation);

        deep_poly_openings
            .lde_composition_poly_proof
            .verify::<BatchedMerkleTreeBackend<Self::Field>>(
                composition_poly_merkle_root,
                *iota,
                &value,
            )
    }

    fn step_4_verify_trace_and_composition_openings<F: IsFFTField, A: AIR<Field = F>>(
        air: &A,
        proof: &StarkProof<Self::Field>,
        challenges: &Challenges<F, A>,
    ) -> bool
    where
        FieldElement<Self::Field>: Serializable,
    {
        challenges
            .iotas
            .iter()
            .zip(&proof.deep_poly_openings)
            .zip(&proof.deep_poly_openings_sym)
            .fold(
                true,
                |mut result, ((iota_n, deep_poly_opening), deep_poly_openings_sym)| {
                    result &= Self::verify_composition_poly_opening(
                        deep_poly_opening,
                        deep_poly_openings_sym,
                        &proof.composition_poly_root,
                        iota_n,
                    );

                    let num_main_columns =
                        air.context().trace_columns - air.number_auxiliary_rap_columns();
                    result &= Self::verify_trace_openings(
                        num_main_columns,
                        proof,
                        deep_poly_opening,
                        deep_poly_openings_sym,
                        *iota_n,
                    );
                    result
                },
            )
    }

    fn verify_fri_layer_openings(
        merkle_root: &Commitment,
        auth_path_sym: &Proof<Commitment>,
        evaluation: &FieldElement<Self::Field>,
        evaluation_sym: &FieldElement<Self::Field>,
        iota: usize,
    ) -> bool
    where
        FieldElement<Self::Field>: Serializable,
    {
        let evaluations = if iota % 2 == 1 {
            vec![evaluation_sym.clone(), evaluation.clone()]
        } else {
            vec![evaluation.clone(), evaluation_sym.clone()]
        };

        auth_path_sym.verify::<BatchedMerkleTreeBackend<Self::Field>>(
            merkle_root,
            iota >> 1,
            &evaluations,
        )
    }

    /// Verify a single FRI query
    /// `zetas`: the vector of all challenges sent by the verifier to the prover at the commit
    /// phase to fold polynomials.
    /// `iota`: the index challenge of this FRI query. This index uniquely determines two elements 𝜐 and -𝜐
    /// of the evaluation domain of FRI layer 0.
    /// `evaluation_point_inv`: precomputed value of 𝜐⁻¹.
    /// `deep_composition_evaluation`: precomputed value of p₀(𝜐), where p₀ is the deep composition polynomial.
    /// `deep_composition_evaluation_sym`: precomputed value of p₀(-𝜐), where p₀ is the deep composition polynomial.
    fn verify_query_and_sym_openings(
        proof: &StarkProof<Self::Field>,
        zetas: &[FieldElement<Self::Field>],
        iota: usize,
        fri_decommitment: &FriDecommitment<Self::Field>,
        evaluation_point_inv: FieldElement<Self::Field>,
        deep_composition_evaluation: &FieldElement<Self::Field>,
        deep_composition_evaluation_sym: &FieldElement<Self::Field>,
    ) -> bool
    where
        FieldElement<Self::Field>: Serializable,
    {
        let fri_layers_merkle_roots = &proof.fri_layers_merkle_roots;
        let evaluation_point_vec: Vec<FieldElement<Self::Field>> =
            core::iter::successors(Some(evaluation_point_inv.square()), |evaluation_point| {
                Some(evaluation_point.square())
            })
            .take(fri_layers_merkle_roots.len())
            .collect();

        let p0_eval = deep_composition_evaluation;
        let p0_eval_sym = deep_composition_evaluation_sym;

        // Reconstruct p₁(𝜐²)
        let mut v =
            (p0_eval + p0_eval_sym) + &zetas[0] * (p0_eval - p0_eval_sym) * evaluation_point_inv;
        let mut index = iota;

        // For each FRI layer, starting from the layer 1: use the proof to verify the validity of values pᵢ(−𝜐^(2ⁱ)) (given by the prover) and
        // pᵢ(𝜐^(2ⁱ)) (computed on the previous iteration by the verifier). Then use them to obtain pᵢ₊₁(𝜐^(2ⁱ⁺¹)).
        // Finally, check that the final value coincides with the given by the prover.
        fri_layers_merkle_roots
            .iter()
            .enumerate()
            .zip(&fri_decommitment.layers_auth_paths)
            .zip(&fri_decommitment.layers_evaluations_sym)
            .zip(evaluation_point_vec)
            .fold(
                true,
                |result,
                 (
                    (((i, merkle_root), auth_path_sym), evaluation_sym),
                    evaluation_point_inv,
                )| {
                    // Verify opening Open(pᵢ(Dₖ), −𝜐^(2ⁱ)) and Open(pᵢ(Dₖ), 𝜐^(2ⁱ)).
                    // `v` is pᵢ(𝜐^(2ⁱ)).
                    // `evaluation_sym` is pᵢ(−𝜐^(2ⁱ)).
                    let openings_ok = Self::verify_fri_layer_openings(
                        merkle_root,
                        auth_path_sym,
                        &v,
                        evaluation_sym,
                        index,
                    );

                    // Update `v` with next value pᵢ₊₁(𝜐^(2ⁱ⁺¹)).
                    v = (&v + evaluation_sym) + &zetas[i + 1] * (&v - evaluation_sym) * evaluation_point_inv;

                    // Update index for next iteration. The index of the squares in the next layer
                    // is obtained by halving the current index. This is due to the bit-reverse
                    // ordering of the elements in the Merkle tree.
                    index >>= 1;

                    if i < fri_decommitment.layers_evaluations_sym.len() - 1 {
                        result & openings_ok
                    } else {
                        // Check that final value is the given by the prover
                        result & (v == proof.fri_last_value) & openings_ok
                    }
                },
            )
    }

    fn reconstruct_deep_composition_poly_evaluations_for_all_queries<A>(
        challenges: &Challenges<Self::Field, A>,
        domain: &Domain<Self::Field>,
        proof: &StarkProof<Self::Field>,
    ) -> DeepPolynomialEvaluations<Self::Field>
    where
        A: AIR<Field = Self::Field>,
    {
        let mut deep_poly_evaluations = Vec::new();
        let mut deep_poly_evaluations_sym = Vec::new();
        for (i, iota) in challenges.iotas.iter().enumerate() {
            let primitive_root =
                &Self::Field::get_primitive_root_of_unity(domain.root_order as u64).unwrap();

            let evaluation_point = Self::query_challenge_to_evaluation_point(*iota, domain);
            deep_poly_evaluations.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                &evaluation_point,
                primitive_root,
                challenges,
                &proof.deep_poly_openings[i].lde_trace_evaluations,
                &proof.deep_poly_openings[i].lde_composition_poly_parts_evaluation,
            ));

            let evaluation_point = Self::query_challenge_to_evaluation_point_sym(*iota, domain);
            deep_poly_evaluations_sym.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                &evaluation_point,
                primitive_root,
                challenges,
                &proof.deep_poly_openings_sym[i].lde_trace_evaluations,
                &proof.deep_poly_openings_sym[i].lde_composition_poly_parts_evaluation,
            ));
        }
        (deep_poly_evaluations, deep_poly_evaluations_sym)
    }

    fn reconstruct_deep_composition_poly_evaluation<A: AIR<Field = Self::Field>>(
        proof: &StarkProof<Self::Field>,
        evaluation_point: &FieldElement<Self::Field>,
        primitive_root: &FieldElement<Self::Field>,
        challenges: &Challenges<Self::Field, A>,
        lde_trace_evaluations: &[FieldElement<Self::Field>],
        lde_composition_poly_parts_evaluation: &[FieldElement<Self::Field>],
    ) -> FieldElement<Self::Field> {
        let mut denoms_trace = (0..proof.trace_ood_evaluations.height)
            .map(|row_idx| evaluation_point - &challenges.z * primitive_root.pow(row_idx as u64))
            .collect::<Vec<FieldElement<Self::Field>>>();
        FieldElement::inplace_batch_inverse(&mut denoms_trace).unwrap();

        let trace_term = (0..proof.trace_ood_evaluations.width)
            .zip(&challenges.trace_term_coeffs)
            .fold(FieldElement::zero(), |trace_terms, (col_idx, coeff_row)| {
                let trace_i = (0..proof.trace_ood_evaluations.height).zip(coeff_row).fold(
                    FieldElement::zero(),
                    |trace_t, (row_idx, coeff)| {
                        let poly_evaluation = (lde_trace_evaluations[col_idx].clone()
                            - proof.trace_ood_evaluations.get_row(row_idx)[col_idx].clone())
                            * &denoms_trace[row_idx];
                        trace_t + &poly_evaluation * coeff
                    },
                );
                trace_terms + trace_i
            });

        let number_of_parts = lde_composition_poly_parts_evaluation.len();
        let z_pow = &challenges.z.pow(number_of_parts);

        let denom_composition = (evaluation_point - z_pow).inv().unwrap();
        let mut h_terms = FieldElement::zero();
        for (j, h_i_upsilon) in lde_composition_poly_parts_evaluation.iter().enumerate() {
            let h_i_zpower = &proof.composition_poly_parts_ood_evaluation[j];
            let h_i_term = (h_i_upsilon - h_i_zpower) * &challenges.gammas[j];
            h_terms += h_i_term;
        }
        h_terms = h_terms * denom_composition;

        trace_term + h_terms
    }

    fn verify<A>(
        proof: &StarkProof<Self::Field>,
        pub_input: &A::PublicInputs,
        proof_options: &ProofOptions,
        mut transcript: impl IsStarkTranscript<Self::Field>,
    ) -> bool
    where
        A: AIR<Field = Self::Field>,
        FieldElement<Self::Field>: Serializable,
    {
        // Verify there are enough queries
        if proof.query_list.len() < proof_options.fri_number_of_queries {
            return false;
        }

        #[cfg(feature = "instruments")]
        println!("- Started step 1: Recover challenges");
        #[cfg(feature = "instruments")]
        let timer1 = Instant::now();

        let air = A::new(proof.trace_length, pub_input, proof_options);
        let domain = Domain::new(&air);

        let challenges = Self::step_1_replay_rounds_and_recover_challenges(
            &air,
            proof,
            &domain,
            &mut transcript,
        );

        // verify grinding
        let security_bits = air.context().proof_options.grinding_factor;
        if security_bits > 0 {
            let nonce_is_valid = proof.nonce.map_or(false, |nonce_value| {
                grinding::is_valid_nonce(&challenges.grinding_seed, nonce_value, security_bits)
            });

            if !nonce_is_valid {
                error!("Grinding factor not satisfied");
                return false;
            }
        }

        #[cfg(feature = "instruments")]
        let elapsed1 = timer1.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed1);

        #[cfg(feature = "instruments")]
        println!("- Started step 2: Verify claimed polynomial");
        #[cfg(feature = "instruments")]
        let timer2 = Instant::now();

        if !Self::step_2_verify_claimed_composition_polynomial(&air, proof, &domain, &challenges) {
            error!("Composition Polynomial verification failed");
            return false;
        }

        #[cfg(feature = "instruments")]
        let elapsed2 = timer2.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed2);
        #[cfg(feature = "instruments")]

        println!("- Started step 3: Verify FRI");
        #[cfg(feature = "instruments")]
        let timer3 = Instant::now();

        if !Self::step_3_verify_fri(proof, &domain, &challenges) {
            error!("FRI verification failed");
            return false;
        }

        #[cfg(feature = "instruments")]
        let elapsed3 = timer3.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed3);

        #[cfg(feature = "instruments")]
        println!("- Started step 4: Verify deep composition polynomial");
        #[cfg(feature = "instruments")]
        let timer4 = Instant::now();

        #[allow(clippy::let_and_return)]
        if !Self::step_4_verify_trace_and_composition_openings(&air, proof, &challenges) {
            error!("DEEP Composition Polynomial verification failed");
            return false;
        }

        #[cfg(feature = "instruments")]
        let elapsed4 = timer4.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed4);

        #[cfg(feature = "instruments")]
        {
            let total_time = elapsed1 + elapsed2 + elapsed3 + elapsed4;
            println!(
                " Fraction of verifying time per step: {:.4} {:.4} {:.4} {:.4}",
                elapsed1.as_nanos() as f64 / total_time.as_nanos() as f64,
                elapsed2.as_nanos() as f64 / total_time.as_nanos() as f64,
                elapsed3.as_nanos() as f64 / total_time.as_nanos() as f64,
                elapsed4.as_nanos() as f64 / total_time.as_nanos() as f64
            );
        }

        true
    }
}
