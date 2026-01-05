use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsSubFieldOf},
    },
    traits::AsBytes,
};

use crate::{
    domain::{new_domain, Domain},
    grinding,
    proof::stark::StarkProof,
    traits::AIR,
    verifier::{Challenges, IsStarkVerifier, Verifier},
};

#[cfg(not(feature = "test_fiat_shamir"))]
use log::error;

// List of airs and their associated proof
type AirsAndProofs<'a, F, E, PI> = Vec<(
    &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = PI>,
    &'a StarkProof<F, E>,
)>;

/// Verifies multiple STARK proofs with their corresponding airs `airs_and_proofs`.
/// Warning: the transcript must be safely initializated before passing it to this method.
pub fn multi_verify<
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: Send + Sync + IsFFTField,
    PI,
>(
    airs_and_proofs: AirsAndProofs<F, E, PI>,
    transcript: &mut impl IsStarkTranscript<E, F>,
) -> bool
where
    FieldElement<F>: AsBytes,
    FieldElement<E>: AsBytes,
{
    // First, replay round 1 for all tables so the transcript state matches the prover,
    // which commits to all traces before sampling any further randomness.
    let mut rap_challenges_vec = Vec::new();
    for (air, proof) in &airs_and_proofs {
        let rap_challenges = replay_round_1(*air, proof, transcript);
        rap_challenges_vec.push(rap_challenges);
    }

    for ((air, proof), rap_challenges) in airs_and_proofs.into_iter().zip(rap_challenges_vec) {
        if !single_table_verify(air, proof, transcript, rap_challenges) {
            return false;
        }
    }
    true
}

/// Replays round 1 of the protocol for a given proof, appending the main and auxiliary
/// trace commitments to the transcript and returning the RAP challenges.
/// We need to split because the prover commits to all traces before sampling any further randomness.
fn replay_round_1<F: IsSubFieldOf<E> + IsFFTField + Send + Sync, E: Send + Sync + IsFFTField, PI>(
    air: &dyn AIR<Field = F, FieldExtension = E, PublicInputs = PI>,
    proof: &StarkProof<F, E>,
    transcript: &mut impl IsStarkTranscript<E, F>,
) -> Vec<FieldElement<E>>
where
    FieldElement<F>: AsBytes,
    FieldElement<E>: AsBytes,
{
    // ===================================
    // ==========|   Round 1   |==========
    // ===================================

    // <<<< Receive commitments:[tⱼ]
    transcript.append_bytes(&proof.lde_trace_main_merkle_root);

    let rap_challenges = air.build_rap_challenges(transcript);

    if let Some(root) = proof.lde_trace_aux_merkle_root {
        transcript.append_bytes(&root);
    }

    rap_challenges
}

fn single_table_verify<
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: Send + Sync + IsFFTField,
    PI,
>(
    air: &dyn AIR<Field = F, FieldExtension = E, PublicInputs = PI>,
    proof: &StarkProof<F, E>,
    transcript: &mut impl IsStarkTranscript<E, F>,
    rap_challenges: Vec<FieldElement<E>>,
) -> bool
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let domain = new_domain(air);

    // Delete comment before merging. Since we don't pass the proof_options to the verifier, we need to access it from the air.
    // Is this correct?
    if proof.query_list.len() < air.options().fri_number_of_queries {
        return false;
    }
    #[cfg(feature = "instruments")]
    println!("- Started step 1: Recover challenges");
    #[cfg(feature = "instruments")]
    let timer1 = Instant::now();

    // Delete before merging: In the previous implementation, we also created the
    // air and domain for each table here.

    let challenges = replay_rounds_after_round_1(air, proof, &domain, transcript, rap_challenges);

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

    if !Verifier::step_2_verify_claimed_composition_polynomial(air, proof, &domain, &challenges) {
        #[cfg(not(feature = "test_fiat_shamir"))]
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

    if !Verifier::<F, E, PI>::step_3_verify_fri(proof, &domain, &challenges) {
        #[cfg(not(feature = "test_fiat_shamir"))]
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
    if !Verifier::<F, E, PI>::step_4_verify_trace_and_composition_openings(proof, &challenges) {
        #[cfg(not(feature = "test_fiat_shamir"))]
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

/// Replays rounds 2, 3 and 4 of the protocol for a given proof, assuming round 1 has
/// already been replayed and the RAP challenges are known. Returns the assembled list
/// of challenges used throughout the verification.
fn replay_rounds_after_round_1<
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: Send + Sync + IsFFTField,
    PI,
>(
    air: &dyn AIR<Field = F, FieldExtension = E, PublicInputs = PI>,
    proof: &StarkProof<F, E>,
    domain: &Domain<F>,
    transcript: &mut impl IsStarkTranscript<E, F>,
    rap_challenges: Vec<FieldElement<E>>,
) -> Challenges<E>
where
    FieldElement<F>: AsBytes,
    FieldElement<E>: AsBytes,
{
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
    let trace_ood_evaluations_columns = proof.trace_ood_evaluations.columns();
    for col in trace_ood_evaluations_columns.iter() {
        for elem in col.iter() {
            transcript.append_field_element(elem);
        }
    }
    // <<<< Receive value: Hᵢ(z^N)
    for element in proof.composition_poly_parts_ood_evaluation.iter() {
        transcript.append_field_element(element);
    }

    // ===================================
    // ==========|   Round 4   |==========
    // ===================================

    let num_terms_composition_poly = proof.composition_poly_parts_ood_evaluation.len();
    let num_terms_trace =
        air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;
    let gamma = transcript.sample_field_element();

    // <<<< Receive challenges: 𝛾, 𝛾'
    let mut deep_composition_coefficients: Vec<_> =
        core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
            .take(num_terms_composition_poly + num_terms_trace)
            .collect();

    let trace_term_coeffs: Vec<_> = deep_composition_coefficients
        .drain(..num_terms_trace)
        .collect::<Vec<_>>()
        .chunks(air.context().transition_offsets.len() * air.step_size())
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
        .collect::<Vec<FieldElement<E>>>();

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
    let iotas = Verifier::<F, E, PI>::sample_query_indexes(number_of_queries, domain, transcript);

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
