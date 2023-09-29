use lambdaworks_math::{
    fft::cpu::bit_reversing::in_place_bit_reverse_permute,
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
    traits::Serializable,
};

use crate::{
    config::{BatchedMerkleTree, BatchedMerkleTreeBackend, Commitment},
    domain::Domain,
    proof::stark::StarkProof,
    prover::IsStarkProver,
    trace::TraceTable,
    traits::AIR,
    transcript::IsStarkTranscript,
    verifier::{Challenges, IsStarkVerifier},
};

pub struct SHARP;

impl SHARP {
    fn apply_permutation<F: IsFFTField>(vector: &mut Vec<FieldElement<F>>, permutation: &[usize]) {
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
    /// This function returns the permutation that converts lambdaworks ordering of rows to the one used in the stone prover
    pub fn get_stone_prover_domain_permutation(
        domain_size: usize,
        blowup_factor: usize,
    ) -> Vec<usize> {
        let mut permutation = Vec::new();
        let n = domain_size;

        let mut indices: Vec<usize> = (0..blowup_factor).collect();
        in_place_bit_reverse_permute(&mut indices);

        for i in indices.iter() {
            for j in 0..n {
                permutation.push(i + j * blowup_factor)
            }
        }

        for coset_indices in permutation.chunks_mut(n) {
            let mut temp = coset_indices.to_owned();
            in_place_bit_reverse_permute(&mut temp);
            for (j, elem) in coset_indices.iter_mut().enumerate() {
                *elem = temp[j];
            }
        }

        permutation.to_vec()
    }
}

impl IsStarkProver for SHARP {
    #[allow(clippy::type_complexity)]
    fn interpolate_and_commit<F>(
        trace: &TraceTable<F>,
        domain: &Domain<F>,
        transcript: &mut impl IsStarkTranscript<F>,
    ) -> (
        Vec<Polynomial<FieldElement<F>>>,
        Vec<Vec<FieldElement<F>>>,
        BatchedMerkleTree<F>,
        Commitment,
    )
    where
        F: IsFFTField,
        FieldElement<F>: Serializable + Send + Sync,
    {
        let trace_polys = trace.compute_trace_polys();

        // Evaluate those polynomials t_j on the large domain D_LDE.
        let lde_trace_evaluations = Self::compute_lde_trace_evaluations(&trace_polys, domain);

        let permutation = Self::get_stone_prover_domain_permutation(
            domain.interpolation_domain_size,
            domain.blowup_factor,
        );

        let mut lde_trace_permuted = lde_trace_evaluations.clone();

        for col in lde_trace_permuted.iter_mut() {
            Self::apply_permutation(col, &permutation);
        }

        // Compute commitments [t_j].
        let lde_trace = TraceTable::new_from_cols(&lde_trace_permuted);
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
}

pub struct SHARV {}

impl IsStarkVerifier for SHARV {
    fn verify_trace_openings<F>(
        proof: &StarkProof<F>,
        deep_poly_opening: &crate::proof::stark::DeepPolynomialOpenings<F>,
        lde_trace_evaluations: &[Vec<FieldElement<F>>],
        iota: usize,
    ) -> bool
    where
        F: IsFFTField,
        FieldElement<F>: Serializable,
    {
        // TODO: this is for the POC. Blowup factor should be more easily accesible from the
        // verifier
        let blowup_factor = proof.trace_length / lde_trace_evaluations[0].len();
        let permutation =
            SHARP::get_stone_prover_domain_permutation(proof.trace_length, blowup_factor);
        proof
            .lde_trace_merkle_roots
            .iter()
            .zip(&deep_poly_opening.lde_trace_merkle_proofs)
            .zip(lde_trace_evaluations)
            .fold(true, |acc, ((merkle_root, merkle_proof), evaluation)| {
                acc & merkle_proof.verify::<BatchedMerkleTreeBackend<F>>(
                    merkle_root,
                    permutation[iota],
                    &evaluation,
                )
            })
    }
}

#[cfg(test)]
pub mod tests {
    use std::num::ParseIntError;

    use lambdaworks_math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };

    use crate::{
        domain::Domain,
        examples::fibonacci_2_cols_shifted::{self, Fibonacci2ColsShifted},
        proof::options::ProofOptions,
        prover::IsStarkProver,
        sharp::{SHARP, SHARV},
        traits::AIR,
        transcript::StoneProverTranscript,
        verifier::IsStarkVerifier,
    };

    fn decode_hex(s: &str) -> Result<Vec<u8>, ParseIntError> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
            .collect()
    }

    #[test]
    fn test_trace_commitment_is_compatible_with_stone_prover_1() {
        let trace = fibonacci_2_cols_shifted::compute_trace(FieldElement::one(), 4);

        let claimed_index = 3;
        let claimed_value = trace.get_row(claimed_index)[0];
        let mut proof_options = ProofOptions::default_test_options();
        proof_options.blowup_factor = 4;
        proof_options.coset_offset = 3;

        let pub_inputs = fibonacci_2_cols_shifted::PublicInputs {
            claimed_value,
            claimed_index,
        };

        let transcript_init_seed = [0xca, 0xfe, 0xca, 0xfe];

        let air = Fibonacci2ColsShifted::new(trace.n_rows(), &pub_inputs, &proof_options);
        let domain = Domain::new(&air);

        let (_, _, _, trace_commitment) = SHARP::interpolate_and_commit(
            &trace,
            &domain,
            &mut StoneProverTranscript::new(&transcript_init_seed),
        );

        assert_eq!(
            &trace_commitment.to_vec(),
            &decode_hex("0eb9dcc0fb1854572a01236753ce05139d392aa3aeafe72abff150fe21175594")
                .unwrap()
        );
    }
    #[test]
    fn test_trace_commitment_is_compatible_with_stone_prover_2() {
        let trace = fibonacci_2_cols_shifted::compute_trace(FieldElement::one(), 4);

        let claimed_index = 3;
        let claimed_value = trace.get_row(claimed_index)[0];
        let mut proof_options = ProofOptions::default_test_options();
        proof_options.blowup_factor = 64;
        proof_options.coset_offset = 3;

        let pub_inputs = fibonacci_2_cols_shifted::PublicInputs {
            claimed_value,
            claimed_index,
        };

        let transcript_init_seed = [0xfa, 0xfa, 0xfa, 0xee];

        let air = Fibonacci2ColsShifted::new(trace.n_rows(), &pub_inputs, &proof_options);
        let domain = Domain::new(&air);

        let (_, _, _, trace_commitment) = SHARP::interpolate_and_commit(
            &trace,
            &domain,
            &mut StoneProverTranscript::new(&transcript_init_seed),
        );

        assert_eq!(
            &trace_commitment.to_vec(),
            &decode_hex("99d8d4342895c4e35a084f8ea993036be06f51e7fa965734ed9c7d41104f0848")
                .unwrap()
        );
    }

    #[test]
    fn test_sharp_compatibility() {
        let trace = fibonacci_2_cols_shifted::compute_trace(FieldElement::one(), 4);

        let claimed_index = 3;
        let claimed_value = trace.get_row(claimed_index)[0];
        let mut proof_options = ProofOptions::default_test_options();
        proof_options.blowup_factor = 4;
        proof_options.coset_offset = 3;

        let pub_inputs = fibonacci_2_cols_shifted::PublicInputs {
            claimed_value,
            claimed_index,
        };

        let transcript_init_seed = [0xca, 0xfe, 0xca, 0xfe];

        let proof = SHARP::prove::<Stark252PrimeField, Fibonacci2ColsShifted<_>>(
            &trace,
            &pub_inputs,
            &proof_options,
            StoneProverTranscript::new(&transcript_init_seed),
        )
        .unwrap();

        let air = Fibonacci2ColsShifted::new(proof.trace_length, &pub_inputs, &proof_options);
        let domain = Domain::new(&air);
        let challenges = SHARV::step_1_replay_rounds_and_recover_challenges(
            &air,
            &proof,
            &domain,
            &mut StoneProverTranscript::new(&transcript_init_seed),
        );

        assert_eq!(
            proof.lde_trace_merkle_roots[0].to_vec(),
            decode_hex("0eb9dcc0fb1854572a01236753ce05139d392aa3aeafe72abff150fe21175594").unwrap()
        );

        let beta = challenges.transition_coeffs[0];
        assert_eq!(
            beta,
            FieldElement::from_hex_unchecked(
                "86105fff7b04ed4068ecccb8dbf1ed223bd45cd26c3532d6c80a818dbd4fa7"
            ),
        );
        assert_eq!(challenges.transition_coeffs[1], beta.pow(2u64));
        assert_eq!(challenges.boundary_coeffs[0], beta.pow(3u64));
        assert_eq!(challenges.boundary_coeffs[1], beta.pow(4u64));
    }
}
