use std::marker::PhantomData;

use lambdaworks_crypto::merkle_tree::{proof::Proof, traits::IsMerkleTreeBackend};
use lambdaworks_math::{
    fft::{cpu::bit_reversing::in_place_bit_reverse_permute, polynomial::FFTPoly},
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
    traits::Serializable,
};

use crate::{
    config::{BatchedMerkleTree, BatchedMerkleTreeBackend, Commitment},
    domain::Domain,
    fri::{fri_commitment::FriLayer, IsFri, fri_decommit::FriDecommitment},
    proof::stark::StarkProof,
    prover::IsStarkProver,
    trace::TraceTable,
    transcript::IsStarkTranscript,
    verifier::IsStarkVerifier,
};

pub struct SHARP<F: IsFFTField> {
    phantom: PhantomData<F>
}

impl<F> SHARP<F> 
where
    F: IsFFTField,
    FieldElement<F>: Serializable
{
    fn apply_permutation<T: Clone>(vector: &mut Vec<T>, permutation: &[usize]) {
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

impl<F> IsStarkProver for SHARP<F>
where
    F: IsFFTField,
    FieldElement<F>: Serializable
{
    type Field = F;
    type MerkleTreeBackend = BatchedMerkleTreeBackend<F>;

    fn fri_commit_phase(
        number_layers: usize,
        p_0: Polynomial<FieldElement<Self::Field>>,
        transcript: &mut impl IsStarkTranscript<Self::Field>,
        coset_offset: &FieldElement<Self::Field>,
        domain_size: usize,
    ) -> (FieldElement<Self::Field>, Vec<FriLayer<Self::Field, Self::MerkleTreeBackend>>)
    {
        SHARPCompatibleFri::fri_commit_phase(
            number_layers,
            p_0,
            transcript,
            coset_offset,
            domain_size,
        )
    }
    
    fn fri_query_phase(fri_layers: &Vec<FriLayer<Self::Field, Self::MerkleTreeBackend>>, iotas: &[usize]) -> Vec<FriDecommitment<Self::Field>>
    {
        SHARPCompatibleFri::fri_query_phase(fri_layers, iotas)
    }

    #[allow(clippy::type_complexity)]
    fn interpolate_and_commit(
        trace: &TraceTable<Self::Field>,
        domain: &Domain<Self::Field>,
        transcript: &mut impl IsStarkTranscript<Self::Field>,
    ) -> (
        Vec<Polynomial<FieldElement<Self::Field>>>,
        Vec<Vec<FieldElement<Self::Field>>>,
        BatchedMerkleTree<F>,
        Commitment,
    )
    where
        F: IsFFTField,
        FieldElement<Self::Field>: Serializable + Send + Sync,
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

    fn commit_composition_polynomial(
        lde_composition_poly_parts_evaluations: &[Vec<FieldElement<Self::Field>>],
        domain: &Domain<F>,
    ) -> (BatchedMerkleTree<F>, Commitment)
    where
        F: IsFFTField,
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

        let permutation = Self::get_stone_prover_domain_permutation(
            domain.interpolation_domain_size,
            domain.blowup_factor,
        );

        Self::apply_permutation(&mut lde_composition_poly_evaluations, &permutation);

        let mut lde_composition_poly_evaluations_merged = Vec::new();
        for chunk in lde_composition_poly_evaluations.chunks(2) {
            let (mut chunk0, chunk1) = (chunk[0].clone(), &chunk[1]);
            chunk0.extend_from_slice(chunk1);
            lde_composition_poly_evaluations_merged.push(chunk0);
        }

        Self::batch_commit(&lde_composition_poly_evaluations_merged)
    }

    fn open_composition_poly(
        domain: &Domain<F>,
        composition_poly_merkle_tree: &BatchedMerkleTree<F>,
        lde_composition_poly_evaluations: &[Vec<FieldElement<Self::Field>>],
        index: usize,
    ) -> (Proof<Commitment>, Vec<FieldElement<Self::Field>>)
    where
        F: IsFFTField,
        FieldElement<Self::Field>: Serializable,
    {
        let permutation = Self::get_stone_prover_domain_permutation(
            domain.interpolation_domain_size,
            domain.blowup_factor,
        );
        let proof = composition_poly_merkle_tree
            .get_proof_by_pos(index)
            .unwrap();

        // Hi openings
        let mut permuted = lde_composition_poly_evaluations.clone().to_vec();
        Self::apply_permutation(&mut permuted[0], &permutation);
        let lde_composition_poly_parts_evaluation: Vec<_> = permuted
            .iter()
            .map(|part| vec![part[index * 2].clone(), part[index * 2 + 1].clone()])
            .flatten()
            .collect();

        (proof, lde_composition_poly_parts_evaluation)
    }

    fn open_trace_polys(
        domain: &Domain<Self::Field>,
        lde_trace_merkle_trees: &Vec<BatchedMerkleTree<F>>,
        lde_trace: &TraceTable<Self::Field>,
        index: usize,
    ) -> (Vec<Proof<Commitment>>, Vec<FieldElement<Self::Field>>)
    where
        F: IsFFTField,
        FieldElement<Self::Field>: Serializable,
    {
        let permutation = Self::get_stone_prover_domain_permutation(
            domain.interpolation_domain_size,
            domain.blowup_factor,
        );
        let lde_trace_evaluations = lde_trace.get_row(permutation[index * 2]).to_vec();

        let index = index;
        // Trace polynomials openings
        #[cfg(feature = "parallel")]
        let merkle_trees_iter = lde_trace_merkle_trees.par_iter();
        #[cfg(not(feature = "parallel"))]
        let merkle_trees_iter = lde_trace_merkle_trees.iter();

        let lde_trace_merkle_proofs: Vec<Proof<[u8; 32]>> = merkle_trees_iter
            .map(|tree| tree.get_proof_by_pos(index * 2).unwrap())
            .collect();

        (lde_trace_merkle_proofs, lde_trace_evaluations)
    }

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
        proof
            .lde_trace_merkle_roots
            .iter()
            .zip(&deep_poly_opening.lde_trace_merkle_proofs)
            .zip(lde_trace_evaluations)
            .fold(true, |acc, ((merkle_root, merkle_proof), evaluation)| {
                acc & merkle_proof.verify::<BatchedMerkleTreeBackend<F>>(
                    merkle_root,
                    iota * 2,
                    &evaluation,
                )
            })
    }

    fn sample_query_indexes<F: IsFFTField>(
        number_of_queries: usize,
        domain: &Domain<F>,
        transcript: &mut impl IsStarkTranscript<F>,
    ) -> Vec<usize> {
        let domain_size = domain.lde_roots_of_unity_coset.len() as u64;
        (0..number_of_queries)
            .map(|_| (transcript.sample_u64(domain_size >> 1)) as usize)
            .collect::<Vec<usize>>()
    }
}

pub struct SHARPCompatibleFri<F> {
    phantom: F
}

impl<F> IsFri for SHARPCompatibleFri<F>
where
    F: IsFFTField,
    FieldElement<F>: Serializable,
{
    type Field = F;
    type MerkleTreeBackend = BatchedMerkleTreeBackend<F>;

    fn new_fri_layer(
        poly: &Polynomial<FieldElement<Self::Field>>,
        coset_offset: &FieldElement<Self::Field>,
        domain_size: usize,
        degree_bound: usize,
    ) -> crate::fri::fri_commitment::FriLayer<Self::Field, Self::MerkleTreeBackend>
    where
        F: IsFFTField,
        FieldElement<F>: Serializable,
    {
        let mut evaluation = poly
            .evaluate_offset_fft(1, Some(domain_size), coset_offset)
            .unwrap(); // TODO: return error

        let blowup_factor = domain_size / (1 << degree_bound);
        let permutation =
            SHARP::get_stone_prover_domain_permutation(domain_size / blowup_factor, blowup_factor);
        SHARP::apply_permutation(&mut evaluation, &permutation);

        println!("Capa");
        for elem in evaluation.iter() {
            println!("{:?}", elem);
        }

        let mut to_commit = Vec::new();
        for chunk in evaluation.chunks(2) {
            to_commit.push(vec![chunk[0].clone(), chunk[1].clone()]);
        }

        // TODO: Fix leaves
        let merkle_tree = BatchedMerkleTree::build(&to_commit);

        FriLayer::new(&evaluation, merkle_tree, coset_offset.clone(), domain_size)
    }


    fn fri_query_phase(
        fri_layers: &Vec<FriLayer<Self::Field, Self::MerkleTreeBackend>>,
        iotas: &[usize],
    ) -> Vec<FriDecommitment<Self::Field>>
    where
        FieldElement<Self::Field>: Serializable,
    {
        if !fri_layers.is_empty() {
            let query_list = iotas
                .iter()
                .map(|iota_s| {
                    // <<<< Receive challenge 𝜄ₛ (iota_s)
                    let mut layers_auth_paths_sym = vec![];
                    let mut layers_evaluations_sym = vec![];
                    let mut layers_evaluations = vec![];
                    let mut layers_auth_paths = vec![];

                    let blowup_factor = fri_layers[0].evaluation.len() / (1 << fri_layers.len());
                    let permutation = SHARP::get_stone_prover_domain_permutation(
                        (1 << fri_layers.len()),
                        blowup_factor,
                    );
                    let permuted_iota = permutation[*iota_s];

                    for layer in fri_layers {
                        // symmetric element
                        let index = iota_s % layer.domain_size;
                        let index_sym = (iota_s + layer.domain_size / 2) % layer.domain_size;
                        let evaluation_sym = layer.evaluation[index_sym].clone();
                        let auth_path_sym = layer.merkle_tree.get_proof_by_pos((permuted_iota % layer.domain_size) >> 1).unwrap();
                        let evaluation = layer.evaluation[index].clone();
                        let auth_path = layer.merkle_tree.get_proof_by_pos((permuted_iota % layer.domain_size) >> 1).unwrap();
                        layers_auth_paths_sym.push(auth_path_sym);
                        layers_evaluations_sym.push(evaluation_sym);
                        layers_evaluations.push(evaluation);
                        layers_auth_paths.push(auth_path);
                    }

                    FriDecommitment {
                        layers_auth_paths_sym,
                        layers_evaluations_sym,
                        layers_evaluations,
                        layers_auth_paths,
                    }
                })
                .collect();

            query_list
        } else {
            vec![]
        }
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
    fn test_sharp_fibonacci_happy_path() {
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

        let proof = SHARP::prove::<Fibonacci2ColsShifted<_>>(
            &trace,
            &pub_inputs,
            &proof_options,
            StoneProverTranscript::new(&transcript_init_seed),
        )
        .unwrap();

        assert!(
            SHARV::verify::<Stark252PrimeField, Fibonacci2ColsShifted<_>>(
                &proof,
                &pub_inputs,
                &proof_options,
                StoneProverTranscript::new(&transcript_init_seed)
            )
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
        proof_options.grinding_factor = 0;

        let pub_inputs = fibonacci_2_cols_shifted::PublicInputs {
            claimed_value,
            claimed_index,
        };

        let transcript_init_seed = [0xca, 0xfe, 0xca, 0xfe];

        let proof = SHARP::prove::<Fibonacci2ColsShifted<_>>(
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

        assert_eq!(challenges.transition_coeffs[0], FieldElement::one());
        let beta = challenges.transition_coeffs[1];
        assert_eq!(
            beta,
            FieldElement::from_hex_unchecked(
                "86105fff7b04ed4068ecccb8dbf1ed223bd45cd26c3532d6c80a818dbd4fa7"
            ),
        );
        assert_eq!(challenges.boundary_coeffs[0], beta.pow(2u64));
        assert_eq!(challenges.boundary_coeffs[1], beta.pow(3u64));

        assert_eq!(
            proof.composition_poly_root.to_vec(),
            decode_hex("7cdd8d5fe3bd62254a417e2e260e0fed4fccdb6c9005e828446f645879394f38").unwrap()
        );

        assert_eq!(
            challenges.z,
            FieldElement::from_hex_unchecked(
                "317629e783794b52cd27ac3a5e418c057fec9dd42f2b537cdb3f24c95b3e550"
            )
        );

        assert_eq!(
            proof.trace_ood_frame_evaluations.get_row(0)[0],
            FieldElement::from_hex_unchecked(
                "70d8181785336cc7e0a0a1078a79ee6541ca0803ed3ff716de5a13c41684037",
            )
        );
        assert_eq!(
            proof.trace_ood_frame_evaluations.get_row(1)[0],
            FieldElement::from_hex_unchecked(
                "29808fc8b7480a69295e4b61600480ae574ca55f8d118100940501b789c1630",
            )
        );
        assert_eq!(
            proof.trace_ood_frame_evaluations.get_row(0)[1],
            FieldElement::from_hex_unchecked(
                "7d8110f21d1543324cc5e472ab82037eaad785707f8cae3d64c5b9034f0abd2",
            )
        );
        assert_eq!(
            proof.trace_ood_frame_evaluations.get_row(1)[1],
            FieldElement::from_hex_unchecked(
                "1b58470130218c122f71399bf1e04cf75a6e8556c4751629d5ce8c02cc4e62d",
            )
        );
        assert_eq!(
            proof.composition_poly_parts_ood_evaluation[0],
            FieldElement::from_hex_unchecked(
                "1c0b7c2275e36d62dfb48c791be122169dcc00c616c63f8efb2c2a504687e85",
            )
        );

        // gamma
        assert_eq!(
            challenges.trace_term_coeffs[0][1],
            FieldElement::from_hex_unchecked(
                "a0c79c1c77ded19520873d9c2440451974d23302e451d13e8124cf82fc15dd"
            )
        );

        assert_eq!(
            challenges.zetas[0],
            FieldElement::from_hex_unchecked(
                "5c6b5a66c9fda19f583f0b10edbaade98d0e458288e62c2fa40e3da2b293cef"
            )
        );

        assert_eq!(
            proof.fri_layers_merkle_roots[1].to_vec(),
            decode_hex("327d47da86f5961ee012b2b0e412de16023ffba97c82bfe85102f00daabd49fb").unwrap()
        );

        assert_eq!(
            challenges.zetas[1],
            FieldElement::from_hex_unchecked(
                "13c337c9dc727bea9eef1f82cab86739f17acdcef562f9e5151708f12891295"
            )
        );

        assert_eq!(
            proof.fri_last_value,
            FieldElement::from_hex_unchecked(
                "43fedf9f9e3d1469309862065c7d7ca0e7e9ce451906e9c01553056f695aec9"
            )
        );

        assert_eq!(
            challenges.iotas[0],
            1
        );

        assert_eq!(
            proof.deep_poly_openings[0].lde_trace_evaluations[0],
            FieldElement::from_hex_unchecked("4de0d56f9cf97dff326c26592fbd4ae9ee756080b12c51cfe4864e9b8734f43")
        );
        //assert_eq!(proof.query_list[0].layers_evaluations_sym);
    }
}
