use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_math::{
    errors::DeserializationError,
    field::{element::FieldElement, traits::IsFFTField},
    traits::{ByteConversion, Deserializable, Serializable},
};

use crate::starks::{
    config::Commitment,
    frame::Frame,
    fri::fri_decommit::FriDecommitment,
    utils::{deserialize_proof, serialize_proof},
};

use std::mem;

#[derive(Debug, Clone)]
pub struct DeepPolynomialOpenings<F: IsFFTField> {
    pub lde_composition_poly_proof: Proof<Commitment>,
    pub lde_composition_poly_even_evaluation: FieldElement<F>,
    pub lde_composition_poly_odd_evaluation: FieldElement<F>,
    pub lde_trace_merkle_proofs: Vec<Proof<Commitment>>,
    pub lde_trace_evaluations: Vec<FieldElement<F>>,
}

#[derive(Debug)]
pub struct StarkProof<F: IsFFTField> {
    // Length of the execution trace
    pub trace_length: usize,
    // Commitments of the trace columns
    // [tⱼ]
    pub lde_trace_merkle_roots: Vec<Commitment>,
    // tⱼ(zgᵏ)
    pub trace_ood_frame_evaluations: Frame<F>,
    // [H₁] and [H₂]
    pub composition_poly_root: Commitment,
    // H₁(z²)
    pub composition_poly_even_ood_evaluation: FieldElement<F>,
    // H₂(z²)
    pub composition_poly_odd_ood_evaluation: FieldElement<F>,
    // [pₖ]
    pub fri_layers_merkle_roots: Vec<Commitment>,
    // pₙ
    pub fri_last_value: FieldElement<F>,
    // Open(p₀(D₀), 𝜐ₛ), Opwn(pₖ(Dₖ), −𝜐ₛ^(2ᵏ))
    pub query_list: Vec<FriDecommitment<F>>,
    // Open(H₁(D_LDE, 𝜐₀), Open(H₂(D_LDE, 𝜐₀), Open(tⱼ(D_LDE), 𝜐₀)
    pub deep_poly_openings: Vec<DeepPolynomialOpenings<F>>,
    // nonce obtained from grinding
    pub nonce: u64,
}

impl<F> Serializable for DeepPolynomialOpenings<F>
where
    F: IsFFTField,
    FieldElement<F>: ByteConversion,
{
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = vec![];
        bytes.extend(serialize_proof(&self.lde_composition_poly_proof));
        let lde_composition_poly_even_evaluation_bytes =
            self.lde_composition_poly_even_evaluation.to_bytes_be();
        let felt_len = lde_composition_poly_even_evaluation_bytes.len();
        bytes.extend(felt_len.to_be_bytes());
        bytes.extend(lde_composition_poly_even_evaluation_bytes);
        bytes.extend(self.lde_composition_poly_odd_evaluation.to_bytes_be());
        bytes.extend(self.lde_trace_merkle_proofs.len().to_be_bytes());
        for proof in &self.lde_trace_merkle_proofs {
            bytes.extend(serialize_proof(proof));
        }
        bytes.extend(self.lde_trace_evaluations.len().to_be_bytes());
        for evaluation in &self.lde_trace_evaluations {
            bytes.extend(evaluation.to_bytes_be());
        }
        bytes
    }
}

impl<F> Deserializable for DeepPolynomialOpenings<F>
where
    F: IsFFTField,
    FieldElement<F>: ByteConversion,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        let mut bytes = bytes;
        let lde_composition_poly_proof;
        (lde_composition_poly_proof, bytes) = deserialize_proof(bytes)?;

        let felt_len = usize::from_be_bytes(
            bytes
                .get(..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );
        bytes = &bytes[8..];

        let lde_composition_poly_even_evaluation = FieldElement::from_bytes_be(
            bytes
                .get(..felt_len)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        )?;
        bytes = &bytes[felt_len..];

        let lde_composition_poly_odd_evaluation = FieldElement::from_bytes_be(
            bytes
                .get(..felt_len)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        )?;
        bytes = &bytes[felt_len..];

        let lde_trace_merkle_proofs_len = usize::from_be_bytes(
            bytes
                .get(..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );
        bytes = &bytes[8..];

        let mut lde_trace_merkle_proofs = vec![];
        for _ in 0..lde_trace_merkle_proofs_len {
            let proof;
            (proof, bytes) = deserialize_proof(bytes)?;
            lde_trace_merkle_proofs.push(proof);
        }

        let lde_trace_evaluations_len = usize::from_be_bytes(
            bytes
                .get(..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );
        bytes = &bytes[8..];

        let mut lde_trace_evaluations = vec![];
        for _ in 0..lde_trace_evaluations_len {
            let evaluation = FieldElement::from_bytes_be(
                bytes
                    .get(..felt_len)
                    .ok_or(DeserializationError::InvalidAmountOfBytes)?
                    .try_into()
                    .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
            )?;
            bytes = &bytes[felt_len..];
            lde_trace_evaluations.push(evaluation);
        }

        Ok(DeepPolynomialOpenings {
            lde_composition_poly_proof,
            lde_composition_poly_even_evaluation,
            lde_composition_poly_odd_evaluation,
            lde_trace_merkle_proofs,
            lde_trace_evaluations,
        })
    }
}

impl<F> Serializable for StarkProof<F>
where
    F: IsFFTField,
    FieldElement<F>: ByteConversion,
{
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = vec![];

        // Serialize trace length
        bytes.extend(self.trace_length.to_be_bytes());

        bytes.extend(self.lde_trace_merkle_roots.len().to_be_bytes());
        for commitment in &self.lde_trace_merkle_roots {
            bytes.extend(commitment);
        }
        let trace_ood_frame_evaluations_bytes = self.trace_ood_frame_evaluations.serialize();
        bytes.extend(trace_ood_frame_evaluations_bytes.len().to_be_bytes());
        bytes.extend(trace_ood_frame_evaluations_bytes);

        bytes.extend(self.composition_poly_root);

        let composition_poly_even_ood_evaluation_bytes =
            self.composition_poly_even_ood_evaluation.to_bytes_be();
        bytes.extend(
            composition_poly_even_ood_evaluation_bytes
                .len()
                .to_be_bytes(),
        );
        bytes.extend(composition_poly_even_ood_evaluation_bytes);
        bytes.extend(self.composition_poly_odd_ood_evaluation.to_bytes_be());

        bytes.extend(self.fri_layers_merkle_roots.len().to_be_bytes());
        for commitment in &self.fri_layers_merkle_roots {
            bytes.extend(commitment);
        }

        bytes.extend(self.fri_last_value.to_bytes_be());

        bytes.extend(self.query_list.len().to_be_bytes());
        for query in &self.query_list {
            let query_bytes = query.serialize();
            bytes.extend(query_bytes.len().to_be_bytes());
            bytes.extend(query_bytes);
        }

        bytes.extend(self.deep_poly_openings.len().to_be_bytes());
        for opening in &self.deep_poly_openings {
            let opening_bytes = opening.serialize();
            bytes.extend(opening_bytes.len().to_be_bytes());
            bytes.extend(opening_bytes);
        }

        // serialize nonce
        bytes.extend(self.nonce.to_be_bytes());

        bytes
    }
}

impl<F> Deserializable for StarkProof<F>
where
    F: IsFFTField,
    FieldElement<F>: ByteConversion,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        let mut bytes = bytes;
        let trace_length_buffer_size = mem::size_of::<usize>();
        let trace_length = usize::from_be_bytes(
            bytes
                .get(..trace_length_buffer_size)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );

        bytes = &bytes[8..];

        let lde_trace_merkle_roots_len = usize::from_be_bytes(
            bytes
                .get(..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );

        bytes = &bytes[8..];

        let mut lde_trace_merkle_roots: Vec<[u8; 32]> = vec![];
        for _ in 0..lde_trace_merkle_roots_len {
            let commitment = bytes
                .get(..32)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?;

            lde_trace_merkle_roots.push(commitment);
            bytes = &bytes[32..];
        }

        let trace_ood_frame_evaluations_len = usize::from_be_bytes(
            bytes
                .get(..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );

        bytes = &bytes[8..];

        let trace_ood_frame_evaluations: Frame<F> = Frame::deserialize(
            bytes
                .get(..trace_ood_frame_evaluations_len)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?,
        )?;

        bytes = &bytes[trace_ood_frame_evaluations_len..];

        let composition_poly_root = bytes
            .get(..32)
            .ok_or(DeserializationError::InvalidAmountOfBytes)?
            .try_into()
            .map_err(|_| DeserializationError::InvalidAmountOfBytes)?;

        bytes = &bytes[32..];

        let felt_len = usize::from_be_bytes(
            bytes
                .get(..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );

        bytes = &bytes[8..];

        let composition_poly_even_ood_evaluation = FieldElement::from_bytes_be(
            bytes
                .get(..felt_len)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?,
        )?;

        bytes = &bytes[felt_len..];

        let composition_poly_odd_ood_evaluation = FieldElement::from_bytes_be(
            bytes
                .get(..felt_len)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?,
        )?;

        bytes = &bytes[felt_len..];

        let fri_layers_merkle_roots_len = usize::from_be_bytes(
            bytes
                .get(..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );

        bytes = &bytes[8..];

        let mut fri_layers_merkle_roots: Vec<[u8; 32]> = vec![];
        for _ in 0..fri_layers_merkle_roots_len {
            let commitment = bytes
                .get(..32)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?;
            fri_layers_merkle_roots.push(commitment);
            bytes = &bytes[32..];
        }

        let fri_last_value = FieldElement::from_bytes_be(
            bytes
                .get(..felt_len)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?,
        )?;

        bytes = &bytes[felt_len..];

        let query_list_len = usize::from_be_bytes(
            bytes
                .get(..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );

        bytes = &bytes[8..];

        let mut query_list = vec![];
        for _ in 0..query_list_len {
            let query_len = usize::from_be_bytes(
                bytes
                    .get(..8)
                    .ok_or(DeserializationError::InvalidAmountOfBytes)?
                    .try_into()
                    .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
            );

            bytes = &bytes[8..];

            let query = FriDecommitment::deserialize(
                bytes
                    .get(..query_len)
                    .ok_or(DeserializationError::InvalidAmountOfBytes)?,
            )?;

            bytes = &bytes[query_len..];

            query_list.push(query);
        }

        let deep_poly_openings_len = usize::from_be_bytes(
            bytes
                .get(..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );

        bytes = &bytes[8..];

        let mut deep_poly_openings = vec![];
        for _ in 0..deep_poly_openings_len {
            let opening_len = usize::from_be_bytes(
                bytes
                    .get(..8)
                    .ok_or(DeserializationError::InvalidAmountOfBytes)?
                    .try_into()
                    .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
            );

            bytes = &bytes[8..];

            let opening = DeepPolynomialOpenings::deserialize(
                bytes
                    .get(..opening_len)
                    .ok_or(DeserializationError::InvalidAmountOfBytes)?,
            )?;

            bytes = &bytes[opening_len..];

            deep_poly_openings.push(opening);
        }

        // deserialize nonce
        let start_nonce = bytes
            .len()
            .checked_sub(core::mem::size_of::<u64>())
            .ok_or(DeserializationError::InvalidAmountOfBytes)?;

        let nonce = u64::from_be_bytes(
            bytes
                .get(start_nonce..)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );

        Ok(StarkProof {
            trace_length,
            lde_trace_merkle_roots,
            trace_ood_frame_evaluations,
            composition_poly_root,
            composition_poly_even_ood_evaluation,
            composition_poly_odd_ood_evaluation,
            fri_layers_merkle_roots,
            fri_last_value,
            query_list,
            deep_poly_openings,
            nonce,
        })
    }
}

#[cfg(test)]
mod test {
    use lambdaworks_crypto::merkle_tree::proof::Proof;
    use lambdaworks_math::field::{
            element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        };
    use proptest::{collection, prelude::*, prop_compose, proptest};

    use lambdaworks_math::traits::{Deserializable, Serializable};

    use crate::starks::{config::{Commitment, COMMITMENT_SIZE}, fri::fri_decommit::FriDecommitment, frame::Frame};

    use super::{DeepPolynomialOpenings, StarkProof};

    type FE = FieldElement<Stark252PrimeField>;

    prop_compose! {
            fn some_commitment()(high in any::<u128>(), low in any::<u128>()) -> Commitment {
                let mut bytes = [0u8; COMMITMENT_SIZE];
                bytes[..16].copy_from_slice(&high.to_be_bytes());
                bytes[16..].copy_from_slice(&low.to_be_bytes());
                bytes
        }
    }

    prop_compose! {
        fn commitment_vec()(vec in collection::vec(some_commitment(), (16_usize, 32_usize))) -> Vec<Commitment> {
            vec
        }
    }

    prop_compose! {
        fn some_proof()(merkle_path in commitment_vec()) -> Proof<Commitment> {
            Proof{merkle_path}
        }
    }

    prop_compose! {
        fn proof_vec()(vec in collection::vec(some_proof(), (4_usize, 8_usize))) -> Vec<Proof<Commitment>> {
            vec
        }
    }

    prop_compose! {
        fn some_felt()(base in any::<u64>(), exponent in any::<u128>()) -> FE {
            FE::from(base).pow(exponent)
        }
    }

    prop_compose! {
        fn field_vec()(vec in collection::vec(some_felt(), (8_usize, 16_usize))) -> Vec<FE> {
            vec
        }
    }

    prop_compose! {
        fn some_fri_decommitment()(
            layers_auth_paths_sym in proof_vec(),
            layers_evaluations_sym in field_vec(),
            layers_evaluations in field_vec(),
            layers_auth_paths in proof_vec()
        ) -> FriDecommitment<Stark252PrimeField> {
            FriDecommitment{
                layers_auth_paths_sym,
                layers_evaluations_sym,
                layers_evaluations,
                layers_auth_paths
            }
        }
    }

    prop_compose! {
        fn fri_decommitment_vec()(vec in collection::vec(some_fri_decommitment(), (16_usize, 32_usize))) -> Vec<FriDecommitment<Stark252PrimeField>> {
            vec
        }
    }

    prop_compose! {
        fn some_deep_polynomial_openings()(
            lde_composition_poly_proof in some_proof(),
            lde_composition_poly_even_evaluation in some_felt(),
            lde_composition_poly_odd_evaluation in some_felt(),
            lde_trace_merkle_proofs in proof_vec(),
            lde_trace_evaluations in field_vec()
        ) -> DeepPolynomialOpenings<Stark252PrimeField> {
            DeepPolynomialOpenings {
                lde_composition_poly_proof,
                lde_composition_poly_even_evaluation,
                lde_composition_poly_odd_evaluation,
                lde_trace_merkle_proofs,
                lde_trace_evaluations
            }
        }
    }

    prop_compose! {
        fn deep_polynomial_openings_vec()(vec in collection::vec(some_deep_polynomial_openings(), (8_usize, 16_usize))) -> Vec<DeepPolynomialOpenings<Stark252PrimeField>> {
            vec
        }
    }

    prop_compose! {
        fn some_frame()(data in field_vec(), row_width in any::<usize>()) -> Frame<Stark252PrimeField> {
            Frame::new(data, row_width)
        }
    }

    prop_compose! {
        fn some_usize()(len in any::<usize>()) -> usize {
            len
        }
    }

    prop_compose! {
        fn some_stark_proof()(
            trace_length in some_usize(),
            lde_trace_merkle_roots in commitment_vec(),
            trace_ood_frame_evaluations in some_frame(),
            composition_poly_root in some_commitment(),
            composition_poly_even_ood_evaluation in some_felt(),
            composition_poly_odd_ood_evaluation in some_felt(),
            fri_layers_merkle_roots in commitment_vec(),
            fri_last_value in some_felt(),
            query_list in fri_decommitment_vec(),
            deep_poly_openings in deep_polynomial_openings_vec()

    ) -> StarkProof<Stark252PrimeField> {
            StarkProof {
                trace_length,
                lde_trace_merkle_roots,
                trace_ood_frame_evaluations,
                composition_poly_root,
                composition_poly_even_ood_evaluation,
                composition_poly_odd_ood_evaluation,
                fri_layers_merkle_roots,
                fri_last_value,
                query_list,
                deep_poly_openings,
                nonce: 0
            }
        }
    }

    proptest! {
        #[test]
        fn test_deep_polynomial_openings_serialization(
            deep_polynomial_openings in some_deep_polynomial_openings()
        ) {
            let serialized = deep_polynomial_openings.serialize();
            let deserialized = DeepPolynomialOpenings::<Stark252PrimeField>::deserialize(&serialized).unwrap();

            for (a, b) in deep_polynomial_openings.lde_trace_merkle_proofs.iter().zip(deserialized.lde_trace_merkle_proofs.iter()) {
                prop_assert_eq!(&a.merkle_path, &b.merkle_path);
            };

            prop_assert_eq!(deep_polynomial_openings.lde_composition_poly_even_evaluation, deserialized.lde_composition_poly_even_evaluation);
            prop_assert_eq!(deep_polynomial_openings.lde_composition_poly_odd_evaluation, deserialized.lde_composition_poly_odd_evaluation);
            prop_assert_eq!(deep_polynomial_openings.lde_composition_poly_proof.merkle_path, deserialized.lde_composition_poly_proof.merkle_path);
            prop_assert_eq!(deep_polynomial_openings.lde_trace_evaluations, deserialized.lde_trace_evaluations);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {cases: 5, .. ProptestConfig::default()})]
        #[test]
        fn test_stark_proof_serialization(
            stark_proof in some_stark_proof()
        ) {
            let serialized = stark_proof.serialize();
            let deserialized = StarkProof::<Stark252PrimeField>::deserialize(&serialized).unwrap();

            prop_assert_eq!(
                stark_proof.lde_trace_merkle_roots,
                deserialized.lde_trace_merkle_roots
            );
            prop_assert_eq!(
                stark_proof.trace_ood_frame_evaluations.num_columns(),
                deserialized.trace_ood_frame_evaluations.num_columns()
            );
            prop_assert_eq!(
                stark_proof.trace_ood_frame_evaluations.num_rows(),
                deserialized.trace_ood_frame_evaluations.num_rows()
            );
            prop_assert_eq!(
                stark_proof.composition_poly_root,
                deserialized.composition_poly_root
            );
            prop_assert_eq!(
                stark_proof.composition_poly_even_ood_evaluation,
                deserialized.composition_poly_even_ood_evaluation
            );
            prop_assert_eq!(
                stark_proof.composition_poly_odd_ood_evaluation,
                deserialized.composition_poly_odd_ood_evaluation
            );
            prop_assert_eq!(
                stark_proof.fri_layers_merkle_roots,
                deserialized.fri_layers_merkle_roots
            );
            prop_assert_eq!(stark_proof.fri_last_value, deserialized.fri_last_value);

            for (a, b) in stark_proof
                .query_list
                .iter()
                .zip(deserialized.query_list.iter())
            {
                for (x, y) in a
                    .clone()
                    .layers_auth_paths_sym
                    .iter()
                    .zip(b.clone().layers_auth_paths_sym.iter())
                {
                    prop_assert_eq!(&x.merkle_path, &y.merkle_path);
                }
                prop_assert_eq!(&a.layers_evaluations_sym, &b.layers_evaluations_sym);
                prop_assert_eq!(&a.layers_evaluations, &b.layers_evaluations);
                for (x, y) in a
                    .clone()
                    .layers_auth_paths
                    .iter()
                    .zip(b.clone().layers_auth_paths.iter())
                {
                    prop_assert_eq!(&x.merkle_path, &y.merkle_path);
                }
            }

            for (a, b) in stark_proof
                .deep_poly_openings
                .iter()
                .zip(deserialized.deep_poly_openings.iter())
            {
                for (x, y) in a
                    .clone()
                    .lde_trace_merkle_proofs
                    .iter()
                    .zip(b.clone().lde_trace_merkle_proofs.iter())
                {
                    prop_assert_eq!(&x.merkle_path, &y.merkle_path);
                }
                prop_assert_eq!(
                    &a.lde_composition_poly_even_evaluation,
                    &b.lde_composition_poly_even_evaluation
                );
                prop_assert_eq!(
                    &a.lde_composition_poly_odd_evaluation,
                    &b.lde_composition_poly_odd_evaluation
                );
                prop_assert_eq!(
                    &a.lde_composition_poly_proof.merkle_path,
                    &b.lde_composition_poly_proof.merkle_path
                );
                prop_assert_eq!(&a.lde_trace_evaluations, &b.lde_trace_evaluations);
            }
        }
    }
}
