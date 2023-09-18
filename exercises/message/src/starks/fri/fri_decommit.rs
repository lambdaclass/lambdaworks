pub use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_math::errors::DeserializationError;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::traits::{ByteConversion, Deserializable, Serializable};

use crate::starks::config::Commitment;
use crate::starks::utils::{deserialize_proof, serialize_proof};

#[derive(Debug, Clone)]
pub struct FriDecommitment<F: IsField> {
    pub layers_auth_paths_sym: Vec<Proof<Commitment>>,
    pub layers_evaluations_sym: Vec<FieldElement<F>>,
    pub layers_auth_paths: Vec<Proof<Commitment>>,
    pub layers_evaluations: Vec<FieldElement<F>>,
}

impl<F> Serializable for FriDecommitment<F>
where
    F: IsField,
    FieldElement<F>: ByteConversion,
{
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = vec![];
        bytes.extend(self.layers_auth_paths_sym.len().to_be_bytes());
        for proof in &self.layers_auth_paths_sym {
            bytes.extend(serialize_proof(proof));
        }
        let felt_len = self.layers_evaluations[0].to_bytes_be().len();
        bytes.extend(felt_len.to_be_bytes());
        bytes.extend(self.layers_evaluations_sym.len().to_be_bytes());
        for evaluation in &self.layers_evaluations_sym {
            bytes.extend(evaluation.to_bytes_be());
        }
        bytes.extend(self.layers_evaluations.len().to_be_bytes());
        for evaluation in &self.layers_evaluations {
            bytes.extend(evaluation.to_bytes_be());
        }
        bytes.extend(self.layers_auth_paths.len().to_be_bytes());
        for proof in &self.layers_auth_paths {
            bytes.extend(serialize_proof(proof));
        }
        bytes
    }
}

impl<F> Deserializable for FriDecommitment<F>
where
    F: IsField,
    FieldElement<F>: ByteConversion,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        let mut bytes = bytes;
        let mut layers_auth_paths_sym = vec![];
        let layers_auth_paths_sym_len = usize::from_be_bytes(
            bytes
                .get(..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );
        bytes = &bytes[8..];

        for _ in 0..layers_auth_paths_sym_len {
            let proof;
            (proof, bytes) = deserialize_proof(bytes)?;
            layers_auth_paths_sym.push(proof);
        }

        let felt_len = usize::from_be_bytes(
            bytes
                .get(..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );
        bytes = &bytes[8..];

        let layers_evaluations_sym_len = usize::from_be_bytes(
            bytes
                .get(..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );
        bytes = &bytes[8..];

        let mut layers_evaluations_sym = vec![];
        for _ in 0..layers_evaluations_sym_len {
            let evaluation = FieldElement::<F>::from_bytes_be(
                bytes
                    .get(..felt_len)
                    .ok_or(DeserializationError::InvalidAmountOfBytes)?
                    .try_into()
                    .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
            )?;
            bytes = &bytes[felt_len..];
            layers_evaluations_sym.push(evaluation);
        }

        let layer_evaluations_len = usize::from_be_bytes(
            bytes
                .get(..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );
        bytes = &bytes[8..];

        let mut layers_evaluations = vec![];
        for _ in 0..layer_evaluations_len {
            let evaluation = FieldElement::<F>::from_bytes_be(
                bytes
                    .get(..felt_len)
                    .ok_or(DeserializationError::InvalidAmountOfBytes)?
                    .try_into()
                    .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
            )?;
            bytes = &bytes[felt_len..];
            layers_evaluations.push(evaluation);
        }

        let mut layers_auth_paths = vec![];
        let layers_auth_paths_len = usize::from_be_bytes(
            bytes
                .get(..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );
        bytes = &bytes[8..];

        for _ in 0..layers_auth_paths_len {
            let proof;
            (proof, bytes) = deserialize_proof(bytes)?;
            layers_auth_paths.push(proof);
        }

        Ok(Self {
            layers_auth_paths_sym,
            layers_evaluations_sym,
            layers_evaluations,
            layers_auth_paths,
        })
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_crypto::merkle_tree::proof::Proof;
    use lambdaworks_math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };
    use proptest::{collection, prelude::*, prop_compose, proptest};

    use lambdaworks_math::traits::{Deserializable, Serializable};

    use crate::starks::config::{Commitment, COMMITMENT_SIZE};

    use super::FriDecommitment;

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
        fn commitment_vec()(vec in collection::vec(some_commitment(), 4)) -> Vec<Commitment> {
            vec
        }
    }

    prop_compose! {
        fn some_proof()(merkle_path in commitment_vec()) -> Proof<Commitment> {
            Proof{merkle_path}
        }
    }

    prop_compose! {
        fn proof_vec()(vec in collection::vec(some_proof(), 4)) -> Vec<Proof<Commitment>> {
            vec
        }
    }

    prop_compose! {
        fn some_felt()(base in any::<u64>(), exponent in any::<u128>()) -> FE {
            FE::from(base).pow(exponent)
        }
    }

    prop_compose! {
        fn field_vec()(vec in collection::vec(some_felt(), 16)) -> Vec<FE> {
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

    proptest! {
        #![proptest_config(ProptestConfig {cases: 5, .. ProptestConfig::default()})]
        #[test]
        fn test_serialize_and_deserialize(fri_decommitment in some_fri_decommitment()) {
            let serialized = fri_decommitment.serialize();
            let deserialized: FriDecommitment<Stark252PrimeField> = FriDecommitment::deserialize(&serialized).unwrap();

            for (a, b) in fri_decommitment.layers_auth_paths_sym.iter().zip(deserialized.layers_auth_paths_sym.iter()) {
                prop_assert_eq!(&a.merkle_path, &b.merkle_path);
            }

            for (a, b) in fri_decommitment.layers_evaluations_sym.iter().zip(deserialized.layers_evaluations_sym.iter()) {
                prop_assert_eq!(a, b);
            }

            for (a, b) in fri_decommitment.layers_evaluations.iter().zip(deserialized.layers_evaluations.iter()) {
                prop_assert_eq!(a, b);
            }

            for (a, b) in fri_decommitment.layers_auth_paths.iter().zip(deserialized.layers_auth_paths.iter()) {
                prop_assert_eq!(&a.merkle_path, &b.merkle_path);
            }
        }
    }
}
