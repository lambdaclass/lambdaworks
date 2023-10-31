pub use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_crypto::merkle_tree::proof::Proof;

use lambdaworks_math::field::traits::IsPrimeField;
use lambdaworks_math::{field::element::FieldElement, traits::ByteConversion};
use serde::de::{SeqAccess, Visitor};

use crate::config::Commitment;

use serde::{ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone)]
pub struct FriDecommitment<F: IsPrimeField> {
    pub layers_auth_paths: Vec<Proof<Commitment>>,
    pub layers_evaluations_sym: Vec<FieldElement<F>>,
}

// #[cfg(feature = "lambdaworks-serde")]
impl<F: IsPrimeField> Serialize for FriDecommitment<F>
where
    FieldElement<F>: ByteConversion,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("FriDecommitment", 2)?;
        state.serialize_field("layers_auth_paths", &self.layers_auth_paths)?;
        state.serialize_field("layers_evaluations_sym", &self.layers_evaluations_sym)?;
        state.end()
    }
}

// #[cfg(feature = "lambdaworks-serde")]
impl<'de, F: IsPrimeField> Deserialize<'de> for FriDecommitment<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Declare fields of the struct
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            LayersAuthPaths,
            LayersEvaluationsSym,
        }

        // Visitor of struct to deserialize
        struct FriDecommitmentVisitor<F: IsPrimeField>(std::marker::PhantomData<F>);

        impl<'de, F: IsPrimeField> Visitor<'de> for FriDecommitmentVisitor<F> {
            type Value = FriDecommitment<F>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct FriDecommitment")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<FriDecommitment<F>, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let layers_auth_paths = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let layers_evaluations_sym = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                Ok(FriDecommitment {
                    layers_auth_paths,
                    layers_evaluations_sym,
                })
            }

            fn visit_map<V>(self, mut map: V) -> Result<FriDecommitment<F>, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut layers_auth_paths = None;
                let mut layers_evaluations_sym = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::LayersAuthPaths => {
                            if layers_auth_paths.is_some() {
                                return Err(serde::de::Error::duplicate_field("layers_auth_paths"));
                            }
                        }
                        Field::LayersEvaluationsSym => {
                            if layers_evaluations_sym.is_some() {
                                return Err(serde::de::Error::duplicate_field(
                                    "layers_evaluations_sym",
                                ));
                            }
                        }
                    }
                }

                let layers_auth_paths = layers_auth_paths
                    .ok_or_else(|| serde::de::Error::missing_field("layers_auth_paths"))?;
                let layers_evaluations_sym = layers_evaluations_sym
                    .ok_or_else(|| serde::de::Error::missing_field("layers_evaluations_sym"))?;
                Ok(FriDecommitment {
                    layers_auth_paths,
                    layers_evaluations_sym,
                })
            }
        }

        const FIELDS: &'static [&'static str] = &["layers_auth_paths", "layers_evaluations_sym"];
        deserializer.deserialize_struct(
            "FriDecommitment",
            FIELDS,
            FriDecommitmentVisitor(std::marker::PhantomData),
        )
    }
}
