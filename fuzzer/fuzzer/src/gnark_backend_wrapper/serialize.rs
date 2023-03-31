use super::GnarkBackendError;
use crate::gnark_backend_wrapper as gnark_backend;
#[cfg(feature = "groth16")]
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;
#[cfg(feature = "groth16")]
use serde::Deserialize;
use std::num::TryFromIntError;

pub fn serialize_felt_unchecked(felt: &gnark_backend::Fr) -> Vec<u8> {
    let mut serialized_felt = Vec::new();
    #[allow(clippy::unwrap_used)]
    felt.serialize_uncompressed(&mut serialized_felt).unwrap();
    // Turn little-endian to big-endian.
    serialized_felt.reverse();
    serialized_felt
}

#[cfg(feature = "groth16")]
pub fn serialize_felt<S>(felt: &gnark_backend::Fr, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::ser::Serializer,
{
    let mut serialized_felt = Vec::new();
    felt.serialize_uncompressed(&mut serialized_felt)
        .map_err(serde::ser::Error::custom)?;
    // Turn little-endian to big-endian.
    serialized_felt.reverse();
    let encoded_coefficient = hex::encode(serialized_felt);
    serializer.serialize_str(&encoded_coefficient)
}

pub fn encode_felts(felts: &[gnark_backend::Fr]) -> Result<String, GnarkBackendError> {
    let mut buff: Vec<u8> = Vec::new();
    let n_felts: u32 = felts
        .len()
        .try_into()
        .map_err(|e: TryFromIntError| GnarkBackendError::SerializeFeltsError(e.to_string()))?;
    buff.extend_from_slice(&n_felts.to_be_bytes());
    buff.extend_from_slice(
        &felts
            .iter()
            .flat_map(serialize_felt_unchecked)
            .collect::<Vec<u8>>(),
    );
    Ok(hex::encode(buff))
}

pub fn serialize_felts<S>(felts: &[gnark_backend::Fr], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::ser::Serializer,
{
    let encoded_buff = encode_felts(felts).map_err(serde::ser::Error::custom)?;
    serializer.serialize_str(&encoded_buff)
}

#[cfg(feature = "groth16")]
pub fn deserialize_felt<'de, D>(deserializer: D) -> Result<gnark_backend::Fr, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let felt_bytes = String::deserialize(deserializer)?;
    let mut decoded = hex::decode(felt_bytes).map_err(serde::de::Error::custom)?;
    // Turn big-endian to little-endian.
    decoded.reverse();
    gnark_backend::Fr::deserialize_uncompressed(decoded.as_slice())
        .map_err(serde::de::Error::custom)
}

#[cfg(feature = "groth16")]
pub fn deserialize_felts<'de, D>(deserializer: D) -> Result<Vec<gnark_backend::Fr>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let serialized_felts = String::deserialize(deserializer)?;

    let mut decoded_felts = hex::decode(serialized_felts).map_err(serde::de::Error::custom)?;

    let n_felts: usize = u32::from_be_bytes(
        decoded_felts
            .get(..4)
            .ok_or_else(|| serde::de::Error::custom("Error getting felts size"))?
            .try_into()
            .map_err(serde::de::Error::custom)?,
    )
    .try_into()
    .map_err(serde::de::Error::custom)?;
    let mut deserialized_felts: Vec<gnark_backend::Fr> = Vec::with_capacity(n_felts);

    decoded_felts
        .get_mut(4..) // Skip the vector length corresponding to the first four bytes.
        .ok_or_else(|| serde::de::Error::custom("Error getting decoded felts"))?
        .chunks_mut(32)
        .try_for_each(|decoded_felt| {
            // Turn big-endian to little-endian.
            decoded_felt.reverse();
            // Here I reference after dereference because I had a mutable reference and I need a non-mutable one.
            let felt: gnark_backend::Fr =
                CanonicalDeserialize::deserialize_uncompressed(&*decoded_felt)
                    .map_err(serde::de::Error::custom)?;
            deserialized_felts.push(felt);
            Ok::<(), D::Error>(())
        })?;

    Ok(deserialized_felts)
}
