//! Legacy Structured Reference String format.
//!
//! This module provides the `StructuredReferenceString` type for backward
//! compatibility with existing SRS files and code.

use alloc::vec::Vec;
use core::mem;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    errors::DeserializationError,
    traits::{AsBytes, Deserializable},
};

/// Legacy Structured Reference String format.
///
/// This type is kept for backward compatibility with existing SRS files.
/// New code should use [`KZGPublicParams`](super::KZGPublicParams) directly.
#[derive(PartialEq, Clone, Debug)]
pub struct StructuredReferenceString<G1Point, G2Point> {
    /// Vector of points in G1 encoding g1, s g1, s^2 g1, s^3 g1, ... s^n g1
    pub powers_main_group: Vec<G1Point>,
    /// Slice of points in G2 encoding g2, s g2
    pub powers_secondary_group: [G2Point; 2],
}

impl<G1Point, G2Point> StructuredReferenceString<G1Point, G2Point>
where
    G1Point: IsGroup,
    G2Point: IsGroup,
{
    /// Creates a new SRS from slices of G1points and a slice of length 2 of G2 points
    pub fn new(powers_main_group: &[G1Point], powers_secondary_group: &[G2Point; 2]) -> Self {
        Self {
            powers_main_group: powers_main_group.into(),
            powers_secondary_group: powers_secondary_group.clone(),
        }
    }
}

#[cfg(feature = "std")]
impl<G1Point, G2Point> StructuredReferenceString<G1Point, G2Point>
where
    G1Point: IsGroup + Deserializable,
    G2Point: IsGroup + Deserializable,
{
    /// Read SRS from file
    pub fn from_file(file_path: &str) -> Result<Self, crate::errors::SrsFromFileError> {
        let bytes = std::fs::read(file_path)?;
        Ok(Self::deserialize(&bytes)?)
    }
}

impl<G1Point, G2Point> AsBytes for StructuredReferenceString<G1Point, G2Point>
where
    G1Point: IsGroup + AsBytes,
    G2Point: IsGroup + AsBytes,
{
    /// Serialize SRS
    fn as_bytes(&self) -> Vec<u8> {
        let mut serialized_data: Vec<u8> = Vec::new();
        // First 4 bytes encodes protocol version
        let protocol_version: [u8; 4] = [0; 4];

        serialized_data.extend(&protocol_version);

        // Second 8 bytes store the amount of G1 elements
        let mut main_group_len_bytes: Vec<u8> = self.powers_main_group.len().to_le_bytes().to_vec();

        // For architectures with less than 64 bits for pointers
        while main_group_len_bytes.len() < 8 {
            main_group_len_bytes.push(0)
        }

        serialized_data.extend(&main_group_len_bytes);

        // G1 elements
        for point in &self.powers_main_group {
            serialized_data.extend(point.as_bytes());
        }

        // G2 elements
        for point in &self.powers_secondary_group {
            serialized_data.extend(point.as_bytes());
        }

        serialized_data
    }
}

impl<G1Point, G2Point> Deserializable for StructuredReferenceString<G1Point, G2Point>
where
    G1Point: IsGroup + Deserializable,
    G2Point: IsGroup + Deserializable,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError> {
        const MAIN_GROUP_LEN_OFFSET: usize = 4;
        const MAIN_GROUP_OFFSET: usize = 12;

        let len_bytes: [u8; 8] = bytes
            .get(MAIN_GROUP_LEN_OFFSET..MAIN_GROUP_OFFSET)
            .ok_or(DeserializationError::InvalidAmountOfBytes)?
            .try_into()
            .map_err(|_| DeserializationError::InvalidAmountOfBytes)?;

        let main_group_len_u64 = u64::from_le_bytes(len_bytes);

        let main_group_len = usize::try_from(main_group_len_u64)
            .map_err(|_| DeserializationError::PointerSizeError)?;

        let mut main_group: Vec<G1Point> = Vec::new();
        let mut secondary_group: Vec<G2Point> = Vec::new();

        let size_g1_point = mem::size_of::<G1Point>();
        let size_g2_point = mem::size_of::<G2Point>();

        for i in 0..main_group_len {
            let start = i * size_g1_point + MAIN_GROUP_OFFSET;
            let end = start + size_g1_point;
            let point_bytes = bytes
                .get(start..end)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?;
            let point = G1Point::deserialize(point_bytes)?;
            main_group.push(point);
        }

        let g2s_offset = size_g1_point * main_group_len + 12;
        for i in 0..2 {
            let start = i * size_g2_point + g2s_offset;
            let end = start + size_g2_point;
            let point_bytes = bytes
                .get(start..end)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?;
            let point = G2Point::deserialize(point_bytes)?;
            secondary_group.push(point);
        }

        let secondary_group_slice = [secondary_group[0].clone(), secondary_group[1].clone()];

        let srs = StructuredReferenceString::new(&main_group, &secondary_group_slice);
        Ok(srs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::cyclic_group::IsGroup;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::{
        curve::BLS12381Curve, twist::BLS12381TwistCurve,
    };
    use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint;
    use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
    use lambdaworks_math::traits::{AsBytes, Deserializable};

    type G1 = ShortWeierstrassJacobianPoint<BLS12381Curve>;
    type G2 = ShortWeierstrassJacobianPoint<BLS12381TwistCurve>;

    #[test]
    fn test_serialize_deserialize_srs() {
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();

        let powers_main_group: Vec<G1> = (0..10)
            .map(|exp| g1.operate_with_self(exp as u64))
            .collect();
        let powers_secondary_group = [g2.clone(), g2.operate_with_self(2_u64)];

        let srs = StructuredReferenceString::new(&powers_main_group, &powers_secondary_group);
        let bytes = srs.as_bytes();
        let deserialized: StructuredReferenceString<G1, G2> =
            StructuredReferenceString::deserialize(&bytes).unwrap();

        assert_eq!(srs, deserialized);
    }
}
