use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::ByteConversion,
};
use sha3::{Digest, Sha3_256};

use super::traits::IsCryptoHash;
pub struct Sha3Hasher;
/// Sha3 Hasher used over fields
/// Notice while it's generic over F, it's only generates enough randomness for fields of at most 256 bits
impl Sha3Hasher {
    pub const fn new() -> Self {
        Self
    }
}
impl<F: IsField> IsCryptoHash<F> for Sha3Hasher {
    fn hash_one(&self, input: FieldElement<F>) -> FieldElement<F>
    where
        FieldElement<F>: ByteConversion,
    {
        let mut hasher = Sha3_256::new();
        hasher.update(input.to_bytes_be());
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&hasher.finalize());
        FieldElement::<F>::from_bytes_le(&result_hash).unwrap()
    }

    fn hash_two(&self, left: FieldElement<F>, right: FieldElement<F>) -> FieldElement<F>
    where
        FieldElement<F>: ByteConversion,
    {
        let mut hasher = Sha3_256::new();
        hasher.update(left.to_bytes_be());
        hasher.update(right.to_bytes_be());
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&hasher.finalize());
        FieldElement::<F>::from_bytes_le(&result_hash).unwrap()
    }
}
