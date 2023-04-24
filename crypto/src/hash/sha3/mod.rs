pub mod helpers;

use std::fmt::Debug;

use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
        traits::IsField,
    },
    traits::ByteConversion,
    unsigned_integer::element::{UnsignedInteger, U256},
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

    pub fn hash_to_field<M: IsModulus<UnsignedInteger<N>> + Clone + Debug, const N: usize>(
        &self,
        msg: &[u8],
        count: u64,
        dst: &[u8],
    ) -> Vec<FieldElement<MontgomeryBackendPrimeField<M, N>>> {
        let order = U256::from("800000000000011000000000000000000000000000000000000000000000001");
        let mut u = vec![FieldElement::zero(); count as usize];
        //L = ceil((ceil(log2(p)) + k) / 8), where k is the security parameter of the cryptosystem (e.g. k = ceil(log2(p) / 2))
        let log2_p = (order.limbs.len() * 8) as f64;
        let k = (log2_p / 2.0).ceil() * 8.0;
        let l = (((log2_p * 8.0) + k) / 8.0).ceil() as u64;
        let len_in_bytes = count * l;
        let pseudo_random_bytes = helpers::expand_message(msg, dst, len_in_bytes).unwrap();
        for i in 0..count {
            let elm_offset = l * i;
            let tv = &pseudo_random_bytes[elm_offset as usize..elm_offset as usize + l as usize];

            u[i as usize] = helpers::os2ip::<M, N>(tv);
        }
        u
    }
}
impl<F: IsField> IsCryptoHash<F> for Sha3Hasher {
    fn hash_one(&self, input: &FieldElement<F>) -> FieldElement<F>
    where
        FieldElement<F>: ByteConversion,
    {
        let mut hasher = Sha3_256::new();
        hasher.update(input.to_bytes_be());
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&hasher.finalize());
        FieldElement::<F>::from_bytes_le(&result_hash).unwrap()
    }

    fn hash_two(&self, left: &FieldElement<F>, right: &FieldElement<F>) -> FieldElement<F>
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

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };

    use proptest::prelude::*;

    use super::Sha3Hasher;

    type F = Stark252PrimeField;

    proptest! {
        #[test]
        fn test_same_message_produce_same_field_elements(msg in ".*", dst in ".*") {
            let hasher = Sha3Hasher::new();
            let field_elements: Vec<FieldElement<F>> = hasher.hash_to_field(msg.as_bytes(), 40, dst.as_bytes());
            let other_field_elements = hasher.hash_to_field(msg.as_bytes(), 40, dst.as_bytes());
            prop_assert_eq!(field_elements, other_field_elements);
        }
    }
}
