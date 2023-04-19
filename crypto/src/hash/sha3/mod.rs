pub mod helpers;

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

    pub fn hash_to_field<F: IsField>(
        &self,
        msg: &[u8],
        count: u64,
        dst: &[u8],
    ) -> Vec<FieldElement<F>> {
        let order = 18446744069414584321_u64 as f64;
        let mut u: Vec<FieldElement<F>> = Vec::with_capacity(count as usize);
        //L = ceil((ceil(log2(p)) + k) / 8), where k is the security parameter of the cryptosystem (e.g., k = 128)
        let l = ((((order.log2().ceil() as u64) + 128) / 8) as f64).ceil() as u64;
        let len_in_bytes = count * l;
        let pseudo_random_bytes = helpers::expand_message(msg, dst, len_in_bytes).unwrap();
        for i in 0..count {
            let elm_offset = l * i;
            let tv = &pseudo_random_bytes[elm_offset as usize..l as usize];
            u[i as usize] = FieldElement::from(helpers::os2ip(tv));
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
    use lambdaworks_math::field::test_fields::u64_test_field::U64TestField;

    use super::Sha3Hasher;

    type F = U64TestField;

    #[test]
    fn test_hash_to_field() {
        let hasher = Sha3Hasher::new();
        let elements = hasher.hash_to_field::<F>(b"test", 3, b"dsttest");
        println!("{:?}", elements);
    }
}
