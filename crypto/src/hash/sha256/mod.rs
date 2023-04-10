use lambdaworks_math::{elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField, traits::ByteConversion};
use lambdaworks_math::field::element::FieldElement;
use sha2::{Sha256};
use super::traits::IsCryptoHash;

pub struct Sha256FieldElements {
    hasher: Sha256
}

impl IsCryptoHash<BLS12381PrimeField> for Sha256FieldElements {
    fn new() -> Self {
        Self { hasher: Sha256::new()}
    }

    fn hash_one(&self, input: FieldElement<BLS12381PrimeField>) -> FieldElement<BLS12381PrimeField> {
        self.update(input.to_bytes_be());
    }

    fn hash_two(
        &self,
        left: FieldElement<BLS12381PrimeField>,
        right: FieldElement<BLS12381PrimeField>,
    ) -> FieldElement<BLS12381PrimeField> {
        todo!()
    }
}
