use lambdaworks_math::{
    field::{self, element::FieldElement},
    traits::ByteConversion,
};

/// Interface to Collision Resistant Hashes.
pub trait IsCryptoHash<F>
where
    F: field::traits::IsField,
{
    /// Hashes a field element into one. Also known as one to one hashing.
    fn hash_one(&self, input: &field::element::FieldElement<F>) -> field::element::FieldElement<F>
    where
        FieldElement<F>: ByteConversion;

    /// Hashes two field elements into one. Also known as two to one hashing.
    fn hash_two(
        &self,
        left: &field::element::FieldElement<F>,
        right: &field::element::FieldElement<F>,
    ) -> field::element::FieldElement<F>
    where
        FieldElement<F>: ByteConversion;
}
