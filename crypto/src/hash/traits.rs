use lambdaworks_math::field;

use crate::errors::HashError;

/// Interface to Collision Resistant Hashes.
pub trait IsCryptoHash<F>
where
    F: field::traits::IsField,
{
    fn new() -> Self;

    /// Hashes a field element into one. Also known as one to one hashing.
    fn hash_one(
        &self,
        input: field::element::FieldElement<F>,
    ) -> Result<field::element::FieldElement<F>, HashError>;

    /// Hashes two field elements into one. Also known as two to one hashing.
    fn hash_two(
        &self,
        left: field::element::FieldElement<F>,
        right: field::element::FieldElement<F>,
    ) -> Result<field::element::FieldElement<F>, HashError>;
}
