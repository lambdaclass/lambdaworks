use lambdaworks_math::field;

/// Interface to Collision Resistant Hashes.
pub trait IsCryptoHash<F>
where
    F: field::traits::IsField,
{
    // TODO: Add a `new` method to generalize the creation of a hash function
    // with non-hardcoded parameters.

    /// Hashes a field element into one. Also known as one to one hashing.
    fn hash_one(input: field::element::FieldElement<F>) -> field::element::FieldElement<F>;

    /// Hashes two field elements into one. Also known as two to one hashing.
    fn hash_two(
        left: field::element::FieldElement<F>,
        right: field::element::FieldElement<F>,
    ) -> field::element::FieldElement<F>;
}
