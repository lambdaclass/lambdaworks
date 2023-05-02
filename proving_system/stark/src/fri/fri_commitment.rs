use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
        traits::IsField,
    },
    unsigned_integer::element::UnsignedInteger,
};

use super::HASHER;
pub use super::{FriMerkleTree, Polynomial};

#[derive(Clone)]
pub struct FriLayer<F: IsField> {
    pub poly: Polynomial<FieldElement<F>>,
    pub domain: Vec<FieldElement<F>>,
    pub evaluation: Vec<FieldElement<F>>,
    pub merkle_tree: FriMerkleTree<F>,
}

impl<M: IsModulus<UnsignedInteger<N>> + Clone, const N: usize>
    FriLayer<MontgomeryBackendPrimeField<M, N>>
{
    pub fn new(
        poly: Polynomial<FieldElement<MontgomeryBackendPrimeField<M, N>>>,
        domain: &[FieldElement<MontgomeryBackendPrimeField<M, N>>],
    ) -> Self {
        let evaluation = poly.evaluate_slice(domain);
        let merkle_tree = FriMerkleTree::build(&evaluation, Box::new(HASHER));

        Self {
            poly,
            domain: domain.to_vec(),
            evaluation: evaluation.to_vec(),
            merkle_tree,
        }
    }
}
