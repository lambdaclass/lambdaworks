use lambdaworks_math::{
    fft::polynomial::FFTPoly,
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField},
    },
    polynomial::Polynomial,
    traits::ByteConversion,
};

use crate::config::FriMerkleTree;

#[derive(Clone)]
pub struct FriLayer<F>
where
    F: IsField,
    FieldElement<F>: ByteConversion,
{
    pub evaluation: Vec<FieldElement<F>>,
    pub merkle_tree: FriMerkleTree<F>,
    pub coset_offset: FieldElement<F>,
    pub domain_size: usize,
}

impl<F> FriLayer<F>
where
    F: IsField + IsFFTField,
    FieldElement<F>: ByteConversion,
{
    pub fn new(
        poly: &Polynomial<FieldElement<F>>,
        coset_offset: &FieldElement<F>,
        domain_size: usize,
    ) -> Self {
        let evaluation = poly
            .evaluate_offset_fft(1, Some(domain_size), coset_offset)
            .unwrap(); // TODO: return error

        let merkle_tree = FriMerkleTree::build(&evaluation);

        Self {
            evaluation,
            merkle_tree,
            coset_offset: coset_offset.clone(),
            domain_size,
        }
    }
}
