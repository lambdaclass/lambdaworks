use crate::{
    cyclic_group::IsCyclicBilinearGroup,
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
};

use super::MSM;

pub struct Naive;

impl<const ORDER: u64, G> MSM<U64PrimeField<ORDER>, G> for Naive
where
    G: IsCyclicBilinearGroup,
{
    /// This function computes the multiscalar multiplication (MSM).
    ///
    /// Assume a group G of order r is given.
    /// Let `hidings = [g_1, ..., g_n]` be a tuple of group points in G and
    /// let `cs = [k_1, ..., k_n]` be a tuple of scalars in the Galois field GF(r).
    ///
    /// Then, with additive notation, `msm(cs, hidings)` computes k_1 * g_1 + .... + k_n * g_n.
    ///
    /// If `hidings` and `cs` are empty, then `msm` returns the zero element of the group.
    ///
    /// Panics if `cs` and `hidings` have different lengths.
    fn msm(&self, cs: &[FieldElement<U64PrimeField<ORDER>>], hidings: &[G]) -> G {
        assert_eq!(
            cs.len(),
            hidings.len(),
            "Slices `cs` and `hidings` must be of the same length to compute `msm`."
        );
        cs.iter()
            .zip(hidings.iter())
            .map(|(&c, h)| h.operate_with_self(*c.value() as u128))
            .reduce(|acc, x| acc.operate_with(&x))
            .unwrap_or_else(G::neutral_element)
    }
}
