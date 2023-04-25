use lambdaworks_math::field::{element::FieldElement, traits::IsField};

pub fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        (usize::MAX >> (n - 1).leading_zeros()) + 1
    }
}

/// Fill a field element slice with 0s until a power of two size is reached, unless it already is.
pub(crate) fn zero_padding<F: IsField>(input: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
    let mut input = input.to_vec();
    input.resize(next_power_of_two(input.len()), FieldElement::zero());
    input
}
