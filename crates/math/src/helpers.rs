#[cfg(feature = "alloc")]
use crate::field::{element::FieldElement, traits::IsFFTField};

pub fn next_power_of_two(n: u64) -> u64 {
    if n <= 1 {
        1
    } else {
        (u64::MAX >> (n - 1).leading_zeros()) + 1
    }
}

#[cfg(feature = "alloc")]
pub fn resize_to_next_power_of_two<F: IsFFTField>(
    trace_colums: &mut [alloc::vec::Vec<FieldElement<F>>],
) {
    trace_colums.iter_mut().for_each(|col| {
        let col_len = col.len() as u64;
        let next_power_of_two_len = next_power_of_two(col_len);
        col.resize(next_power_of_two_len as usize, FieldElement::<F>::zero())
    })
}
