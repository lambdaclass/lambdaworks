#[cfg(feature = "std")]
use crate::field::{element::FieldElement, traits::IsFFTField};

pub fn next_power_of_two(n: u64) -> u64 {
    if n <= 1 {
        1
    } else {
        (u64::MAX >> (n - 1).leading_zeros()) + 1
    }
}

#[cfg(feature = "std")]
pub fn resize_to_next_power_of_two<F: IsFFTField>(trace_colums: &mut [Vec<FieldElement<F>>]) {
    trace_colums.iter_mut().for_each(|col| {
        // TODO: Remove this unwrap. This may panic if the usize cant be
        // casted into a u64.
        let col_len = col.len().try_into().unwrap();
        let next_power_of_two_len = next_power_of_two(col_len);
        col.resize(next_power_of_two_len as usize, FieldElement::<F>::zero())
    })
}
