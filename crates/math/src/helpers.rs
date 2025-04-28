#[cfg(feature = "alloc")]
use crate::field::{element::FieldElement, traits::IsFFTField};

/// Computes the power of two that is equal or greater than n
pub fn next_power_of_two(n: u64) -> u64 {
    if n <= 1 {
        1
    } else {
        (u64::MAX >> (n - 1).leading_zeros()) + 1
    }
}

/// Pads the trace table with zeros until the length of the columns of the trace
/// is equal to a power of 2
/// This is required to ensure that we can use the radix-2 Cooley-Tukey FFT algorithm
#[cfg(feature = "alloc")]
pub fn resize_to_next_power_of_two<F: IsFFTField>(
    trace_colums: &mut [alloc::vec::Vec<FieldElement<F>>],
) {
    trace_colums.iter_mut().for_each(|col| {
        // TODO: Remove this unwrap. This may panic if the usize cant be
        // casted into a u64.
        let col_len = col.len().try_into().unwrap();
        let next_power_of_two_len = next_power_of_two(col_len);
        col.resize(next_power_of_two_len as usize, FieldElement::<F>::zero())
    })
}
