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
        // Convert usize to u64 safely, handling potential overflow
        let col_len = match col.len().try_into() {
            Ok(len) => len,
            Err(_) => {
                // If usize is larger than u64::MAX, use u64::MAX as a reasonable fallback
                u64::MAX
            }
        };
        let next_power_of_two_len = next_power_of_two(col_len);
        col.resize(next_power_of_two_len as usize, FieldElement::<F>::zero())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::test_fields::u64_test_field::U64TestField;

    #[test]
    fn test_resize_to_next_power_of_two() {
        let mut trace_columns = vec![
            vec![FieldElement::<U64TestField>::one(); 5],
            vec![FieldElement::<U64TestField>::one(); 3],
        ];
        
        resize_to_next_power_of_two(&mut trace_columns);
        
        // First column should be resized to 8 (next power of 2 after 5)
        assert_eq!(trace_columns[0].len(), 8);
        // Second column should be resized to 4 (next power of 2 after 3)
        assert_eq!(trace_columns[1].len(), 4);
        
        // Check that the original values are preserved
        for i in 0..5 {
            assert_eq!(trace_columns[0][i], FieldElement::<U64TestField>::one());
        }
        
        for i in 0..3 {
            assert_eq!(trace_columns[1][i], FieldElement::<U64TestField>::one());
        }
        
        // Check that new elements are zeros
        for i in 5..8 {
            assert_eq!(trace_columns[0][i], FieldElement::<U64TestField>::zero());
        }
        
        for i in 3..4 {
            assert_eq!(trace_columns[1][i], FieldElement::<U64TestField>::zero());
        }
    }
}
