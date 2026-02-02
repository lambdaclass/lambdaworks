extern crate alloc;
use crate::field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field};
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
/// fft in place algorithm used to evaluate a polynomial of degree 2^n - 1 in 2^n points.
/// Input must be of size 2^n for some n.
pub fn cfft(
    input: &mut [FieldElement<Mersenne31Field>],
    twiddles: Vec<Vec<FieldElement<Mersenne31Field>>>,
) {
    // If the input size is 2^n, then log_2_size is n.
    let log_2_size = input.len().trailing_zeros();

    // The cfft has n layers.
    (0..log_2_size).for_each(|i| {
        // In each layer i we split the current input in chunks of size 2^{i+1}.
        let chunk_size = 1 << (i + 1);
        let half_chunk_size = 1 << i;
        input.chunks_mut(chunk_size).for_each(|chunk| {
            // We split each chunk in half, calling the first half hi_part and the second hal low_part.
            let (hi_part, low_part) = chunk.split_at_mut(half_chunk_size);

            // We apply the corresponding butterfly for every element j of the high and low part.
            hi_part
                .iter_mut()
                .zip(low_part)
                .enumerate()
                .for_each(|(j, (hi, low))| {
                    let temp = *low * twiddles[i as usize][j];
                    *low = *hi - temp;
                    *hi += temp
                });
        });
    });
}

#[cfg(feature = "alloc")]
/// The inverse fft algorithm used to interpolate 2^n points.
/// Input must be of size 2^n for some n.
pub fn icfft(
    input: &mut [FieldElement<Mersenne31Field>],
    twiddles: Vec<Vec<FieldElement<Mersenne31Field>>>,
) {
    // If the input size is 2^n, then log_2_size is n.
    let log_2_size = input.len().trailing_zeros();

    // The icfft has n layers.
    (0..log_2_size).for_each(|i| {
        // In each layer i we split the current input in chunks of size 2^{n - i}.
        let chunk_size = 1 << (log_2_size - i);
        let half_chunk_size = chunk_size >> 1;
        input.chunks_mut(chunk_size).for_each(|chunk| {
            // We split each chunk in half, calling the first half hi_part and the second hal low_part.
            let (hi_part, low_part) = chunk.split_at_mut(half_chunk_size);

            // We apply the corresponding butterfly for every element j of the high and low part.
            hi_part
                .iter_mut()
                .zip(low_part)
                .enumerate()
                .for_each(|(j, (hi, low))| {
                    let temp = *hi + *low;
                    *low = (*hi - *low) * twiddles[i as usize][j];
                    *hi = temp;
                });
        });
    });
}

/// This function permutes a slice of field elements to order the result of the cfft in the natural way.
/// We call the natural order to [P(x0, y0), P(x1, y1), P(x2, y2), ...],
/// where (x0, y0) is the first point of the corresponding coset.
/// The cfft doesn't return the evaluations in the natural order.
/// For example, if we apply the cfft to 8 coefficients of a polynomial of degree 7 we'll get the evaluations in this order:
/// [P(x0, y0), P(x2, y2), P(x4, y4), P(x6, y6), P(x7, y7), P(x5, y5), P(x3, y3), P(x1, y1)],
/// where the even indices are found first in ascending order and then the odd indices in descending order.
/// This function permutes the slice [0, 2, 4, 6, 7, 5, 3, 1] into [0, 1, 2, 3, 4, 5, 6, 7].
#[deprecated(note = "Use order_cfft_result_in_place for better performance")]
pub fn order_cfft_result_naive(
    input: &[FieldElement<Mersenne31Field>],
) -> Vec<FieldElement<Mersenne31Field>> {
    let mut result = Vec::new();
    let length = input.len();
    for i in 0..length / 2 {
        result.push(input[i]); // We push the left index.
        result.push(input[length - i - 1]); // We push the right index.
    }
    result
}

/// In-place permutation to order the result of the cfft in the natural way.
///
/// This is an optimized version of `order_cfft_result_naive` that performs the permutation
/// in-place without allocating a new vector (only a small bitvector for tracking visited elements).
///
/// The cfft returns evaluations in this order: `[P(x0), P(x2), P(x4), ..., P(x_{n-1}), ..., P(x3), P(x1)]`
/// (even indices ascending, then odd indices descending).
///
/// This function permutes them to natural order: `[P(x0), P(x1), P(x2), ..., P(x_{n-1})]`.
///
/// # Algorithm
///
/// The permutation can be described as:
/// - `output[2*i] = input[i]` for `i < n/2`
/// - `output[2*i+1] = input[n-1-i]` for `i < n/2`
///
/// Equivalently, for each destination position `dst`:
/// - `output[dst] = input[src_for_dst(dst)]`
///   where `src_for_dst(dst) = dst/2` if dst is even, `src_for_dst(dst) = n-1-dst/2` if dst is odd.
///
/// We follow cycles to perform swaps in-place, using a bitvector to track visited positions.
pub fn order_cfft_result_in_place(input: &mut [FieldElement<Mersenne31Field>]) {
    let n = input.len();
    if n <= 2 {
        return;
    }

    // Compute source index for a destination position:
    // The value at `output[dst]` comes from `input[src_for_dst(dst)]`
    // dst even: output[dst] = input[dst/2]
    // dst odd:  output[dst] = input[n-1-dst/2]
    let src_for_dst = |dst: usize| -> usize {
        if dst.is_multiple_of(2) {
            dst / 2
        } else {
            n - 1 - dst / 2
        }
    };

    // Track visited positions with a bitvector (n bits = n/64 u64s, much smaller than n field elements)
    let mut visited = vec![0u64; n.div_ceil(64)];

    let is_visited = |visited: &[u64], i: usize| -> bool { (visited[i / 64] >> (i % 64)) & 1 == 1 };

    let mark_visited = |visited: &mut [u64], i: usize| {
        visited[i / 64] |= 1u64 << (i % 64);
    };

    for start in 0..n {
        if is_visited(&visited, start) {
            continue;
        }

        // For cycle following with a permutation where we know the SOURCE for each destination,
        // we need to follow the chain: dst <- src_for_dst(dst) <- src_for_dst(src_for_dst(dst)) <- ...
        // Save the original value at `start`, then pull values from their sources
        let mut dst = start;
        let temp = input[dst];

        loop {
            mark_visited(&mut visited, dst);
            let src = src_for_dst(dst);

            if src == start {
                // Cycle complete, place the saved value
                input[dst] = temp;
                break;
            }

            // Pull the value from source to destination
            input[dst] = input[src];
            dst = src;
        }
    }
}

/// This function permutes a slice of field elements to order the input of the icfft in a specific way.
/// For example, if we want to interpolate 8 points we should input them in the icfft in this order:
/// [(x0, y0), (x2, y2), (x4, y4), (x6, y6), (x7, y7), (x5, y5), (x3, y3), (x1, y1)],
/// where the even indices are found first in ascending order and then the odd indices in descending order.
/// This function permutes the slice [0, 1, 2, 3, 4, 5, 6, 7] into [0, 2, 4, 6, 7, 5, 3, 1].
#[deprecated(note = "Use order_icfft_input_in_place for better performance")]
pub fn order_icfft_input_naive(
    input: &mut [FieldElement<Mersenne31Field>],
) -> Vec<FieldElement<Mersenne31Field>> {
    let mut result = Vec::new();

    // We push the even indices.
    (0..input.len()).step_by(2).for_each(|i| {
        result.push(input[i]);
    });

    // We push the odd indices.
    (1..input.len()).step_by(2).rev().for_each(|i| {
        result.push(input[i]);
    });
    result
}

/// In-place permutation to order the input for icfft.
///
/// This is an optimized version of `order_icfft_input_naive` that performs the permutation
/// in-place without allocating a new vector (only a small bitvector for tracking visited elements).
///
/// Natural order input: `[P(x0), P(x1), P(x2), ..., P(x_{n-1})]`
///
/// Output for icfft: `[P(x0), P(x2), P(x4), ..., P(x_{n-1}), ..., P(x3), P(x1)]`
/// (even indices ascending, then odd indices descending).
///
/// # Algorithm
///
/// The permutation is the inverse of `order_cfft_result_in_place`:
/// - `output[i] = input[2*i]` for `i < n/2`
/// - `output[n-1-i] = input[2*i+1]` for `i < n/2`
///
/// Equivalently, for each destination position `dst`:
/// - If `dst < n/2`: `output[dst] = input[2*dst]`
/// - If `dst >= n/2`: `output[dst] = input[2*(n-1-dst)+1]`
///
/// We follow cycles to perform swaps in-place, using a bitvector to track visited positions.
pub fn order_icfft_input_in_place(input: &mut [FieldElement<Mersenne31Field>]) {
    let n = input.len();
    if n <= 2 {
        return;
    }

    let half = n / 2;

    // Compute source index for a destination position:
    // The value at `output[dst]` comes from `input[src_for_dst(dst)]`
    // dst < n/2:  output[dst] = input[2*dst]
    // dst >= n/2: output[dst] = input[2*(n-1-dst)+1]
    let src_for_dst = |dst: usize| -> usize {
        if dst < half {
            2 * dst
        } else {
            2 * (n - 1 - dst) + 1
        }
    };

    // Track visited positions with a bitvector
    let mut visited = vec![0u64; n.div_ceil(64)];

    let is_visited = |visited: &[u64], i: usize| -> bool { (visited[i / 64] >> (i % 64)) & 1 == 1 };

    let mark_visited = |visited: &mut [u64], i: usize| {
        visited[i / 64] |= 1u64 << (i % 64);
    };

    for start in 0..n {
        if is_visited(&visited, start) {
            continue;
        }

        // For cycle following with a permutation where we know the SOURCE for each destination,
        // we need to follow the chain: dst <- src_for_dst(dst) <- src_for_dst(src_for_dst(dst)) <- ...
        // Save the original value at `start`, then pull values from their sources
        let mut dst = start;
        let temp = input[dst];

        loop {
            mark_visited(&mut visited, dst);
            let src = src_for_dst(dst);

            if src == start {
                // Cycle complete, place the saved value
                input[dst] = temp;
                break;
            }

            // Pull the value from source to destination
            input[dst] = input[src];
            dst = src;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type FE = FieldElement<Mersenne31Field>;

    #[test]
    fn ordering_cfft_result_works_for_4_points() {
        let expected_slice = [FE::from(0), FE::from(1), FE::from(2), FE::from(3)];

        let slice = [FE::from(0), FE::from(2), FE::from(3), FE::from(1)];

        #[allow(deprecated)]
        let res = order_cfft_result_naive(&slice);

        assert_eq!(res, expected_slice)
    }

    #[test]
    fn ordering_cfft_result_works_for_16_points() {
        let expected_slice = [
            FE::from(0),
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
            FE::from(9),
            FE::from(10),
            FE::from(11),
            FE::from(12),
            FE::from(13),
            FE::from(14),
            FE::from(15),
        ];

        let slice = [
            FE::from(0),
            FE::from(2),
            FE::from(4),
            FE::from(6),
            FE::from(8),
            FE::from(10),
            FE::from(12),
            FE::from(14),
            FE::from(15),
            FE::from(13),
            FE::from(11),
            FE::from(9),
            FE::from(7),
            FE::from(5),
            FE::from(3),
            FE::from(1),
        ];

        #[allow(deprecated)]
        let res = order_cfft_result_naive(&slice);

        assert_eq!(res, expected_slice)
    }

    #[test]
    fn from_natural_to_icfft_input_order_works() {
        let mut slice = [
            FE::from(0),
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
            FE::from(9),
            FE::from(10),
            FE::from(11),
            FE::from(12),
            FE::from(13),
            FE::from(14),
            FE::from(15),
        ];

        let expected_slice = [
            FE::from(0),
            FE::from(2),
            FE::from(4),
            FE::from(6),
            FE::from(8),
            FE::from(10),
            FE::from(12),
            FE::from(14),
            FE::from(15),
            FE::from(13),
            FE::from(11),
            FE::from(9),
            FE::from(7),
            FE::from(5),
            FE::from(3),
            FE::from(1),
        ];

        #[allow(deprecated)]
        let res = order_icfft_input_naive(&mut slice);

        assert_eq!(res, expected_slice)
    }

    // Tests for in-place versions

    #[test]
    fn ordering_cfft_result_in_place_works_for_4_points() {
        let expected_slice = [FE::from(0), FE::from(1), FE::from(2), FE::from(3)];

        let mut slice = [FE::from(0), FE::from(2), FE::from(3), FE::from(1)];

        order_cfft_result_in_place(&mut slice);

        assert_eq!(slice, expected_slice)
    }

    #[test]
    fn ordering_cfft_result_in_place_works_for_8_points() {
        let expected_slice = [
            FE::from(0),
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
        ];

        // CFFT order: even ascending, then odd descending
        // [0, 2, 4, 6, 7, 5, 3, 1]
        let mut slice = [
            FE::from(0),
            FE::from(2),
            FE::from(4),
            FE::from(6),
            FE::from(7),
            FE::from(5),
            FE::from(3),
            FE::from(1),
        ];

        order_cfft_result_in_place(&mut slice);

        assert_eq!(slice, expected_slice)
    }

    #[test]
    fn ordering_cfft_result_in_place_works_for_16_points() {
        let expected_slice = [
            FE::from(0),
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
            FE::from(9),
            FE::from(10),
            FE::from(11),
            FE::from(12),
            FE::from(13),
            FE::from(14),
            FE::from(15),
        ];

        let mut slice = [
            FE::from(0),
            FE::from(2),
            FE::from(4),
            FE::from(6),
            FE::from(8),
            FE::from(10),
            FE::from(12),
            FE::from(14),
            FE::from(15),
            FE::from(13),
            FE::from(11),
            FE::from(9),
            FE::from(7),
            FE::from(5),
            FE::from(3),
            FE::from(1),
        ];

        order_cfft_result_in_place(&mut slice);

        assert_eq!(slice, expected_slice)
    }

    #[test]
    fn ordering_cfft_result_in_place_matches_naive() {
        let slice = [
            FE::from(0),
            FE::from(2),
            FE::from(4),
            FE::from(6),
            FE::from(8),
            FE::from(10),
            FE::from(12),
            FE::from(14),
            FE::from(15),
            FE::from(13),
            FE::from(11),
            FE::from(9),
            FE::from(7),
            FE::from(5),
            FE::from(3),
            FE::from(1),
        ];

        #[allow(deprecated)]
        let naive_result = order_cfft_result_naive(&slice);

        let mut in_place_slice = slice;
        order_cfft_result_in_place(&mut in_place_slice);

        assert_eq!(in_place_slice.to_vec(), naive_result)
    }

    #[test]
    fn order_icfft_input_in_place_works_for_4_points() {
        let expected_slice = [FE::from(0), FE::from(2), FE::from(3), FE::from(1)];

        let mut slice = [FE::from(0), FE::from(1), FE::from(2), FE::from(3)];

        order_icfft_input_in_place(&mut slice);

        assert_eq!(slice, expected_slice)
    }

    #[test]
    fn order_icfft_input_in_place_works_for_8_points() {
        // Natural order
        let mut slice = [
            FE::from(0),
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
        ];

        // Expected ICFFT order: even ascending, then odd descending
        let expected_slice = [
            FE::from(0),
            FE::from(2),
            FE::from(4),
            FE::from(6),
            FE::from(7),
            FE::from(5),
            FE::from(3),
            FE::from(1),
        ];

        order_icfft_input_in_place(&mut slice);

        assert_eq!(slice, expected_slice)
    }

    #[test]
    fn order_icfft_input_in_place_works_for_16_points() {
        let mut slice = [
            FE::from(0),
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
            FE::from(9),
            FE::from(10),
            FE::from(11),
            FE::from(12),
            FE::from(13),
            FE::from(14),
            FE::from(15),
        ];

        let expected_slice = [
            FE::from(0),
            FE::from(2),
            FE::from(4),
            FE::from(6),
            FE::from(8),
            FE::from(10),
            FE::from(12),
            FE::from(14),
            FE::from(15),
            FE::from(13),
            FE::from(11),
            FE::from(9),
            FE::from(7),
            FE::from(5),
            FE::from(3),
            FE::from(1),
        ];

        order_icfft_input_in_place(&mut slice);

        assert_eq!(slice, expected_slice)
    }

    #[test]
    fn order_icfft_input_in_place_matches_naive() {
        let mut slice_for_naive = [
            FE::from(0),
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
            FE::from(9),
            FE::from(10),
            FE::from(11),
            FE::from(12),
            FE::from(13),
            FE::from(14),
            FE::from(15),
        ];

        let mut slice_for_in_place = slice_for_naive;

        #[allow(deprecated)]
        let naive_result = order_icfft_input_naive(&mut slice_for_naive);

        order_icfft_input_in_place(&mut slice_for_in_place);

        assert_eq!(slice_for_in_place.to_vec(), naive_result)
    }

    #[test]
    fn cfft_icfft_orderings_are_inverse() {
        // Start with natural order
        let original = [
            FE::from(0),
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
        ];

        // Apply icfft input ordering (natural -> cfft order)
        let mut slice = original;
        order_icfft_input_in_place(&mut slice);

        // Apply cfft result ordering (cfft order -> natural)
        order_cfft_result_in_place(&mut slice);

        // Should be back to original
        assert_eq!(slice, original)
    }

    #[test]
    fn ordering_edge_case_size_2() {
        // Size 2 is an edge case
        let mut slice = [FE::from(0), FE::from(1)];
        order_cfft_result_in_place(&mut slice);
        // Size 2: input [0, 1] (even ascending: 0, odd descending: 1)
        // Natural order would be [0, 1]
        assert_eq!(slice, [FE::from(0), FE::from(1)]);

        let mut slice2 = [FE::from(0), FE::from(1)];
        order_icfft_input_in_place(&mut slice2);
        // Natural [0, 1] -> ICFFT order [0, 1]
        assert_eq!(slice2, [FE::from(0), FE::from(1)]);
    }
}
