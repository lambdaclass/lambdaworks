extern crate alloc;
use crate::field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field};

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
pub fn order_cfft_result_naive(
    input: &mut [FieldElement<Mersenne31Field>],
) -> Vec<FieldElement<Mersenne31Field>> {
    let mut result = Vec::new();
    let length = input.len();
    for i in 0..length / 2 {
        result.push(input[i]); // We push the left index.
        result.push(input[length - i - 1]); // We push the right index.
    }
    result
}

/// This function permutes a slice of field elements to order the input of the icfft in a specific way.
/// For example, if we want to interpolate 8 points we should input them in the icfft in this order:
/// [(x0, y0), (x2, y2), (x4, y4), (x6, y6), (x7, y7), (x5, y5), (x3, y3), (x1, y1)],
/// where the even indices are found first in ascending order and then the odd indices in descending order.
/// This function permutes the slice [0, 1, 2, 3, 4, 5, 6, 7] into [0, 2, 4, 6, 7, 5, 3, 1].
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

// We are not using this fucntion.
pub fn reverse_cfft_index(index: usize, length: usize) -> usize {
    if index < (length >> 1) {
        // index < length / 2
        index << 1 // index * 2
    } else {
        (((length - 1) - index) << 1) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type FE = FieldElement<Mersenne31Field>;

    #[test]
    fn ordering_cfft_result_works_for_4_points() {
        let expected_slice = [FE::from(0), FE::from(1), FE::from(2), FE::from(3)];

        let mut slice = [FE::from(0), FE::from(2), FE::from(3), FE::from(1)];

        let res = order_cfft_result_naive(&mut slice);

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

        let res = order_cfft_result_naive(&mut slice);

        assert_eq!(res, expected_slice)
    }

    #[test]
    fn reverse_cfft_index_works() {
        let mut reversed: Vec<usize> = Vec::with_capacity(16);
        for i in 0..reversed.capacity() {
            reversed.push(reverse_cfft_index(i, reversed.capacity()));
        }
        assert_eq!(
            reversed[..],
            [0, 2, 4, 6, 8, 10, 12, 14, 15, 13, 11, 9, 7, 5, 3, 1]
        );
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

        let res = order_icfft_input_naive(&mut slice);

        assert_eq!(res, expected_slice)
    }
}
