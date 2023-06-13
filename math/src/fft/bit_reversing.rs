/// In-place bit-reverse permutation algorithm. Requires input length to be a power of two.
pub fn in_place_bit_reverse_permute<E>(input: &mut [E]) {
    for i in 0..input.len() {
        let bit_reversed_index = reverse_index(&i, input.len() as u64);
        if bit_reversed_index > i {
            input.swap(i, bit_reversed_index);
        }
    }
}

/// Reverses the `log2(size)` first bits of `i`
pub fn reverse_index(i: &usize, size: u64) -> usize {
    if size == 1 {
        *i
    } else {
        i.reverse_bits() >> (usize::BITS - size.trailing_zeros())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // TODO: proptest would be better.
    #[test]
    fn bit_reverse_permutation_works() {
        let mut reversed: Vec<usize> = Vec::with_capacity(16);
        for i in 0..reversed.capacity() {
            reversed.push(reverse_index(&i, reversed.capacity() as u64));
        }
        assert_eq!(
            reversed[..],
            [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
        );

        in_place_bit_reverse_permute(&mut reversed[..]);
        assert_eq!(
            reversed[..],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );
    }

    #[test]
    fn bit_reverse_permutation_edge_case() {
        let mut edge_case = [0];

        in_place_bit_reverse_permute(&mut edge_case[..]);
        assert_eq!(edge_case[..], [0]);
    }
}
