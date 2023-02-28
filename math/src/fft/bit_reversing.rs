fn in_place_bit_reverse_permute<E>(input: &mut [E]) {
    for i in 0..input.len() {
        let bit_reversed_index = i.reverse_bits() >> (usize::BITS - input.len().trailing_zeros());
        if bit_reversed_index > i {
            input.swap(i, bit_reversed_index);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bit_reverse_permutation_works() {
        let mut reversed: Vec<usize> = Vec::with_capacity(16);
        for i in 0..reversed.capacity() {
            reversed.push(
                i.clone().reverse_bits() >> (usize::BITS - reversed.capacity().trailing_zeros()),
            );
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
}
