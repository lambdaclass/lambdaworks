use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    traits::ByteConversion,
    unsigned_integer::element::UnsignedInteger,
};

/// Converts a hash, represented by an array of bytes, into a vector of field elements.
/// `l * count` must be equal to the length of `pseudo_random_bytes`.
/// For more info, see
/// https://www.ietf.org/id/draft-irtf-cfrg-hash-to-curve-16.html#name-hashing-to-a-finite-field
pub fn hash_to_field<M: IsModulus<UnsignedInteger<N>> + Clone, const N: usize>(
    pseudo_random_bytes: &[u8],
    count: usize,
    l: usize,
) -> Vec<FieldElement<MontgomeryBackendPrimeField<M, N>>> {
    // TODO: remove this and return Result instead
    assert_eq!(l * count, pseudo_random_bytes.len());
    let mut u = vec![FieldElement::zero(); count];
    for (i, item) in u.iter_mut().enumerate() {
        let elm_offset = l * i;
        let tv = &pseudo_random_bytes[elm_offset..elm_offset + l];

        *item = os2ip::<M, N>(tv);
    }
    u
}

/// Converts an octet string to a nonnegative integer.
/// For more info, see https://www.rfc-editor.org/rfc/pdfrfc/rfc8017.txt.pdf
fn os2ip<M: IsModulus<UnsignedInteger<N>> + Clone, const N: usize>(
    x: &[u8],
) -> FieldElement<MontgomeryBackendPrimeField<M, N>> {
    let mut aux_x = x.to_vec();
    aux_x.reverse();
    let two_to_the_nth = build_two_to_the_nth();
    let mut j = 0_u32;
    let mut item_vec = Vec::with_capacity(N * 8);
    let mut result = FieldElement::zero();
    for item_u8 in aux_x.iter() {
        item_vec.push(*item_u8);
        if item_vec.len() == item_vec.capacity() {
            result += FieldElement::from_bytes_le(&item_vec).unwrap() * two_to_the_nth.pow(j);
            item_vec.clear();
            j += 1;
        }
    }
    result
}

/// Builds a `FieldElement` for `2^(N*16)`, where `N` is the number of limbs of the `UnsignedInteger`
/// used for the prime field.
fn build_two_to_the_nth<M: IsModulus<UnsignedInteger<N>> + Clone, const N: usize>(
) -> FieldElement<MontgomeryBackendPrimeField<M, N>> {
    // The hex used to build the FieldElement is a 1 followed by N * 16 zeros
    let mut two_to_the_nth = String::with_capacity(N * 16);
    for _ in 0..two_to_the_nth.capacity() - 1 {
        two_to_the_nth.push('1');
    }
    FieldElement::from_hex_unchecked(&two_to_the_nth) + FieldElement::one()
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::{
        field::{
            element::FieldElement,
            fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
        },
        unsigned_integer::element::UnsignedInteger,
    };

    use crate::hash::hash_to_field::hash_to_field;

    type F = MontgomeryBackendPrimeField<U64, 1>;

    #[derive(Clone, Debug)]
    struct U64;
    impl IsModulus<UnsignedInteger<1>> for U64 {
        const MODULUS: UnsignedInteger<1> = UnsignedInteger::from_u64(18446744069414584321_u64);
    }

    #[test]
    fn test_same_message_produce_same_field_elements() {
        let input = &(0..40).map(|_| rand::random()).collect::<Vec<u8>>();
        let field_elements: Vec<FieldElement<F>> = hash_to_field(input, 40, 1);
        let other_field_elements = hash_to_field(input, 40, 1);
        assert_eq!(field_elements, other_field_elements);
    }
}
