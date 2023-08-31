use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::UnsignedInteger,
};

/// Converts a hash, represented by an array of bytes, into a vector of field elements
/// For more info, see
/// https://www.ietf.org/id/draft-irtf-cfrg-hash-to-curve-16.html#name-hashing-to-a-finite-field
pub fn hash_to_field<M: IsModulus<UnsignedInteger<N>> + Clone, const N: usize>(
    pseudo_random_bytes: &[u8],
    count: usize,
) -> Vec<FieldElement<MontgomeryBackendPrimeField<M, N>>> {
    let order = M::MODULUS;
    let l = compute_length(order);
    let mut u = vec![FieldElement::zero(); count];
    for (i, item) in u.iter_mut().enumerate() {
        let elm_offset = l * i;
        let tv = &pseudo_random_bytes[elm_offset..elm_offset + l];

        *item = os2ip::<M, N>(tv);
    }
    u
}

fn compute_length<const N: usize>(order: UnsignedInteger<N>) -> usize {
    //L = ceil((ceil(log2(p)) + k) / 8), where k is the security parameter of the cryptosystem (e.g. k = ceil(log2(p) / 2))
    let log2_p = order.limbs.len() << 3;
    ((log2_p << 3) + (log2_p << 2)) >> 3
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
    let mut item_hex = String::with_capacity(N * 16);
    let mut result = FieldElement::zero();
    for item_u8 in aux_x.iter() {
        item_hex += &format!("{:x}", item_u8);
        if item_hex.len() == item_hex.capacity() {
            result += FieldElement::from_hex_unchecked(&item_hex) * two_to_the_nth.pow(j);
            item_hex.clear();
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

    use crate::hash::{hash_to_field::hash_to_field, sha3::Sha3Hasher};

    type F = MontgomeryBackendPrimeField<U64, 1>;

    #[derive(Clone, Debug)]
    struct U64;
    impl IsModulus<UnsignedInteger<1>> for U64 {
        const MODULUS: UnsignedInteger<1> = UnsignedInteger::from_u64(18446744069414584321_u64);
    }

    #[test]
    fn test_same_message_produce_same_field_elements() {
        let input = Sha3Hasher::expand_message(b"helloworld", b"dsttest", 500).unwrap();
        let field_elements: Vec<FieldElement<F>> = hash_to_field(&input, 40);
        let other_field_elements = hash_to_field(&input, 40);
        assert_eq!(field_elements, other_field_elements);
    }
}
