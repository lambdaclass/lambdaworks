use std::fmt::Debug;

use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::UnsignedInteger,
};

pub fn hash_to_field<M: IsModulus<UnsignedInteger<N>> + Clone + Debug, const N: usize>(
    pseudo_random_bytes: &[u8],
    count: usize,
) -> Vec<FieldElement<MontgomeryBackendPrimeField<M, N>>> {
    let order = M::MODULUS;
    let mut u = vec![FieldElement::zero(); count as usize];
    //L = ceil((ceil(log2(p)) + k) / 8), where k is the security parameter of the cryptosystem (e.g. k = ceil(log2(p) / 2))
    let log2_p = (order.limbs.len() * 8) as f64;
    let k = (log2_p / 2.0).ceil() * 8.0;
    let l = (((log2_p * 8.0) + k) / 8.0).ceil() as usize;
    for i in 0..count {
        let elm_offset = l * i;
        let tv = &pseudo_random_bytes[elm_offset as usize..elm_offset as usize + l as usize];

        u[i as usize] = os2ip::<M, N>(tv);
    }
    u
}

/// Converts an octet string to a nonnegative integer.
/// For more info, see https://www.rfc-editor.org/rfc/pdfrfc/rfc8017.txt.pdf
fn os2ip<M: IsModulus<UnsignedInteger<N>> + Clone + Debug, const N: usize>(
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
            result += FieldElement::from_hex(&item_hex) * two_to_the_nth.pow(j);
            item_hex.clear();
            j += 1;
        }
    }
    result
}

/// Builds a `FieldElement` for `2^(N*16)`, where `N` is the number of limbs of the `UnsignedInteger`
/// used for the prime field.
fn build_two_to_the_nth<M: IsModulus<UnsignedInteger<N>> + Clone + Debug, const N: usize>(
) -> FieldElement<MontgomeryBackendPrimeField<M, N>> {
    // The hex used to build the FieldElement is a 1 followed by N * 16 zeros
    let mut two_to_the_nth = String::with_capacity(N * 16);
    for _ in 0..two_to_the_nth.capacity() - 1 {
        two_to_the_nth.push('1');
    }
    FieldElement::from_hex(&two_to_the_nth) + FieldElement::one()
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
    use rand::random;

    use crate::hash::hash_to_field::hash_to_field;

    type F = MontgomeryBackendPrimeField<U64, 1>;

    #[derive(Clone, Debug)]
    struct U64;
    impl IsModulus<UnsignedInteger<1>> for U64 {
        const MODULUS: UnsignedInteger<1> = UnsignedInteger::from_u64(18446744069414584321_u64);
    }

    #[test]
    fn test_same_message_produce_same_field_elements() {
        let input = (0..500).map(|_| random::<u8>()).collect::<Vec<u8>>();
        let field_elements: Vec<FieldElement<F>> = hash_to_field(&input, 40);
        let other_field_elements = hash_to_field(&input, 40);
        assert_eq!(field_elements, other_field_elements);
    }
}
