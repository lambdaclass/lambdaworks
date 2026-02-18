use crate::field::fields::u64_prime_field::U64PrimeField;
use crate::field::traits::IsFFTField;

/// Dilithium prime field: q = 8380417
///
/// q - 1 = 8380416 = 2^13 * 1023, where 1023 = 3 * 11 * 31.
/// This gives TWO_ADICITY = 13, which supports NTT of size up to 2^13 = 8192.
/// Dilithium uses N = 256 = 2^8, so this is more than sufficient.
pub type DilithiumField = U64PrimeField<8380417>;

impl IsFFTField for DilithiumField {
    const TWO_ADICITY: u64 = 13;

    // Primitive 2^13-th root of unity: 10^1023 mod q = 1938117
    // where 10 is a primitive root of Z_q* and 1023 = (q-1)/2^13.
    // Satisfies: 1938117^(2^13) ≡ 1 (mod q) and 1938117^(2^12) ≡ -1 (mod q).
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: u64 = 1938117;

    fn field_name() -> &'static str {
        "dilithium_q"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::element::FieldElement;
    use crate::field::traits::IsPrimeField;
    use alloc::vec::Vec;

    type FE = FieldElement<DilithiumField>;

    const Q: u64 = 8380417;

    #[test]
    fn field_order_is_correct() {
        assert_eq!(FE::from(Q - 1) + FE::from(1), FE::from(0));
    }

    #[test]
    fn basic_arithmetic() {
        let a = FE::from(2u64);
        let b = FE::from(3u64);
        assert_eq!(a + b, FE::from(5u64));
        assert_eq!(a * b, FE::from(6u64));
        assert_eq!(b - a, FE::from(1u64));
    }

    #[test]
    fn mul_order_minus_1_squared_is_one() {
        let a = FE::from(Q - 1);
        assert_eq!(a * a, FE::from(1u64));
    }

    #[test]
    fn inv_works() {
        let a = FE::from(1753u64);
        let inv_a = a.inv().unwrap();
        assert_eq!(a * inv_a, FE::from(1u64));
    }

    #[test]
    fn negation_works() {
        let a = FE::from(42u64);
        assert_eq!(a + (-a), FE::from(0u64));
    }

    #[test]
    fn two_adicity_is_correct() {
        // q - 1 = 2^13 * 1023
        let q_minus_1 = Q - 1;
        assert_eq!(q_minus_1 >> 13, 1023);
        assert_eq!(1023 % 2, 1); // odd part is odd
    }

    #[test]
    fn primitive_root_of_unity_order() {
        let w = FE::new(DilithiumField::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY);
        // w^(2^13) = 1
        assert_eq!(w.pow(1u64 << 13), FE::from(1u64));
        // w^(2^12) != 1 (should be -1)
        assert_eq!(w.pow(1u64 << 12), FE::from(Q - 1));
    }

    #[test]
    fn get_primitive_root_of_unity_for_dilithium_ntt() {
        // Dilithium uses N=256=2^8, so we need an 8th-order root
        let w256 = DilithiumField::get_primitive_root_of_unity(8).unwrap();
        assert_eq!(w256.pow(256u64), FE::from(1u64));
        assert_ne!(w256.pow(128u64), FE::from(1u64));
    }

    #[test]
    fn fft_round_trip() {
        use crate::polynomial::Polynomial;

        let coeffs: Vec<FE> = (0..8).map(|i| FE::from(i as u64)).collect();
        let poly = Polynomial::new(&coeffs);
        let evals = Polynomial::evaluate_fft::<DilithiumField>(&poly, 1, None).unwrap();
        let recovered = Polynomial::interpolate_fft::<DilithiumField>(&evals).unwrap();
        assert_eq!(poly, recovered);
    }

    #[test]
    fn field_bit_size() {
        assert_eq!(DilithiumField::field_bit_size(), 23);
    }
}
