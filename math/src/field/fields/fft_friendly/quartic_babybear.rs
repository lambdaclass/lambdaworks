use crate::{
    field::{
        element::FieldElement,
        errors::FieldError,
        fields::fft_friendly::babybear::Babybear31PrimeField,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
    traits::ByteConversion,
};

// BETA = 11
// -BETA = -11 is the non-residue.
// We are implementig the extension of Baby Bear of degree 4 using the irreducible polynomial x^4 + 11.
pub const BETA: FieldElement<Babybear31PrimeField> =
    FieldElement::<Babybear31PrimeField>::from_hex_unchecked("b");

#[derive(Clone, Debug)]
pub struct Degree4BabyBearExtensionField;

impl IsField for Degree4BabyBearExtensionField {
    type BaseType = [FieldElement<Babybear31PrimeField>; 4];

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] + &b[0], &a[1] + &b[1], &a[2] + &b[2], &a[3] + &b[3]]
    }

    // Result of multiplying two polynomials a = a0 + a1 * x + a2 * x^2 + a3 * x^3 and
    // b = b0 + b1 * x + b2 * x^2 + b3 * x^3 by applying distribution and taking
    // the remainder of the division by x^4 + 11.
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [
            &a[0] * &b[0] - BETA * (&a[1] * &b[3] + &a[3] * &b[1] + &a[2] * &b[2]),
            &a[0] * &b[1] + &a[1] * &b[0] - BETA * (&a[2] * &b[3] + &a[3] * &b[2]),
            &a[0] * &b[2] + &a[2] * &b[0] + &a[1] * &b[1] - BETA * (&a[3] * &b[3]),
            &a[0] * &b[3] + &a[3] * &b[0] + &a[1] * &b[2] + &a[2] * &b[1],
        ]
    }

    fn square(a: &Self::BaseType) -> Self::BaseType {
        [
            &a[0].square() - BETA * ((&a[1] * &a[3]).double() + &a[2].square()),
            (&a[0] * &a[1] - BETA * (&a[2] * &a[3])).double(),
            (&a[0] * &a[2]).double() + &a[1].square() - BETA * (&a[3].square()),
            (&a[0] * &a[3] + &a[1] * &a[2]).double(),
        ]
    }
    /// Returns the component wise subtraction of `a` and `b`
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] - &b[0], &a[1] - &b[1], &a[2] - &b[2], &a[3] - &b[3]]
    }

    /// Returns the component wise negation of `a`
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-&a[0], -&a[1], -&a[2], -&a[3]]
    }

    /// Returns the multiplicative inverse of `a`
    ///
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let mut b0 = &a[0] * &a[0] + BETA * (&a[1] * (&a[3] + &a[3]) - &a[2] * &a[2]);
        let mut b2 = &a[0] * (&a[2] + &a[2]) - &a[1] * &a[1] + BETA * (&a[3] * &a[3]);
        let c = &b0.square() + BETA * b2.square();
        let c_inv = c.inv()?;
        b0 = b0 * &c_inv;
        b2 = b2 * &c_inv;
        Ok([
            &a[0] * &b0 + BETA * &a[2] * &b2,
            -&a[1] * &b0 - BETA * &a[3] * &b2,
            -&a[0] * &b2 + &a[2] * &b0,
            &a[1] * &b2 - &a[3] * &b0,
        ])
    }

    /// Returns the division of `a` and `b`
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        <Self as IsField>::mul(a, &Self::inv(b).unwrap())
    }

    /// Returns a boolean indicating whether `a` and `b` are equal component wise.
    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
    }

    /// Returns the additive neutral element of the field extension.
    fn zero() -> Self::BaseType {
        [
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// Returns the multiplicative neutral element of the field extension.
    fn one() -> Self::BaseType {
        [
            FieldElement::one(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> Self::BaseType {
        [
            FieldElement::from(x),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    /// Note: for this case this is simply the identity, because the components
    /// already have correct representations.
    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }

    fn double(a: &Self::BaseType) -> Self::BaseType {
        <Degree4BabyBearExtensionField as IsField>::add(a, a)
    }

    fn pow<T>(a: &Self::BaseType, mut exponent: T) -> Self::BaseType
    where
        T: crate::unsigned_integer::traits::IsUnsignedInteger,
    {
        let zero = T::from(0);
        let one = T::from(1);

        if exponent == zero {
            Self::one()
        } else if exponent == one {
            a.clone()
        } else {
            let mut result = a.clone();

            while exponent & one == zero {
                result = Self::square(&result);
                exponent >>= 1;
            }

            if exponent == zero {
                result
            } else {
                let mut base = result.clone();
                exponent >>= 1;

                while exponent != zero {
                    base = Self::square(&base);
                    if exponent & one == one {
                        result = <Degree4BabyBearExtensionField as IsField>::mul(&result, &base);
                    }
                    exponent >>= 1;
                }

                result
            }
        }
    }
}

impl IsSubFieldOf<Degree4BabyBearExtensionField> for Babybear31PrimeField {
    fn mul(
        a: &Self::BaseType,
        b: &<Degree4BabyBearExtensionField as IsField>::BaseType,
    ) -> <Degree4BabyBearExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::mul(a, b[0].value()));
        let c1 = FieldElement::from_raw(<Self as IsField>::mul(a, b[1].value()));
        let c2 = FieldElement::from_raw(<Self as IsField>::mul(a, b[2].value()));
        let c3 = FieldElement::from_raw(<Self as IsField>::mul(a, b[3].value()));

        [c0, c1, c2, c3]
    }

    fn add(
        a: &Self::BaseType,
        b: &<Degree4BabyBearExtensionField as IsField>::BaseType,
    ) -> <Degree4BabyBearExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::add(a, b[0].value()));
        let c1 = FieldElement::from_raw(*b[1].value());
        let c2 = FieldElement::from_raw(*b[2].value());
        let c3 = FieldElement::from_raw(*b[3].value());

        [c0, c1, c2, c3]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree4BabyBearExtensionField as IsField>::BaseType,
    ) -> <Degree4BabyBearExtensionField as IsField>::BaseType {
        let b_inv = Degree4BabyBearExtensionField::inv(b).unwrap();
        <Self as IsSubFieldOf<Degree4BabyBearExtensionField>>::mul(a, &b_inv)
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree4BabyBearExtensionField as IsField>::BaseType,
    ) -> <Degree4BabyBearExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::sub(a, b[0].value()));
        let c1 = FieldElement::from_raw(<Self as IsField>::neg(b[1].value()));
        let c2 = FieldElement::from_raw(<Self as IsField>::neg(b[2].value()));
        let c3 = FieldElement::from_raw(<Self as IsField>::neg(b[3].value()));
        [c0, c1, c2, c3]
    }

    fn embed(a: Self::BaseType) -> <Degree4BabyBearExtensionField as IsField>::BaseType {
        [
            FieldElement::from_raw(a),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree4BabyBearExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

#[cfg(feature = "lambdaworks-serde-binary")]
impl ByteConversion for [FieldElement<Babybear31PrimeField>; 4] {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        unimplemented!()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        unimplemented!()
    }

    fn from_bytes_be(_bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        unimplemented!()
    }

    fn from_bytes_le(_bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}

impl IsFFTField for Degree4BabyBearExtensionField {
    const TWO_ADICITY: u64 = 29;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: Self::BaseType = [
        FieldElement::from_hex_unchecked("0"),
        FieldElement::from_hex_unchecked("0"),
        FieldElement::from_hex_unchecked("0"),
        FieldElement::from_hex_unchecked("771F1C8"),
    ];
}

#[cfg(test)]
mod tests {

    use crate::{
        fft::cpu::roots_of_unity::{
            get_powers_of_primitive_root, get_powers_of_primitive_root_coset,
        },
        field::traits::RootsConfig,
        polynomial::Polynomial,
    };

    use super::*;

    type FpE = FieldElement<Babybear31PrimeField>;
    type Fp4E = FieldElement<Degree4BabyBearExtensionField>;

    #[test]
    fn test_add() {
        let a = Fp4E::new([FpE::from(0), FpE::from(1), FpE::from(2), FpE::from(3)]);
        let b = Fp4E::new([-FpE::from(2), FpE::from(4), FpE::from(6), -FpE::from(8)]);
        let expected_result = Fp4E::new([
            FpE::from(0) - FpE::from(2),
            FpE::from(1) + FpE::from(4),
            FpE::from(2) + FpE::from(6),
            FpE::from(3) - FpE::from(8),
        ]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_sub() {
        let a = Fp4E::new([FpE::from(0), FpE::from(1), FpE::from(2), FpE::from(3)]);
        let b = Fp4E::new([-FpE::from(2), FpE::from(4), FpE::from(6), -FpE::from(8)]);
        let expected_result = Fp4E::new([
            FpE::from(0) + FpE::from(2),
            FpE::from(1) - FpE::from(4),
            FpE::from(2) - FpE::from(6),
            FpE::from(3) + FpE::from(8),
        ]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_mul_by_0() {
        let a = Fp4E::new([FpE::from(4), FpE::from(1), FpE::from(2), FpE::from(3)]);
        let b = Fp4E::new([FpE::zero(), FpE::zero(), FpE::zero(), FpE::zero()]);
        assert_eq!(&a * &b, b);
    }

    #[test]
    fn test_mul_by_1() {
        let a = Fp4E::new([FpE::from(4), FpE::from(1), FpE::from(2), FpE::from(3)]);
        let b = Fp4E::new([FpE::one(), FpE::zero(), FpE::zero(), FpE::zero()]);
        assert_eq!(&a * b, a);
    }

    #[test]
    fn test_mul() {
        let a = Fp4E::new([FpE::from(0), FpE::from(1), FpE::from(2), FpE::from(3)]);
        let b = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);
        let expected_result = Fp4E::new([
            -FpE::from(352),
            -FpE::from(372),
            -FpE::from(256),
            FpE::from(20),
        ]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_pow() {
        let a = Fp4E::new([FpE::from(0), FpE::from(1), FpE::from(2), FpE::from(3)]);
        let expected_result = &a * &a * &a;
        assert_eq!(a.pow(3u64), expected_result);
    }

    #[test]
    fn test_inv_of_one_is_one() {
        let a = Fp4E::one();
        assert_eq!(a.inv().unwrap(), a);
    }

    #[test]
    fn test_mul_by_inv_is_identity() {
        let a = Fp4E::from(123456);
        assert_eq!(&a * a.inv().unwrap(), Fp4E::one());
    }

    #[test]
    fn test_mul_as_subfield() {
        let a = FpE::from(2);
        let b = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);
        let expected_result = Fp4E::new([
            FpE::from(2) * FpE::from(2),
            FpE::from(4) * FpE::from(2),
            FpE::from(6) * FpE::from(2),
            FpE::from(8) * FpE::from(2),
        ]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_double_equals_sum_two_times() {
        let a = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);

        assert_eq!(a.double(), &a + &a);
    }

    #[test]
    fn test_mul_group_generator_pow_order_is_one() {
        let generator = Fp4E::new([FpE::from(8), FpE::from(1), FpE::zero(), FpE::zero()]);
        let extension_order: u128 = 78000001_u128.pow(4);
        assert_eq!(generator.pow(extension_order), generator);
    }

    #[test]
    fn test_two_adic_primitve_root_of_unity() {
        let generator = Fp4E::new(Degree4BabyBearExtensionField::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY);
        assert_eq!(
            generator.pow(2u64.pow(Degree4BabyBearExtensionField::TWO_ADICITY as u32)),
            Fp4E::one()
        );
    }

    #[test]
    fn test_fft() {
        let c0 = Fp4E::new([FpE::from(1), FpE::from(2), FpE::from(3), FpE::from(4)]);
        let c1 = Fp4E::new([FpE::from(2), FpE::from(3), FpE::from(4), FpE::from(5)]);
        let c2 = Fp4E::new([FpE::from(3), FpE::from(4), FpE::from(5), FpE::from(6)]);
        let c3 = Fp4E::new([FpE::from(4), FpE::from(5), FpE::from(6), FpE::from(7)]);
        let c4 = Fp4E::new([FpE::from(5), FpE::from(6), FpE::from(7), FpE::from(8)]);
        let c5 = Fp4E::new([FpE::from(6), FpE::from(7), FpE::from(8), FpE::from(9)]);
        let c6 = Fp4E::new([FpE::from(7), FpE::from(8), FpE::from(9), FpE::from(0)]);
        let c7 = Fp4E::new([FpE::from(8), FpE::from(9), FpE::from(0), FpE::from(1)]);

        let poly = Polynomial::new(&[c0, c1, c2, c3, c4, c5, c6, c7]);
        let evaluations =
            Polynomial::evaluate_fft::<Degree4BabyBearExtensionField>(&poly, 1, None).unwrap();
        let poly_interpol =
            Polynomial::interpolate_fft::<Degree4BabyBearExtensionField>(&evaluations).unwrap();

        assert_eq!(poly, poly_interpol)
    }

    #[test]
    fn test_fft_and_naive_evaluation() {
        let c0 = Fp4E::new([FpE::from(1), FpE::from(2), FpE::from(3), FpE::from(4)]);
        let c1 = Fp4E::new([FpE::from(2), FpE::from(3), FpE::from(4), FpE::from(5)]);
        let c2 = Fp4E::new([FpE::from(3), FpE::from(4), FpE::from(5), FpE::from(6)]);
        let c3 = Fp4E::new([FpE::from(4), FpE::from(5), FpE::from(6), FpE::from(7)]);
        let c4 = Fp4E::new([FpE::from(5), FpE::from(6), FpE::from(7), FpE::from(8)]);
        let c5 = Fp4E::new([FpE::from(6), FpE::from(7), FpE::from(8), FpE::from(9)]);
        let c6 = Fp4E::new([FpE::from(7), FpE::from(8), FpE::from(9), FpE::from(0)]);
        let c7 = Fp4E::new([FpE::from(8), FpE::from(9), FpE::from(0), FpE::from(1)]);

        let poly = Polynomial::new(&[c0, c1, c2, c3, c4, c5, c6, c7]);

        let len = poly.coeff_len().next_power_of_two();
        let order = len.trailing_zeros();
        let twiddles =
            get_powers_of_primitive_root(order.into(), len, RootsConfig::Natural).unwrap();

        let fft_eval =
            Polynomial::evaluate_fft::<Degree4BabyBearExtensionField>(&poly, 1, None).unwrap();
        let naive_eval = poly.evaluate_slice(&twiddles);

        assert_eq!(fft_eval, naive_eval);
    }

    #[test]
    fn gen_fft_coset_and_naive_evaluation() {
        let c0 = Fp4E::new([FpE::from(1), FpE::from(2), FpE::from(3), FpE::from(4)]);
        let c1 = Fp4E::new([FpE::from(2), FpE::from(3), FpE::from(4), FpE::from(5)]);
        let c2 = Fp4E::new([FpE::from(3), FpE::from(4), FpE::from(5), FpE::from(6)]);
        let c3 = Fp4E::new([FpE::from(4), FpE::from(5), FpE::from(6), FpE::from(7)]);
        let c4 = Fp4E::new([FpE::from(5), FpE::from(6), FpE::from(7), FpE::from(8)]);
        let c5 = Fp4E::new([FpE::from(6), FpE::from(7), FpE::from(8), FpE::from(9)]);
        let c6 = Fp4E::new([FpE::from(7), FpE::from(8), FpE::from(9), FpE::from(0)]);
        let c7 = Fp4E::new([FpE::from(8), FpE::from(9), FpE::from(0), FpE::from(1)]);

        let poly = Polynomial::new(&[c0, c1, c2, c3, c4, c5, c6, c7]);

        let offset = Fp4E::new([FpE::from(10), FpE::from(11), FpE::from(12), FpE::from(13)]);
        let blowup_factor = 4;

        let len = poly.coeff_len().next_power_of_two();
        let order = (len * blowup_factor).trailing_zeros();
        let twiddles =
            get_powers_of_primitive_root_coset(order.into(), len * blowup_factor, &offset).unwrap();

        let fft_eval = Polynomial::evaluate_offset_fft::<Degree4BabyBearExtensionField>(
            &poly,
            blowup_factor,
            None,
            &offset,
        )
        .unwrap();
        let naive_eval = poly.evaluate_slice(&twiddles);

        assert_eq!(fft_eval, naive_eval);
    }

    #[test]
    fn test_fft_and_naive_interpolate() {
        let c0 = Fp4E::new([FpE::from(1), FpE::from(2), FpE::from(3), FpE::from(4)]);
        let c1 = Fp4E::new([FpE::from(2), FpE::from(3), FpE::from(4), FpE::from(5)]);
        let c2 = Fp4E::new([FpE::from(3), FpE::from(4), FpE::from(5), FpE::from(6)]);
        let c3 = Fp4E::new([FpE::from(4), FpE::from(5), FpE::from(6), FpE::from(7)]);
        let c4 = Fp4E::new([FpE::from(5), FpE::from(6), FpE::from(7), FpE::from(8)]);
        let c5 = Fp4E::new([FpE::from(6), FpE::from(7), FpE::from(8), FpE::from(9)]);
        let c6 = Fp4E::new([FpE::from(7), FpE::from(8), FpE::from(9), FpE::from(0)]);
        let c7 = Fp4E::new([FpE::from(8), FpE::from(9), FpE::from(0), FpE::from(1)]);

        let fft_evals = [c0, c1, c2, c3, c4, c5, c6, c7];
        let order = fft_evals.len().trailing_zeros() as u64;
        let twiddles: Vec<FieldElement<Degree4BabyBearExtensionField>> =
            get_powers_of_primitive_root(order, 1 << order, RootsConfig::Natural).unwrap();

        let naive_poly = Polynomial::interpolate(&twiddles, &fft_evals).unwrap();
        let fft_poly =
            Polynomial::interpolate_fft::<Degree4BabyBearExtensionField>(&fft_evals).unwrap();

        assert_eq!(fft_poly, naive_poly)
    }
    /*
    #[test]
    fn gen_fft_and_naive_coset_interpolate(
        fft_evals: &[FieldElement<F>],
    ) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<F>>) {
        let order = fft_evals.len().trailing_zeros() as u64;
        let twiddles = get_powers_of_primitive_root_coset(order, 1 << order, offset).unwrap();
        let offset = Fp4E::new([FpE::from(10), FpE::from(11), FpE::from(12), FpE::from(13)]);

        let naive_poly = Polynomial::interpolate(&twiddles, fft_evals).unwrap();
        let fft_poly = Polynomial::interpolate_offset_fft(fft_evals, &offset).unwrap();
    }*/
}
