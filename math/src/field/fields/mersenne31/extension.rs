use crate::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::LevelTwoResidue,
    field::{
        element::FieldElement,
        errors::FieldError,
        extensions::{
            cubic::{CubicExtensionField, HasCubicNonResidue},
            quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
        },
        traits::{IsField, IsSubFieldOf},
    },
};

use super::field::Mersenne31Field;

type FpE = FieldElement<Mersenne31Field>;

//Note: The inverse calculation in mersenne31/plonky3 differs from the default quadratic extension so I implemented the complex extension.
//////////////////
#[derive(Clone, Debug)]
pub struct Degree2ExtensionField;

impl IsField for Degree2ExtensionField {
    //Elements represents a[0] = real, a[1] = imaginary
    type BaseType = [FpE; 2];

    /// Returns the component wise addition of `a` and `b`
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1]]
    }

    /// Returns the multiplication of `a` and `b` using the following
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let a0b0 = a[0] * b[0];
        let a1b1 = a[1] * b[1];
        let z = (a[0] + a[1]) * (b[0] + b[1]);
        [a0b0 - a1b1, z - a0b0 - a1b1]
    }

    fn square(a: &Self::BaseType) -> Self::BaseType {
        let [a0, a1] = a;
        let v0 = a0 * a1;
        let c0 = (a0 + a1) * (a0 - a1);
        let c1 = v0.double();
        [c0, c1]
    }
    /// Returns the component wise subtraction of `a` and `b`
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1]]
    }

    /// Returns the component wise negation of `a`
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-a[0], -a[1]]
    }

    /// Returns the multiplicative inverse of `a`
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let inv_norm = (a[0].square() + a[1].square()).inv()?;
        Ok([a[0] * inv_norm, -a[1] * inv_norm])
    }

    /// Returns the division of `a` and `b`
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        <Self as IsField>::mul(a, &Self::inv(b).unwrap())
    }

    /// Returns a boolean indicating whether `a` and `b` are equal component wise.
    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    /// Returns the additive neutral element of the field extension.
    fn zero() -> Self::BaseType {
        [FpE::zero(), FpE::zero()]
    }

    /// Returns the multiplicative neutral element of the field extension.
    fn one() -> Self::BaseType {
        [FpE::one(), FpE::zero()]
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> Self::BaseType {
        [FpE::from(x), FpE::zero()]
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    /// Note: for this case this is simply the identity, because the components
    /// already have correct representations.
    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }
}

impl IsSubFieldOf<Degree2ExtensionField> for Mersenne31Field {
    fn mul(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        let c0 = FpE::from(a) * b[0];
        let c1 = FpE::from(a) * b[1];
        [c0, c1]
    }

    fn add(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::add(a, b[0].value()));
        let c1 = FieldElement::from_raw(*b[1].value());
        [c0, c1]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        let b_inv = Degree2ExtensionField::inv(b).unwrap();
        <Self as IsSubFieldOf<Degree2ExtensionField>>::mul(a, &b_inv)
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::sub(a, b[0].value()));
        let c1 = FieldElement::from_raw(<Self as IsField>::neg(b[1].value()));
        [c0, c1]
    }

    fn embed(a: Self::BaseType) -> <Degree2ExtensionField as IsField>::BaseType {
        [FieldElement::from_raw(a), FieldElement::zero()]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree2ExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

type Fp2E = FieldElement<Degree2ExtensionField>;

/// Extension of degree 4 defined with lambdaworks quadratic extension to test the correctness of Degree4ExtensionField
#[derive(Debug, Clone)]
pub struct Mersenne31LevelTwoResidue;
impl HasQuadraticNonResidue<Degree2ExtensionField> for Mersenne31LevelTwoResidue {
    fn residue() -> Fp2E {
        Fp2E::new([FpE::from(2), FpE::one()])
    }
}
pub type Degree4ExtensionFieldV2 =
    QuadraticExtensionField<Degree2ExtensionField, Mersenne31LevelTwoResidue>;

/// I = 0 + 1 * i
pub const I: Fp2E = Fp2E::const_from_raw([FpE::const_from_raw(0), FpE::const_from_raw(1)]);

/// TWO_PLUS_I = 2 + 1 is the non-residue of Fp2 used for the Fp4 extension.
pub const TWO_PLUS_I: Fp2E = Fp2E::const_from_raw([FpE::const_from_raw(2), FpE::const_from_raw(1)]);

pub fn mul_fp2_by_nonresidue(a: &Fp2E) -> Fp2E {
    Fp2E::new([
        a.value()[0].double() - a.value()[1],
        &a.value()[1].double() + &a.value()[0],
    ])
}
#[derive(Clone, Debug)]
pub struct Degree4ExtensionField;

impl IsField for Degree4ExtensionField {
    type BaseType = [Fp2E; 2];

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] + &b[0], &a[1] + &b[1]]
    }

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] - &b[0], &a[1] - &b[1]]
    }

    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-&a[0], -&a[1]]
    }

    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        /*
        // VERSION 1 (distribution by hand):
        // a = a0 + a1 * u, b = b0 + b1 * u, where
        // a0 = a00 + a01 * i, a1 = a11 + a11 * i, etc.
        let [a00, a01] = a[0].value();
        let [a10, a11] = a[1].value();
        let [b00, b01] = b[0].value();
        let [b10, b11] = b[1].value();

        let a10b10 = a10 * b10;
        let a10b11 = a10 * b11;
        let a11b10 = b10 * a11;
        let a11b11 = a11 * b11;

        let c00 = a00 * b00 - a01 * b01 + a10b10.double() - a10b11 - a11b10 - (a11b11).double();
        let c01 = a00 * b01 + a01 * b00 + a10b10 + a10b11.double() + a11b10.double() - a11b11;
        let c10 = a00 * b10 - a01 * b11 + a10 * b00 - b01 * a11;
        let c11 = a00 * b11 + a01 * b10 + a10 * b01 + a11 * b00;

        [Fp2E::new([c00, c01]), Fp2E::new([c10, c11])]
        */

        // VERSION 2 (paper):
        let a0b0 = &a[0] * &b[0];
        let a1b1 = &a[1] * &b[1];
        [
            &a0b0 + mul_fp2_by_nonresidue(&a1b1),
            (&a[0] + &a[1]) * (&b[0] + &b[1]) - a0b0 - a1b1,
        ]
    }

    fn square(a: &Self::BaseType) -> Self::BaseType {
        // a = a0 + a1 * u, where
        // a0 = a00 + a01 * i and a1 = a11 + a11 * i
        let [a00, a01] = a[0].value();
        let [a10, a11] = a[1].value();

        let a10a10 = a10 * a10;
        let a10a11 = a10 * a11;
        let a11a11 = a11 * a11;

        let c00 = a00 * a00 - a01 * a01 + a10a10.double() - a10a11.double() - (a11a11).double();
        let c01 = (a00 * a01).double() + a10a10 + a10a11.double().double() - a11a11;
        let c10 = (a00 * a10).double() - (a01 * a11).double();
        let c11 = (a00 * a11).double() + (a01 * a10).double();

        [Fp2E::new([c00, c01]), Fp2E::new([c10, c11])]
    }

    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        // VERSION 1:
        // let a1_square = a[1].square();
        // let inv_norm = (a[0].square() - a1_square.double() - a1_square * I).inv()?;

        // VERSION 2:
        let inv_norm = (a[0].square() - mul_fp2_by_nonresidue(&a[1].square())).inv()?;
        Ok([&a[0] * &inv_norm, -&a[1] * &inv_norm])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        <Self as IsField>::mul(a, &Self::inv(b).unwrap())
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    fn zero() -> Self::BaseType {
        [Fp2E::zero(), Fp2E::zero()]
    }

    fn one() -> Self::BaseType {
        [Fp2E::one(), Fp2E::zero()]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [Fp2E::from(x), Fp2E::zero()]
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }
}

impl IsSubFieldOf<Degree4ExtensionField> for Mersenne31Field {
    fn mul(
        a: &Self::BaseType,
        b: &<Degree4ExtensionField as IsField>::BaseType,
    ) -> <Degree4ExtensionField as IsField>::BaseType {
        let c0 = FpE::from(a) * &b[0];
        let c1 = FpE::from(a) * &b[1];
        [c0, c1]
    }

    fn add(
        a: &Self::BaseType,
        b: &<Degree4ExtensionField as IsField>::BaseType,
    ) -> <Degree4ExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsSubFieldOf<Degree2ExtensionField>>::add(
            a,
            b[0].value(),
        ));
        let c1 = FieldElement::from_raw(*b[1].value());
        [c0, c1]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree4ExtensionField as IsField>::BaseType,
    ) -> <Degree4ExtensionField as IsField>::BaseType {
        let b_inv = Degree4ExtensionField::inv(b).unwrap();
        <Self as IsSubFieldOf<Degree4ExtensionField>>::mul(a, &b_inv)
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree4ExtensionField as IsField>::BaseType,
    ) -> <Degree4ExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsSubFieldOf<Degree2ExtensionField>>::sub(
            a,
            b[0].value(),
        ));
        let c1 = FieldElement::from_raw(<Degree2ExtensionField as IsField>::neg(b[1].value()));
        [c0, c1]
    }

    fn embed(a: Self::BaseType) -> <Degree4ExtensionField as IsField>::BaseType {
        [
            Fp2E::from_raw(<Self as IsSubFieldOf<Degree2ExtensionField>>::embed(a)),
            Fp2E::zero(),
        ]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree4ExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        // TODO: Repace this for with a map similarly to this:
        // b.into_iter().map(|x| x.to_raw()).collect()
        let mut result = Vec::new();
        for fp2e in b {
            result.push(fp2e.value()[0].to_raw());
            result.push(fp2e.value()[1].to_raw());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use core::ops::Neg;

    use crate::field::fields::mersenne31::field::MERSENNE_31_PRIME_FIELD_ORDER;

    use super::*;

    type Fp2E = FieldElement<Degree2ExtensionField>;
    type Fp4E = FieldElement<Degree4ExtensionField>;

    #[test]
    fn add_real_one_plus_one_is_two() {
        assert_eq!(Fp2E::one() + Fp2E::one(), Fp2E::from(2))
    }

    #[test]
    fn add_real_neg_one_plus_one_is_zero() {
        assert_eq!(Fp2E::one() + Fp2E::one().neg(), Fp2E::zero())
    }

    #[test]
    fn add_real_neg_one_plus_two_is_one() {
        assert_eq!(Fp2E::one().neg() + Fp2E::from(2), Fp2E::one())
    }

    #[test]
    fn add_real_neg_one_plus_neg_one_is_order_sub_two() {
        assert_eq!(
            Fp2E::one().neg() + Fp2E::one().neg(),
            Fp2E::new([FpE::from(&(MERSENNE_31_PRIME_FIELD_ORDER - 2)), FpE::zero()])
        )
    }

    #[test]
    fn add_complex_one_plus_one_two() {
        //Manually declare the complex part to one
        let one_i = Fp2E::new([FpE::zero(), FpE::one()]);
        let two_i = Fp2E::new([FpE::zero(), FpE::from(2)]);
        assert_eq!(&one_i + &one_i, two_i)
    }

    #[test]
    fn add_complex_neg_one_plus_one_is_zero() {
        //Manually declare the complex part to one
        let neg_one_i = Fp2E::new([FpE::zero(), -FpE::one()]);
        let one_i = Fp2E::new([FpE::zero(), FpE::one()]);
        assert_eq!(neg_one_i + one_i, Fp2E::zero())
    }

    #[test]
    fn add_complex_neg_one_plus_two_is_one() {
        let neg_one_i = Fp2E::new([FpE::zero(), -FpE::one()]);
        let two_i = Fp2E::new([FpE::zero(), FpE::from(2)]);
        let one_i = Fp2E::new([FpE::zero(), FpE::one()]);
        assert_eq!(&neg_one_i + &two_i, one_i)
    }

    #[test]
    fn add_complex_neg_one_plus_neg_one_imag_is_order_sub_two() {
        let neg_one_i = Fp2E::new([FpE::zero(), -FpE::one()]);
        assert_eq!(
            (&neg_one_i + &neg_one_i).value()[1],
            FpE::from(&(MERSENNE_31_PRIME_FIELD_ORDER - 2))
        )
    }

    #[test]
    fn add_order() {
        let a = Fp2E::new([-FpE::one(), FpE::one()]);
        let b = Fp2E::new([
            FpE::from(2),
            FpE::from(&(MERSENNE_31_PRIME_FIELD_ORDER - 2)),
        ]);
        let c = Fp2E::new([FpE::one(), -FpE::one()]);
        assert_eq!(&a + &b, c)
    }

    #[test]
    fn add_equal_zero() {
        let a = Fp2E::new([-FpE::one(), -FpE::one()]);
        let b = Fp2E::new([FpE::one(), FpE::one()]);
        assert_eq!(&a + &b, Fp2E::zero())
    }

    #[test]
    fn add_plus_one() {
        let a = Fp2E::new([FpE::one(), FpE::from(2)]);
        let b = Fp2E::new([FpE::one(), FpE::one()]);
        let c = Fp2E::new([FpE::from(2), FpE::from(3)]);
        assert_eq!(&a + &b, c)
    }

    #[test]
    fn sub_real_one_sub_one_is_zero() {
        assert_eq!(&Fp2E::one() - &Fp2E::one(), Fp2E::zero())
    }

    #[test]
    fn sub_real_two_sub_two_is_zero() {
        assert_eq!(&Fp2E::from(2) - &Fp2E::from(2), Fp2E::zero())
    }

    #[test]
    fn sub_real_neg_one_sub_neg_one_is_zero() {
        assert_eq!(Fp2E::one().neg() - Fp2E::one().neg(), Fp2E::zero())
    }

    #[test]
    fn sub_real_two_sub_one_is_one() {
        assert_eq!(Fp2E::from(2) - Fp2E::one(), Fp2E::one())
    }

    #[test]
    fn sub_real_neg_one_sub_zero_is_neg_one() {
        assert_eq!(Fp2E::one().neg() - Fp2E::zero(), Fp2E::one().neg())
    }

    #[test]
    fn sub_complex_one_sub_one_is_zero() {
        let one = Fp2E::new([FpE::zero(), FpE::one()]);
        assert_eq!(&one - &one, Fp2E::zero())
    }

    #[test]
    fn sub_complex_two_sub_two_is_zero() {
        let two = Fp2E::new([FpE::zero(), FpE::from(2)]);
        assert_eq!(&two - &two, Fp2E::zero())
    }

    #[test]
    fn sub_complex_neg_one_sub_neg_one_is_zero() {
        let neg_one = Fp2E::new([FpE::zero(), -FpE::one()]);
        assert_eq!(&neg_one - &neg_one, Fp2E::zero())
    }

    #[test]
    fn sub_complex_two_sub_one_is_one() {
        let two = Fp2E::new([FpE::zero(), FpE::from(2)]);
        let one = Fp2E::new([FpE::zero(), FpE::one()]);
        assert_eq!(&two - &one, one)
    }

    #[test]
    fn sub_complex_neg_one_sub_zero_is_neg_one() {
        let neg_one = Fp2E::new([FpE::zero(), -FpE::one()]);
        assert_eq!(&neg_one - &Fp2E::zero(), neg_one)
    }

    #[test]
    fn mul_fp2_is_correct() {
        let a = Fp2E::new([FpE::from(2), FpE::from(2)]);
        let b = Fp2E::new([FpE::from(4), FpE::from(5)]);
        let c = Fp2E::new([-FpE::from(2), FpE::from(18)]);
        assert_eq!(&a * &b, c)
    }

    #[test]
    fn square_equals_mul_by_itself() {
        let a = Fp2E::new([FpE::from(2), FpE::from(3)]);
        assert_eq!(a.square(), &a * &a)
    }

    #[test]
    fn mul_fpe_by_fp2e() {
        let a = FpE::from(3);
        let b = Fp2E::new([FpE::from(2), FpE::from(4)]);
        assert_eq!(a * b, Fp2E::new([FpE::from(6), FpE::from(12)]))
    }

    #[test]
    fn mul_fp4_is_correct() {
        let a = Fp4E::new([
            Fp2E::new([FpE::from(2), FpE::from(3)]),
            Fp2E::new([FpE::from(4), FpE::from(5)]),
        ]);

        let b = Fp4E::new([
            Fp2E::new([FpE::from(6), FpE::from(7)]),
            Fp2E::new([FpE::from(8), FpE::from(9)]),
        ]);

        let a2 = FieldElement::<Degree4ExtensionFieldV2>::new([
            Fp2E::new([FpE::from(2), FpE::from(3)]),
            Fp2E::new([FpE::from(4), FpE::from(5)]),
        ]);

        let b2 = FieldElement::<Degree4ExtensionFieldV2>::new([
            Fp2E::new([FpE::from(6), FpE::from(7)]),
            Fp2E::new([FpE::from(8), FpE::from(9)]),
        ]);

        assert_eq!((&a * &b).value(), (a2 * b2).value())
    }

    #[test]
    fn mul_fp4_is_correct_2() {
        let a = Fp4E::new([
            Fp2E::new([FpE::from(2147483647), FpE::from(2147483648)]),
            Fp2E::new([FpE::from(2147483649), FpE::from(2147483650)]),
        ]);

        let b = Fp4E::new([
            Fp2E::new([FpE::from(6), FpE::from(7)]),
            Fp2E::new([FpE::from(8), FpE::from(9)]),
        ]);

        let a2 = FieldElement::<Degree4ExtensionFieldV2>::new([
            Fp2E::new([FpE::from(2147483647), FpE::from(2147483648)]),
            Fp2E::new([FpE::from(2147483649), FpE::from(2147483650)]),
        ]);

        let b2 = FieldElement::<Degree4ExtensionFieldV2>::new([
            Fp2E::new([FpE::from(6), FpE::from(7)]),
            Fp2E::new([FpE::from(8), FpE::from(9)]),
        ]);

        assert_eq!((&a * &b).value(), (a2 * b2).value())
    }

    #[test]
    fn mul_fp4_by_zero_is_zero() {
        let a = Fp4E::new([
            Fp2E::new([FpE::from(2), FpE::from(3)]),
            Fp2E::new([FpE::from(4), FpE::from(5)]),
        ]);
        assert_eq!(Fp4E::zero(), a * Fp4E::zero())
    }

    #[test]
    fn mul_fp4_by_one_is_identity() {
        let a = Fp4E::new([
            Fp2E::new([FpE::from(2), FpE::from(3)]),
            Fp2E::new([FpE::from(4), FpE::from(5)]),
        ]);
        assert_eq!(a, a.clone() * Fp4E::one())
    }

    #[test]
    fn square_fp4_is_correct() {
        let a = Fp4E::new([
            Fp2E::new([FpE::from(2), FpE::from(3)]),
            Fp2E::new([FpE::from(4), FpE::from(5)]),
        ]);

        let a2 = FieldElement::<Degree4ExtensionFieldV2>::new([
            Fp2E::new([FpE::from(2), FpE::from(3)]),
            Fp2E::new([FpE::from(4), FpE::from(5)]),
        ]);

        assert_eq!(a.square().value(), a2.square().value())
    }

    #[test]
    fn square_fp4_equals_mul_two_times() {
        let a = Fp4E::new([
            Fp2E::new([FpE::from(3), FpE::from(4)]),
            Fp2E::new([FpE::from(5), FpE::from(6)]),
        ]);

        assert_eq!(a.square(), &a * &a)
    }

    #[test]
    fn fp4_mul_by_inv_is_one() {
        let a = Fp4E::new([
            Fp2E::new([FpE::from(2147483647), FpE::from(2147483648)]),
            Fp2E::new([FpE::from(2147483649), FpE::from(2147483650)]),
        ]);

        assert_eq!(&a * a.inv().unwrap(), Fp4E::one())
    }

    #[test]
    fn embed_fp_with_fp4() {
        let a = FpE::from(3);
        let a_extension = Fp4E::from(3);
        assert_eq!(a.to_extension::<Degree4ExtensionField>(), a_extension);
    }

    #[test]
    fn add_fp_and_fp4() {
        let a = FpE::from(3);
        let a_extension = Fp4E::from(3);
        let b = Fp4E::from(2);
        assert_eq!(a + &b, a_extension + b);
    }

    #[test]
    fn mul_fp_by_fp4() {
        let a = FpE::from(30000000000);
        let a_extension = a.clone().to_extension::<Degree4ExtensionField>();
        let b = Fp4E::new([
            Fp2E::new([FpE::from(1), FpE::from(2)]),
            Fp2E::new([FpE::from(3), FpE::from(4)]),
        ]);
        assert_eq!(a * &b, a_extension * b);
    }
}
