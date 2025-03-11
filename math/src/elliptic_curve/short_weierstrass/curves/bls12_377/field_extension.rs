use crate::field::{
    element::FieldElement,
    errors::FieldError,
    extensions::{
        cubic::{CubicExtensionField, HasCubicNonResidue},
        quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
    },
    fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    traits::{IsField, IsSubFieldOf},
};
use crate::traits::ByteConversion;
use crate::unsigned_integer::element::U384;

pub const BLS12377_PRIME_FIELD_ORDER: U384 = U384::from_hex_unchecked("1ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001");
pub const FP2_RESIDUE: FieldElement<BLS12377PrimeField> =FieldElement::from_hex_unchecked("0x1ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508bffffffffffc");

// FPBLS12377
#[derive(Clone, Debug)]
pub struct BLS12377FieldModulus;
impl IsModulus<U384> for BLS12377FieldModulus {
    const MODULUS: U384 = BLS12377_PRIME_FIELD_ORDER;
}

pub type BLS12377PrimeField = MontgomeryBackendPrimeField<BLS12377FieldModulus, 6>;
type Fp2E = FieldElement<Degree2ExtensionField>;

#[derive(Clone, Debug)]
pub struct Degree2ExtensionField;

impl IsField for Degree2ExtensionField {
    type BaseType = [FieldElement<BLS12377PrimeField>; 2];

    const EXTENSION_DEGREE: usize = 2;

    /// Returns the component wise addition of `a` and `b`
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] + &b[0], &a[1] + &b[1]]
    }

    /// Returns the multiplication of `a` and `b` using the following
    /// equation:
    /// (a0 + a1 * t) * (b0 + b1 * t) = a0 * b0 + a1 * b1 * Self::residue() + (a0 * b1 + a1 * b0) * t
    /// where `t.pow(2)` equals `Q::residue()`.
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let a0b0 = &a[0] * &b[0];
        let a1b1 = &a[1] * &b[1];
        [&a0b0 + &a1b1 * &FP2_RESIDUE, &a[0] * &b[1] + &a[1] * &b[0]]
    }

    fn square(a: &Self::BaseType) -> Self::BaseType {
        let [a0, a1] = a;
        let v0 = a0 * a1;
        let c0 = (a0 + a1) * (a0 + &FP2_RESIDUE * a1) - &v0 - &FP2_RESIDUE * &v0;
        let c1 = &v0 + &v0;
        [c0, c1]
    }
    /// Returns the component wise subtraction of `a` and `b`
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] - &b[0], &a[1] - &b[1]]
    }

    /// Returns the component wise negation of `a`
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-&a[0], -&a[1]]
    }

    /// Returns the multiplicative inverse of `a`
    /// This uses the equality `(a0 + a1 * t) * (a0 - a1 * t) = a0.pow(2) - a1.pow(2) * Q::residue()`
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let inv_norm = (a[0].square() - FP2_RESIDUE * a[1].square()).inv()?;
        Ok([&a[0] * &inv_norm, -&a[1] * inv_norm])
    }

    /// Returns the division of `a` and `b`
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = &Self::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(<Self as IsField>::mul(a, b_inv))
    }

    /// Returns a boolean indicating whether `a` and `b` are equal component wise.
    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    /// Returns the additive neutral element of the field extension.
    fn zero() -> Self::BaseType {
        [FieldElement::zero(), FieldElement::zero()]
    }

    /// Returns the multiplicative neutral element of the field extension.
    fn one() -> Self::BaseType {
        [FieldElement::one(), FieldElement::zero()]
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> Self::BaseType {
        [FieldElement::from(x), FieldElement::zero()]
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    /// Note: for this case this is simply the identity, because the components
    /// already have correct representations.
    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }
}

impl IsSubFieldOf<Degree2ExtensionField> for BLS12377PrimeField {
    fn mul(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::mul(a, b[0].value()));
        let c1 = FieldElement::from_raw(<Self as IsField>::mul(a, b[1].value()));
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
    ) -> Result<<Degree2ExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Degree2ExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Degree2ExtensionField>>::mul(
            a, &b_inv,
        ))
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

impl ByteConversion for FieldElement<Degree2ExtensionField> {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_be(&self.value()[0]);
        byte_slice.extend(ByteConversion::to_bytes_be(&self.value()[1]));
        byte_slice
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_le(&self.value()[0]);
        byte_slice.extend(ByteConversion::to_bytes_le(&self.value()[1]));
        byte_slice
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: core::marker::Sized,
    {
        const BYTES_PER_FIELD: usize = 48;
        let x0 = FieldElement::from_bytes_be(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        Ok(Self::new([x0, x1]))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: core::marker::Sized,
    {
        const BYTES_PER_FIELD: usize = 48;
        let x0 = FieldElement::from_bytes_le(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        Ok(Self::new([x0, x1]))
    }
}

impl FieldElement<Degree2ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([FieldElement::new(U384::from(a_hex)), FieldElement::zero()])
    }

    pub fn conjugate(&self) -> Self {
        let [a, b] = self.value();
        Self::new([a.clone(), -b])
    }
}

#[derive(Debug, Clone)]
pub struct BLS12377Residue;
impl HasQuadraticNonResidue<BLS12377PrimeField> for BLS12377Residue {
    fn residue() -> FieldElement<BLS12377PrimeField> {
        FP2_RESIDUE
    }
}

#[derive(Debug, Clone)]
pub struct LevelTwoResidue;

impl HasCubicNonResidue<Degree2ExtensionField> for LevelTwoResidue {
    fn residue() -> FieldElement<Degree2ExtensionField> {
        FieldElement::new([
            FieldElement::new(U384::from("0")),
            FieldElement::new(U384::from("1")),
        ])
    }
}
pub type Degree6ExtensionField = CubicExtensionField<Degree2ExtensionField, LevelTwoResidue>;
type Fp6E = FieldElement<Degree6ExtensionField>;

#[derive(Debug, Clone)]
pub struct LevelThreeResidue;
impl HasQuadraticNonResidue<Degree6ExtensionField> for LevelThreeResidue {
    fn residue() -> FieldElement<Degree6ExtensionField> {
        FieldElement::new([
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::zero(),
        ])
    }
}

/// We define Fp12 = Fp6 [w] / (w^2 - v)
pub type Degree12ExtensionField = QuadraticExtensionField<Degree6ExtensionField, LevelThreeResidue>;
pub type Fp12E = FieldElement<Degree12ExtensionField>;

impl FieldElement<Degree6ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([
            FieldElement::new([FieldElement::new(U384::from(a_hex)), FieldElement::zero()]),
            FieldElement::zero(),
            FieldElement::zero(),
        ])
    }
}

impl FieldElement<BLS12377PrimeField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new(U384::from(a_hex))
    }
}

impl FieldElement<Degree4ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([Fp2E::new_base(a_hex), Fp2E::zero()])
    }
}

impl FieldElement<Degree12ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([
            FieldElement::<Degree6ExtensionField>::new_base(a_hex),
            FieldElement::zero(),
        ])
    }

    pub fn from_coefficients(coefficients: &[&str; 12]) -> Self {
        FieldElement::<Degree12ExtensionField>::new([
            FieldElement::new([
                FieldElement::new([
                    FieldElement::new(U384::from(coefficients[0])),
                    FieldElement::new(U384::from(coefficients[1])),
                ]),
                FieldElement::new([
                    FieldElement::new(U384::from(coefficients[2])),
                    FieldElement::new(U384::from(coefficients[3])),
                ]),
                FieldElement::new([
                    FieldElement::new(U384::from(coefficients[4])),
                    FieldElement::new(U384::from(coefficients[5])),
                ]),
            ]),
            FieldElement::new([
                FieldElement::new([
                    FieldElement::new(U384::from(coefficients[6])),
                    FieldElement::new(U384::from(coefficients[7])),
                ]),
                FieldElement::new([
                    FieldElement::new(U384::from(coefficients[8])),
                    FieldElement::new(U384::from(coefficients[9])),
                ]),
                FieldElement::new([
                    FieldElement::new(U384::from(coefficients[10])),
                    FieldElement::new(U384::from(coefficients[11])),
                ]),
            ]),
        ])
    }
}
impl HasQuadraticNonResidue<Degree2ExtensionField> for LevelTwoResidue {
    fn residue() -> FieldElement<Degree2ExtensionField> {
        FieldElement::new([
            FieldElement::new(U384::from("0")),
            FieldElement::new(U384::from("1")),
        ])
    }
}
pub type Degree4ExtensionField = QuadraticExtensionField<Degree2ExtensionField, LevelTwoResidue>;

/// Computes the multiplication of an element of fp2 by the level two non-residue (0+u).
pub fn mul_fp2_by_nonresidue(a: &Fp2E) -> Fp2E {
    let c0 = FP2_RESIDUE * &a.value()[1]; // c0 = -5 * a1
    let c1 = a.value()[0].clone(); // c1 = a0
    Fp2E::new([c0, c1])
}
/// Computes the multiplication of an element of fp6 by the level three non-residue v.
pub fn mul_fp6_by_nonresidue(a: &Fp6E) -> Fp6E {
    let [a0, a1, a2] = a.value();

    let c0 = mul_fp2_by_nonresidue(a2);
    let c1 = a0.clone();
    let c2 = a1.clone();

    Fp6E::new([c0, c1, c2])
}

///Multiplication between a = a0 + a1 * w and b = b0 + b1 * w with
/// b1 = b10 + b11 * v + 0 * v^2 which is the case of the line used
/// in the miller loop.
pub fn sparse_fp12_mul(a: &Fp12E, b: &Fp12E) -> Fp12E {
    let [a0, a1] = a.value();
    let [b0, b1] = b.value();
    let b00 = &b0.value()[0];
    let [b10, b11, _] = b1.value();

    let t0 = a0 * b0;
    let t1 = a1 * b1;
    let c0 = &t0 + mul_fp6_by_nonresidue(&t1);
    let t2 = Fp6E::new([b10 + b00, b11.clone(), Fp2E::zero()]);
    let mut c1 = (a0 + a1) * t2;
    c1 = c1 - t0 - t1;

    Fp12E::new([c0, c1])
}

#[cfg(test)]
mod tests {

    use super::*;
    type FpE = FieldElement<BLS12377PrimeField>;
    type Fp2E = FieldElement<Degree2ExtensionField>;
    type Fp6E = FieldElement<Degree6ExtensionField>;
    type Fp12E = FieldElement<Degree12ExtensionField>;

    #[test]
    fn embed_base_field_with_degree_2_extension() {
        let a = FpE::from(3);
        let a_extension = Fp2E::from(3);
        assert_eq!(a.to_extension::<Degree2ExtensionField>(), a_extension);
    }
    #[test]
    fn add_base_field_with_degree_2_extension() {
        let a = FpE::from(3);
        let a_extension = Fp2E::from(3);
        let b = Fp2E::from(2);
        assert_eq!(a + &b, a_extension + b);
    }
    #[test]
    fn mul_degree_2_with_degree_6_extension() {
        let a = Fp2E::new([FpE::from(3), FpE::from(4)]);
        let a_extension = a.clone().to_extension::<Degree2ExtensionField>();
        let b = Fp6E::from(2);
        assert_eq!(a * &b, a_extension * b);
    }
    #[test]
    fn div_degree_6_degree_12_extension() {
        let a = Fp6E::from(3);
        let a_extension = Fp12E::from(3);
        let b = Fp12E::from(2);
        assert_eq!((a / &b).unwrap(), (a_extension / b).unwrap());
    }

    #[test]
    fn double_equals_sum_two_times() {
        let a = FpE::from(3);
        assert_eq!(a.double(), a.clone() + a);
    }
    #[test]
    fn base_field_sum_is_asociative() {
        let a = FpE::from(3);
        let b = FpE::from(2);
        let c = &a + &b;
        assert_eq!(a.double() + b, a + c);
    }
    #[test]
    fn degree_2_extension_mul_is_conmutative() {
        let a = Fp2E::from(3);
        let b = Fp2E::new([FpE::from(2), FpE::from(4)]);
        assert_eq!(&a * &b, b * a);
    }

    #[test]
    fn base_field_pow_p_is_identity() {
        let a = FpE::from(3);
        assert_eq!(a.pow(BLS12377_PRIME_FIELD_ORDER), a);
    }
    #[test]
    fn square_equals_mul_two_times() {
        let a = FpE::from(3);
        assert_eq!(a.square(), a.clone() * a);
    }
    #[test]
    fn test_mul_fp2_by_nonresidue() {
        // Element in Fp2: a = 3 + 5u
        let a = Fp2E::new([FpE::from(3), FpE::from(5)]);
        // Expected result: (-25) + 3u
        let expected = Fp2E::new([-FpE::from(25), FpE::from(3)]);
        assert_eq!(mul_fp2_by_nonresidue(&a), expected);
    }
}
