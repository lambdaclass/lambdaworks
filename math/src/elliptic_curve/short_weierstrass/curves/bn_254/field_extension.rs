use crate::field::{
    element::FieldElement, errors::FieldError, extensions::{
        cubic::{CubicExtensionField, HasCubicNonResidue},
        quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
    }, fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField}, traits::{IsField, IsSubFieldOf}
};
use crate::unsigned_integer::element::U256;

#[cfg(feature = "std")]
use crate::traits::ByteConversion;

pub const BN254_PRIME_FIELD_ORDER: U256 =
    U256::from_hex_unchecked("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47");

// Fp for BN254
#[derive(Clone, Debug)]
pub struct BN254FieldModulus;
impl IsModulus<U256> for BN254FieldModulus {
    const MODULUS: U256 = BN254_PRIME_FIELD_ORDER;
}

pub type BN254PrimeField = MontgomeryBackendPrimeField<BN254FieldModulus, 4>;

#[derive(Clone, Debug)]
pub struct Degree2ExtensionField;

impl IsField for Degree2ExtensionField {
    type BaseType = [FieldElement<BN254PrimeField>; 2];

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
        let sum_a = &a[0] + &a[1];
        let sum_b = &b[0] + &b[1];
        let z = &sum_a * &sum_b;
        [&a0b0 - &a1b1, z - &a0b0 - &a1b1]
    }
    fn square(a: &Self::BaseType) -> Self::BaseType {
        let [a0, a1] = a;
        let v0 = a0 * a1;
        let c0 = (a0 + a1) * (a0 - a1);
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
        let inv_norm = (a[0].pow(2_u64) + a[1].pow(2_u64)).inv()?;
        Ok([&a[0] * &inv_norm, -&a[1] * inv_norm])
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

impl IsSubFieldOf<Degree2ExtensionField> for BN254PrimeField {
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

#[derive(Debug, Clone)]
pub struct BN254Residue;
impl HasQuadraticNonResidue<BN254PrimeField> for BN254Residue {
    fn residue() -> FieldElement<BN254PrimeField> {
        -FieldElement::one()
    }
}

#[cfg(feature = "std")]
impl ByteConversion for FieldElement<Degree2ExtensionField> {
    #[cfg(feature = "std")]
    fn to_bytes_be(&self) -> Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_be(&self.value()[0]);
        byte_slice.extend(ByteConversion::to_bytes_be(&self.value()[1]));
        byte_slice
    }

    #[cfg(feature = "std")]
    fn to_bytes_le(&self) -> Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_le(&self.value()[0]);
        byte_slice.extend(ByteConversion::to_bytes_le(&self.value()[1]));
        byte_slice
    }

    #[cfg(feature = "std")]
    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: std::marker::Sized,
    {
        const BYTES_PER_FIELD: usize = 32;
        let x0 = FieldElement::from_bytes_be(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        Ok(Self::new([x0, x1]))
    }

    #[cfg(feature = "std")]
    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: std::marker::Sized,
    {
        const BYTES_PER_FIELD: usize = 32;
        let x0 = FieldElement::from_bytes_le(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        Ok(Self::new([x0, x1]))
    }
}

#[derive(Debug, Clone)]
pub struct LevelTwoResidue;

impl HasQuadraticNonResidue<Degree2ExtensionField> for LevelTwoResidue {
    fn residue() -> FieldElement<Degree2ExtensionField> {
        FieldElement::new([FieldElement::from(9), FieldElement::one()])
    }
}

impl HasCubicNonResidue<Degree2ExtensionField> for LevelTwoResidue {
    fn residue() -> FieldElement<Degree2ExtensionField> {
        FieldElement::new([FieldElement::from(9), FieldElement::one()])
    }
}


/// Fp4 = Fp2 [V] / (V^2 - (9+u))
pub type Degree4ExtensionField = QuadraticExtensionField<Degree2ExtensionField, LevelTwoResidue>;

type Fp2E = FieldElement<Degree2ExtensionField>;

pub fn mul_fp2_by_nonresidue(fe: &Fp2E) -> Fp2E {
    // (c0 + c1 * u) * (9 + u) = (9 * c0 - c1) + (9c1 + c0) * u
    let f = fe.double().double().double(); 
    let c0 = &f.value()[0] - &fe.value()[1] + &fe.value()[0]; // 9c0 - c1
    let c1 = &f.value()[1] + &fe.value()[1] + &fe.value()[0]; // 9c1 + c0
    Fp2E::new([c0, c1])
}

pub type Degree6ExtensionField = CubicExtensionField<Degree2ExtensionField, LevelTwoResidue>;

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

pub type Degree12ExtensionField = QuadraticExtensionField<Degree6ExtensionField, LevelThreeResidue>;

impl FieldElement<BN254PrimeField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new(U256::from(a_hex))
    }
}

impl FieldElement<Degree2ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([FieldElement::new(U256::from(a_hex)), FieldElement::zero()])
    }
    pub fn conjugate(&self) -> Self {
        let [a, b] = self.value();
        Self::new([a.clone(), -b])
    }
}

impl FieldElement<Degree4ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([
            Fp2E::new_base(a_hex),
            Fp2E::zero(),
        ])
    }
}


impl FieldElement<Degree6ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([
            FieldElement::new([FieldElement::new(U256::from(a_hex)), FieldElement::zero()]),
            FieldElement::zero(),
            FieldElement::zero(),
        ])
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
                    FieldElement::new(U256::from(coefficients[0])),
                    FieldElement::new(U256::from(coefficients[1])),
                ]),
                FieldElement::new([
                    FieldElement::new(U256::from(coefficients[2])),
                    FieldElement::new(U256::from(coefficients[3])),
                ]),
                FieldElement::new([
                    FieldElement::new(U256::from(coefficients[4])),
                    FieldElement::new(U256::from(coefficients[5])),
                ]),
            ]),
            FieldElement::new([
                FieldElement::new([
                    FieldElement::new(U256::from(coefficients[6])),
                    FieldElement::new(U256::from(coefficients[7])),
                ]),
                FieldElement::new([
                    FieldElement::new(U256::from(coefficients[8])),
                    FieldElement::new(U256::from(coefficients[9])),
                ]),
                FieldElement::new([
                    FieldElement::new(U256::from(coefficients[10])),
                    FieldElement::new(U256::from(coefficients[11])),
                ]),
            ]),
        ])
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    type FpE = FieldElement<BN254PrimeField>;
    type Fp2E = FieldElement<Degree2ExtensionField>;
    type Fp4E = FieldElement<Degree4ExtensionField>;
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
    fn mul_degree_2_with_degree_4_extension() {
        let a = Fp2E::new([FpE::from(3), FpE::from(4)]);
        let a_extension = a.clone().to_extension::<Degree4ExtensionField>();
        let b = Fp4E::from(2);
        assert_eq!(a * &b, a_extension * b);
    }
    

    #[test]
    fn div_degree_6_degree_12_extension() {
        let a = Fp6E::from(3);
        let a_extension = Fp12E::from(3);
        let b = Fp12E::from(2);
        assert_eq!(a / &b, a_extension / b);
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
        assert_eq!(a.pow(BN254_PRIME_FIELD_ORDER), a);
    }

    #[test]
    fn mul_fp2_by_nonresidue2_is_correct() {
        let a = Fp2E::new([FpE::from(2), FpE::from(4)]);
        assert_eq!(&a * <LevelTwoResidue as HasQuadraticNonResidue<Degree2ExtensionField>>::residue(), mul_fp2_by_nonresidue(&a))
    }
}