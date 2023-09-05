use crate::unsigned_integer::element::U384;
use crate::{
    field::{
        element::FieldElement,
        errors::FieldError,
        extensions::{
            cubic::{CubicExtensionField, HasCubicNonResidue},
            quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
        },
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
        traits::IsField,
    },
    traits::ByteConversion,
};

pub const BLS12381_PRIME_FIELD_ORDER: U384 = U384::from_hex_unchecked("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");

// FPBLS12381
#[derive(Clone, Debug)]
pub struct BLS12381FieldModulus;
impl IsModulus<U384> for BLS12381FieldModulus {
    const MODULUS: U384 = BLS12381_PRIME_FIELD_ORDER;
}

pub type BLS12381PrimeField = MontgomeryBackendPrimeField<BLS12381FieldModulus, 6>;

//////////////////
#[derive(Clone, Debug)]
pub struct Degree2ExtensionField;

impl IsField for Degree2ExtensionField {
    type BaseType = [FieldElement<BLS12381PrimeField>; 2];

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
        let z = (&a[0] + &a[1]) * (&b[0] + &b[1]);
        [&a0b0 - &a1b1, z - a0b0 - a1b1]
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
        Self::mul(a, &Self::inv(b).unwrap())
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

impl ByteConversion for FieldElement<Degree2ExtensionField> {
    fn to_bytes_be(&self) -> Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_be(&self.value()[0]);
        byte_slice.extend(ByteConversion::to_bytes_be(&self.value()[1]));
        byte_slice
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_le(&self.value()[0]);
        byte_slice.extend(ByteConversion::to_bytes_le(&self.value()[1]));
        byte_slice
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: std::marker::Sized,
    {
        const BYTES_PER_FIELD: usize = 48;
        let x0 = FieldElement::from_bytes_be(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        Ok(Self::new([x0, x1]))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: std::marker::Sized,
    {
        const BYTES_PER_FIELD: usize = 48;
        let x0 = FieldElement::from_bytes_le(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        Ok(Self::new([x0, x1]))
    }
}

///////////////
#[derive(Debug, Clone)]
pub struct LevelTwoResidue;
impl HasCubicNonResidue for LevelTwoResidue {
    type BaseField = Degree2ExtensionField;

    fn residue() -> FieldElement<Degree2ExtensionField> {
        FieldElement::new([
            FieldElement::new(U384::from("1")),
            FieldElement::new(U384::from("1")),
        ])
    }
}

pub type Degree6ExtensionField = CubicExtensionField<LevelTwoResidue>;

#[derive(Debug, Clone)]
pub struct LevelThreeResidue;
impl HasQuadraticNonResidue for LevelThreeResidue {
    type BaseField = Degree6ExtensionField;

    fn residue() -> FieldElement<Degree6ExtensionField> {
        FieldElement::new([
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::zero(),
        ])
    }
}

pub type Degree12ExtensionField = QuadraticExtensionField<LevelThreeResidue>;

impl FieldElement<BLS12381PrimeField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new(U384::from(a_hex))
    }
}

impl FieldElement<Degree2ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([FieldElement::new(U384::from(a_hex)), FieldElement::zero()])
    }
}

impl FieldElement<Degree6ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([
            FieldElement::new([FieldElement::new(U384::from(a_hex)), FieldElement::zero()]),
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

#[cfg(test)]
mod tests {
    use crate::elliptic_curve::{
        short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve, traits::IsEllipticCurve,
    };

    use super::*;
    type Fp12E = FieldElement<Degree12ExtensionField>;

    #[test]
    fn element_squared_1() {
        // base = 1 + u + (1 + u)v + (1 + u)v^2 + ((1+u) + (1 + u)v + (1+ u)v^2)w
        let element_ones =
            Fp12E::from_coefficients(&["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]);
        let element_ones_squared =
            Fp12E::from_coefficients(&["1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaa1",
            "c",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaa5",
            "c",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaa9",
            "c",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaa3",
            "c",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaa7",
            "c",
            "0",
            "c"]);
        assert_eq!(element_ones.pow(2_u16), element_ones_squared);
        assert_eq!(element_ones.square(), element_ones_squared);
    }

    #[test]
    fn element_squared_2() {
        let element_sequence =
            Fp12E::from_coefficients(&["1", "2", "5", "6", "9", "a", "3", "4", "7", "8", "b", "c"]);

        let element_sequence_squared = Fp12E::from_coefficients(&["1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa87d",
        "199",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa851",
        "20b",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa955",
        "1cd",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa845",
        "1e8",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa8a9",
        "202",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa5d",
        "16c"]);

        assert_eq!(element_sequence.pow(2_u16), element_sequence_squared);
        assert_eq!(element_sequence.square(), element_sequence_squared);
    }

    #[test]
    fn to_fp12_unnormalized_computes_correctly() {
        let g = BLS12381TwistCurve::generator();
        let expectedx = Fp12E::from_coefficients(&[
            "0",
            "0",
            "24aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8",
            "13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0"
        ]);
        let expectedy = Fp12E::from_coefficients(&[
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801",
            "606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be",
            "0",
            "0"
        ]);
        let [g_to_fp12_x, g_to_fp12_y] = g.to_fp12_unnormalized();
        assert_eq!(g_to_fp12_x, expectedx);
        assert_eq!(g_to_fp12_y, expectedy);
    }
}
