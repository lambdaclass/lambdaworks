use crate::cyclic_group::HasCyclicBilinearGroupStructure;
use crate::elliptic_curve::traits::{HasDistortionMap, HasEllipticCurveOperations};
use crate::field::element::FieldElement;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Represents an elliptic curve point using the projective short Weierstrass form:
/// y^2 * z = x^3 + a * x * z^2 + b * z^3,
/// where `x`, `y` and `z` variables are field elements.
#[derive(Debug, Clone)]
pub struct EllipticCurveElement<E: HasEllipticCurveOperations> {
    value: [FieldElement<E::BaseField>; 3],
    elliptic_curve: PhantomData<E>,
}

impl<E: HasEllipticCurveOperations> EllipticCurveElement<E> {
    /// Creates an elliptic curve point giving the projective [x: y: z] coordinates.
    pub fn new(value: [FieldElement<E::BaseField>; 3]) -> Self {
        assert_eq!(
            E::defining_equation(&value),
            FieldElement::zero(),
            "Point ({:?}) does not belong to the elliptic curve.",
            &value
        );
        Self {
            value,
            elliptic_curve: PhantomData,
        }
    }

    /// Returns the `x` coordinate of the point.
    pub fn x(&self) -> &FieldElement<E::BaseField> {
        &self.value[0]
    }

    /// Returns the `y` coordinate of the point.
    pub fn y(&self) -> &FieldElement<E::BaseField> {
        &self.value[1]
    }

    /// Returns the `z` coordinate of the point.
    pub fn z(&self) -> &FieldElement<E::BaseField> {
        &self.value[2]
    }

    /// Creates the same point in affine coordinates. That is,
    /// returns [x / z: y / z: 1] where `self` is [x: y: z].
    /// Panics if `self` is the point at infinity.
    pub fn to_affine(&self) -> Self {
        Self {
            value: E::affine(&self.value),
            elliptic_curve: PhantomData,
        }
    }

    /// Returns the Weil pairing between `self` and `other`.
    pub fn weil_pairing(&self, other: &Self) -> FieldElement<E::BaseField> {
        E::weil_pairing(&self.value, &other.value)
    }

    /// Returns the Tate pairing between `self` and `other`.
    pub fn tate_pairing(&self, other: &Self) -> FieldElement<E::BaseField> {
        E::tate_pairing(&self.value, &other.value)
    }
}

impl<E: HasEllipticCurveOperations> PartialEq for EllipticCurveElement<E> {
    fn eq(&self, other: &Self) -> bool {
        E::eq(&self.value, &other.value)
    }
}

impl<E: HasEllipticCurveOperations> Eq for EllipticCurveElement<E> {}

impl<E: HasEllipticCurveOperations + HasDistortionMap> HasCyclicBilinearGroupStructure
    for EllipticCurveElement<E>
{
    type PairingOutput = FieldElement<E::BaseField>;

    fn generator() -> Self {
        Self::new([
            E::generator_affine_x(),
            E::generator_affine_y(),
            FieldElement::one(),
        ])
    }

    fn neutral_element() -> Self {
        Self::new(E::neutral_element())
    }

    /// Computes the addition of `self` and `other`.
    /// Taken from "Moonmath" (Algorithm 7, page 89)
    fn operate_with(&self, other: &Self) -> Self {
        Self::new(E::add(&self.value, &other.value))
    }

    /// Computes a Type 1 Tate pairing between `self` and `other.
    /// See "Pairing for beginners" from Craig Costello, section 4.2 Pairing types, page 58.
    /// Note that a distorsion map is applied to `other` before using the Tate pairing.
    /// So this method can be called with two field extension elements from the base field.
    fn pairing(&self, other: &Self) -> Self::PairingOutput {
        let [qx, qy, qz] = E::distorsion_map(&other.value);
        Self::tate_pairing(self, &Self::new([qx, qy, qz]))
    }
}
