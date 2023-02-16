use crate::{
    cyclic_group::IsGroup,
    field::{element::FieldElement, traits::IsField},
};
use std::fmt::Debug;

pub trait IsEllipticCurve {
    /// BaseField is the field used for each of the coordinates of a point p
    /// belonging to the curve.
    type BaseField: IsField + Clone + Debug;

    /// The representation of the point. For example it can be projective
    /// coordinates, affine coordinates, XYZZ, depending on the curve and its
    /// possible optimizations.
    type PointRepresentation: IsGroup;

    /// Returns the generator of the main subgroup.
    fn generator() -> Self::PointRepresentation;

    /// Returns an affine point.
    // TODO: Return a result when the point does not belong to the curve.
    fn create_point_from_affine(
        x: FieldElement<Self::BaseField>,
        y: FieldElement<Self::BaseField>,
    ) -> Self::PointRepresentation;
}

pub trait HasPairing {
    type LhsGroup;
    type RhsGroup;
    type OutputGroup;

    fn pairing(a: &Self::LhsGroup, b: &Self::RhsGroup) -> Self::OutputGroup;
}
