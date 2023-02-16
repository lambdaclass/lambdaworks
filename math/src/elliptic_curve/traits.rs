use crate::{
    cyclic_group::IsGroup,
    field::{element::FieldElement, traits::IsField},
};
use std::fmt::Debug;

#[derive(Debug, PartialEq, Eq)]
pub enum EllipticCurveError {
    InvalidPoint,
}

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
    fn create_point_from_affine(
        x: FieldElement<Self::BaseField>,
        y: FieldElement<Self::BaseField>,
    ) -> Result<Self::PointRepresentation, EllipticCurveError>;
}
