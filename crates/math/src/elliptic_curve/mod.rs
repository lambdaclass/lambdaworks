/// Batch operations for elliptic curves using Montgomery's trick.
#[cfg(feature = "alloc")]
pub mod batch;
pub mod edwards;
pub mod montgomery;
/// Implementation of ProjectivePoint, a generic projective point in a curve.
pub mod point;
pub mod short_weierstrass;
pub mod traits;
