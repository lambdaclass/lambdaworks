//! KZG Opening Proof type.

use core::fmt;
use lambdaworks_math::elliptic_curve::traits::IsPairing;

/// KZG Opening Proof - a single elliptic curve point in G1.
///
/// The proof that p(z) = y is the commitment to the quotient polynomial:
/// q(x) = (p(x) - y) / (x - z)
///
/// The proof point is: π = q(τ)·G1
///
/// Verification uses the pairing equation:
/// e(C - y·G1, G2) = e(π, τ·G2 - z·G2)
pub struct KZGProof<P: IsPairing> {
    /// The proof point (quotient commitment) in G1.
    pub point: P::G1Point,
}

impl<P: IsPairing> KZGProof<P> {
    /// Create a new proof from a G1 point.
    pub fn new(point: P::G1Point) -> Self {
        Self { point }
    }

    /// Get a reference to the proof point.
    pub fn point(&self) -> &P::G1Point {
        &self.point
    }

    /// Consume the proof and return the inner point.
    pub fn into_point(self) -> P::G1Point {
        self.point
    }
}

impl<P: IsPairing> Clone for KZGProof<P>
where
    P::G1Point: Clone,
{
    fn clone(&self) -> Self {
        Self {
            point: self.point.clone(),
        }
    }
}

impl<P: IsPairing> fmt::Debug for KZGProof<P>
where
    P::G1Point: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KZGProof")
            .field("point", &self.point)
            .finish()
    }
}

impl<P: IsPairing> PartialEq for KZGProof<P>
where
    P::G1Point: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point
    }
}

impl<P: IsPairing> Eq for KZGProof<P> where P::G1Point: Eq {}
