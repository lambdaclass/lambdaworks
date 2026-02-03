//! KZG Commitment type.

use core::fmt;
use lambdaworks_math::elliptic_curve::traits::IsPairing;

/// KZG Commitment - a single elliptic curve point in G1.
///
/// The commitment to a polynomial p(x) is computed as:
/// C = p(τ)·G1 = Σ coefficients[i] * τ^i · G1
///
/// where τ is the secret from the trusted setup.
pub struct KZGCommitment<P: IsPairing> {
    /// The commitment point in G1.
    pub point: P::G1Point,
}

impl<P: IsPairing> KZGCommitment<P> {
    /// Create a new commitment from a G1 point.
    pub fn new(point: P::G1Point) -> Self {
        Self { point }
    }

    /// Get a reference to the commitment point.
    pub fn point(&self) -> &P::G1Point {
        &self.point
    }

    /// Consume the commitment and return the inner point.
    pub fn into_point(self) -> P::G1Point {
        self.point
    }
}

impl<P: IsPairing> Clone for KZGCommitment<P>
where
    P::G1Point: Clone,
{
    fn clone(&self) -> Self {
        Self {
            point: self.point.clone(),
        }
    }
}

impl<P: IsPairing> fmt::Debug for KZGCommitment<P>
where
    P::G1Point: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KZGCommitment")
            .field("point", &self.point)
            .finish()
    }
}

impl<P: IsPairing> PartialEq for KZGCommitment<P>
where
    P::G1Point: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point
    }
}

impl<P: IsPairing> Eq for KZGCommitment<P> where P::G1Point: Eq {}
