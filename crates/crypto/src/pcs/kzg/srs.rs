//! KZG Structured Reference String (SRS) types.

use lambdaworks_math::elliptic_curve::traits::IsPairing;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// KZG Public Parameters (Structured Reference String).
///
/// Generated during a trusted setup ceremony, these parameters
/// encode powers of a secret τ in the exponent of group elements.
#[derive(Clone, Debug)]
pub struct KZGPublicParams<P: IsPairing> {
    /// Powers of τ in G1: [g1, τ·g1, τ²·g1, ..., τ^n·g1]
    pub powers_of_g1: Vec<P::G1Point>,

    /// Powers of τ in G2: [g2, τ·g2]
    /// Only two elements needed for standard KZG verification.
    pub powers_of_g2: Vec<P::G2Point>,

    /// Maximum polynomial degree supported.
    pub max_degree: usize,
}

impl<P: IsPairing> KZGPublicParams<P> {
    /// Create new public parameters.
    ///
    /// # Arguments
    ///
    /// * `powers_of_g1` - Powers of τ in G1.
    /// * `powers_of_g2` - Powers of τ in G2 (at least [g2, τ·g2]).
    pub fn new(powers_of_g1: Vec<P::G1Point>, powers_of_g2: Vec<P::G2Point>) -> Self {
        let max_degree = powers_of_g1.len().saturating_sub(1);
        Self {
            powers_of_g1,
            powers_of_g2,
            max_degree,
        }
    }

    /// Returns the maximum polynomial degree supported.
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }
}

/// KZG Committer Key.
///
/// A subset of the public parameters used by the prover
/// to commit to polynomials and generate opening proofs.
#[derive(Clone, Debug)]
pub struct KZGCommitterKey<P: IsPairing> {
    /// Powers of τ in G1, truncated to the supported degree.
    pub powers_of_g1: Vec<P::G1Point>,

    /// Maximum polynomial degree supported by this key.
    pub max_degree: usize,
}

impl<P: IsPairing> KZGCommitterKey<P> {
    /// Returns the maximum polynomial degree supported.
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }
}

/// KZG Verifier Key.
///
/// Minimal data needed for verification.
#[derive(Clone, Debug)]
pub struct KZGVerifierKey<P: IsPairing> {
    /// Generator of G1.
    pub g1: P::G1Point,

    /// Generator of G2.
    pub g2: P::G2Point,

    /// τ·g2 (tau times the G2 generator).
    pub tau_g2: P::G2Point,
}

impl<P: IsPairing> KZGVerifierKey<P> {
    /// Create a new verifier key.
    pub fn new(g1: P::G1Point, g2: P::G2Point, tau_g2: P::G2Point) -> Self {
        Self { g1, g2, tau_g2 }
    }
}
