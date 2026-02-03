//! KZG Structured Reference String (SRS) types.

use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::traits::IsPairing;
use lambdaworks_math::errors::DeserializationError;
use lambdaworks_math::traits::{AsBytes, Deserializable};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::commitments::kzg::StructuredReferenceString;

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

    /// Create public parameters from an existing SRS.
    ///
    /// This provides interoperability with the legacy `StructuredReferenceString` type.
    pub fn from_srs(srs: StructuredReferenceString<P::G1Point, P::G2Point>) -> Self
    where
        P::G1Point: IsGroup,
        P::G2Point: IsGroup,
    {
        let powers_of_g2 = srs.powers_secondary_group.to_vec();
        Self::new(srs.powers_main_group, powers_of_g2)
    }

    /// Convert to the legacy SRS format.
    ///
    /// # Panics
    ///
    /// Panics if `powers_of_g2` doesn't contain exactly 2 elements.
    pub fn to_srs(&self) -> StructuredReferenceString<P::G1Point, P::G2Point>
    where
        P::G1Point: IsGroup + Clone,
        P::G2Point: IsGroup + Clone,
    {
        assert!(
            self.powers_of_g2.len() >= 2,
            "Need at least 2 G2 points for SRS"
        );
        let powers_secondary_group = [self.powers_of_g2[0].clone(), self.powers_of_g2[1].clone()];
        StructuredReferenceString::new(&self.powers_of_g1, &powers_secondary_group)
    }
}

/// Load public parameters from a file.
#[cfg(feature = "std")]
impl<P: IsPairing> KZGPublicParams<P>
where
    P::G1Point: IsGroup + Deserializable,
    P::G2Point: IsGroup + Deserializable,
{
    /// Load public parameters from a file.
    ///
    /// The file format is compatible with `StructuredReferenceString`.
    pub fn from_file(path: &str) -> Result<Self, crate::errors::SrsFromFileError> {
        let srs = StructuredReferenceString::<P::G1Point, P::G2Point>::from_file(path)?;
        Ok(Self::from_srs(srs))
    }
}

/// Serialization support for public parameters.
impl<P: IsPairing> AsBytes for KZGPublicParams<P>
where
    P::G1Point: IsGroup + AsBytes + Clone,
    P::G2Point: IsGroup + AsBytes + Clone,
{
    fn as_bytes(&self) -> Vec<u8> {
        self.to_srs().as_bytes()
    }
}

/// Deserialization support for public parameters.
impl<P: IsPairing> Deserializable for KZGPublicParams<P>
where
    P::G1Point: IsGroup + Deserializable,
    P::G2Point: IsGroup + Deserializable,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError> {
        let srs = StructuredReferenceString::<P::G1Point, P::G2Point>::deserialize(bytes)?;
        Ok(Self::from_srs(srs))
    }
}

/// KZG Committer Key.
///
/// A subset of the public parameters used by the prover
/// to commit to polynomials and generate opening proofs.
pub struct KZGCommitterKey<P: IsPairing> {
    /// Powers of τ in G1, truncated to the supported degree.
    pub powers_of_g1: Vec<P::G1Point>,

    /// Maximum polynomial degree supported by this key.
    pub max_degree: usize,
}

impl<P: IsPairing> Clone for KZGCommitterKey<P>
where
    P::G1Point: Clone,
{
    fn clone(&self) -> Self {
        Self {
            powers_of_g1: self.powers_of_g1.clone(),
            max_degree: self.max_degree,
        }
    }
}

impl<P: IsPairing> core::fmt::Debug for KZGCommitterKey<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KZGCommitterKey")
            .field("max_degree", &self.max_degree)
            .field("num_powers", &self.powers_of_g1.len())
            .finish()
    }
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
pub struct KZGVerifierKey<P: IsPairing> {
    /// Generator of G1.
    pub g1: P::G1Point,

    /// Generator of G2.
    pub g2: P::G2Point,

    /// τ·g2 (tau times the G2 generator).
    pub tau_g2: P::G2Point,
}

impl<P: IsPairing> Clone for KZGVerifierKey<P>
where
    P::G1Point: Clone,
    P::G2Point: Clone,
{
    fn clone(&self) -> Self {
        Self {
            g1: self.g1.clone(),
            g2: self.g2.clone(),
            tau_g2: self.tau_g2.clone(),
        }
    }
}

impl<P: IsPairing> core::fmt::Debug for KZGVerifierKey<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KZGVerifierKey").finish()
    }
}

impl<P: IsPairing> KZGVerifierKey<P> {
    /// Create a new verifier key.
    pub fn new(g1: P::G1Point, g2: P::G2Point, tau_g2: P::G2Point) -> Self {
        Self { g1, g2, tau_g2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use lambdaworks_math::cyclic_group::IsGroup;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::{
        curve::BLS12381Curve, default_types::FrField, pairing::BLS12381AtePairing,
        twist::BLS12381TwistCurve,
    };
    use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::unsigned_integer::element::U256;

    type G1Point =
        <BLS12381AtePairing as lambdaworks_math::elliptic_curve::traits::IsPairing>::G1Point;
    type G2Point =
        <BLS12381AtePairing as lambdaworks_math::elliptic_curve::traits::IsPairing>::G2Point;

    fn create_test_params(max_degree: usize) -> KZGPublicParams<BLS12381AtePairing> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let toxic_waste = FieldElement::<FrField>::new(U256 {
            limbs: [
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
            ],
        });

        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();

        let powers_of_g1: Vec<G1Point> = (0..=max_degree)
            .map(|exp| g1.operate_with_self(toxic_waste.pow(exp as u128).canonical()))
            .collect();

        let powers_of_g2: Vec<G2Point> =
            vec![g2.clone(), g2.operate_with_self(toxic_waste.canonical())];

        KZGPublicParams::new(powers_of_g1, powers_of_g2)
    }

    #[test]
    fn test_srs_conversion_roundtrip() {
        let pp = create_test_params(10);

        // Convert to legacy SRS
        let srs = pp.to_srs();
        assert_eq!(srs.powers_main_group.len(), pp.powers_of_g1.len());

        // Convert back
        let pp2 = KZGPublicParams::<BLS12381AtePairing>::from_srs(srs);
        assert_eq!(pp2.max_degree, pp.max_degree);
        assert_eq!(pp2.powers_of_g1.len(), pp.powers_of_g1.len());
    }

    #[test]
    fn test_serialize_deserialize() {
        let pp = create_test_params(5);
        let bytes = pp.as_bytes();
        let pp2 = KZGPublicParams::<BLS12381AtePairing>::deserialize(&bytes)
            .expect("deserialization of freshly serialized params should succeed");

        assert_eq!(pp2.max_degree, pp.max_degree);
        assert_eq!(pp2.powers_of_g1.len(), pp.powers_of_g1.len());
        assert_eq!(pp2.powers_of_g2.len(), pp.powers_of_g2.len());
    }
}
