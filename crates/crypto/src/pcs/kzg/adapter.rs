//! KZG adapter for legacy `IsCommitmentScheme` compatibility.
//!
//! # Safety Invariants
//!
//! This adapter uses `expect()` in trait method implementations because the legacy
//! `IsCommitmentScheme` trait does not return `Result` types. The following invariants
//! ensure these `expect()` calls will not panic in normal usage:
//!
//! 1. **Commit**: Callers must ensure polynomial degree does not exceed `max_degree`.
//!    The KZG commit operation only fails if the polynomial degree exceeds the SRS size.
//!
//! 2. **Open**: If commit succeeded, open will succeed because the quotient polynomial
//!    has degree at most (original_degree - 1).
//!
//! 3. **Verify**: Pairing computation only fails with malformed curve points, which
//!    cannot occur with well-formed commitments produced by this library.

use super::{KZGCommitment, KZGCommitterKey, KZGProof, KZGVerifierKey, KZG};
use crate::pcs::compat::IsCommitmentScheme;
use crate::pcs::traits::PolynomialCommitmentScheme;

use alloc::borrow::ToOwned;
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::traits::IsPairing;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsPrimeField;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;

/// Adapter that wraps the new KZG PCS for legacy compatibility.
///
/// This adapter stores committer and verifier keys internally,
/// implementing the instance-method based `IsCommitmentScheme` trait.
///
/// # Panics
///
/// Methods may panic if polynomial degrees exceed the configured `max_degree`.
/// See module-level documentation for invariants.
pub struct KZGAdapter<P: IsPairing> {
    /// Committer key for commit and open operations.
    pub committer_key: KZGCommitterKey<P>,
    /// Verifier key for verification operations.
    pub verifier_key: KZGVerifierKey<P>,
}

impl<P: IsPairing> Clone for KZGAdapter<P>
where
    P::G1Point: Clone,
    P::G2Point: Clone,
{
    fn clone(&self) -> Self {
        Self {
            committer_key: self.committer_key.clone(),
            verifier_key: self.verifier_key.clone(),
        }
    }
}

impl<P: IsPairing> KZGAdapter<P>
where
    P::G1Point: Clone,
    P::G2Point: Clone,
{
    /// Create from a structured reference string.
    ///
    /// This provides backward compatibility with code that used `KZG::new(srs)`.
    ///
    /// # Panics
    ///
    /// Panics if the SRS is empty.
    pub fn new(srs: super::StructuredReferenceString<P::G1Point, P::G2Point>) -> Self {
        let pp: super::KZGPublicParams<P> = super::KZGPublicParams::from_srs(srs);
        let committer_key = KZGCommitterKey {
            powers_of_g1: pp.powers_of_g1.clone(),
            max_degree: pp.max_degree,
        };
        let verifier_key = KZGVerifierKey {
            g1: pp.powers_of_g1.first().cloned().unwrap_or_else(|| {
                panic!("SRS must have at least one element")
            }),
            g2: pp.powers_of_g2[0].clone(),
            tau_g2: pp.powers_of_g2[1].clone(),
        };
        Self::from_keys(committer_key, verifier_key)
    }

    /// Create a new adapter from committer and verifier keys.
    pub fn from_keys(committer_key: KZGCommitterKey<P>, verifier_key: KZGVerifierKey<P>) -> Self {
        Self {
            committer_key,
            verifier_key,
        }
    }

    /// Create from public parameters by trimming to a specific degree.
    pub fn from_params<const N: usize, F>(
        pp: &super::KZGPublicParams<P>,
        supported_degree: usize,
    ) -> Result<Self, crate::pcs::error::PCSError>
    where
        F: IsPrimeField<CanonicalType = UnsignedInteger<N>>,
        P::G1Point: PartialEq + Eq,
    {
        let (ck, vk) = KZG::<F, P>::trim(pp, supported_degree)?;
        Ok(Self::from_keys(ck, vk))
    }

    /// Get the maximum degree supported.
    pub fn max_degree(&self) -> usize {
        self.committer_key.max_degree
    }
}

impl<const N: usize, F, P> IsCommitmentScheme<F> for KZGAdapter<P>
where
    F: IsPrimeField<CanonicalType = UnsignedInteger<N>>,
    P: IsPairing,
    P::G1Point: IsGroup + Clone + PartialEq + Eq,
    P::G2Point: Clone,
{
    type Commitment = P::G1Point;

    fn commit(&self, p: &Polynomial<FieldElement<F>>) -> Self::Commitment {
        // Invariant: poly.degree() <= self.max_degree() (see module docs)
        KZG::<F, P>::commit(&self.committer_key, p)
            .expect("polynomial degree exceeds max_degree")
            .0
            .point
    }

    fn open(
        &self,
        x: &FieldElement<F>,
        _y: &FieldElement<F>,
        p: &Polynomial<FieldElement<F>>,
    ) -> Self::Commitment {
        let state = super::KZGCommitmentState::default();
        // Invariant: quotient degree < poly degree <= max_degree (see module docs)
        let proof = KZG::<F, P>::open(&self.committer_key, p, &state, x)
            .expect("quotient degree exceeds capacity");
        proof.point
    }

    fn open_batch(
        &self,
        x: &FieldElement<F>,
        ys: &[FieldElement<F>],
        polynomials: &[Polynomial<FieldElement<F>>],
        upsilon: &FieldElement<F>,
    ) -> Self::Commitment {
        // Use Horner's method to combine polynomials with random linear combination
        let acc_polynomial = polynomials
            .iter()
            .rev()
            .fold(Polynomial::zero(), |acc, polynomial| {
                acc * upsilon.to_owned() + polynomial
            });

        let acc_y = ys
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, y| acc * upsilon.to_owned() + y);

        self.open(x, &acc_y, &acc_polynomial)
    }

    fn verify(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        p_commitment: &Self::Commitment,
        proof: &Self::Commitment,
    ) -> bool {
        let commitment = KZGCommitment {
            point: p_commitment.clone(),
        };
        let proof = KZGProof {
            point: proof.clone(),
        };

        // Invariant: well-formed curve points from this library (see module docs)
        KZG::<F, P>::verify(&self.verifier_key, &commitment, x, y, &proof)
            .expect("pairing computation failed")
    }

    fn verify_batch(
        &self,
        x: &FieldElement<F>,
        ys: &[FieldElement<F>],
        p_commitments: &[Self::Commitment],
        proof: &Self::Commitment,
        upsilon: &FieldElement<F>,
    ) -> bool {
        // Combine commitments using random linear combination
        let acc_commitment =
            p_commitments
                .iter()
                .rev()
                .fold(P::G1Point::neutral_element(), |acc, point| {
                    acc.operate_with_self(upsilon.to_owned().canonical())
                        .operate_with(point)
                });

        let acc_y = ys
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, y| acc * upsilon.to_owned() + y);

        self.verify(x, &acc_y, &acc_commitment, proof)
    }
}
