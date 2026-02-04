//! KZG (Kate-Zaverucha-Goldberg) Polynomial Commitment Scheme.
//!
//! This module provides a KZG implementation that implements the
//! [`PolynomialCommitmentScheme`] trait.
//!
//! # Overview
//!
//! KZG is a pairing-based polynomial commitment scheme with:
//! - **Constant-size commitments**: A single elliptic curve point
//! - **Constant-size proofs**: A single elliptic curve point
//! - **Trusted setup**: Requires a Structured Reference String (SRS)
//!
//! # Security
//!
//! Security relies on the hardness of the discrete logarithm problem
//! in the pairing groups and the knowledge-of-exponent assumption.
//!
//! # References
//!
//! - [KZG10 Paper](https://www.iacr.org/archive/asiacrypt2010/6477178/6477178.pdf)

mod adapter;
mod commitment;
mod legacy_srs;
mod proof;
mod srs;

use crate::pcs::error::PCSError;
use crate::pcs::traits::PolynomialCommitmentScheme;
#[cfg(feature = "alloc")]
use crate::pcs::traits::BatchPCS;

pub use adapter::KZGAdapter;
pub use commitment::KZGCommitment;
pub use legacy_srs::StructuredReferenceString;
pub use proof::KZGProof;
pub use srs::{KZGCommitterKey, KZGPublicParams, KZGVerifierKey};

use alloc::vec::Vec;
use core::marker::PhantomData;
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::traits::IsPairing;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsPrimeField;
use lambdaworks_math::msm::pippenger::msm;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;

/// KZG Polynomial Commitment Scheme.
///
/// # Type Parameters
///
/// - `F`: The scalar field of the pairing.
/// - `P`: The pairing implementation (e.g., BLS12-381).
#[derive(Clone, Debug)]
pub struct KZG<F: IsPrimeField, P: IsPairing> {
    _field: PhantomData<F>,
    _pairing: PhantomData<P>,
}

impl<F: IsPrimeField, P: IsPairing> KZG<F, P> {
    /// Create a new KZG instance.
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
            _pairing: PhantomData,
        }
    }
}

impl<F: IsPrimeField, P: IsPairing> Default for KZG<F, P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Commitment state for KZG (empty for non-hiding, randomness for hiding).
#[derive(Clone, Debug)]
pub struct KZGCommitmentState<F: IsPrimeField> {
    /// Blinding factor for hiding commitments.
    pub blinding_factor: Option<FieldElement<F>>,
}

impl<F: IsPrimeField> Default for KZGCommitmentState<F> {
    fn default() -> Self {
        Self {
            blinding_factor: None,
        }
    }
}

impl<const N: usize, F, P> PolynomialCommitmentScheme<F> for KZG<F, P>
where
    F: IsPrimeField<CanonicalType = UnsignedInteger<N>>,
    P: IsPairing,
    P::G1Point: Clone + PartialEq + Eq,
    P::G2Point: Clone,
{
    type PublicParameters = KZGPublicParams<P>;
    type CommitterKey = KZGCommitterKey<P>;
    type VerifierKey = KZGVerifierKey<P>;
    type Commitment = KZGCommitment<P>;
    type CommitmentState = KZGCommitmentState<F>;
    type Proof = KZGProof<P>;

    fn max_degree(&self) -> usize {
        // This would be determined by the SRS size
        // For now, return a placeholder
        usize::MAX
    }

    #[cfg(feature = "std")]
    fn setup<R: rand::RngCore>(
        _max_degree: usize,
        _rng: &mut R,
    ) -> Result<Self::PublicParameters, PCSError> {
        // Note: In production, SRS should be loaded from a trusted setup ceremony.
        // This placeholder exists because proper SRS generation requires
        // careful handling of toxic waste.
        Err(PCSError::setup(
            "KZG setup not yet implemented - use load_srs or from_existing_srs instead",
        ))
    }

    fn trim(
        pp: &Self::PublicParameters,
        supported_degree: usize,
    ) -> Result<(Self::CommitterKey, Self::VerifierKey), PCSError> {
        if supported_degree > pp.max_degree {
            return Err(PCSError::degree_too_large(pp.max_degree, supported_degree));
        }

        if pp.powers_of_g1.len() <= supported_degree {
            return Err(PCSError::setup(
                "Insufficient G1 powers in public parameters",
            ));
        }

        if pp.powers_of_g2.len() < 2 {
            return Err(PCSError::setup(
                "Public parameters must contain at least 2 G2 points",
            ));
        }

        let ck = KZGCommitterKey {
            powers_of_g1: pp.powers_of_g1[..=supported_degree].to_vec(),
            max_degree: supported_degree,
        };

        let vk = KZGVerifierKey {
            g1: pp.powers_of_g1[0].clone(),
            g2: pp.powers_of_g2[0].clone(),
            tau_g2: pp.powers_of_g2[1].clone(),
        };

        Ok((ck, vk))
    }

    #[inline]
    fn commit(
        ck: &Self::CommitterKey,
        polynomial: &Polynomial<FieldElement<F>>,
    ) -> Result<(Self::Commitment, Self::CommitmentState), PCSError> {
        let degree = polynomial.degree();
        if degree > ck.max_degree {
            return Err(PCSError::degree_too_large(ck.max_degree, degree));
        }

        // Convert coefficients to canonical form for MSM
        let coefficients: Vec<_> = polynomial
            .coefficients
            .iter()
            .map(|c| c.canonical())
            .collect();

        // Compute commitment using multi-scalar multiplication:
        // C = Σ coefficients[i] * powers_of_g1[i]
        let point = msm(&coefficients, &ck.powers_of_g1[..coefficients.len()])
            .map_err(|_| PCSError::commitment("MSM failed during commitment"))?;

        let commitment = KZGCommitment::new(point);
        let state = KZGCommitmentState::default();

        Ok((commitment, state))
    }

    #[inline]
    fn open(
        ck: &Self::CommitterKey,
        polynomial: &Polynomial<FieldElement<F>>,
        _commitment_state: &Self::CommitmentState,
        point: &FieldElement<F>,
    ) -> Result<Self::Proof, PCSError> {
        // Compute the evaluation y = p(point)
        let evaluation = polynomial.evaluate(point);

        // Compute quotient polynomial: q(x) = (p(x) - y) / (x - point)
        let mut quotient = polynomial - &evaluation;
        quotient.ruffini_division_inplace(point);

        // Commit to the quotient polynomial to get the proof
        let coefficients: Vec<_> = quotient
            .coefficients
            .iter()
            .map(|c| c.canonical())
            .collect();

        if coefficients.len() > ck.powers_of_g1.len() {
            return Err(PCSError::opening(
                "Quotient polynomial degree exceeds committer key capacity",
            ));
        }

        let proof_point = msm(&coefficients, &ck.powers_of_g1[..coefficients.len()])
            .map_err(|_| PCSError::opening("MSM failed during proof generation"))?;

        Ok(KZGProof::new(proof_point))
    }

    #[inline]
    fn verify(
        vk: &Self::VerifierKey,
        commitment: &Self::Commitment,
        point: &FieldElement<F>,
        evaluation: &FieldElement<F>,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        // Verify using pairing equation:
        // e(C - y·G1, G2) = e(π, τ·G2 - z·G2)
        //
        // Rearranged as:
        // e(C - y·G1, G2) · e(-π, τ·G2 - z·G2) = 1

        let g1 = &vk.g1;
        let g2 = &vk.g2;
        let tau_g2 = &vk.tau_g2;

        // Compute C - y·G1
        let y_g1 = g1.operate_with_self(evaluation.canonical());
        let lhs_g1 = commitment.point.operate_with(&y_g1.neg());

        // Compute τ·G2 - z·G2
        let z_g2 = g2.operate_with_self(point.canonical());
        let rhs_g2 = tau_g2.operate_with(&z_g2.neg());

        // Compute pairing: e(C - y·G1, G2) · e(-π, τ·G2 - z·G2)
        let result = P::compute_batch(&[(&lhs_g1, g2), (&proof.point.neg(), &rhs_g2)]);

        match result {
            Ok(pairing_result) => Ok(pairing_result == FieldElement::one()),
            Err(_) => Err(PCSError::PairingCheckFailed),
        }
    }
}

/// Implement batch operations for KZG.
///
/// Batch operations use random linear combinations (via the challenge parameter)
/// to combine multiple polynomials/commitments into a single proof/verification.
#[cfg(feature = "alloc")]
impl<const N: usize, F, P> BatchPCS<F> for KZG<F, P>
where
    F: IsPrimeField<CanonicalType = UnsignedInteger<N>>,
    P: IsPairing,
    P::G1Point: Clone + PartialEq + Eq,
    P::G2Point: Clone,
{
    type BatchProof = KZGProof<P>;

    fn batch_open_single_point(
        ck: &KZGCommitterKey<P>,
        polynomials: &[Polynomial<FieldElement<F>>],
        _commitment_states: &[KZGCommitmentState<F>],
        point: &FieldElement<F>,
        challenge: &FieldElement<F>,
    ) -> Result<KZGProof<P>, PCSError> {
        if polynomials.is_empty() {
            return Err(PCSError::opening("Cannot batch open empty polynomial list"));
        }

        // Compute evaluations
        let evaluations: Vec<_> = polynomials.iter().map(|p| p.evaluate(point)).collect();

        // Combine polynomials using random linear combination (Horner's method)
        // acc = p_n + challenge * (p_{n-1} + challenge * (p_{n-2} + ...))
        let combined_poly = polynomials
            .iter()
            .rev()
            .fold(Polynomial::zero(), |acc, poly| {
                acc * challenge.clone() + poly
            });

        // Combine evaluations similarly
        let combined_eval = evaluations
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, eval| {
                acc * challenge.clone() + eval
            });

        // Compute quotient: q(x) = (combined_poly(x) - combined_eval) / (x - point)
        let mut quotient = &combined_poly - &combined_eval;
        quotient.ruffini_division_inplace(point);

        // Commit to quotient
        let coefficients: Vec<_> = quotient
            .coefficients
            .iter()
            .map(|c| c.canonical())
            .collect();

        if coefficients.len() > ck.powers_of_g1.len() {
            return Err(PCSError::opening(
                "Quotient polynomial degree exceeds committer key capacity",
            ));
        }

        let proof_point = msm(&coefficients, &ck.powers_of_g1[..coefficients.len()])
            .map_err(|_| PCSError::opening("MSM failed during batch proof generation"))?;

        Ok(KZGProof::new(proof_point))
    }

    fn batch_verify_single_point(
        vk: &KZGVerifierKey<P>,
        commitments: &[KZGCommitment<P>],
        point: &FieldElement<F>,
        evaluations: &[FieldElement<F>],
        proof: &KZGProof<P>,
        challenge: &FieldElement<F>,
    ) -> Result<bool, PCSError> {
        if commitments.len() != evaluations.len() {
            return Err(PCSError::length_mismatch(commitments.len(), evaluations.len()));
        }

        if commitments.is_empty() {
            return Err(PCSError::verification(
                "Cannot batch verify empty commitment list",
            ));
        }

        // Combine commitments using random linear combination (Horner's method)
        // acc_commitment = C_n + challenge * (C_{n-1} + challenge * ...)
        let combined_commitment = commitments
            .iter()
            .rev()
            .fold(P::G1Point::neutral_element(), |acc, c| {
                acc.operate_with_self(challenge.canonical())
                    .operate_with(&c.point)
            });

        // Combine evaluations similarly
        let combined_eval = evaluations
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, eval| {
                acc * challenge.clone() + eval
            });

        // Verify using the standard pairing check with combined values
        let g1 = &vk.g1;
        let g2 = &vk.g2;
        let tau_g2 = &vk.tau_g2;

        // Compute combined_commitment - combined_eval·G1
        let eval_g1 = g1.operate_with_self(combined_eval.canonical());
        let lhs_g1 = combined_commitment.operate_with(&eval_g1.neg());

        // Compute τ·G2 - point·G2
        let point_g2 = g2.operate_with_self(point.canonical());
        let rhs_g2 = tau_g2.operate_with(&point_g2.neg());

        // Pairing check: e(lhs_g1, G2) · e(-π, rhs_g2) = 1
        let result = P::compute_batch(&[(&lhs_g1, g2), (&proof.point.neg(), &rhs_g2)]);

        match result {
            Ok(pairing_result) => Ok(pairing_result == FieldElement::one()),
            Err(_) => Err(PCSError::PairingCheckFailed),
        }
    }

    fn batch_open_multi_point(
        _ck: &KZGCommitterKey<P>,
        _polynomials: &[Polynomial<FieldElement<F>>],
        _commitment_states: &[KZGCommitmentState<F>],
        _points: &[FieldElement<F>],
        _challenge: &FieldElement<F>,
    ) -> Result<KZGProof<P>, PCSError> {
        // Multi-point batch opening requires more complex techniques
        // (e.g., Kate-Zaverucha-Goldberg multi-point scheme or Shplonk)
        Err(PCSError::opening(
            "Multi-point batch opening not yet implemented - use single-point batch or individual openings",
        ))
    }

    fn batch_verify_multi_point(
        _vk: &KZGVerifierKey<P>,
        _commitments: &[KZGCommitment<P>],
        _points: &[FieldElement<F>],
        _evaluations: &[Vec<FieldElement<F>>],
        _proof: &KZGProof<P>,
        _challenge: &FieldElement<F>,
    ) -> Result<bool, PCSError> {
        // Multi-point batch verification requires the corresponding opening implementation
        Err(PCSError::verification(
            "Multi-point batch verification not yet implemented",
        ))
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
    use lambdaworks_math::unsigned_integer::element::U256;

    type TestKZG = KZG<FrField, BLS12381AtePairing>;
    type G1Point =
        <BLS12381AtePairing as lambdaworks_math::elliptic_curve::traits::IsPairing>::G1Point;
    type G2Point =
        <BLS12381AtePairing as lambdaworks_math::elliptic_curve::traits::IsPairing>::G2Point;

    /// Create a test SRS with random toxic waste
    fn create_test_srs(max_degree: usize) -> KZGPublicParams<BLS12381AtePairing> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Generate random toxic waste (τ)
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

        // Generate powers of τ in G1: [g1, τ·g1, τ²·g1, ..., τ^n·g1]
        let powers_of_g1: Vec<G1Point> = (0..=max_degree)
            .map(|exp| g1.operate_with_self(toxic_waste.pow(exp as u128).canonical()))
            .collect();

        // Generate powers of τ in G2: [g2, τ·g2]
        let powers_of_g2: Vec<G2Point> =
            vec![g2.clone(), g2.operate_with_self(toxic_waste.canonical())];

        KZGPublicParams::new(powers_of_g1, powers_of_g2)
    }

    #[test]
    fn test_kzg_new() {
        let _kzg: TestKZG = KZG::new();
    }

    #[test]
    fn test_kzg_trim() {
        let pp = create_test_srs(10);
        let result = TestKZG::trim(&pp, 5);
        assert!(result.is_ok());

        let (ck, vk) = result.unwrap();
        assert_eq!(ck.max_degree, 5);
        assert_eq!(ck.powers_of_g1.len(), 6); // 0 to 5 inclusive
        assert_eq!(vk.g1, pp.powers_of_g1[0]);
    }

    #[test]
    fn test_kzg_trim_degree_too_large() {
        let pp = create_test_srs(5);
        let result = TestKZG::trim(&pp, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_kzg_commit_constant_polynomial() {
        let pp = create_test_srs(10);
        let (ck, _vk) = TestKZG::trim(&pp, 10).unwrap();

        // Constant polynomial p(x) = 42
        let p = Polynomial::new(&[FieldElement::<FrField>::from(42u64)]);
        let result = TestKZG::commit(&ck, &p);
        assert!(result.is_ok());
    }

    #[test]
    fn test_kzg_commit_linear_polynomial() {
        let pp = create_test_srs(10);
        let (ck, _vk) = TestKZG::trim(&pp, 10).unwrap();

        // Linear polynomial p(x) = 1 + x
        let p = Polynomial::new(&[FieldElement::<FrField>::one(), FieldElement::one()]);
        let result = TestKZG::commit(&ck, &p);
        assert!(result.is_ok());
    }

    #[test]
    fn test_kzg_full_workflow() {
        let pp = create_test_srs(10);
        let (ck, vk) = TestKZG::trim(&pp, 10).unwrap();

        // Polynomial p(x) = 1 + x
        let p = Polynomial::new(&[FieldElement::<FrField>::one(), FieldElement::one()]);

        // Commit
        let (commitment, state) = TestKZG::commit(&ck, &p).unwrap();

        // Choose evaluation point x = 3
        let x = FieldElement::<FrField>::from(3u64);
        let y = p.evaluate(&x); // y = 1 + 3 = 4

        // Open (create proof)
        let proof = TestKZG::open(&ck, &p, &state, &x).unwrap();

        // Verify
        let is_valid = TestKZG::verify(&vk, &commitment, &x, &y, &proof).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_kzg_verification_fails_for_wrong_evaluation() {
        let pp = create_test_srs(10);
        let (ck, vk) = TestKZG::trim(&pp, 10).unwrap();

        // Polynomial p(x) = 1 + x
        let p = Polynomial::new(&[FieldElement::<FrField>::one(), FieldElement::one()]);

        // Commit
        let (commitment, state) = TestKZG::commit(&ck, &p).unwrap();

        // Choose evaluation point x = 3
        let x = FieldElement::<FrField>::from(3u64);

        // Open with correct proof
        let proof = TestKZG::open(&ck, &p, &state, &x).unwrap();

        // Try to verify with wrong evaluation (should fail)
        let wrong_y = FieldElement::<FrField>::from(999u64);
        let is_valid = TestKZG::verify(&vk, &commitment, &x, &wrong_y, &proof).unwrap();
        assert!(!is_valid);
    }

    #[test]
    fn test_kzg_quadratic_polynomial() {
        let pp = create_test_srs(10);
        let (ck, vk) = TestKZG::trim(&pp, 10).unwrap();

        // Polynomial p(x) = 1 + 2x + 3x²
        let p = Polynomial::new(&[
            FieldElement::<FrField>::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
        ]);

        // Commit
        let (commitment, state) = TestKZG::commit(&ck, &p).unwrap();

        // Choose evaluation point x = 2
        let x = FieldElement::<FrField>::from(2u64);
        let y = p.evaluate(&x); // y = 1 + 4 + 12 = 17

        // Open
        let proof = TestKZG::open(&ck, &p, &state, &x).unwrap();

        // Verify
        let is_valid = TestKZG::verify(&vk, &commitment, &x, &y, &proof).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_kzg_evaluation_at_negative_one() {
        let pp = create_test_srs(10);
        let (ck, vk) = TestKZG::trim(&pp, 10).unwrap();

        // Polynomial p(x) = 1 + x (root at x = -1)
        let p = Polynomial::new(&[FieldElement::<FrField>::one(), FieldElement::one()]);

        // Commit
        let (commitment, state) = TestKZG::commit(&ck, &p).unwrap();

        // Evaluate at x = -1
        let x = -FieldElement::<FrField>::one();
        let y = p.evaluate(&x); // y = 1 + (-1) = 0
        assert_eq!(y, FieldElement::zero());

        // Open
        let proof = TestKZG::open(&ck, &p, &state, &x).unwrap();

        // Verify
        let is_valid = TestKZG::verify(&vk, &commitment, &x, &y, &proof).unwrap();
        assert!(is_valid);
    }

    // ==================== Batch Operation Tests ====================

    #[test]
    #[cfg(feature = "alloc")]
    fn test_kzg_batch_single_polynomial() {
        use crate::pcs::traits::BatchPCS;

        let pp = create_test_srs(10);
        let (ck, vk) = TestKZG::trim(&pp, 10).unwrap();

        // Single polynomial p(x) = 1 + x
        let p = Polynomial::new(&[FieldElement::<FrField>::one(), FieldElement::one()]);
        let (commitment, state) = TestKZG::commit(&ck, &p).unwrap();

        let x = FieldElement::<FrField>::from(3u64);
        let y = p.evaluate(&x);

        // Batch open with a single polynomial
        let challenge = FieldElement::<FrField>::from(7u64);
        let proof =
            TestKZG::batch_open_single_point(&ck, &[p], &[state], &x, &challenge).unwrap();

        // Batch verify
        let is_valid =
            TestKZG::batch_verify_single_point(&vk, &[commitment], &x, &[y], &proof, &challenge)
                .unwrap();
        assert!(is_valid);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn test_kzg_batch_two_polynomials() {
        use crate::pcs::traits::BatchPCS;

        let pp = create_test_srs(10);
        let (ck, vk) = TestKZG::trim(&pp, 10).unwrap();

        // Two polynomials
        let p0 = Polynomial::new(&[FieldElement::<FrField>::from(9000u64)]); // constant
        let p1 = Polynomial::new(&[
            FieldElement::<FrField>::from(1u64),
            FieldElement::from(2u64),
            -FieldElement::from(1u64),
        ]); // 1 + 2x - x²

        let (c0, s0) = TestKZG::commit(&ck, &p0).unwrap();
        let (c1, s1) = TestKZG::commit(&ck, &p1).unwrap();

        let x = FieldElement::<FrField>::from(3u64);
        let y0 = p0.evaluate(&x); // 9000
        let y1 = p1.evaluate(&x); // 1 + 6 - 9 = -2

        let challenge = FieldElement::<FrField>::from(5u64);

        // Batch open
        let proof = TestKZG::batch_open_single_point(
            &ck,
            &[p0, p1],
            &[s0, s1],
            &x,
            &challenge,
        )
        .unwrap();

        // Batch verify
        let is_valid = TestKZG::batch_verify_single_point(
            &vk,
            &[c0, c1],
            &x,
            &[y0, y1],
            &proof,
            &challenge,
        )
        .unwrap();
        assert!(is_valid);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn test_kzg_batch_verification_fails_for_wrong_evaluation() {
        use crate::pcs::traits::BatchPCS;

        let pp = create_test_srs(10);
        let (ck, vk) = TestKZG::trim(&pp, 10).unwrap();

        let p0 = Polynomial::new(&[FieldElement::<FrField>::from(100u64)]);
        let p1 = Polynomial::new(&[FieldElement::<FrField>::one(), FieldElement::one()]);

        let (c0, s0) = TestKZG::commit(&ck, &p0).unwrap();
        let (c1, s1) = TestKZG::commit(&ck, &p1).unwrap();

        let x = FieldElement::<FrField>::from(2u64);
        let y0 = p0.evaluate(&x);
        let _y1 = p1.evaluate(&x);
        let wrong_y1 = FieldElement::<FrField>::from(999u64); // Wrong!

        let challenge = FieldElement::<FrField>::from(3u64);

        let proof = TestKZG::batch_open_single_point(
            &ck,
            &[p0, p1],
            &[s0, s1],
            &x,
            &challenge,
        )
        .unwrap();

        // Should fail with wrong evaluation
        let is_valid = TestKZG::batch_verify_single_point(
            &vk,
            &[c0, c1],
            &x,
            &[y0, wrong_y1],
            &proof,
            &challenge,
        )
        .unwrap();
        assert!(!is_valid);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn test_kzg_batch_three_polynomials() {
        use crate::pcs::traits::BatchPCS;

        let pp = create_test_srs(10);
        let (ck, vk) = TestKZG::trim(&pp, 10).unwrap();

        // Three polynomials of varying degrees
        let p0 = Polynomial::new(&[FieldElement::<FrField>::from(42u64)]);
        let p1 = Polynomial::new(&[FieldElement::<FrField>::one(), FieldElement::from(2u64)]);
        let p2 = Polynomial::new(&[
            FieldElement::<FrField>::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
        ]);

        let (c0, s0) = TestKZG::commit(&ck, &p0).unwrap();
        let (c1, s1) = TestKZG::commit(&ck, &p1).unwrap();
        let (c2, s2) = TestKZG::commit(&ck, &p2).unwrap();

        let x = FieldElement::<FrField>::from(5u64);
        let y0 = p0.evaluate(&x);
        let y1 = p1.evaluate(&x);
        let y2 = p2.evaluate(&x);

        let challenge = FieldElement::<FrField>::from(11u64);

        let proof = TestKZG::batch_open_single_point(
            &ck,
            &[p0, p1, p2],
            &[s0, s1, s2],
            &x,
            &challenge,
        )
        .unwrap();

        let is_valid = TestKZG::batch_verify_single_point(
            &vk,
            &[c0, c1, c2],
            &x,
            &[y0, y1, y2],
            &proof,
            &challenge,
        )
        .unwrap();
        assert!(is_valid);
    }
}
