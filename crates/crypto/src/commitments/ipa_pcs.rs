//! IPA-based Polynomial Commitment Scheme
//!
//! This module wraps the Inner Product Argument to create a polynomial commitment
//! scheme with the following properties:
//!
//! - **Transparent setup**: No trusted ceremony required (unlike KZG)
//! - **Post-quantum concern**: Not based on pairings, so potentially more future-proof
//! - **Logarithmic proof size**: O(log n) proof size for degree n polynomials
//! - **Linear verification**: O(n) verification time (can be amortized)
//!
//! # How It Works
//!
//! A polynomial `p(X) = c_0 + c_1*X + c_2*X^2 + ... + c_n*X^n` is committed by
//! treating its coefficients as a vector and computing a Pedersen vector commitment.
//!
//! To prove that `p(z) = y` for some evaluation point `z`:
//! 1. Note that `p(z) = <coeffs, [1, z, z^2, ..., z^n]>` (inner product)
//! 2. Use IPA to prove this inner product relation
//!
//! # Reference
//!
//! - Halo paper, Section 3: <https://eprint.iacr.org/2019/1021.pdf>
//! - Bulletproofs paper: <https://eprint.iacr.org/2017/1066.pdf>

use alloc::vec::Vec;
use core::marker::PhantomData;

use lambdaworks_math::{
    elliptic_curve::traits::IsEllipticCurve,
    field::{element::FieldElement, traits::IsPrimeField},
    polynomial::Polynomial,
    traits::{AsBytes, ByteConversion},
    unsigned_integer::element::UnsignedInteger,
};

use crate::fiat_shamir::is_transcript::IsTranscript;

use super::{
    ipa::{compute_powers, inner_product, IPAError, IPAProof, IPAProver, IPAVerifier},
    pedersen::PedersenParams,
};

/// IPA-based polynomial commitment scheme.
///
/// This provides a transparent (no trusted setup) polynomial commitment scheme
/// based on the Inner Product Argument.
///
/// # Type Parameters
///
/// * `E` - The elliptic curve for Pedersen commitments
/// * `F` - The scalar field (must match the curve's scalar field)
pub struct IPAPolynomialCommitment<E: IsEllipticCurve, F: IsPrimeField> {
    params: PedersenParams<E>,
    max_degree: usize,
    _marker: PhantomData<F>,
}

impl<E: IsEllipticCurve, F: IsPrimeField> Clone for IPAPolynomialCommitment<E, F>
where
    E::PointRepresentation: Clone,
{
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            max_degree: self.max_degree,
            _marker: PhantomData,
        }
    }
}

impl<E: IsEllipticCurve, F: IsPrimeField> core::fmt::Debug for IPAPolynomialCommitment<E, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IPAPolynomialCommitment")
            .field("max_degree", &self.max_degree)
            .finish()
    }
}

/// A commitment to a polynomial.
pub struct PolynomialCommitment<E: IsEllipticCurve> {
    /// The commitment point C = <coeffs, G>
    pub commitment: E::PointRepresentation,
    /// The degree of the committed polynomial (for verification)
    pub degree: usize,
}

impl<E: IsEllipticCurve> Clone for PolynomialCommitment<E>
where
    E::PointRepresentation: Clone,
{
    fn clone(&self) -> Self {
        Self {
            commitment: self.commitment.clone(),
            degree: self.degree,
        }
    }
}

impl<E: IsEllipticCurve> core::fmt::Debug for PolynomialCommitment<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PolynomialCommitment")
            .field("degree", &self.degree)
            .finish()
    }
}

/// An opening proof for a polynomial evaluation.
///
/// Proves that a committed polynomial evaluates to a specific value at a point.
pub struct OpeningProof<E: IsEllipticCurve, F: IsPrimeField> {
    /// The IPA proof
    pub ipa_proof: IPAProof<E, F>,
    /// The claimed evaluation value
    pub evaluation: FieldElement<F>,
}

impl<E: IsEllipticCurve, F: IsPrimeField> Clone for OpeningProof<E, F>
where
    E::PointRepresentation: Clone,
{
    fn clone(&self) -> Self {
        Self {
            ipa_proof: self.ipa_proof.clone(),
            evaluation: self.evaluation.clone(),
        }
    }
}

impl<E: IsEllipticCurve, F: IsPrimeField> core::fmt::Debug for OpeningProof<E, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OpeningProof")
            .field("num_rounds", &self.ipa_proof.num_rounds())
            .finish()
    }
}

impl<E: IsEllipticCurve, F: IsPrimeField> IPAPolynomialCommitment<E, F>
where
    FieldElement<E::BaseField>: ByteConversion,
    E::PointRepresentation: AsBytes,
{
    /// Create a new IPA polynomial commitment scheme.
    ///
    /// # Arguments
    ///
    /// * `max_degree` - Maximum polynomial degree supported.
    ///   Must be one less than a power of 2 (e.g., 3, 7, 15, 31, ...)
    ///   This is because IPA requires power-of-2 vector lengths.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Support polynomials up to degree 15 (16 coefficients)
    /// let pcs = IPAPolynomialCommitment::<PallasCurve, VestaField>::new(15);
    /// ```
    pub fn new(max_degree: usize) -> Self {
        // We need max_degree + 1 coefficients, which must be a power of 2
        let num_coeffs = max_degree + 1;
        let padded_size = num_coeffs.next_power_of_two();

        let params = PedersenParams::<E>::new(padded_size);

        Self {
            params,
            max_degree,
            _marker: PhantomData,
        }
    }

    /// Create from existing Pedersen parameters.
    ///
    /// Useful when you want to share parameters across multiple instances.
    pub fn from_params(params: PedersenParams<E>, max_degree: usize) -> Self {
        Self {
            params,
            max_degree,
            _marker: PhantomData,
        }
    }

    /// Returns the maximum polynomial degree this scheme supports.
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Commit to a polynomial.
    ///
    /// Creates a binding commitment to the polynomial's coefficients.
    /// The commitment hides nothing about the polynomial (use blinding if needed).
    ///
    /// # Arguments
    ///
    /// * `poly` - The polynomial to commit to
    ///
    /// # Returns
    ///
    /// A polynomial commitment, or an error if the degree is too large.
    pub fn commit<const N: usize>(
        &self,
        poly: &Polynomial<FieldElement<F>>,
    ) -> Result<PolynomialCommitment<E>, IPAError>
    where
        F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>,
    {
        let degree = poly.degree();
        if degree > self.max_degree {
            return Err(IPAError::LengthMismatch {
                expected: self.max_degree,
                actual: degree,
            });
        }

        // Pad coefficients to power of 2
        let coeffs = self.pad_coefficients(&poly.coefficients);

        // Create commitment using Pedersen
        let commitment = self.params.commit_without_blinding(&coeffs)?;

        Ok(PolynomialCommitment { commitment, degree })
    }

    /// Open a polynomial commitment at a point.
    ///
    /// Creates a proof that `p(point) = evaluation`.
    ///
    /// # Arguments
    ///
    /// * `poly` - The polynomial (must match the commitment)
    /// * `point` - The evaluation point
    /// * `transcript` - Fiat-Shamir transcript
    ///
    /// # Returns
    ///
    /// An opening proof containing the evaluation and IPA proof.
    pub fn open<const N: usize, T>(
        &self,
        poly: &Polynomial<FieldElement<F>>,
        point: &FieldElement<F>,
        transcript: &mut T,
    ) -> Result<OpeningProof<E, F>, IPAError>
    where
        F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>,
        T: IsTranscript<F>,
    {
        // Compute evaluation
        let evaluation = poly.evaluate(point);

        // Pad coefficients to power of 2
        let coeffs = self.pad_coefficients(&poly.coefficients);
        let n = coeffs.len();

        // Compute powers of the evaluation point: [1, z, z^2, ..., z^{n-1}]
        let powers = compute_powers(point, n);

        // Add point and evaluation to transcript for binding
        transcript.append_field_element(point);
        transcript.append_field_element(&evaluation);

        // Create IPA proof for <coeffs, powers> = evaluation
        let prover = IPAProver::<E, F>::new(self.params.clone());
        let ipa_proof = prover.prove(&coeffs, &powers, transcript)?;

        Ok(OpeningProof {
            ipa_proof,
            evaluation,
        })
    }

    /// Verify an opening proof.
    ///
    /// Checks that the proof demonstrates `p(point) = proof.evaluation` for
    /// the polynomial committed in `commitment`.
    ///
    /// # Arguments
    ///
    /// * `commitment` - The polynomial commitment
    /// * `point` - The evaluation point
    /// * `proof` - The opening proof
    /// * `transcript` - Fiat-Shamir transcript (must match prover's)
    ///
    /// # Returns
    ///
    /// `true` if the proof is valid, `false` otherwise.
    pub fn verify<const N: usize, T>(
        &self,
        commitment: &PolynomialCommitment<E>,
        point: &FieldElement<F>,
        proof: &OpeningProof<E, F>,
        transcript: &mut T,
    ) -> Result<bool, IPAError>
    where
        F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>,
        T: IsTranscript<F>,
    {
        // Determine the vector size from the proof
        let n = proof.ipa_proof.original_size();

        // Compute powers of the evaluation point
        let powers = compute_powers(point, n);

        // Add point and evaluation to transcript (must match prover)
        transcript.append_field_element(point);
        transcript.append_field_element(&proof.evaluation);

        // Verify IPA proof
        let verifier = IPAVerifier::<E, F>::new(self.params.clone());
        verifier.verify(
            &commitment.commitment,
            &powers,
            &proof.evaluation,
            &proof.ipa_proof,
            transcript,
        )
    }

    /// Pad polynomial coefficients to the next power of 2.
    fn pad_coefficients(&self, coeffs: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
        let target_len = (coeffs.len()).next_power_of_two();
        let mut padded = coeffs.to_vec();
        padded.resize(target_len, FieldElement::zero());
        padded
    }
}

/// Convenience function to evaluate a polynomial at a point.
///
/// This is mathematically equivalent to inner product:
/// `p(z) = c_0 + c_1*z + c_2*z^2 + ... = <[c_0, c_1, ...], [1, z, z^2, ...]>`
pub fn evaluate_as_inner_product<F: IsPrimeField>(
    coeffs: &[FieldElement<F>],
    point: &FieldElement<F>,
) -> FieldElement<F> {
    let powers = compute_powers(point, coeffs.len());
    inner_product(coeffs, &powers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::{
        cyclic_group::IsGroup,
        elliptic_curve::short_weierstrass::curves::pallas::curve::PallasCurve,
        field::{element::FieldElement, fields::vesta_field::Vesta255PrimeField},
        polynomial::Polynomial,
    };

    type FE = FieldElement<Vesta255PrimeField>;
    type Transcript = DefaultTranscript<Vesta255PrimeField>;
    type PCS = IPAPolynomialCommitment<PallasCurve, Vesta255PrimeField>;

    #[test]
    fn test_polynomial_commitment_basic() {
        // Support polynomials up to degree 3 (4 coefficients, power of 2)
        let pcs = PCS::new(3);

        // p(x) = 1 + 2x + 3x^2
        let poly = Polynomial::new(&[FE::from(1), FE::from(2), FE::from(3)]);

        let commitment = pcs.commit(&poly).expect("commitment should succeed");
        assert!(!commitment.commitment.is_neutral_element());
        assert_eq!(commitment.degree, 2);
    }

    #[test]
    fn test_polynomial_open_verify() {
        let pcs = PCS::new(3);

        // p(x) = 1 + 2x + 3x^2
        let poly = Polynomial::new(&[FE::from(1), FE::from(2), FE::from(3)]);

        let commitment = pcs.commit(&poly).expect("commitment should succeed");

        // Evaluate at x = 2: p(2) = 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        let point = FE::from(2);
        let expected_eval = FE::from(17);
        assert_eq!(poly.evaluate(&point), expected_eval);

        // Create opening proof
        let mut prover_transcript = Transcript::default();
        let proof = pcs
            .open(&poly, &point, &mut prover_transcript)
            .expect("opening should succeed");

        assert_eq!(proof.evaluation, expected_eval);

        // Verify proof
        let mut verifier_transcript = Transcript::default();
        let result = pcs
            .verify(&commitment, &point, &proof, &mut verifier_transcript)
            .expect("verification should not error");

        assert!(result, "Valid opening proof should verify");
    }

    #[test]
    fn test_polynomial_constant() {
        let pcs = PCS::new(3);

        // p(x) = 42 (constant polynomial)
        let poly = Polynomial::new(&[FE::from(42)]);

        let commitment = pcs.commit(&poly).expect("commitment should succeed");

        // Evaluate at any point should give 42
        let point = FE::from(999);

        let mut prover_transcript = Transcript::default();
        let proof = pcs
            .open(&poly, &point, &mut prover_transcript)
            .expect("opening should succeed");

        assert_eq!(proof.evaluation, FE::from(42));

        let mut verifier_transcript = Transcript::default();
        let result = pcs
            .verify(&commitment, &point, &proof, &mut verifier_transcript)
            .expect("verification should not error");

        assert!(result);
    }

    #[test]
    fn test_polynomial_linear() {
        let pcs = PCS::new(3);

        // p(x) = 3 + 5x
        let poly = Polynomial::new(&[FE::from(3), FE::from(5)]);

        let commitment = pcs.commit(&poly).expect("commitment should succeed");

        // p(7) = 3 + 5*7 = 3 + 35 = 38
        let point = FE::from(7);
        let expected = FE::from(38);

        let mut prover_transcript = Transcript::default();
        let proof = pcs
            .open(&poly, &point, &mut prover_transcript)
            .expect("opening should succeed");

        assert_eq!(proof.evaluation, expected);

        let mut verifier_transcript = Transcript::default();
        let result = pcs
            .verify(&commitment, &point, &proof, &mut verifier_transcript)
            .expect("verification should not error");

        assert!(result);
    }

    #[test]
    fn test_polynomial_wrong_evaluation() {
        let pcs = PCS::new(3);

        let poly = Polynomial::new(&[FE::from(1), FE::from(2), FE::from(3)]);
        let commitment = pcs.commit(&poly).expect("commitment should succeed");

        let point = FE::from(2);

        let mut prover_transcript = Transcript::default();
        let mut proof = pcs
            .open(&poly, &point, &mut prover_transcript)
            .expect("opening should succeed");

        // Tamper with the evaluation
        proof.evaluation = FE::from(9999);

        let mut verifier_transcript = Transcript::default();
        let result = pcs
            .verify(&commitment, &point, &proof, &mut verifier_transcript)
            .expect("verification should not error");

        assert!(!result, "Tampered proof should not verify");
    }

    #[test]
    fn test_polynomial_degree_too_large() {
        let pcs = PCS::new(3); // Max degree 3

        // Degree 4 polynomial
        let poly = Polynomial::new(&[
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
        ]);

        let result = pcs.commit(&poly);
        assert!(matches!(result, Err(IPAError::LengthMismatch { .. })));
    }

    #[test]
    fn test_evaluate_as_inner_product() {
        let coeffs = vec![FE::from(1), FE::from(2), FE::from(3)];
        let point = FE::from(2);

        // p(2) = 1 + 2*2 + 3*4 = 17
        let result = evaluate_as_inner_product(&coeffs, &point);
        assert_eq!(result, FE::from(17));

        // Compare with Polynomial::evaluate
        let poly = Polynomial::new(&coeffs);
        assert_eq!(result, poly.evaluate(&point));
    }

    #[test]
    fn test_polynomial_at_zero() {
        let pcs = PCS::new(3);

        // p(x) = 5 + 3x + 2x^2
        let poly = Polynomial::new(&[FE::from(5), FE::from(3), FE::from(2)]);

        let commitment = pcs.commit(&poly).expect("commitment should succeed");

        // p(0) = 5
        let point = FE::zero();

        let mut prover_transcript = Transcript::default();
        let proof = pcs
            .open(&poly, &point, &mut prover_transcript)
            .expect("opening should succeed");

        assert_eq!(proof.evaluation, FE::from(5));

        let mut verifier_transcript = Transcript::default();
        let result = pcs
            .verify(&commitment, &point, &proof, &mut verifier_transcript)
            .expect("verification should not error");

        assert!(result);
    }

    #[test]
    fn test_multiple_openings_same_polynomial() {
        let pcs = PCS::new(7);

        // p(x) = 1 + x + x^2 + x^3
        let poly = Polynomial::new(&[FE::from(1), FE::from(1), FE::from(1), FE::from(1)]);

        let commitment = pcs.commit(&poly).expect("commitment should succeed");

        // Open at multiple points
        for i in 1..=5 {
            let point = FE::from(i as u64);
            let expected = poly.evaluate(&point);

            let mut prover_transcript = Transcript::default();
            let proof = pcs
                .open(&poly, &point, &mut prover_transcript)
                .expect("opening should succeed");

            assert_eq!(proof.evaluation, expected);

            let mut verifier_transcript = Transcript::default();
            let result = pcs
                .verify(&commitment, &point, &proof, &mut verifier_transcript)
                .expect("verification should not error");

            assert!(result, "Opening at point {} should verify", i);
        }
    }
}
