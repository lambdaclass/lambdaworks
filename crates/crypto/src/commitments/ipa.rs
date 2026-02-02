//! Inner Product Argument (IPA) Protocol
//!
//! This module implements the Inner Product Argument, a key building block for:
//! - Bulletproofs range proofs
//! - Halo 2 recursive SNARKs
//! - Any system requiring transparent polynomial commitments
//!
//! # Protocol Overview
//!
//! The IPA protocol proves knowledge of vectors `a`, `b` such that:
//! - The prover knows `a` committed in `P = <a, G> + <a, b> * U`
//! - The inner product `<a, b> = z` equals the claimed value
//!
//! The proof has logarithmic size in the vector length, achieved through
//! recursive halving of the problem.
//!
//! # Mathematical Background
//!
//! At each round, the prover:
//! 1. Splits vectors: `a = (a_L, a_R)`, `b = (b_L, b_R)`, `G = (G_L, G_R)`
//! 2. Computes cross-terms:
//!    - `L = <a_L, G_R> + <a_L, b_R> * U`
//!    - `R = <a_R, G_L> + <a_R, b_L> * U`
//! 3. Receives challenge `x` from verifier (via Fiat-Shamir)
//! 4. Folds vectors:
//!    - `a' = x^{-1} * a_L + x * a_R`
//!    - `b' = x * b_L + x^{-1} * b_R`
//!    - `G' = x * G_L + x^{-1} * G_R`
//!
//! This continues until vectors have length 1, producing a logarithmic proof.
//!
//! # Reference
//!
//! - Bulletproofs paper, Section 3: <https://eprint.iacr.org/2017/1066.pdf>
//! - Halo paper: <https://eprint.iacr.org/2019/1021.pdf>

use alloc::vec::Vec;
use core::marker::PhantomData;

use core::fmt::Debug;

use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::traits::IsEllipticCurve,
    field::{element::FieldElement, traits::IsPrimeField},
    msm::pippenger::msm,
    traits::{AsBytes, ByteConversion},
    unsigned_integer::element::UnsignedInteger,
};

use crate::fiat_shamir::is_transcript::IsTranscript;

use super::pedersen::{PedersenError, PedersenParams};

/// Error types for IPA operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IPAError {
    /// Vectors have mismatched lengths
    LengthMismatch { expected: usize, actual: usize },
    /// Vector length is not a power of 2
    NotPowerOfTwo { length: usize },
    /// Empty vectors provided
    EmptyVectors,
    /// Invalid proof structure
    InvalidProof,
    /// MSM computation failed
    MsmError,
    /// Field inversion failed
    InversionError,
    /// Pedersen commitment error
    PedersenError(PedersenError),
}

impl From<PedersenError> for IPAError {
    fn from(err: PedersenError) -> Self {
        IPAError::PedersenError(err)
    }
}

/// IPA proof structure
///
/// Contains the logarithmic-sized proof for an inner product claim.
/// For vectors of length `n`, the proof contains `log2(n)` L and R points.
pub struct IPAProof<E: IsEllipticCurve, F: IsPrimeField> {
    /// Left folding commitments: `L_0, L_1, ..., L_{log(n)-1}`
    pub l_vec: Vec<E::PointRepresentation>,
    /// Right folding commitments: `R_0, R_1, ..., R_{log(n)-1}`
    pub r_vec: Vec<E::PointRepresentation>,
    /// Final scalar `a` after all folding rounds
    pub a_final: FieldElement<F>,
    /// Final scalar `b` after all folding rounds
    pub b_final: FieldElement<F>,
}

impl<E: IsEllipticCurve, F: IsPrimeField> Clone for IPAProof<E, F>
where
    E::PointRepresentation: Clone,
{
    fn clone(&self) -> Self {
        Self {
            l_vec: self.l_vec.clone(),
            r_vec: self.r_vec.clone(),
            a_final: self.a_final.clone(),
            b_final: self.b_final.clone(),
        }
    }
}

impl<E: IsEllipticCurve, F: IsPrimeField> Debug for IPAProof<E, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IPAProof")
            .field("num_rounds", &self.l_vec.len())
            .field("a_final", &"...")
            .field("b_final", &"...")
            .finish()
    }
}

impl<E: IsEllipticCurve, F: IsPrimeField> IPAProof<E, F> {
    /// Returns the number of rounds (log2 of original vector size)
    pub fn num_rounds(&self) -> usize {
        self.l_vec.len()
    }

    /// Returns the original vector size this proof was created for
    pub fn original_size(&self) -> usize {
        1 << self.num_rounds()
    }
}

/// Inner Product Argument prover
///
/// Creates proofs that demonstrate knowledge of vectors `a`, `b` such that
/// their inner product equals a claimed value.
#[derive(Clone)]
pub struct IPAProver<E: IsEllipticCurve, F: IsPrimeField> {
    params: PedersenParams<E>,
    _marker: PhantomData<F>,
}

impl<E: IsEllipticCurve, F: IsPrimeField> Debug for IPAProver<E, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IPAProver")
            .field("max_size", &self.params.max_size())
            .finish()
    }
}

impl<E: IsEllipticCurve, F: IsPrimeField> IPAProver<E, F>
where
    FieldElement<E::BaseField>: ByteConversion,
    E::PointRepresentation: AsBytes,
{
    /// Create a new IPA prover with the given parameters
    pub fn new(params: PedersenParams<E>) -> Self {
        Self {
            params,
            _marker: PhantomData,
        }
    }

    /// Create an IPA proof for vectors `a` and `b`.
    ///
    /// Proves knowledge of `a`, `b` such that:
    /// - The commitment `P = <a, G> + <a,b> * U` is correctly formed
    /// - `<a, b> = z` (the inner product equals the claimed value)
    ///
    /// # Arguments
    ///
    /// * `a` - First vector (the prover's secret)
    /// * `b` - Second vector (typically public, e.g., powers of evaluation point)
    /// * `transcript` - Fiat-Shamir transcript for challenge generation
    ///
    /// # Returns
    ///
    /// An IPA proof with logarithmic size.
    ///
    /// # Panics
    ///
    /// Panics if vector lengths are not equal or not a power of 2.
    pub fn prove<const N: usize, T>(
        &self,
        a: &[FieldElement<F>],
        b: &[FieldElement<F>],
        transcript: &mut T,
    ) -> Result<IPAProof<E, F>, IPAError>
    where
        F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>,
        T: IsTranscript<F>,
    {
        // Validate inputs
        if a.is_empty() || b.is_empty() {
            return Err(IPAError::EmptyVectors);
        }
        if a.len() != b.len() {
            return Err(IPAError::LengthMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        let n = a.len();
        if !n.is_power_of_two() {
            return Err(IPAError::NotPowerOfTwo { length: n });
        }
        if n > self.params.max_size() {
            return Err(IPAError::LengthMismatch {
                expected: self.params.max_size(),
                actual: n,
            });
        }

        let num_rounds = n.ilog2() as usize;
        let mut l_vec = Vec::with_capacity(num_rounds);
        let mut r_vec = Vec::with_capacity(num_rounds);

        // Working copies of vectors
        let mut a_vec = a.to_vec();
        let mut b_vec = b.to_vec();
        let mut g_vec = self.params.g_vec[..n].to_vec();

        // Get the U generator for inner product binding
        let u = &self.params.u;

        // Main IPA loop: recursive halving
        // Reference: Bulletproofs paper, Protocol 1 (Section 3.1)
        while a_vec.len() > 1 {
            let half = a_vec.len() / 2;

            // Split vectors
            let (a_l, a_r) = a_vec.split_at(half);
            let (b_l, b_r) = b_vec.split_at(half);
            let (g_l, g_r) = g_vec.split_at(half);

            // Compute inner products for cross-terms
            // c_L = <a_L, b_R>
            // c_R = <a_R, b_L>
            let c_l = inner_product(a_l, b_r);
            let c_r = inner_product(a_r, b_l);

            // Compute L = <a_L, G_R> + c_L * U
            // This is the "left" cross-term commitment
            let a_l_scalars: Vec<UnsignedInteger<N>> =
                a_l.iter().map(|x| x.representative()).collect();
            let l_msm = msm(&a_l_scalars, g_r).map_err(|_| IPAError::MsmError)?;
            let l = l_msm.operate_with(&u.operate_with_self(c_l.representative()));

            // Compute R = <a_R, G_L> + c_R * U
            // This is the "right" cross-term commitment
            let a_r_scalars: Vec<UnsignedInteger<N>> =
                a_r.iter().map(|x| x.representative()).collect();
            let r_msm = msm(&a_r_scalars, g_l).map_err(|_| IPAError::MsmError)?;
            let r = r_msm.operate_with(&u.operate_with_self(c_r.representative()));

            // Append L, R to transcript and get challenge
            transcript.append_bytes(&l.as_bytes());
            transcript.append_bytes(&r.as_bytes());
            let x: FieldElement<F> = transcript.sample_field_element();

            // Compute x_inv for folding
            let x_inv = x.inv().map_err(|_| IPAError::InversionError)?;

            // Store L, R in proof
            l_vec.push(l);
            r_vec.push(r);

            // Fold vectors according to Bulletproofs/HALO IPA protocol:
            // a' = x^{-1} * a_L + x * a_R
            let a_new: Vec<FieldElement<F>> = a_l
                .iter()
                .zip(a_r.iter())
                .map(|(al, ar)| &x_inv * al + &x * ar)
                .collect();

            // b' = x * b_L + x^{-1} * b_R
            let b_new: Vec<FieldElement<F>> = b_l
                .iter()
                .zip(b_r.iter())
                .map(|(bl, br)| &x * bl + &x_inv * br)
                .collect();

            // G' = x * G_L + x^{-1} * G_R
            let g_new: Vec<E::PointRepresentation> = g_l
                .iter()
                .zip(g_r.iter())
                .map(|(gl, gr)| {
                    gl.operate_with_self(x.representative())
                        .operate_with(&gr.operate_with_self(x_inv.representative()))
                })
                .collect();

            a_vec = a_new;
            b_vec = b_new;
            g_vec = g_new;
        }

        Ok(IPAProof {
            l_vec,
            r_vec,
            a_final: a_vec[0].clone(),
            b_final: b_vec[0].clone(),
        })
    }
}

/// Inner Product Argument verifier
///
/// Verifies IPA proofs created by [`IPAProver`].
#[derive(Clone)]
pub struct IPAVerifier<E: IsEllipticCurve, F: IsPrimeField> {
    params: PedersenParams<E>,
    _marker: PhantomData<F>,
}

impl<E: IsEllipticCurve, F: IsPrimeField> Debug for IPAVerifier<E, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IPAVerifier")
            .field("max_size", &self.params.max_size())
            .finish()
    }
}

impl<E: IsEllipticCurve, F: IsPrimeField> IPAVerifier<E, F>
where
    FieldElement<E::BaseField>: ByteConversion,
    E::PointRepresentation: AsBytes,
{
    /// Create a new IPA verifier with the given parameters
    pub fn new(params: PedersenParams<E>) -> Self {
        Self {
            params,
            _marker: PhantomData,
        }
    }

    /// Verify an IPA proof.
    ///
    /// Checks that the proof demonstrates knowledge of vectors `a`, `b` with:
    /// - Commitment `P = <a, G>`
    /// - Inner product `<a, b> = z`
    ///
    /// The verification uses the folding relation to check that after applying
    /// all the challenges, the final commitment matches `a_final * G_final + z_final * U`
    /// where z_final = a_final * b_final.
    ///
    /// # Arguments
    ///
    /// * `commitment` - The commitment to vector `a`: P = <a, G>
    /// * `b` - The public vector (e.g., powers of evaluation point)
    /// * `claimed_ip` - The claimed inner product value `<a, b>`
    /// * `proof` - The IPA proof to verify
    /// * `transcript` - Fiat-Shamir transcript (must match prover's)
    ///
    /// # Returns
    ///
    /// `true` if the proof is valid, `false` otherwise.
    pub fn verify<const N: usize, T>(
        &self,
        commitment: &E::PointRepresentation,
        b: &[FieldElement<F>],
        claimed_ip: &FieldElement<F>,
        proof: &IPAProof<E, F>,
        transcript: &mut T,
    ) -> Result<bool, IPAError>
    where
        F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>,
        T: IsTranscript<F>,
    {
        let n = b.len();
        if !n.is_power_of_two() {
            return Err(IPAError::NotPowerOfTwo { length: n });
        }

        let num_rounds = n.ilog2() as usize;
        if proof.l_vec.len() != num_rounds || proof.r_vec.len() != num_rounds {
            return Err(IPAError::InvalidProof);
        }

        // Reconstruct challenges from transcript
        let mut challenges = Vec::with_capacity(num_rounds);
        for i in 0..num_rounds {
            transcript.append_bytes(&proof.l_vec[i].as_bytes());
            transcript.append_bytes(&proof.r_vec[i].as_bytes());
            let x: FieldElement<F> = transcript.sample_field_element();
            challenges.push(x);
        }

        // Compute the folded commitment using the recurrence relation
        // C' = C + z*U + sum_{i}(x_i^{-2} * L_i + x_i^{2} * R_i)
        //
        // Starting with C + z*U (where z is the claimed inner product),
        // after each round the commitment is updated by adding x^{-2} * L + x^2 * R
        let mut p_prime = commitment.clone();
        p_prime =
            p_prime.operate_with(&self.params.u.operate_with_self(claimed_ip.representative()));

        for (i, x) in challenges.iter().enumerate().take(num_rounds) {
            let x_sq = x * x;
            let x_inv = x.inv().map_err(|_| IPAError::InversionError)?;
            let x_inv_sq = &x_inv * &x_inv;

            // C' = C + x^{-2} * L + x^2 * R
            let l_term = proof.l_vec[i].operate_with_self(x_inv_sq.representative());
            let r_term = proof.r_vec[i].operate_with_self(x_sq.representative());

            p_prime = p_prime.operate_with(&l_term).operate_with(&r_term);
        }

        // Compute the folded generator G'
        // The scalar for G_i depends on the binary representation of i and the challenges
        let g_prime = self.compute_folded_generator::<N>(&challenges, n)?;

        // Compute the folded b' from the public b values and challenges
        let b_prime = self.compute_folded_b(&challenges, b)?;

        // Check that b_prime matches proof.b_final
        // This ensures the prover used the correct b vector
        if b_prime != proof.b_final {
            return Ok(false);
        }

        // Verify the final equation: P' == a_final * G' + (a_final * b_final) * U
        // This follows from the IPA folding invariant
        let expected = g_prime.operate_with_self(proof.a_final.representative());
        let ip_term = &proof.a_final * &proof.b_final;
        let expected =
            expected.operate_with(&self.params.u.operate_with_self(ip_term.representative()));

        Ok(p_prime == expected)
    }

    /// Compute the folded generator G' given challenges.
    ///
    /// The folded generator is a linear combination of the original generators
    /// with coefficients determined by the challenge values.
    ///
    /// After k rounds of folding with challenges x_1, ..., x_k, the final generator is:
    /// G' = sum_{i=0}^{n-1} s_i * G_i
    ///
    /// This follows from the folding rule: G' = x * G_L + x^{-1} * G_R at each step.
    /// - Left half elements (bit = 0) get multiplied by x
    /// - Right half elements (bit = 1) get multiplied by x^{-1}
    fn compute_folded_generator<const N: usize>(
        &self,
        challenges: &[FieldElement<F>],
        n: usize,
    ) -> Result<E::PointRepresentation, IPAError>
    where
        F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>,
    {
        let k = challenges.len();
        let mut scalars = Vec::with_capacity(n);

        for i in 0..n {
            let mut scalar = FieldElement::<F>::one();
            for j in 0..k {
                // Check bit j of i (from LSB)
                // bit j tells us which half element i was in during round (k-1-j)
                let x = &challenges[k - 1 - j];
                let bit = (i >> j) & 1;
                if bit == 0 {
                    // Left half: multiply by x
                    scalar = &scalar * x;
                } else {
                    // Right half: multiply by x^{-1}
                    let x_inv = x.inv().map_err(|_| IPAError::InversionError)?;
                    scalar = &scalar * &x_inv;
                }
            }
            scalars.push(scalar);
        }

        let scalar_reps: Vec<UnsignedInteger<N>> =
            scalars.iter().map(|s| s.representative()).collect();
        msm(&scalar_reps, &self.params.g_vec[..n]).map_err(|_| IPAError::MsmError)
    }

    /// Compute the folded b' from the original b values and challenges.
    fn compute_folded_b(
        &self,
        challenges: &[FieldElement<F>],
        b: &[FieldElement<F>],
    ) -> Result<FieldElement<F>, IPAError> {
        // Fold b using the same logic as the prover
        let mut b_vec = b.to_vec();

        for x in challenges.iter() {
            let x_inv = x.inv().map_err(|_| IPAError::InversionError)?;
            let half = b_vec.len() / 2;
            let (b_l, b_r) = b_vec.split_at(half);

            // b' = x * b_L + x^{-1} * b_R (matching prover's folding)
            let b_new: Vec<FieldElement<F>> = b_l
                .iter()
                .zip(b_r.iter())
                .map(|(bl, br)| x * bl + &x_inv * br)
                .collect();

            b_vec = b_new;
        }

        Ok(b_vec[0].clone())
    }
}

/// Compute the inner product of two vectors.
///
/// Returns `sum_{i}(a_i * b_i)`.
pub fn inner_product<F: IsPrimeField>(
    a: &[FieldElement<F>],
    b: &[FieldElement<F>],
) -> FieldElement<F> {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have equal length");
    a.iter()
        .zip(b.iter())
        .fold(FieldElement::zero(), |acc, (ai, bi)| acc + ai * bi)
}

/// Compute powers of a field element: `[1, x, x^2, ..., x^{n-1}]`
pub fn compute_powers<F: IsPrimeField>(x: &FieldElement<F>, n: usize) -> Vec<FieldElement<F>> {
    let mut powers = Vec::with_capacity(n);
    let mut current = FieldElement::one();
    for _ in 0..n {
        powers.push(current.clone());
        current = &current * x;
    }
    powers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::{
        elliptic_curve::short_weierstrass::curves::pallas::curve::PallasCurve,
        field::{element::FieldElement, fields::vesta_field::Vesta255PrimeField},
    };

    type FE = FieldElement<Vesta255PrimeField>;
    type Transcript = DefaultTranscript<Vesta255PrimeField>;

    #[test]
    fn test_inner_product() {
        let a = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let b = vec![FE::from(5), FE::from(6), FE::from(7), FE::from(8)];

        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let ip = inner_product(&a, &b);
        assert_eq!(ip, FE::from(70));
    }

    #[test]
    fn test_compute_powers() {
        let x = FE::from(3);
        let powers = compute_powers(&x, 4);

        assert_eq!(powers[0], FE::from(1)); // 3^0
        assert_eq!(powers[1], FE::from(3)); // 3^1
        assert_eq!(powers[2], FE::from(9)); // 3^2
        assert_eq!(powers[3], FE::from(27)); // 3^3
    }

    #[test]
    fn test_ipa_manual_verification() {
        // This test manually verifies the IPA folding invariants
        use lambdaworks_math::cyclic_group::IsGroup;
        use lambdaworks_math::msm::pippenger::msm;

        let n = 2; // Simplest case: just one round
        let params = PedersenParams::<PallasCurve>::new(n);

        let a: Vec<FE> = vec![FE::from(1), FE::from(2)];
        let b: Vec<FE> = vec![FE::from(3), FE::from(4)];

        // <a, b> = 1*3 + 2*4 = 3 + 8 = 11
        let z = inner_product(&a, &b);
        assert_eq!(z, FE::from(11));

        // Initial commitment: C = <a, G> + z*U
        let a_scalars: Vec<_> = a.iter().map(|x| x.representative()).collect();
        let c_base = msm(&a_scalars, &params.g_vec[..n]).expect("msm should work");
        let c = c_base.operate_with(&params.u.operate_with_self(z.representative()));

        // Split vectors
        let (a_l, a_r) = (&a[0..1], &a[1..2]);
        let (b_l, b_r) = (&b[0..1], &b[1..2]);
        let (g_l, g_r) = (&params.g_vec[0..1], &params.g_vec[1..2]);

        // Compute L and R
        // L = <a_L, G_R> + <a_L, b_R> * U
        let c_l = inner_product(a_l, b_r); // 1*4 = 4
                                           // R = <a_R, G_L> + <a_R, b_L> * U
        let c_r = inner_product(a_r, b_l); // 2*3 = 6
        assert_eq!(c_l, FE::from(4));
        assert_eq!(c_r, FE::from(6));

        // L = <a_L, G_R> + c_L * U
        let a_l_scalars: Vec<_> = a_l.iter().map(|x| x.representative()).collect();
        let l_base = msm(&a_l_scalars, g_r).expect("msm should work");
        let l = l_base.operate_with(&params.u.operate_with_self(c_l.representative()));

        // R = <a_R, G_L> + c_R * U
        let a_r_scalars: Vec<_> = a_r.iter().map(|x| x.representative()).collect();
        let r_base = msm(&a_r_scalars, g_l).expect("msm should work");
        let r = r_base.operate_with(&params.u.operate_with_self(c_r.representative()));

        // Use a fixed challenge for testing
        let x = FE::from(5);
        let x_inv = x.inv().expect("should invert");
        let x_sq = &x * &x;
        let x_inv_sq = &x_inv * &x_inv;

        // Compute C' = C + x^{-2} * L + x^{2} * R (correct HALO/Bulletproofs equation)
        let c_prime = c
            .operate_with(&l.operate_with_self(x_inv_sq.representative()))
            .operate_with(&r.operate_with_self(x_sq.representative()));

        // Fold vectors according to HALO/Bulletproofs:
        // a' = x^{-1} * a_L + x * a_R
        let a_prime = vec![&x_inv * &a_l[0] + &x * &a_r[0]]; // (1/5)*1 + 5*2 = 1/5 + 10

        // b' = x * b_L + x^{-1} * b_R
        let b_prime = vec![&x * &b_l[0] + &x_inv * &b_r[0]]; // 5*3 + (1/5)*4 = 15 + 4/5

        // G' = x * G_L + x^{-1} * G_R
        let g_prime = vec![g_l[0]
            .operate_with_self(x.representative())
            .operate_with(&g_r[0].operate_with_self(x_inv.representative()))];

        // Verify: C' should equal a'[0] * G'[0] + (a'[0] * b'[0]) * U
        let expected = g_prime[0].operate_with_self(a_prime[0].representative());
        let ip_term = &a_prime[0] * &b_prime[0];
        let expected = expected.operate_with(&params.u.operate_with_self(ip_term.representative()));

        assert_eq!(c_prime, expected, "IPA folding invariant should hold");
    }

    #[test]
    fn test_ipa_prove_verify_small() {
        let n = 4; // Must be power of 2
        let params = PedersenParams::<PallasCurve>::new(n);

        let prover = IPAProver::<PallasCurve, Vesta255PrimeField>::new(params.clone());
        let verifier = IPAVerifier::<PallasCurve, Vesta255PrimeField>::new(params.clone());

        // Create test vectors
        let a: Vec<FE> = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let b: Vec<FE> = vec![FE::from(5), FE::from(6), FE::from(7), FE::from(8)];

        // Compute expected inner product
        let claimed_ip = inner_product(&a, &b);
        assert_eq!(claimed_ip, FE::from(70));

        // Create commitment: P = <a, G>
        let commitment = params
            .commit_without_blinding(&a)
            .expect("commitment should succeed");

        // Create proof
        let mut prover_transcript = Transcript::default();
        let proof = prover
            .prove(&a, &b, &mut prover_transcript)
            .expect("proof generation should succeed");

        // Verify proof
        let mut verifier_transcript = Transcript::default();
        let result = verifier
            .verify(
                &commitment,
                &b,
                &claimed_ip,
                &proof,
                &mut verifier_transcript,
            )
            .expect("verification should not fail");

        assert!(result, "Valid proof should verify");
    }

    #[test]
    fn test_ipa_proof_structure() {
        let n = 8;
        let params = PedersenParams::<PallasCurve>::new(n);
        let prover = IPAProver::<PallasCurve, Vesta255PrimeField>::new(params);

        let a: Vec<FE> = (1..=8).map(FE::from).collect();
        let b: Vec<FE> = (1..=8).map(FE::from).collect();

        let mut transcript = Transcript::default();
        let proof = prover
            .prove(&a, &b, &mut transcript)
            .expect("proof should succeed");

        // For n=8, we need log2(8) = 3 rounds
        assert_eq!(proof.num_rounds(), 3);
        assert_eq!(proof.l_vec.len(), 3);
        assert_eq!(proof.r_vec.len(), 3);
        assert_eq!(proof.original_size(), 8);
    }

    #[test]
    fn test_ipa_invalid_wrong_ip() {
        let n = 4;
        let params = PedersenParams::<PallasCurve>::new(n);

        let prover = IPAProver::<PallasCurve, Vesta255PrimeField>::new(params.clone());
        let verifier = IPAVerifier::<PallasCurve, Vesta255PrimeField>::new(params.clone());

        let a: Vec<FE> = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let b: Vec<FE> = vec![FE::from(5), FE::from(6), FE::from(7), FE::from(8)];

        let commitment = params
            .commit_without_blinding(&a)
            .expect("commitment should succeed");

        let mut prover_transcript = Transcript::default();
        let proof = prover
            .prove(&a, &b, &mut prover_transcript)
            .expect("proof generation should succeed");

        // Try to verify with wrong inner product claim
        let wrong_ip = FE::from(999);
        let mut verifier_transcript = Transcript::default();
        let result = verifier
            .verify(&commitment, &b, &wrong_ip, &proof, &mut verifier_transcript)
            .expect("verification should not error");

        assert!(!result, "Proof with wrong IP claim should not verify");
    }

    #[test]
    fn test_ipa_length_mismatch() {
        let params = PedersenParams::<PallasCurve>::new(4);
        let prover = IPAProver::<PallasCurve, Vesta255PrimeField>::new(params);

        let a: Vec<FE> = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let b: Vec<FE> = vec![FE::from(5), FE::from(6)]; // Different length!

        let mut transcript = Transcript::default();
        let result = prover.prove(&a, &b, &mut transcript);

        assert!(matches!(result, Err(IPAError::LengthMismatch { .. })));
    }

    #[test]
    fn test_ipa_not_power_of_two() {
        let params = PedersenParams::<PallasCurve>::new(8);
        let prover = IPAProver::<PallasCurve, Vesta255PrimeField>::new(params);

        let a: Vec<FE> = vec![FE::from(1), FE::from(2), FE::from(3)]; // 3 is not power of 2
        let b: Vec<FE> = vec![FE::from(5), FE::from(6), FE::from(7)];

        let mut transcript = Transcript::default();
        let result = prover.prove(&a, &b, &mut transcript);

        assert!(matches!(result, Err(IPAError::NotPowerOfTwo { .. })));
    }

    #[test]
    fn test_ipa_empty_vectors() {
        let params = PedersenParams::<PallasCurve>::new(4);
        let prover = IPAProver::<PallasCurve, Vesta255PrimeField>::new(params);

        let a: Vec<FE> = vec![];
        let b: Vec<FE> = vec![];

        let mut transcript = Transcript::default();
        let result = prover.prove(&a, &b, &mut transcript);

        assert!(matches!(result, Err(IPAError::EmptyVectors)));
    }
}
