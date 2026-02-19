use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::errors::DeserializationError;
use lambdaworks_math::fft::errors::FFTError;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::traits::{
    deserialize_field_element_with_length, deserialize_with_length, AsBytes, Deserializable,
    IsRandomFieldElementGenerator,
};
use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::setup::{
    new_strong_fiat_shamir_transcript, CommonPreprocessedInput, VerificationKey, Witness,
};
use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_math::{
    field::element::FieldElement,
    polynomial::{self, Polynomial},
};
use lambdaworks_math::{
    field::traits::{HasDefaultTranscript, IsField},
    traits::ByteConversion,
};

/// Errors that can occur during PLONK proving or verification.
#[derive(Debug)]
pub enum ProverError {
    /// Division by zero in field operations
    DivisionByZero,
    /// Error during FFT operation
    FFTError(String),
    /// Failed to get primitive root of unity for the given order
    PrimitiveRootNotFound(u64),
    /// Batch inversion failed (likely due to zero element)
    BatchInversionFailed,
    /// Setup error
    SetupError(String),
    /// Commitment error
    CommitmentError(String),
}

impl std::fmt::Display for ProverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProverError::DivisionByZero => write!(f, "Division by zero"),
            ProverError::FFTError(msg) => write!(f, "FFT error: {}", msg),
            ProverError::PrimitiveRootNotFound(order) => {
                write!(f, "Primitive root not found for order {}", order)
            }
            ProverError::BatchInversionFailed => write!(f, "Batch inversion failed"),
            ProverError::SetupError(msg) => write!(f, "Setup error: {}", msg),
            ProverError::CommitmentError(msg) => write!(f, "Commitment error: {}", msg),
        }
    }
}

impl std::error::Error for ProverError {}

impl From<FFTError> for ProverError {
    fn from(err: FFTError) -> Self {
        ProverError::FFTError(format!("{:?}", err))
    }
}

/// PLONK proof structure.
///
/// # Challenge Schedule (Fiat-Shamir)
/// - Round 2: β, γ (permutation challenges)
/// - Round 3: α (gate/permutation combination)
/// - Round 4: ζ (evaluation point)
/// - Round 5: υ (batching challenge)
///
/// # Key Polynomials
/// - `Z_H`: Vanishing polynomial for domain H
/// - `z`: Permutation polynomial encoding copy constraints
/// - `p`: Combined constraint polynomial (gates + permutation)
/// - `t = p / Z_H`: Quotient polynomial
/// - `a, b, c`: Wire assignment polynomials
/// - `S_σ1, S_σ2, S_σ3`: Copy permutation polynomials
///
/// # Quotient Polynomial Split (gnark-compatible)
///
/// The quotient polynomial `t(X)` has degree approximately 3n and is split into
/// three parts for efficient commitment:
///
/// ```text
/// t(X) = t_lo(X) + X^(n+2) · t_mid(X) + X^(2n+4) · t_hi(X)
/// ```
///
/// The exponents `n+2` and `2n+4` (instead of `n` and `2n`) account for the
/// blinding factors added to ensure zero-knowledge. Each part has degree at
/// most `n+1`, allowing commitment with an SRS of size `n+3`.
///
/// This approach follows gnark's implementation, which differs slightly from
/// the original PLONK paper's `n` and `2n` exponents.
///
/// # Linearization
/// The polynomial `p` is linearized for efficient verification:
/// `linearized_p = p_non_constant + p_constant`
/// where `p_non_constant` contains terms with polynomial factors (e.g., `b(ζ)·Q_R(X)`)
/// and `p_constant` contains the rest (e.g., `PI(ζ)`).
pub struct Proof<F: IsField, CS: IsCommitmentScheme<F>> {
    // Round 1.
    /// Commitment to the wire polynomial `a(x)`
    pub a_1: CS::Commitment,
    /// Commitment to the wire polynomial `b(x)`
    pub b_1: CS::Commitment,
    /// Commitment to the wire polynomial `c(x)`
    pub c_1: CS::Commitment,

    // Round 2.
    /// Commitment to the copy constraints polynomial `z(x)`
    pub z_1: CS::Commitment,

    // Round 3.
    /// Commitment to the low part of the quotient polynomial t(X)
    pub t_lo_1: CS::Commitment,
    /// Commitment to the middle part of the quotient polynomial t(X)
    pub t_mid_1: CS::Commitment,
    /// Commitment to the high part of the quotient polynomial t(X)
    pub t_hi_1: CS::Commitment,

    // Round 4.
    /// Value of `a(ζ)`.
    pub a_zeta: FieldElement<F>,
    /// Value of `b(ζ)`.
    pub b_zeta: FieldElement<F>,
    /// Value of `c(ζ)`.
    pub c_zeta: FieldElement<F>,
    /// Value of `S_σ1(ζ)`.
    pub s1_zeta: FieldElement<F>,
    /// Value of `S_σ2(ζ)`.
    pub s2_zeta: FieldElement<F>,
    /// Value of `z(ζω)`.
    pub z_zeta_omega: FieldElement<F>,

    // Round 5
    /// Value of `p_non_constant(ζ)`.
    pub p_non_constant_zeta: FieldElement<F>,
    ///  Value of `t(ζ)`.
    pub t_zeta: FieldElement<F>,
    /// Batch opening proof for all the evaluations at ζ
    pub w_zeta_1: CS::Commitment,
    /// Single opening proof for `z(ζω)`.
    pub w_zeta_omega_1: CS::Commitment,
}

impl<F, CS> AsBytes for Proof<F, CS>
where
    F: IsField,
    CS: IsCommitmentScheme<F>,
    FieldElement<F>: ByteConversion,
    CS::Commitment: AsBytes,
{
    fn as_bytes(&self) -> Vec<u8> {
        let mut serialized_proof: Vec<u8> = Vec::new();

        // Serialize field elements with length prefix
        for element in [
            &self.a_zeta,
            &self.b_zeta,
            &self.c_zeta,
            &self.s1_zeta,
            &self.s2_zeta,
            &self.z_zeta_omega,
            &self.p_non_constant_zeta,
            &self.t_zeta,
        ] {
            let serialized_element = element.to_bytes_be();
            serialized_proof.extend_from_slice(&(serialized_element.len() as u32).to_be_bytes());
            serialized_proof.extend_from_slice(&serialized_element);
        }

        // Serialize commitments using shared helper
        for commitment in [
            &self.a_1,
            &self.b_1,
            &self.c_1,
            &self.z_1,
            &self.t_lo_1,
            &self.t_mid_1,
            &self.t_hi_1,
            &self.w_zeta_1,
            &self.w_zeta_omega_1,
        ] {
            serialized_proof.extend(lambdaworks_math::traits::serialize_with_length(commitment));
        }

        serialized_proof
    }
}

impl<F, CS> Deserializable for Proof<F, CS>
where
    F: IsField,
    CS: IsCommitmentScheme<F>,
    FieldElement<F>: ByteConversion,
    CS::Commitment: Deserializable,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        let (offset, a_zeta) = deserialize_field_element_with_length(bytes, 0)?;
        let (offset, b_zeta) = deserialize_field_element_with_length(bytes, offset)?;
        let (offset, c_zeta) = deserialize_field_element_with_length(bytes, offset)?;
        let (offset, s1_zeta) = deserialize_field_element_with_length(bytes, offset)?;
        let (offset, s2_zeta) = deserialize_field_element_with_length(bytes, offset)?;
        let (offset, z_zeta_omega) = deserialize_field_element_with_length(bytes, offset)?;
        let (offset, p_non_constant_zeta) = deserialize_field_element_with_length(bytes, offset)?;
        let (offset, t_zeta) = deserialize_field_element_with_length(bytes, offset)?;

        let (offset, a_1) = deserialize_with_length(bytes, offset)?;
        let (offset, b_1) = deserialize_with_length(bytes, offset)?;
        let (offset, c_1) = deserialize_with_length(bytes, offset)?;
        let (offset, z_1) = deserialize_with_length(bytes, offset)?;
        let (offset, t_lo_1) = deserialize_with_length(bytes, offset)?;
        let (offset, t_mid_1) = deserialize_with_length(bytes, offset)?;
        let (offset, t_hi_1) = deserialize_with_length(bytes, offset)?;
        let (offset, w_zeta_1) = deserialize_with_length(bytes, offset)?;
        let (_, w_zeta_omega_1) = deserialize_with_length(bytes, offset)?;

        Ok(Proof {
            a_1,
            b_1,
            c_1,
            z_1,
            t_lo_1,
            t_mid_1,
            t_hi_1,
            a_zeta,
            b_zeta,
            c_zeta,
            s1_zeta,
            s2_zeta,
            z_zeta_omega,
            p_non_constant_zeta,
            t_zeta,
            w_zeta_1,
            w_zeta_omega_1,
        })
    }
}

pub struct Prover<F: IsField, CS: IsCommitmentScheme<F>, R: IsRandomFieldElementGenerator<F>> {
    commitment_scheme: CS,
    random_generator: R,
    phantom: PhantomData<F>,
}

struct Round1Result<F: IsField, Hiding> {
    a_1: Hiding,
    b_1: Hiding,
    c_1: Hiding,
    p_a: Polynomial<FieldElement<F>>,
    p_b: Polynomial<FieldElement<F>>,
    p_c: Polynomial<FieldElement<F>>,
}

struct Round2Result<F: IsField, Hiding> {
    z_1: Hiding,
    p_z: Polynomial<FieldElement<F>>,
    beta: FieldElement<F>,
    gamma: FieldElement<F>,
}

struct Round3Result<F: IsField, Hiding> {
    t_lo_1: Hiding,
    t_mid_1: Hiding,
    t_hi_1: Hiding,
    p_t_lo: Polynomial<FieldElement<F>>,
    p_t_mid: Polynomial<FieldElement<F>>,
    p_t_hi: Polynomial<FieldElement<F>>,
    alpha: FieldElement<F>,
}

struct Round4Result<F: IsField> {
    a_zeta: FieldElement<F>,
    b_zeta: FieldElement<F>,
    c_zeta: FieldElement<F>,
    s1_zeta: FieldElement<F>,
    s2_zeta: FieldElement<F>,
    z_zeta_omega: FieldElement<F>,
    zeta: FieldElement<F>,
}

struct Round5Result<F: IsField, Hiding> {
    w_zeta_1: Hiding,
    w_zeta_omega_1: Hiding,
    p_non_constant_zeta: FieldElement<F>,
    t_zeta: FieldElement<F>,
}

impl<F, CS, R> Prover<F, CS, R>
where
    F: IsField + IsFFTField + HasDefaultTranscript + Sync,
    CS: IsCommitmentScheme<F> + Sync,
    FieldElement<F>: ByteConversion,
    CS::Commitment: AsBytes + Send + Sync,
    R: IsRandomFieldElementGenerator<F> + Sync,
{
    pub fn new(commitment_scheme: CS, random_generator: R) -> Self {
        Self {
            commitment_scheme,
            random_generator,
            phantom: PhantomData,
        }
    }

    fn blind_polynomial(
        &self,
        target: &Polynomial<FieldElement<F>>,
        blinder: &Polynomial<FieldElement<F>>,
        n: u64,
    ) -> Polynomial<FieldElement<F>>
    where
        F: IsField,
        R: IsRandomFieldElementGenerator<F>,
    {
        let bs: Vec<FieldElement<F>> = (0..n).map(|_| self.random_generator.generate()).collect();
        let random_part = Polynomial::new(&bs);
        target + blinder * random_part
    }

    fn round_1(
        &self,
        witness: &Witness<F>,
        common_preprocessed_input: &CommonPreprocessedInput<F>,
    ) -> Round1Result<F, CS::Commitment> {
        let p_a = Polynomial::interpolate_fft::<F>(&witness.a)
            .expect("xs and ys have equal length and xs are unique");
        let p_b = Polynomial::interpolate_fft::<F>(&witness.b)
            .expect("xs and ys have equal length and xs are unique");
        let p_c = Polynomial::interpolate_fft::<F>(&witness.c)
            .expect("xs and ys have equal length and xs are unique");

        let z_h = Polynomial::new_monomial(FieldElement::one(), common_preprocessed_input.n)
            - FieldElement::<F>::one();
        let p_a = self.blind_polynomial(&p_a, &z_h, 2);
        let p_b = self.blind_polynomial(&p_b, &z_h, 2);
        let p_c = self.blind_polynomial(&p_c, &z_h, 2);

        #[cfg(feature = "parallel")]
        let (a_1, b_1, c_1) = {
            let (a_1, (b_1, c_1)) = rayon::join(
                || self.commitment_scheme.commit(&p_a),
                || {
                    rayon::join(
                        || self.commitment_scheme.commit(&p_b),
                        || self.commitment_scheme.commit(&p_c),
                    )
                },
            );
            (a_1, b_1, c_1)
        };
        #[cfg(not(feature = "parallel"))]
        let (a_1, b_1, c_1) = (
            self.commitment_scheme.commit(&p_a),
            self.commitment_scheme.commit(&p_b),
            self.commitment_scheme.commit(&p_c),
        );

        Round1Result {
            a_1,
            b_1,
            c_1,
            p_a,
            p_b,
            p_c,
        }
    }

    fn round_2(
        &self,
        witness: &Witness<F>,
        common_preprocessed_input: &CommonPreprocessedInput<F>,
        beta: FieldElement<F>,
        gamma: FieldElement<F>,
    ) -> Result<Round2Result<F, CS::Commitment>, ProverError> {
        let cpi = common_preprocessed_input;
        let (s1, s2, s3) = (&cpi.s1_lagrange, &cpi.s2_lagrange, &cpi.s3_lagrange);

        let k2 = &cpi.k1 * &cpi.k1;

        let lp = |w: &FieldElement<F>, eta: &FieldElement<F>| w + &beta * eta + &gamma;

        // Compute all numerators and denominators first.
        // We need n-1 factors to compute n coefficients: z[0]=1, z[i+1]=z[i]*factor[i] for i in 0..n-1.
        // This matches the original loop range `0..cpi.n - 1`.
        let n_minus_1 = cpi.n - 1;
        let mut numerators = Vec::with_capacity(n_minus_1);
        let mut denominators = Vec::with_capacity(n_minus_1);

        for i in 0..n_minus_1 {
            let (a_i, b_i, c_i) = (&witness.a[i], &witness.b[i], &witness.c[i]);
            let num = lp(a_i, &cpi.domain[i])
                * lp(b_i, &(&cpi.domain[i] * &cpi.k1))
                * lp(c_i, &(&cpi.domain[i] * &k2));
            let den = lp(a_i, &s1[i]) * lp(b_i, &s2[i]) * lp(c_i, &s3[i]);
            numerators.push(num);
            denominators.push(den);
        }

        // Batch invert all denominators at once (much faster than n-1 individual inversions)
        FieldElement::inplace_batch_inverse(&mut denominators).expect(
            "batch inversion failed in permutation polynomial: beta and gamma should prevent zeros",
        );

        // Compute coefficients using the inverted denominators
        let mut coefficients: Vec<FieldElement<F>> = Vec::with_capacity(cpi.n);
        coefficients.push(FieldElement::one());

        for i in 0..n_minus_1 {
            let factor = &numerators[i] * &denominators[i];
            let new_term = coefficients.last().expect("coefficients non-empty") * &factor;
            coefficients.push(new_term);
        }

        let p_z = Polynomial::interpolate_fft::<F>(&coefficients)?;
        let z_h = Polynomial::new_monomial(FieldElement::one(), common_preprocessed_input.n)
            - FieldElement::<F>::one();
        let p_z = self.blind_polynomial(&p_z, &z_h, 3);
        let z_1 = self.commitment_scheme.commit(&p_z);
        Ok(Round2Result {
            z_1,
            p_z,
            beta,
            gamma,
        })
    }

    fn round_3(
        &self,
        common_preprocessed_input: &CommonPreprocessedInput<F>,
        public_input: &[FieldElement<F>],
        Round1Result { p_a, p_b, p_c, .. }: &Round1Result<F, CS::Commitment>,
        Round2Result {
            p_z, beta, gamma, ..
        }: &Round2Result<F, CS::Commitment>,
        alpha: FieldElement<F>,
    ) -> Result<Round3Result<F, CS::Commitment>, ProverError> {
        let cpi = common_preprocessed_input;
        let k2 = &cpi.k1 * &cpi.k1;

        let z_x_omega_coefficients: Vec<FieldElement<F>> = p_z
            .coefficients()
            .iter()
            .enumerate()
            .map(|(i, x)| x * &cpi.domain[i % cpi.n])
            .collect();
        let z_x_omega = Polynomial::new(&z_x_omega_coefficients);
        let mut e1 = vec![FieldElement::<F>::zero(); cpi.domain.len()];
        e1[0] = FieldElement::one();
        let l1 = Polynomial::interpolate_fft::<F>(&e1)?;
        let mut p_pi_y = public_input.to_vec();
        p_pi_y.append(&mut vec![FieldElement::zero(); cpi.n - public_input.len()]);
        let p_pi = Polynomial::interpolate_fft::<F>(&p_pi_y)?;

        // Compute p using FFT evaluation form for efficiency
        // Quotient polynomial degree bound: 4n covers standard PLONK gates
        // (constraint polynomial degree is ~4n, divided by zh of degree n gives ~3n)
        let degree = 4 * cpi.n;
        let offset = &cpi.k1;
        // All 15 polynomials need coset FFT evaluation at the same degree/offset.
        // These evaluations are completely independent and can run in parallel.
        let polys_to_eval: Vec<&Polynomial<FieldElement<F>>> = vec![
            p_a, p_b, p_c, // 0-2: wire polynomials
            &cpi.ql, &cpi.qr, &cpi.qm, &cpi.qo, &cpi.qc, // 3-7: gate polynomials
            &p_pi,   // 8: public input
            p_z, &z_x_omega, // 9-10: permutation
            &cpi.s1, &cpi.s2, &cpi.s3, // 11-13: sigma polynomials
            &l1,     // 14: L1 polynomial
        ];

        #[cfg(feature = "parallel")]
        let evals: Vec<Vec<FieldElement<F>>> = polys_to_eval
            .par_iter()
            .map(|poly| {
                Polynomial::evaluate_offset_fft(poly, 1, Some(degree), offset)
                    .expect("FFT evaluation must be within field's two-adicity limit")
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let evals: Vec<Vec<FieldElement<F>>> = polys_to_eval
            .iter()
            .map(|poly| {
                Polynomial::evaluate_offset_fft(poly, 1, Some(degree), offset)
                    .expect("FFT evaluation must be within field's two-adicity limit")
            })
            .collect();

        let p_a_eval = &evals[0];
        let p_b_eval = &evals[1];
        let p_c_eval = &evals[2];
        let ql_eval = &evals[3];
        let qr_eval = &evals[4];
        let qm_eval = &evals[5];
        let qo_eval = &evals[6];
        let qc_eval = &evals[7];
        let p_pi_eval = &evals[8];
        let p_z_eval = &evals[9];
        let p_z_x_omega_eval = &evals[10];
        let p_s1_eval = &evals[11];
        let p_s2_eval = &evals[12];
        let p_s3_eval = &evals[13];
        let l1_eval = &evals[14];

        // p_x = X (identity polynomial), so p_x(offset * ω^i) = offset * ω^i.
        // Generate the coset directly instead of using FFT.
        let omega = F::get_primitive_root_of_unity(degree.trailing_zeros() as u64)
            .expect("primitive root exists for degree");
        let p_x_eval: Vec<_> = (0..degree)
            .scan(offset.clone(), |current, _| {
                let val = current.clone();
                *current = &*current * &omega;
                Some(val)
            })
            .collect();
        debug_assert_eq!(
            p_x_eval.len(),
            p_a_eval.len(),
            "p_x_eval length must match FFT evaluation length"
        );

        let p_constraints_eval: Vec<_> = p_a_eval
            .iter()
            .zip(p_b_eval.iter())
            .zip(p_c_eval.iter())
            .zip(ql_eval.iter())
            .zip(qr_eval.iter())
            .zip(qm_eval.iter())
            .zip(qo_eval.iter())
            .zip(qc_eval.iter())
            .zip(p_pi_eval.iter())
            .map(|((((((((a, b), c), ql), qr), qm), qo), qc), pi)| {
                a * b * qm + a * ql + b * qr + c * qo + qc + pi
            })
            .collect();

        let f_eval: Vec<_> = p_a_eval
            .iter()
            .zip(p_b_eval.iter())
            .zip(p_c_eval.iter())
            .zip(p_x_eval.iter())
            .map(|(((a, b), c), x)| {
                (a + x * beta + gamma)
                    * (b + x * beta * &cpi.k1 + gamma)
                    * (c + x * beta * &k2 + gamma)
            })
            .collect();

        let g_eval: Vec<_> = p_a_eval
            .iter()
            .zip(p_b_eval.iter())
            .zip(p_c_eval.iter())
            .zip(p_s1_eval.iter())
            .zip(p_s2_eval.iter())
            .zip(p_s3_eval.iter())
            .map(|(((((a, b), c), s1), s2), s3)| {
                (a + s1 * beta + gamma) * (b + s2 * beta + gamma) * (c + s3 * beta + gamma)
            })
            .collect();

        let p_permutation_1_eval: Vec<_> = g_eval
            .iter()
            .zip(f_eval.iter())
            .zip(p_z_eval.iter())
            .zip(p_z_x_omega_eval.iter())
            .map(|(((g, f), z), y)| g * y - f * z)
            .collect();

        let p_permutation_2_eval: Vec<_> = p_z_eval
            .iter()
            .zip(l1_eval.iter())
            .map(|(z, l)| (z - FieldElement::<F>::one()) * l)
            .collect();

        let p_eval: Vec<_> = p_permutation_2_eval
            .iter()
            .zip(p_permutation_1_eval.iter())
            .zip(p_constraints_eval.iter())
            .map(|((p2, p1), co)| (p2 * &alpha + p1) * &alpha + co)
            .collect();

        // Optimization: Z_H(x) = x^n - 1 has only 4 distinct values on a coset of size 4n.
        // On coset {offset * ω^i : i = 0..4n-1} where ω is primitive 4n-th root:
        //   Z_H(offset * ω^i) = offset^n * (ω^n)^i - 1
        // Since ω^n is a 4th root of unity, (ω^n)^i cycles through 4 values.
        //
        // SAFETY: This optimization assumes degree == 4 * n. If degree changes (see TODO above),
        // this optimization must be revisited.
        debug_assert_eq!(
            degree,
            4 * cpi.n,
            "Z_H optimization requires degree == 4n; if degree formula changes, update this code"
        );
        let omega_4n = F::get_primitive_root_of_unity(degree.trailing_zeros() as u64)
            .expect("primitive root exists for degree");
        let omega_n = omega_4n.pow(cpi.n as u64); // ω^n where ω is 4n-th root; this is a 4th root of unity
        let offset_to_n = offset.pow(cpi.n as u64);

        // Compute the 4 distinct Z_H values and their inverses
        // Use multiplication chain for small powers (faster than pow)
        let omega_n_sq = &omega_n * &omega_n;
        let omega_n_cubed = &omega_n_sq * &omega_n;
        let mut zh_base = [
            &offset_to_n - FieldElement::<F>::one(), // i ≡ 0 (mod 4)
            &offset_to_n * &omega_n - FieldElement::<F>::one(), // i ≡ 1 (mod 4)
            &offset_to_n * &omega_n_sq - FieldElement::<F>::one(), // i ≡ 2 (mod 4)
            &offset_to_n * &omega_n_cubed - FieldElement::<F>::one(), // i ≡ 3 (mod 4)
        ];
        FieldElement::inplace_batch_inverse(&mut zh_base)
            .expect("Z_H evaluations are non-zero on coset offset from roots of unity");

        // Build full evaluation vector by cycling through the 4 values
        let zh_eval: Vec<_> = (0..degree).map(|i| zh_base[i % 4].clone()).collect();
        let c: Vec<_> = p_eval
            .iter()
            .zip(zh_eval.iter())
            .map(|(a, b)| a * b)
            .collect();
        let mut t = Polynomial::interpolate_offset_fft(&c, offset)
            .expect("FFT interpolation of quotient polynomial must succeed");

        // Split quotient polynomial into 3 parts following gnark's approach:
        // t(X) = t_lo(X) + X^(n+2) * t_mid(X) + X^(2n+4) * t_hi(X)
        polynomial::pad_with_zero_coefficients_to_length(&mut t, 3 * (&cpi.n + 2));
        let p_t_lo = Polynomial::new(&t.coefficients[..&cpi.n + 2]);
        let p_t_mid = Polynomial::new(&t.coefficients[&cpi.n + 2..2 * (&cpi.n + 2)]);
        let p_t_hi = Polynomial::new(&t.coefficients[2 * (&cpi.n + 2)..3 * (&cpi.n + 2)]);

        let b_0 = self.random_generator.generate();
        let b_1 = self.random_generator.generate();

        let p_t_lo = &p_t_lo + &b_0 * Polynomial::new_monomial(FieldElement::one(), cpi.n + 2);
        let p_t_mid =
            &p_t_mid - b_0 + &b_1 * Polynomial::new_monomial(FieldElement::one(), cpi.n + 2);
        let p_t_hi = &p_t_hi - b_1;

        #[cfg(feature = "parallel")]
        let (t_lo_1, t_mid_1, t_hi_1) = {
            let (t_lo_1, (t_mid_1, t_hi_1)) = rayon::join(
                || self.commitment_scheme.commit(&p_t_lo),
                || {
                    rayon::join(
                        || self.commitment_scheme.commit(&p_t_mid),
                        || self.commitment_scheme.commit(&p_t_hi),
                    )
                },
            );
            (t_lo_1, t_mid_1, t_hi_1)
        };
        #[cfg(not(feature = "parallel"))]
        let (t_lo_1, t_mid_1, t_hi_1) = (
            self.commitment_scheme.commit(&p_t_lo),
            self.commitment_scheme.commit(&p_t_mid),
            self.commitment_scheme.commit(&p_t_hi),
        );

        Ok(Round3Result {
            t_lo_1,
            t_mid_1,
            t_hi_1,
            p_t_lo,
            p_t_mid,
            p_t_hi,
            alpha,
        })
    }

    fn round_4(
        &self,
        CommonPreprocessedInput { s1, s2, omega, .. }: &CommonPreprocessedInput<F>,
        Round1Result { p_a, p_b, p_c, .. }: &Round1Result<F, CS::Commitment>,
        Round2Result { p_z, .. }: &Round2Result<F, CS::Commitment>,
        zeta: FieldElement<F>,
    ) -> Round4Result<F> {
        let a_zeta = p_a.evaluate(&zeta);
        let b_zeta = p_b.evaluate(&zeta);
        let c_zeta = p_c.evaluate(&zeta);
        let s1_zeta = s1.evaluate(&zeta);
        let s2_zeta = s2.evaluate(&zeta);
        let z_zeta_omega = p_z.evaluate(&(&zeta * omega));
        Round4Result {
            a_zeta,
            b_zeta,
            c_zeta,
            s1_zeta,
            s2_zeta,
            z_zeta_omega,
            zeta,
        }
    }

    fn round_5(
        &self,
        common_preprocessed_input: &CommonPreprocessedInput<F>,
        round_1: &Round1Result<F, CS::Commitment>,
        round_2: &Round2Result<F, CS::Commitment>,
        round_3: &Round3Result<F, CS::Commitment>,
        round_4: &Round4Result<F>,
        upsilon: FieldElement<F>,
    ) -> Result<Round5Result<F, CS::Commitment>, ProverError> {
        let cpi = common_preprocessed_input;
        let (r1, r2, r3, r4) = (round_1, round_2, round_3, round_4);
        // Precompute variables
        let k2 = &cpi.k1 * &cpi.k1;

        // Compute zeta powers efficiently: zeta^n, zeta^(n+2), zeta^(2n+4)
        // Start with zeta^n, then derive others to avoid redundant exponentiations
        // Following gnark's approach: quotient split uses n+2 and 2n+4 exponents
        // TODO: Paper says n and 2n, but Gnark uses n+2 and 2n+4
        let zeta_n = r4.zeta.pow(cpi.n as u64);
        let zeta_sq = &r4.zeta * &r4.zeta;
        let zeta_n_plus_2 = &zeta_n * &zeta_sq; // zeta^(n+2) = zeta^n * zeta^2
        let zeta_2n_plus_4 = &zeta_n_plus_2 * &zeta_n_plus_2; // zeta^(2n+4) = (zeta^(n+2))^2

        let zeta_raised_n = Polynomial::new_monomial(zeta_n_plus_2, 0);
        let zeta_raised_2n = Polynomial::new_monomial(zeta_2n_plus_4, 0);

        // zeta is sampled outside the set of roots of unity so zeta != 1, and n != 0.
        let l1_zeta = ((&zeta_n - FieldElement::<F>::one())
            / ((&r4.zeta - FieldElement::<F>::one()) * FieldElement::<F>::from(cpi.n as u64)))
        .expect("zeta is outside roots of unity so denominator is non-zero");

        let mut p_non_constant = &cpi.qm * &r4.a_zeta * &r4.b_zeta
            + &r4.a_zeta * &cpi.ql
            + &r4.b_zeta * &cpi.qr
            + &r4.c_zeta * &cpi.qo
            + &cpi.qc;

        let r_2_1 = (&r4.a_zeta + &r2.beta * &r4.zeta + &r2.gamma)
            * (&r4.b_zeta + &r2.beta * &cpi.k1 * &r4.zeta + &r2.gamma)
            * (&r4.c_zeta + &r2.beta * &k2 * &r4.zeta + &r2.gamma)
            * &r2.p_z;
        let r_2_2 = (&r4.a_zeta + &r2.beta * &r4.s1_zeta + &r2.gamma)
            * (&r4.b_zeta + &r2.beta * &r4.s2_zeta + &r2.gamma)
            * &r2.beta
            * &r4.z_zeta_omega
            * &cpi.s3;
        let alpha_squared = &r3.alpha * &r3.alpha;
        p_non_constant += (r_2_2 - r_2_1) * &r3.alpha;

        let r_3 = &r2.p_z * l1_zeta;
        p_non_constant += r_3 * &alpha_squared;

        let partial_t = &r3.p_t_lo + zeta_raised_n * &r3.p_t_mid + zeta_raised_2n * &r3.p_t_hi;

        // TODO: Refactor to remove clones.
        let polynomials = vec![
            partial_t,
            p_non_constant,
            r1.p_a.clone(),
            r1.p_b.clone(),
            r1.p_c.clone(),
            cpi.s1.clone(),
            cpi.s2.clone(),
        ];
        let ys: Vec<FieldElement<F>> = polynomials.iter().map(|p| p.evaluate(&r4.zeta)).collect();

        #[cfg(feature = "parallel")]
        let (w_zeta_1, w_zeta_omega_1) = {
            let zeta_omega = &r4.zeta * &cpi.omega;
            rayon::join(
                || {
                    self.commitment_scheme
                        .open_batch(&r4.zeta, &ys, &polynomials, &upsilon)
                },
                || {
                    self.commitment_scheme
                        .open(&zeta_omega, &r4.z_zeta_omega, &r2.p_z)
                },
            )
        };
        #[cfg(not(feature = "parallel"))]
        let (w_zeta_1, w_zeta_omega_1) = {
            let w_zeta_1 = self
                .commitment_scheme
                .open_batch(&r4.zeta, &ys, &polynomials, &upsilon);
            let w_zeta_omega_1 =
                self.commitment_scheme
                    .open(&(&r4.zeta * &cpi.omega), &r4.z_zeta_omega, &r2.p_z);
            (w_zeta_1, w_zeta_omega_1)
        };

        Ok(Round5Result {
            w_zeta_1,
            w_zeta_omega_1,
            p_non_constant_zeta: ys[1].clone(),
            t_zeta: ys[0].clone(),
        })
    }

    /// Generates a PLONK proof for the given witness and public inputs.
    ///
    /// # Arguments
    /// * `witness` - The witness assignment (values for all variables)
    /// * `public_input` - The public input values
    /// * `common_preprocessed_input` - Preprocessed circuit data from setup
    /// * `vk` - The verification key
    ///
    /// # Returns
    /// * `Ok(Proof)` on success
    /// * `Err(ProverError)` if proving fails (e.g., FFT errors, invalid witness)
    pub fn prove(
        &self,
        witness: &Witness<F>,
        public_input: &[FieldElement<F>],
        common_preprocessed_input: &CommonPreprocessedInput<F>,
        vk: &VerificationKey<CS::Commitment>,
    ) -> Result<Proof<F, CS>, ProverError> {
        let mut transcript = new_strong_fiat_shamir_transcript::<F, CS>(vk, public_input);

        // Round 1: Commit to wire polynomials
        let round_1 = self.round_1(witness, common_preprocessed_input);
        transcript.append_bytes(&round_1.a_1.as_bytes());
        transcript.append_bytes(&round_1.b_1.as_bytes());
        transcript.append_bytes(&round_1.c_1.as_bytes());

        // Round 2: Commit to permutation polynomial
        let beta = transcript.sample_field_element();
        let gamma = transcript.sample_field_element();

        let round_2 = self.round_2(witness, common_preprocessed_input, beta, gamma)?;
        transcript.append_bytes(&round_2.z_1.as_bytes());

        // Round 3: Compute and commit to quotient polynomial
        let alpha = transcript.sample_field_element();
        let round_3 = self.round_3(
            common_preprocessed_input,
            public_input,
            &round_1,
            &round_2,
            alpha,
        )?;
        transcript.append_bytes(&round_3.t_lo_1.as_bytes());
        transcript.append_bytes(&round_3.t_mid_1.as_bytes());
        transcript.append_bytes(&round_3.t_hi_1.as_bytes());

        // Round 4: Evaluate polynomials at challenge point
        let zeta = transcript.sample_field_element();
        let round_4 = self.round_4(common_preprocessed_input, &round_1, &round_2, zeta);

        transcript.append_field_element(&round_4.a_zeta);
        transcript.append_field_element(&round_4.b_zeta);
        transcript.append_field_element(&round_4.c_zeta);
        transcript.append_field_element(&round_4.s1_zeta);
        transcript.append_field_element(&round_4.s2_zeta);
        transcript.append_field_element(&round_4.z_zeta_omega);

        // Round 5: Compute opening proofs
        let upsilon = transcript.sample_field_element();
        let round_5 = self.round_5(
            common_preprocessed_input,
            &round_1,
            &round_2,
            &round_3,
            &round_4,
            upsilon,
        )?;

        Ok(Proof {
            a_1: round_1.a_1,
            b_1: round_1.b_1,
            c_1: round_1.c_1,
            z_1: round_2.z_1,
            t_lo_1: round_3.t_lo_1,
            t_mid_1: round_3.t_mid_1,
            t_hi_1: round_3.t_hi_1,
            a_zeta: round_4.a_zeta,
            b_zeta: round_4.b_zeta,
            c_zeta: round_4.c_zeta,
            s1_zeta: round_4.s1_zeta,
            s2_zeta: round_4.s2_zeta,
            z_zeta_omega: round_4.z_zeta_omega,
            w_zeta_1: round_5.w_zeta_1,
            w_zeta_omega_1: round_5.w_zeta_omega_1,
            p_non_constant_zeta: round_5.p_non_constant_zeta,
            t_zeta: round_5.t_zeta,
        })
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::{
                curves::bls12_381::{curve::BLS12381Curve, default_types::FrElement},
                point::ShortWeierstrassJacobianPoint,
            },
            traits::IsEllipticCurve,
        },
    };

    use crate::{
        test_utils::circuit_1::{test_common_preprocessed_input_1, test_witness_1},
        test_utils::utils::{test_srs, FpElement, TestRandomFieldGenerator, KZG},
    };

    use super::*;

    fn alpha() -> FrElement {
        FrElement::from_hex_unchecked(
            "583cfb0df2ef98f2131d717bc6aadd571c5302597c135cab7c00435817bf6e50",
        )
    }

    fn beta() -> FrElement {
        FrElement::from_hex_unchecked(
            "bdda7414bdf5bf42b77cbb3af4a82f32ec7622dd6c71575bede021e6e4609d4",
        )
    }

    fn gamma() -> FrElement {
        FrElement::from_hex_unchecked(
            "58f6690d9b36e62e4a0aef27612819288df2a3ff5bf01597cf06779503f51583",
        )
    }

    fn zeta() -> FrElement {
        FrElement::from_hex_unchecked(
            "2a4040abb941ee5e2a42602a7a60d282a430a4cf099fa3bb0ba8f4da628ec59a",
        )
    }

    fn upsilon() -> FrElement {
        FrElement::from_hex_unchecked(
            "2d15959489a2a8e44693221ca7cbdcab15253d6bae9fd7fe0664cff02fe4f1cf",
        )
    }

    #[test]
    fn test_round_1() {
        let witness = test_witness_1(FrElement::from(2), FrElement::from(2));
        let common_preprocessed_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_preprocessed_input.n);
        let kzg = KZG::new(srs);
        let random_generator = TestRandomFieldGenerator {};

        let prover = Prover::new(kzg, random_generator);
        let round_1 = prover.round_1(&witness, &common_preprocessed_input);
        let a_1_expected = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"),
            FpElement::from_hex_unchecked("114d1d6855d545a8aa7d76c8cf2e21f267816aef1db507c96655b9d5caac42364e6f38ba0ecb751bad54dcd6b939c2ca"),
        ).unwrap();
        let b_1_expected = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("44ed7c3ed015c6a39c350cd06d03b48d3e1f5eaf7a256c5b6203886e6e78cd9b76623d163da4dfb0f2491e7cc06408"),
            FpElement::from_hex_unchecked("14c4464d2556fdfdc8e31068ef8d953608e511569a236c825f2ddab4fe04af03aba29e38b9b2b6221243124d235f4c67"),
        ).unwrap();
        let c_1_expected = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("7726dc031bd26122395153ca428d5e6dea0a64c1f9b3b1bb2f2508a5eb6ea0ea0363294fad3160858bc87e46d3422fd"),
            FpElement::from_hex_unchecked("8db0c15bfd77df7fe66284c3b04e6043eaba99ef6a845d4f7255fd0da95f2fb8e474df2e7f8e1a38829f7a9612a9b87"),
        ).unwrap();
        assert_eq!(round_1.a_1, a_1_expected);
        assert_eq!(round_1.b_1, b_1_expected);
        assert_eq!(round_1.c_1, c_1_expected);
    }

    #[test]
    fn test_round_2() {
        let witness = test_witness_1(FrElement::from(2), FrElement::from(2));
        let common_preprocessed_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_preprocessed_input.n);
        let kzg = KZG::new(srs);
        let random_generator = TestRandomFieldGenerator {};
        let prover = Prover::new(kzg, random_generator);

        let result_2 = prover
            .round_2(&witness, &common_preprocessed_input, beta(), gamma())
            .unwrap();
        let z_1_expected = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("3e8322968c3496cf1b5786d4d71d158a646ec90c14edf04e758038e1f88dcdfe8443fcecbb75f3074a872a380391742"),
            FpElement::from_hex_unchecked("11eac40d09796ff150004e7b858d83ddd9fe995dced0b3fbd7535d6e361729b25d488799da61fdf1d7b5022684053327"),
        ).unwrap();
        assert_eq!(result_2.z_1, z_1_expected);
    }

    #[test]
    fn test_round_3() {
        let witness = test_witness_1(FrElement::from(2), FrElement::from(2));
        let common_preprocessed_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_preprocessed_input.n);
        let kzg = KZG::new(srs);
        let public_input = vec![FieldElement::from(2_u64), FieldElement::from(4)];
        let random_generator = TestRandomFieldGenerator {};
        let prover = Prover::new(kzg, random_generator);
        let round_1 = prover.round_1(&witness, &common_preprocessed_input);
        let round_2 = prover
            .round_2(&witness, &common_preprocessed_input, beta(), gamma())
            .unwrap();
        let round_3 = prover
            .round_3(
                &common_preprocessed_input,
                &public_input,
                &round_1,
                &round_2,
                alpha(),
            )
            .unwrap();

        let t_lo_1_expected = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("9f511a769e77e87537b0749d65f467532fbf0f9dc1bcc912c333741be9d0a613f61e5fe595996964646ce30794701e5"),
            FpElement::from_hex_unchecked("89fd6bb571323912210517237d6121144fc01ba2756f47c12c9cc94fc9197313867d68530f152dc8d447f10fcf75a6c"),
        ).unwrap();
        let t_mid_1_expected = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("f96d8a93f3f5be2ab2819891f41c9f883cacea63da423e6ed1701765fcd659fc11e056a48c554f5df3a9c6603d48ca8"),
            FpElement::from_hex_unchecked("14fa74fa049b7276007b739f3b8cfeac09e8cfabd4f858b6b99798c81124c34851960bebda90133cb03c981c08c8b6d3"),
        ).unwrap();
        let t_hi_1_expected = ShortWeierstrassJacobianPoint::<BLS12381Curve>::neutral_element();

        assert_eq!(round_3.t_lo_1, t_lo_1_expected);
        assert_eq!(round_3.t_mid_1, t_mid_1_expected);
        assert_eq!(round_3.t_hi_1, t_hi_1_expected);
    }

    #[test]
    fn test_round_4() {
        let witness = test_witness_1(FrElement::from(2), FrElement::from(2));
        let common_preprocessed_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_preprocessed_input.n);
        let kzg = KZG::new(srs);
        let random_generator = TestRandomFieldGenerator {};
        let prover = Prover::new(kzg, random_generator);

        let round_1 = prover.round_1(&witness, &common_preprocessed_input);
        let round_2 = prover
            .round_2(&witness, &common_preprocessed_input, beta(), gamma())
            .unwrap();

        let round_4 = prover.round_4(&common_preprocessed_input, &round_1, &round_2, zeta());
        let expected_a_value = FrElement::from_hex_unchecked(
            "2c090a95b57f1f493b7b747bba34fef7772fd72f97d718ed69549641a823eb2e",
        );
        let expected_b_value = FrElement::from_hex_unchecked(
            "5975959d91369ba4e7a03c6ae94b7fe98e8b61b7bf9af63c8ae0759e17ac0c7e",
        );
        let expected_c_value = FrElement::from_hex_unchecked(
            "6bf31edeb4344b7d2df2cb1bd40b4d13e182d9cb09f89591fa043c1a34b4a93",
        );
        let expected_z_value = FrElement::from_hex_unchecked(
            "38e2ec8e7c3dab29e2b8e9c8ea152914b8fe4612e91f2902c80238efcf21f4ee",
        );
        let expected_s1_value = FrElement::from_hex_unchecked(
            "472f66db4fb6947d9ed9808241fe82324bc08aa2a54be93179db8e564e1137d4",
        );
        let expected_s2_value = FrElement::from_hex_unchecked(
            "5588f1239c24efe0538868d0f716984e69c6980e586864f615e4b0621fdc6f81",
        );

        assert_eq!(round_4.a_zeta, expected_a_value);
        assert_eq!(round_4.b_zeta, expected_b_value);
        assert_eq!(round_4.c_zeta, expected_c_value);
        assert_eq!(round_4.z_zeta_omega, expected_z_value);
        assert_eq!(round_4.s1_zeta, expected_s1_value);
        assert_eq!(round_4.s2_zeta, expected_s2_value);
    }

    #[test]
    fn test_round_5() {
        let witness = test_witness_1(FrElement::from(2), FrElement::from(2));
        let common_preprocessed_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_preprocessed_input.n);
        let kzg = KZG::new(srs);
        let public_input = vec![FieldElement::from(2_u64), FieldElement::from(4)];
        let random_generator = TestRandomFieldGenerator {};
        let prover = Prover::new(kzg, random_generator);

        let round_1 = prover.round_1(&witness, &common_preprocessed_input);
        let round_2 = prover
            .round_2(&witness, &common_preprocessed_input, beta(), gamma())
            .unwrap();

        let round_3 = prover
            .round_3(
                &common_preprocessed_input,
                &public_input,
                &round_1,
                &round_2,
                alpha(),
            )
            .unwrap();

        let round_4 = prover.round_4(&common_preprocessed_input, &round_1, &round_2, zeta());

        let expected_w_zeta_1 = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("fa6250b80a418f0548b132ac264ff9915b2076c0c2548da9316ae19ffa35bbcf905d9f02f9274739608045ef83a4757"),
            FpElement::from_hex_unchecked("17713ade2dbd66e923d4092a5d2da98202959dd65a15e9f7791fab3c0dd08788aa9b4a1cb21d04e0c43bd29225472145"),
        ).unwrap();
        let expected_w_zeta_omega_1 = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("4484f08f8eaccf28bab8ee9539e6e7f4059cb1ce77b9b18e9e452f387163dc0b845f4874bf6445399e650d362799ff5"),
            FpElement::from_hex_unchecked("1254347a0fa2ac856917825a5cff5f9583d39a52edbc2be5bb10fabd0c04d23019bcb963404345743120310fd734a61a"),
        ).unwrap();

        let round_5 = prover
            .round_5(
                &common_preprocessed_input,
                &round_1,
                &round_2,
                &round_3,
                &round_4,
                upsilon(),
            )
            .unwrap();
        assert_eq!(round_5.w_zeta_1, expected_w_zeta_1);
        assert_eq!(round_5.w_zeta_omega_1, expected_w_zeta_omega_1);
    }
}
