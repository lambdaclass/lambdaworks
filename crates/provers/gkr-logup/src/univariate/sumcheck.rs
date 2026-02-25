//! Univariate sumcheck protocol for LogUp-GKR Phase 2.
//!
//! Reduces the inner product check `sum_i u_f(omega^i) * c_i(t) = v`
//! to a single point evaluation via polynomial division.
//!
//! **Identity:** `u_f(X) * C_t(X) - v/N = q(X) * (X^N - 1) + X * r'(X)`
//!
//! where `q(X)` and `r'(X)` are auxiliary polynomials. This holds iff the
//! sum of `u_f(omega^i) * C_t(omega^i)` equals `v`.
//!
//! **Protocol:**
//! 1. Prover commits to `q(X)` and `r'(X)` via PCS
//! 2. Challenge `z` sampled from transcript
//! 3. Prover batch-opens `u_f`, `q`, `r'` at `z`
//! 4. Verifier computes `C_t(z)` via Lagrange interpolation
//! 5. Verifier checks: `u_f(z) * C_t(z) - v/N == q(z) * (z^N - 1) + z * r'(z)`

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::polynomial::Polynomial;

use crate::univariate::pcs::PcsError;

/// Result of the sumcheck prover: the auxiliary polynomial coefficients.
pub struct SumcheckProverResult<F: IsFFTField> {
    /// Polynomial `u_f` in coefficient form (interpolated from evals on H).
    pub u_f_poly: Polynomial<FieldElement<F>>,
    /// Quotient polynomial `q(X)` from dividing `u_f * C_t - v/N` by `Z_H = X^N - 1`.
    pub q_poly: Polynomial<FieldElement<F>>,
    /// Remainder polynomial `r'(X)` where `r(X) = X * r'(X)` is the remainder after division.
    pub r_prime_poly: Polynomial<FieldElement<F>>,
}

/// Run the sumcheck prover.
///
/// Given evaluations of `u_f` and `C_t` on H (the cyclic domain of size N),
/// and the claimed sum `v`, compute the auxiliary polynomials `q` and `r'`.
pub fn prove_sumcheck<F>(
    u_f_evals: &[FieldElement<F>],
    lagrange_evals: &[FieldElement<F>],
    claimed_sum: &FieldElement<F>,
) -> Result<SumcheckProverResult<F>, PcsError>
where
    F: IsFFTField,
{
    let n = u_f_evals.len();
    if n != lagrange_evals.len() || !n.is_power_of_two() || n == 0 {
        return Err(PcsError::InvalidInput(
            "u_f and lagrange evals must have the same power-of-2 length".into(),
        ));
    }

    // Interpolate u_f and C_t from their evaluations on H
    let u_f_poly = Polynomial::interpolate_fft::<F>(u_f_evals)
        .map_err(|e| PcsError::InternalError(format!("FFT interpolation of u_f failed: {e}")))?;

    let c_t_poly = Polynomial::interpolate_fft::<F>(lagrange_evals)
        .map_err(|e| PcsError::InternalError(format!("FFT interpolation of C_t failed: {e}")))?;

    // Compute product = u_f * C_t (degree < 2N)
    let product = u_f_poly.mul_with_ref(&c_t_poly);

    // Subtract v/N from the constant term
    let n_fe = FieldElement::<F>::from(n as u64);
    let n_inv = n_fe
        .inv()
        .map_err(|_| PcsError::InternalError("N is not invertible".into()))?;
    let v_over_n = claimed_sum * &n_inv;

    let shifted = &product - &Polynomial::new(&[v_over_n]);

    // Build Z_H(X) = X^N - 1
    let mut z_h_coeffs = vec![FieldElement::<F>::zero(); n + 1];
    z_h_coeffs[0] = -FieldElement::<F>::one();
    z_h_coeffs[n] = FieldElement::one();
    let z_h = Polynomial::new(&z_h_coeffs);

    // Divide: shifted = q * Z_H + r
    let (q_poly, r_poly) = shifted
        .long_division_with_remainder(&z_h)
        .map_err(|e| PcsError::InternalError(format!("division failed: {e}")))?;

    // r(X) should have r[0] == 0 (the constant term is zero when the sum is correct)
    let r_coeffs = r_poly.coefficients();
    if !r_coeffs.is_empty() && r_coeffs[0] != FieldElement::zero() {
        return Err(PcsError::InternalError(
            "sumcheck failed: remainder constant term is nonzero".into(),
        ));
    }

    // r'(X) = r(X) / X — shift coefficients down by one
    let r_prime_coeffs: Vec<FieldElement<F>> = if r_coeffs.len() > 1 {
        r_coeffs[1..].to_vec()
    } else {
        vec![]
    };
    let r_prime_poly = Polynomial::new(&r_prime_coeffs);

    Ok(SumcheckProverResult {
        u_f_poly,
        q_poly,
        r_prime_poly,
    })
}

/// Verify the univariate sumcheck equation at point `z`:
///
/// `u_f(z) * C_t(z) - v/N == q(z) * (z^N - 1) + z * r'(z)`
pub fn verify_sumcheck_at_z<F: IsFFTField>(
    u_f_z: &FieldElement<F>,
    c_t_z: &FieldElement<F>,
    claimed_sum: &FieldElement<F>,
    q_z: &FieldElement<F>,
    r_prime_z: &FieldElement<F>,
    z: &FieldElement<F>,
    n: usize,
) -> bool {
    let n_fe = FieldElement::<F>::from(n as u64);
    let n_inv = match n_fe.inv() {
        Ok(inv) => inv,
        Err(_) => return false,
    };
    let v_over_n = claimed_sum * &n_inv;

    // LHS = u_f(z) * C_t(z) - v/N
    let lhs = u_f_z * c_t_z - &v_over_n;

    // RHS = q(z) * (z^N - 1) + z * r'(z)
    let z_n = z.pow(n);
    let z_h_z = &z_n - FieldElement::<F>::one();
    let rhs = q_z * &z_h_z + z * r_prime_z;

    lhs == rhs
}

/// Compute `C_t(z)` — the evaluation of the Lagrange interpolation polynomial at `z`.
///
/// Uses the barycentric formula:
/// `C_t(z) = (z^N - 1)/N * sum_i c_i * omega^i / (z - omega^i)`
///
/// This is O(N) field multiplications + one batch inversion.
pub fn evaluate_lagrange_at_z<F: IsFFTField>(
    lagrange_evals: &[FieldElement<F>],
    z: &FieldElement<F>,
    n: usize,
) -> Result<FieldElement<F>, PcsError> {
    if lagrange_evals.len() != n || !n.is_power_of_two() || n == 0 {
        return Err(PcsError::InvalidInput(
            "invalid lagrange evals length".into(),
        ));
    }

    let log_n = n.trailing_zeros() as u64;
    let omega = F::get_primitive_root_of_unity(log_n)
        .map_err(|_| PcsError::InternalError("no root of unity".into()))?;

    // Compute z^N - 1
    let z_n = z.pow(n);
    let z_n_minus_1 = &z_n - FieldElement::<F>::one();

    // If z is in the domain (z^N == 1), we can directly look up the value
    if z_n_minus_1 == FieldElement::zero() {
        // z is a root of unity; find which one
        let mut omega_i = FieldElement::<F>::one();
        for c_i in lagrange_evals {
            if *z == omega_i {
                return Ok(c_i.clone());
            }
            omega_i = &omega_i * &omega;
        }
        return Err(PcsError::InternalError(
            "z is in domain but not found".into(),
        ));
    }

    // Compute denominators: z - omega^i for each i
    let mut omega_i = FieldElement::<F>::one();
    let mut denoms = Vec::with_capacity(n);
    for _ in 0..n {
        denoms.push(z - &omega_i);
        omega_i = &omega_i * &omega;
    }

    // Batch inversion
    let inv_denoms = batch_inverse(&denoms)?;

    // Sum: sum_i c_i * omega^i / (z - omega^i)
    let mut sum = FieldElement::<F>::zero();
    let mut omega_i = FieldElement::<F>::one();
    for (c_i, inv_d) in lagrange_evals.iter().zip(inv_denoms.iter()) {
        sum += c_i * &omega_i * inv_d;
        omega_i = &omega_i * &omega;
    }

    // C_t(z) = (z^N - 1) / N * sum
    let n_fe = FieldElement::<F>::from(n as u64);
    let n_inv = n_fe
        .inv()
        .map_err(|_| PcsError::InternalError("N not invertible".into()))?;

    Ok(&z_n_minus_1 * &n_inv * &sum)
}

/// Montgomery batch inversion: compute inverses of all elements using O(N) muls + 1 inversion.
pub fn batch_inverse<F: IsFFTField>(
    elements: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, PcsError> {
    let n = elements.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Compute prefix products
    let mut prefix = Vec::with_capacity(n);
    prefix.push(elements[0].clone());
    for i in 1..n {
        prefix.push(&prefix[i - 1] * &elements[i]);
    }

    // Invert the total product
    let mut inv_total = prefix[n - 1]
        .inv()
        .map_err(|_| PcsError::InternalError("batch inverse: zero element encountered".into()))?;

    // Unwind to get individual inverses
    let mut result = vec![FieldElement::<F>::zero(); n];
    for i in (1..n).rev() {
        result[i] = &inv_total * &prefix[i - 1];
        inv_total = &inv_total * &elements[i];
    }
    result[0] = inv_total;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::univariate::lagrange_column::compute_lagrange_column;
    use lambdaworks_math::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;

    type F = Degree4BabyBearExtensionField;
    type FE = FieldElement<F>;

    #[test]
    fn sumcheck_roundtrip_size_4() {
        let u_f_evals: Vec<FE> = (1..=4).map(|i| FE::from(i as u64)).collect();
        let t = vec![FE::from(5u64), FE::from(7u64)];
        let lagrange_evals = compute_lagrange_column(&t);

        // Compute the sum
        let sum: FE = u_f_evals
            .iter()
            .zip(lagrange_evals.iter())
            .fold(FE::zero(), |acc, (u, c)| acc + u * c);

        let result = prove_sumcheck::<F>(&u_f_evals, &lagrange_evals, &sum).unwrap();

        // Verify at a random z
        let z = FE::from(42u64);
        let u_f_z = result.u_f_poly.evaluate(&z);
        let c_t_z = evaluate_lagrange_at_z(&lagrange_evals, &z, 4).unwrap();
        let q_z = result.q_poly.evaluate(&z);
        let r_prime_z = result.r_prime_poly.evaluate(&z);

        assert!(verify_sumcheck_at_z(
            &u_f_z, &c_t_z, &sum, &q_z, &r_prime_z, &z, 4
        ));
    }

    #[test]
    fn sumcheck_roundtrip_size_8() {
        let u_f_evals: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let t = vec![FE::from(3u64), FE::from(11u64), FE::from(17u64)];
        let lagrange_evals = compute_lagrange_column(&t);

        let sum: FE = u_f_evals
            .iter()
            .zip(lagrange_evals.iter())
            .fold(FE::zero(), |acc, (u, c)| acc + u * c);

        let result = prove_sumcheck::<F>(&u_f_evals, &lagrange_evals, &sum).unwrap();

        let z = FE::from(99u64);
        let u_f_z = result.u_f_poly.evaluate(&z);
        let c_t_z = evaluate_lagrange_at_z(&lagrange_evals, &z, 8).unwrap();
        let q_z = result.q_poly.evaluate(&z);
        let r_prime_z = result.r_prime_poly.evaluate(&z);

        assert!(verify_sumcheck_at_z(
            &u_f_z, &c_t_z, &sum, &q_z, &r_prime_z, &z, 8
        ));
    }

    #[test]
    fn sumcheck_roundtrip_size_16() {
        let u_f_evals: Vec<FE> = (1..=16).map(|i| FE::from(i as u64)).collect();
        let t = vec![
            FE::from(2u64),
            FE::from(5u64),
            FE::from(9u64),
            FE::from(13u64),
        ];
        let lagrange_evals = compute_lagrange_column(&t);

        let sum: FE = u_f_evals
            .iter()
            .zip(lagrange_evals.iter())
            .fold(FE::zero(), |acc, (u, c)| acc + u * c);

        let result = prove_sumcheck::<F>(&u_f_evals, &lagrange_evals, &sum).unwrap();

        let z = FE::from(200u64);
        let u_f_z = result.u_f_poly.evaluate(&z);
        let c_t_z = evaluate_lagrange_at_z(&lagrange_evals, &z, 16).unwrap();
        let q_z = result.q_poly.evaluate(&z);
        let r_prime_z = result.r_prime_poly.evaluate(&z);

        assert!(verify_sumcheck_at_z(
            &u_f_z, &c_t_z, &sum, &q_z, &r_prime_z, &z, 16
        ));
    }

    #[test]
    fn sumcheck_wrong_sum_rejected() {
        let u_f_evals: Vec<FE> = (1..=4).map(|i| FE::from(i as u64)).collect();
        let t = vec![FE::from(5u64), FE::from(7u64)];
        let lagrange_evals = compute_lagrange_column(&t);

        // Compute correct sum, then tamper
        let sum: FE = u_f_evals
            .iter()
            .zip(lagrange_evals.iter())
            .fold(FE::zero(), |acc, (u, c)| acc + u * c);
        let wrong_sum = sum + FE::one();

        // prove_sumcheck should fail because the constant term of r won't be zero
        let result = prove_sumcheck::<F>(&u_f_evals, &lagrange_evals, &wrong_sum);
        assert!(result.is_err());
    }

    #[test]
    fn batch_inverse_basic() {
        let elements: Vec<FE> = (1..=5).map(|i| FE::from(i as u64)).collect();
        let inverses = batch_inverse::<F>(&elements).unwrap();

        for (elem, inv) in elements.iter().zip(inverses.iter()) {
            let product = elem * inv;
            assert_eq!(product, FE::one());
        }
    }

    #[test]
    fn evaluate_lagrange_at_z_consistent_with_interpolation() {
        // Check that evaluate_lagrange_at_z matches direct polynomial interpolation + evaluation
        let t = vec![FE::from(3u64), FE::from(7u64), FE::from(11u64)];
        let lagrange_evals = compute_lagrange_column(&t);
        let n = lagrange_evals.len(); // 8

        let z = FE::from(42u64);
        let c_t_z = evaluate_lagrange_at_z(&lagrange_evals, &z, n).unwrap();

        // Compare with direct interpolation
        let c_t_poly = Polynomial::interpolate_fft::<F>(&lagrange_evals).unwrap();
        let expected = c_t_poly.evaluate(&z);

        assert_eq!(c_t_z, expected);
    }
}
