//! Barycentric Lagrange interpolation on cosets.
//!
//! Given evaluations `y_i = P(h * omega^i)` on a coset `{h * omega^i}_{i=0..N-1}`,
//! evaluates `P(z)` for arbitrary `z` without recovering the coefficient form.
//!
//! # Formula
//!
//! ```text
//! P(z) = [(z^N - h^N) / (N * h^{N-1})] * sum_{i=0}^{N-1} omega^i * y_i / (z - h * omega^i)
//! ```
//!
//! where `omega` is a primitive N-th root of unity and `h` is the coset offset.
//!
//! # Derivation
//!
//! The vanishing polynomial of the coset `{h * omega^i}` is `V(z) = z^N - h^N`.
//! Its derivative at the i-th point is `V'(x_i) = N * h^{N-1} * omega^{-i}`,
//! giving barycentric weight `w_i = omega^i / (N * h^{N-1})`.
//!
//! # Mixed-field advantage
//!
//! When evaluations are in an extension field E but the coset is in the base field F,
//! the weights `omega^i` and denominators `1/(z - h * omega^i)` are computed entirely
//! in the base field. Only the final multiply with `y_i` crosses into the extension,
//! making the inner loop BF×EF (cost ~d) instead of EF×EF (cost ~d²).

use crate::field::element::FieldElement;
use crate::field::traits::{IsFFTField, IsField, IsSubFieldOf};

/// Evaluates a polynomial at `z` given its evaluations on a coset `{h * omega^i}`.
///
/// The coset offset `h` and primitive root `omega` are in the base field `F`,
/// while evaluations and `z` may be in an extension field `E`.
///
/// # Arguments
///
/// - `evaluations`: Values `y_i = P(h * omega^i)` for `i = 0..N-1` (in field E)
/// - `coset_offset`: The coset offset `h` (in base field F)
/// - `lde_primitive_root`: The primitive N-th root of unity `omega` (in base field F)
/// - `z`: The point at which to evaluate `P` (in field E)
///
/// # Returns
///
/// `P(z)` computed via the barycentric formula.
///
/// # Panics
///
/// Panics if `evaluations` is empty or if `z` coincides with a coset point.
pub fn barycentric_evaluate_on_coset<F, E>(
    evaluations: &[FieldElement<E>],
    coset_offset: &FieldElement<F>,
    lde_primitive_root: &FieldElement<F>,
    z: &FieldElement<E>,
) -> FieldElement<E>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = evaluations.len();
    assert!(n > 0, "evaluations must be non-empty");

    // All base-field computations first.
    // Prefactor: (z^N - h^N) / (N * h^{N-1})
    let h_n: FieldElement<F> = coset_offset.pow(n);
    let n_inv: FieldElement<F> = FieldElement::<F>::from(n as u64)
        .inv()
        .expect("N is a power of two in an FFT field, always invertible");
    let h_n_minus_1_inv: FieldElement<F> = coset_offset
        .pow(n - 1)
        .inv()
        .expect("coset offset must be invertible");
    // BF scalar: 1 / (N * h^{N-1})
    let bf_scalar: FieldElement<F> = &n_inv * &h_n_minus_1_inv;

    // z^N in extension field
    let z_n: FieldElement<E> = z.pow(n);
    // vanishing = z^N - h^N = -(h^N - z^N), using BF - EF → EF via IsSubFieldOf
    let vanishing: FieldElement<E> = -(&h_n - &z_n);

    // Pre-compute all denominators (z - h*omega^i) and batch-invert them.
    // This replaces N expensive field inversions with a single inversion
    // plus 3*(N-1) cheap multiplications (Montgomery's trick).
    let mut denoms: Vec<FieldElement<E>> = Vec::with_capacity(n);
    {
        let mut coset_pt = coset_offset.clone();
        for _ in 0..n {
            // denom = z - coset_pt = -(coset_pt - z), using BF - EF → EF
            denoms.push(-(&coset_pt - z));
            coset_pt = &coset_pt * lde_primitive_root;
        }
    }
    FieldElement::inplace_batch_inverse(&mut denoms)
        .expect("z should not coincide with any coset point");

    // Accumulate: sum_{i=0}^{N-1} (omega^i * denom_inv_i) * y_i
    //
    // omega^i stays in base field. BF × EF multiplications use native
    // IsSubFieldOf operators (cost ~d vs ~d²).
    let mut acc = FieldElement::<E>::zero();
    let mut omega_i = FieldElement::<F>::one();

    for (y_i, denom_inv) in evaluations.iter().zip(denoms.iter()) {
        let weight_times_inv: FieldElement<E> = &omega_i * denom_inv;
        acc += &weight_times_inv * y_i;
        omega_i = &omega_i * lde_primitive_root;
    }

    // Final: prefactor * acc = (bf_scalar × vanishing) * acc
    // bf_scalar (BF) × vanishing (EF) via native IsSubFieldOf mul
    let prefactor: FieldElement<E> = &bf_scalar * &vanishing;
    &prefactor * &acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use crate::polynomial::Polynomial;

    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    /// Helper: evaluate polynomial on coset {h * omega^i} via FFT.
    fn evaluate_on_coset(
        poly: &Polynomial<FpE>,
        coset_offset: &FpE,
        domain_size: usize,
    ) -> Vec<FpE> {
        Polynomial::evaluate_offset_fft::<F>(
            poly,
            domain_size / poly.coeff_len(),
            None,
            coset_offset,
        )
        .expect("FFT evaluation failed")
    }

    #[test]
    fn barycentric_matches_horner_degree_3() {
        // P(x) = x^3 + 2x + 1
        let poly = Polynomial::new(&[FpE::from(1u64), FpE::from(2u64), FpE::zero(), FpE::one()]);

        let coset_offset = FpE::from(7u64);
        let domain_size = 8; // 4 coefficients, blowup = 2
        let evals = evaluate_on_coset(&poly, &coset_offset, domain_size);

        let order = domain_size.trailing_zeros() as u64;
        let omega = F::get_primitive_root_of_unity(order).expect("root exists");

        for z_val in [3u64, 17, 42, 1000, 999999937] {
            let z = FpE::from(z_val);
            let bary = barycentric_evaluate_on_coset::<F, F>(&evals, &coset_offset, &omega, &z);
            let horner = poly.evaluate(&z);
            assert_eq!(
                bary, horner,
                "mismatch at z={z_val}: bary={bary:?} horner={horner:?}"
            );
        }
    }

    #[test]
    fn barycentric_matches_horner_degree_1() {
        // P(x) = 5x + 3
        let poly = Polynomial::new(&[FpE::from(3u64), FpE::from(5u64)]);

        let coset_offset = FpE::from(11u64);
        let domain_size = 4; // 2 coefficients, blowup = 2
        let evals = evaluate_on_coset(&poly, &coset_offset, domain_size);

        let order = domain_size.trailing_zeros() as u64;
        let omega = F::get_primitive_root_of_unity(order).expect("root exists");

        for z_val in [1u64, 2, 100, 2u64.pow(32) - 1] {
            let z = FpE::from(z_val);
            let bary = barycentric_evaluate_on_coset::<F, F>(&evals, &coset_offset, &omega, &z);
            let horner = poly.evaluate(&z);
            assert_eq!(bary, horner, "mismatch at z={z_val}");
        }
    }

    #[test]
    fn barycentric_matches_horner_large_degree() {
        // Random-ish polynomial of degree 255
        let coeffs: Vec<FpE> = (0..256).map(|i| FpE::from(i as u64 * 7 + 13)).collect();
        let poly = Polynomial::new(&coeffs);

        let coset_offset = FpE::from(7u64);
        let domain_size = 1024; // 256 coeffs, blowup = 4
        let evals = evaluate_on_coset(&poly, &coset_offset, domain_size);

        let order = domain_size.trailing_zeros() as u64;
        let omega = F::get_primitive_root_of_unity(order).expect("root exists");

        for z_val in [99u64, 12345, 999999937] {
            let z = FpE::from(z_val);
            let bary = barycentric_evaluate_on_coset::<F, F>(&evals, &coset_offset, &omega, &z);
            let horner = poly.evaluate(&z);
            assert_eq!(bary, horner, "mismatch at z={z_val}");
        }
    }

    #[test]
    fn barycentric_on_standard_domain() {
        // Test with h = 1 (standard domain, not a coset)
        let poly = Polynomial::new(&[FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);

        let coset_offset = FpE::one();
        let domain_size = 8;
        let evals = evaluate_on_coset(&poly, &coset_offset, domain_size);

        let order = domain_size.trailing_zeros() as u64;
        let omega = F::get_primitive_root_of_unity(order).expect("root exists");

        let z = FpE::from(42u64);
        let bary = barycentric_evaluate_on_coset::<F, F>(&evals, &coset_offset, &omega, &z);
        let horner = poly.evaluate(&z);
        assert_eq!(bary, horner);
    }
}
