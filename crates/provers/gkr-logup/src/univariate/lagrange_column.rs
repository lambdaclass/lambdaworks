use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

/// Computes the Lagrange column `c_i = eq(iota(i), t)` for all `i in [0, 2^n)`.
///
/// This is the evaluation of the multilinear eq polynomial at the point `t`, but
/// indexed in **cyclic domain order** (LSB-first bit indexing) rather than
/// MLE lexicographic order.
///
/// Butterfly algorithm processing variables in **reverse** order to match
/// the `DenseMultilinearPolynomial` indexing convention.
///
/// # Formula (adapted to `{0,1}^n`)
///
/// `c_i = eq(iota(i), t)` where `iota(i)` maps index `i` to the hypercube point
/// matching the MLE's layout: variable 0 occupies the MSB of the index.
///
/// This is the same butterfly as `gen_eq_evals` (with `v = 1`).
pub fn compute_lagrange_column<F: IsField>(
    evaluation_point: &[FieldElement<F>],
) -> Vec<FieldElement<F>> {
    let n = evaluation_point.len();
    let mut evals = Vec::with_capacity(1 << n);
    evals.push(FieldElement::one());

    // Reverse order matches MLE lexicographic convention (same as gen_eq_evals)
    for t_k in evaluation_point.iter().rev() {
        let len = evals.len();
        for j in 0..len {
            // tmp = evals[j] * t_k (the i_k=1 branch)
            let tmp = &evals[j] * t_k;
            // evals[j] = evals[j] * (1 - t_k) = evals[j] - tmp (the i_k=0 branch)
            evals.push(tmp.clone());
            evals[j] = &evals[j] - &tmp;
        }
    }

    evals
}

/// Error returned when Lagrange column constraints fail.
#[derive(Debug)]
pub enum LagrangeColumnError {
    /// Constraint eq. 10 failed: `c[0] != prod(1 - t_k)`.
    InitialValueFailed,
    /// Constraint eq. 11 failed at position `(k, i)`.
    PeriodicConstraintFailed { k: usize, i: usize },
    /// Column length doesn't match `2^n`.
    InvalidColumnLength,
    /// Vectors have mismatched lengths in inner product.
    LengthMismatch { expected: usize, got: usize },
    /// Division by zero: some `t_k == 1`.
    DivisionByZero { k: usize },
}

impl core::fmt::Display for LagrangeColumnError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InitialValueFailed => write!(f, "Lagrange column initial value c[0] mismatch"),
            Self::PeriodicConstraintFailed { k, i } => {
                write!(
                    f,
                    "Lagrange column periodic constraint failed at k={k}, i={i}"
                )
            }
            Self::InvalidColumnLength => write!(f, "column length is not 2^n"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            Self::DivisionByZero { k } => write!(f, "t_{k} = 1, ratio undefined"),
        }
    }
}

/// Verifies the periodic constraints (eqs. 10, 11 from Section 5.1) on a Lagrange column.
///
/// - **Eq. 10**: `c[0] = prod_{k=0}^{n-1} (1 - t_k)`
/// - **Eq. 11**: For each `k = 1..n`, for each `i` that is a multiple of `2^k` in `[0, 2^n)`:
///   `c[i + 2^{k-1}] = c[i] * t_{k-1} / (1 - t_{k-1})`
pub fn verify_lagrange_column_constraints<F: IsField>(
    column: &[FieldElement<F>],
    evaluation_point: &[FieldElement<F>],
) -> Result<(), LagrangeColumnError> {
    let n = evaluation_point.len();
    if column.len() != 1 << n {
        return Err(LagrangeColumnError::InvalidColumnLength);
    }

    // Eq. 10: c[0] = prod(1 - t_k)
    let expected_c0: FieldElement<F> = evaluation_point
        .iter()
        .fold(FieldElement::one(), |acc, t_k| {
            acc * (FieldElement::<F>::one() - t_k)
        });

    if column[0] != expected_c0 {
        return Err(LagrangeColumnError::InitialValueFailed);
    }

    // Eq. 11: The butterfly processes variables in reverse order (t_{n-1}, t_{n-2}, ..., t_0).
    // At level k (k=1..n), the variable is t_{n-k}.
    // For each i that is a multiple of 2^k in [0, 2^n):
    //   c[i + 2^{k-1}] == c[i] * t_{n-k} / (1 - t_{n-k})
    for k in 1..=n {
        let var_idx = n - k;
        let t_var = &evaluation_point[var_idx];
        let one_minus_t = FieldElement::<F>::one() - t_var;

        let step = 1 << k;
        let half = 1 << (k - 1);

        if one_minus_t == FieldElement::<F>::zero() {
            // t_k == 1: the (1-t_k) factor is zero, so all "bit-0" entries
            // (those at positions i that are multiples of 2^k) must be zero.
            // The "bit-1" entries (at i + half) are unconstrained by this ratio
            // but are covered by other levels.
            let mut i = 0;
            while i < (1 << n) {
                if column[i] != FieldElement::<F>::zero() {
                    return Err(LagrangeColumnError::PeriodicConstraintFailed { k, i });
                }
                i += step;
            }
        } else {
            let inv = one_minus_t
                .inv()
                .map_err(|_| LagrangeColumnError::DivisionByZero { k: var_idx })?;
            let ratio = t_var * &inv;

            let mut i = 0;
            while i < (1 << n) {
                let expected = &column[i] * &ratio;
                if column[i + half] != expected {
                    return Err(LagrangeColumnError::PeriodicConstraintFailed { k, i });
                }
                i += step;
            }
        }
    }

    Ok(())
}

/// Computes the inner product of univariate evaluations with the Lagrange column.
///
/// `f(t) = sum_{i=0}^{2^n - 1} u_f(omega^i) * c_i`
///
/// This equals the multilinear evaluation of the polynomial at point `t`.
pub fn inner_product<F: IsField>(
    univariate_values: &[FieldElement<F>],
    lagrange_column: &[FieldElement<F>],
) -> Result<FieldElement<F>, LagrangeColumnError> {
    if univariate_values.len() != lagrange_column.len() {
        return Err(LagrangeColumnError::LengthMismatch {
            expected: lagrange_column.len(),
            got: univariate_values.len(),
        });
    }
    Ok(univariate_values
        .iter()
        .zip(lagrange_column.iter())
        .fold(FieldElement::zero(), |acc, (u, c)| acc + u * c))
}

/// Computes the combined inner product for multiple columns using random linear combination.
///
/// Given columns `[col_0, col_1, ...]` and combining factor `lambda`:
/// `combined = col_0 + lambda * col_1 + lambda^2 * col_2 + ...`
/// then returns `inner_product(combined, lagrange_column)`.
pub fn combined_inner_product<F: IsField>(
    columns: &[&[FieldElement<F>]],
    lagrange_column: &[FieldElement<F>],
    lambda: &FieldElement<F>,
) -> Result<FieldElement<F>, LagrangeColumnError> {
    if columns.is_empty() {
        return Ok(FieldElement::zero());
    }

    let n = lagrange_column.len();

    // Combine columns with lambda powers, then inner product with lagrange
    let mut combined = vec![FieldElement::<F>::zero(); n];
    let mut lambda_power = FieldElement::<F>::one();
    for col in columns {
        if col.len() != n {
            return Err(LagrangeColumnError::LengthMismatch {
                expected: n,
                got: col.len(),
            });
        }
        for i in 0..n {
            combined[i] = &combined[i] + &(&lambda_power * &col[i]);
        }
        lambda_power = &lambda_power * lambda;
    }

    inner_product(&combined, lagrange_column)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
    use lambdaworks_math::polynomial::dense_multilinear_poly::eq_eval;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn test_lagrange_column_n1() {
        // n=1, t = [7]
        // c_0 = eq((0), (7)) = 1-7 = -6 = 95
        // c_1 = eq((1), (7)) = 7
        let t = vec![FE::from(7)];
        let col = compute_lagrange_column(&t);

        assert_eq!(col.len(), 2);
        assert_eq!(col[0], FE::one() - FE::from(7));
        assert_eq!(col[1], FE::from(7));
    }

    #[test]
    fn test_lagrange_column_n2() {
        // n=2, t = [t0, t1] = [3, 5]
        // MLE convention: variable 0 is MSB of index.
        // c[0] = (1-t0)(1-t1), c[1] = (1-t0)*t1, c[2] = t0*(1-t1), c[3] = t0*t1
        let t = vec![FE::from(3), FE::from(5)];
        let col = compute_lagrange_column(&t);

        assert_eq!(col.len(), 4);

        let zero = FE::zero();
        let one = FE::one();
        // MLE convention: index i → (x0, x1) where x0 = i>>1, x1 = i&1
        assert_eq!(col[0], eq_eval(&[zero, zero], &t)); // i=0
        assert_eq!(col[1], eq_eval(&[zero, one], &t)); // i=1: x0=0, x1=1
        assert_eq!(col[2], eq_eval(&[one, zero], &t)); // i=2: x0=1, x1=0
        assert_eq!(col[3], eq_eval(&[one, one], &t)); // i=3
    }

    #[test]
    fn test_lagrange_column_n3() {
        // n=3, t = [2, 7, 11]
        let t = vec![FE::from(2), FE::from(7), FE::from(11)];
        let col = compute_lagrange_column(&t);
        assert_eq!(col.len(), 8);

        // MLE convention: index i → (x0, x1, x2) where x_k = (i >> (n-1-k)) & 1
        let n = 3;
        for i in 0..8u64 {
            let bits: Vec<FE> = (0..n).map(|k| FE::from((i >> (n - 1 - k)) & 1)).collect();
            assert_eq!(col[i as usize], eq_eval(&bits, &t), "mismatch at i={i}");
        }
    }

    #[test]
    fn test_column_sums_to_one() {
        let t = vec![FE::from(3), FE::from(7), FE::from(13)];
        let col = compute_lagrange_column(&t);

        let sum: FE = col.iter().fold(FE::zero(), |acc, c| acc + c);
        assert_eq!(sum, FE::one());
    }

    #[test]
    fn test_inner_product_equals_mle_eval() {
        // Key correctness property: inner_product(mle_values, lagrange_col) = MLE(t)
        //
        // MLE convention: evals[i] = f(x_0, ..., x_{n-1}) where x_k = (i >> (n-1-k)) & 1
        // So evals[0]=f(0,0), evals[1]=f(0,1), evals[2]=f(1,0), evals[3]=f(1,1)
        let t = vec![FE::from(3), FE::from(7)];
        let col = compute_lagrange_column(&t);

        let mle_values = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];

        let ip = inner_product(&mle_values, &col).unwrap();

        // MLE evaluation: sum_i evals[i] * eq(x(i), t)
        let zero = FE::zero();
        let one = FE::one();
        let expected = FE::from(1) * eq_eval(&[zero, zero], &t) // i=0
            + FE::from(2) * eq_eval(&[zero, one], &t) // i=1: x0=0, x1=1
            + FE::from(3) * eq_eval(&[one, zero], &t) // i=2: x0=1, x1=0
            + FE::from(4) * eq_eval(&[one, one], &t); // i=3

        assert_eq!(ip, expected);
    }

    #[test]
    fn test_constraints_pass_for_valid_column() {
        let t = vec![FE::from(3), FE::from(7), FE::from(13)];
        let col = compute_lagrange_column(&t);
        assert!(verify_lagrange_column_constraints(&col, &t).is_ok());
    }

    #[test]
    fn test_constraints_fail_for_tampered_column() {
        let t = vec![FE::from(3), FE::from(7), FE::from(13)];
        let mut col = compute_lagrange_column(&t);
        col[3] += FE::one(); // tamper
        assert!(verify_lagrange_column_constraints(&col, &t).is_err());
    }

    #[test]
    fn test_constraints_fail_for_wrong_length() {
        let t = vec![FE::from(3), FE::from(7)];
        let col = vec![FE::one(); 5]; // not power of 2
        assert!(matches!(
            verify_lagrange_column_constraints(&col, &t),
            Err(LagrangeColumnError::InvalidColumnLength)
        ));
    }

    #[test]
    fn test_combined_inner_product_single_column() {
        let t = vec![FE::from(3), FE::from(7)];
        let col = compute_lagrange_column(&t);
        let values = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let lambda = FE::from(5);

        let result = combined_inner_product(&[&values], &col, &lambda).unwrap();
        let expected = inner_product(&values, &col).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_combined_inner_product_two_columns() {
        let t = vec![FE::from(3), FE::from(7)];
        let col = compute_lagrange_column(&t);
        let v1 = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let v2 = vec![FE::from(5), FE::from(6), FE::from(7), FE::from(8)];
        let lambda = FE::from(10);

        let result = combined_inner_product(&[&v1, &v2], &col, &lambda).unwrap();

        // combined[i] = v1[i] + lambda * v2[i]
        let combined: Vec<FE> = v1
            .iter()
            .zip(v2.iter())
            .map(|(a, b)| a + lambda * b)
            .collect();
        let expected = inner_product(&combined, &col).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_lagrange_column_with_t_k_one() {
        // When t_k == 1, the (1-t_k) factor is zero.
        // compute_lagrange_column should still produce correct values,
        // and verify_lagrange_column_constraints should accept them.
        //
        // t = [1, 7]: MLE convention: x_0 = MSB = i>>1, x_1 = LSB = i&1
        // eq(x, t) = prod_k [x_k * t_k + (1-x_k)*(1-t_k)]
        // With t_0 = 1: (1-t_0)=0, so entries where x_0=0 are zero (indices 0,1)
        let t = vec![FE::from(1), FE::from(7)];
        let col = compute_lagrange_column(&t);
        assert_eq!(col.len(), 4);

        // col[0] = eq((0,0), (1,7)) = (1-1)*(1-7) = 0
        // col[1] = eq((0,1), (1,7)) = (1-1)*7 = 0
        // col[2] = eq((1,0), (1,7)) = 1*(1-7) = -6 = 95
        // col[3] = eq((1,1), (1,7)) = 1*7 = 7
        assert_eq!(col[0], FE::zero());
        assert_eq!(col[1], FE::zero());
        assert_ne!(col[2], FE::zero());
        assert_ne!(col[3], FE::zero());

        // Constraints should pass (no DivisionByZero)
        assert!(verify_lagrange_column_constraints(&col, &t).is_ok());
    }

    #[test]
    fn test_lagrange_column_with_all_t_one() {
        // Edge case: all t_k == 1 → only the (1,1,...,1) index is nonzero
        let t = vec![FE::from(1), FE::from(1), FE::from(1)];
        let col = compute_lagrange_column(&t);
        assert_eq!(col.len(), 8);

        // Only col[7] (= eq(111, (1,1,1))) should be nonzero (= 1)
        for (i, val) in col.iter().enumerate().take(7) {
            assert_eq!(*val, FE::zero(), "col[{i}] should be zero");
        }
        assert_eq!(col[7], FE::one());

        assert!(verify_lagrange_column_constraints(&col, &t).is_ok());
    }
}
