//! Multilinear extension (MLE) utilities for Spartan.
//!
//! Provides functions to encode R1CS matrices and witness vectors as MLEs,
//! and to compute key operations like eq_poly and matrix-vector product MLEs.

use lambdaworks_math::field::{element::FieldElement, traits::IsField};
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

use crate::errors::SpartanError;
use crate::sparse_matrix::SparseMatrix;

/// Returns the smallest power of 2 that is >= n (returns 1 for n = 0).
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        n.next_power_of_two()
    }
}

/// Encodes a witness vector z as a dense multilinear polynomial (MLE).
///
/// The witness z is padded to the next power of 2 with zeros.
/// The resulting MLE has log2(padded_len) variables.
///
/// The result at boolean point (b0, b1, ..., b_{k-1}) where bi in {0,1}
/// equals z[i] where i is the integer with binary representation b0b1...b_{k-1}.
pub fn encode_witness<F: IsField>(z: &[FieldElement<F>]) -> DenseMultilinearPolynomial<F>
where
    F::BaseType: Send + Sync,
{
    let n = z.len();
    // Ensure at least 2 evaluations so the MLE has at least 1 variable (required by sumcheck)
    let padded_len = next_power_of_two(n).max(2);
    let mut results = z.to_vec();
    results.resize(padded_len, FieldElement::zero());
    DenseMultilinearPolynomial::new(results)
}

/// Computes eq(tau, x) multilinear polynomial evaluations on {0,1}^s.
///
/// The equality polynomial is:
/// eq(tau, x) = prod_i (tau_i * x_i + (1 - tau_i) * (1 - x_i))
///
/// This evaluates to 1 when x = tau (over the boolean hypercube) and 0 otherwise.
///
/// The evaluations are computed efficiently using the "expansion" algorithm:
/// Start with [1], then for each tau_i expand: new[2k] = old[k] * (1-tau_i), new[2k+1] = old[k] * tau_i
pub fn eq_poly<F: IsField>(tau: &[FieldElement<F>]) -> DenseMultilinearPolynomial<F>
where
    F::BaseType: Send + Sync,
{
    let s = tau.len();
    let n = 1 << s;
    let mut current = vec![FieldElement::one(); 1];

    for tau_i in tau.iter() {
        let half = current.len();
        let one_minus_tau_i = FieldElement::<F>::one() - tau_i;
        let mut new_vals = vec![FieldElement::zero(); half * 2];
        for k in 0..half {
            new_vals[2 * k] = &current[k] * &one_minus_tau_i;
            new_vals[2 * k + 1] = &current[k] * tau_i;
        }
        current = new_vals;
    }

    debug_assert_eq!(current.len(), n);
    DenseMultilinearPolynomial::new(current)
}

/// Computes eq(r_x, binary(i)) for all i in [0, num_rows).
///
/// This is the Lagrange basis computation of the equality polynomial at a fixed point r_x.
/// Returns a vector of size num_rows_padded where entry i = prod_j (r_x[j]*b_j + (1-r_x[j])*(1-b_j))
/// and b = binary(i).
///
/// Used in matrix_vector_product_mle to compute the "row selector" weights.
pub fn eq_evals<F: IsField>(r_x: &[FieldElement<F>], size: usize) -> Vec<FieldElement<F>>
where
    F::BaseType: Send + Sync,
{
    // Build eq evaluations over {0,1}^{log(size)} using expansion algorithm
    let eq_mle = eq_poly(r_x);
    let eq_ev = eq_mle.evals().to_vec();
    // Pad or truncate to `size`
    let mut result = eq_ev;
    result.resize(size, FieldElement::zero());
    result
}

/// Computes the matrix-vector product MLE: MZ(r_x) as a function of column index.
///
/// Given sparse matrix M (m x n), a fixed point r_x (of length log m), computes:
///
///   MZ(r_x)[j] = sum_{i=0}^{m_padded - 1} M[i][j] * eq(r_x, binary(i))
///
/// Returns a DenseMultilinearPolynomial in log(n_padded) variables
/// whose result at boolean point binary(j) equals MZ(r_x)[j].
pub fn matrix_vector_product_mle<F: IsField>(
    matrix: &SparseMatrix<F>,
    num_rows_padded: usize,
    num_cols_padded: usize,
    r_x: &[FieldElement<F>],
) -> Result<DenseMultilinearPolynomial<F>, SpartanError>
where
    F::BaseType: Send + Sync,
{
    let eq_weights = eq_evals(r_x, num_rows_padded);
    let mut result = vec![FieldElement::zero(); num_cols_padded];

    for entry in &matrix.entries {
        if entry.row < num_rows_padded && entry.col < num_cols_padded {
            result[entry.col] = &result[entry.col] + &eq_weights[entry.row] * &entry.val;
        }
    }

    Ok(DenseMultilinearPolynomial::new(result))
}

/// Computes AZ(r_x) = sum_i eq(r_x, i) * <A[i], z>.
///
/// Iterates only non-zero entries of the sparse matrix.
pub fn mz_eval<F: IsField>(
    matrix: &SparseMatrix<F>,
    z: &[FieldElement<F>],
    num_rows_padded: usize,
    r_x: &[FieldElement<F>],
) -> FieldElement<F>
where
    F::BaseType: Send + Sync,
{
    let eq_weights = eq_evals(r_x, num_rows_padded);
    let mut result = FieldElement::zero();

    for entry in &matrix.entries {
        if entry.row < num_rows_padded {
            result += &eq_weights[entry.row] * &entry.val * &z[entry.col];
        }
    }

    result
}

/// Converts a witness index `i` into its boolean point for the witness MLE.
///
/// `DenseMultilinearPolynomial` uses MSB-first ordering: the bit representation of `i`
/// with `n` bits has the most significant bit as variable 0.
/// That is, `point[k] = (i >> (n-1-k)) & 1`.
///
/// Used to open the witness MLE at specific positions, e.g. to verify that
/// z_tilde(bits(i)) == public_inputs[i-1] for each public input index i.
pub fn index_to_multilinear_point<F: IsField>(i: usize, n: usize) -> Vec<FieldElement<F>>
where
    F::BaseType: Send + Sync,
{
    (0..n)
        .map(|k| {
            if (i >> (n - 1 - k)) & 1 == 1 {
                FieldElement::one()
            } else {
                FieldElement::zero()
            }
        })
        .collect()
}

/// Computes a matrix MLE at (r_x, r_y).
///
/// Computes sum_{i,j} M[i][j] * eq(r_x, binary(i)) * eq(r_y, binary(j))
///
/// Iterates only non-zero entries of the sparse matrix.
pub fn matrix_mle_eval<F: IsField>(
    matrix: &SparseMatrix<F>,
    num_rows_padded: usize,
    num_cols_padded: usize,
    r_x: &[FieldElement<F>],
    r_y: &[FieldElement<F>],
) -> FieldElement<F>
where
    F::BaseType: Send + Sync,
{
    let eq_x = eq_evals(r_x, num_rows_padded);
    let eq_y = eq_evals(r_y, num_cols_padded);
    let mut result = FieldElement::zero();

    for entry in &matrix.entries {
        if entry.row < num_rows_padded && entry.col < num_cols_padded {
            result += &eq_x[entry.row] * &eq_y[entry.col] * &entry.val;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn test_encode_witness_evaluations() {
        let z = vec![FE::from(1), FE::from(7), FE::from(3), FE::from(5)];
        let z_mle = encode_witness(&z);

        assert_eq!(z_mle.num_vars(), 2);
        assert_eq!(
            z_mle.evaluate(vec![FE::zero(), FE::zero()]).unwrap(),
            FE::from(1)
        );
        assert_eq!(
            z_mle.evaluate(vec![FE::zero(), FE::one()]).unwrap(),
            FE::from(7)
        );
        assert_eq!(
            z_mle.evaluate(vec![FE::one(), FE::zero()]).unwrap(),
            FE::from(3)
        );
        assert_eq!(
            z_mle.evaluate(vec![FE::one(), FE::one()]).unwrap(),
            FE::from(5)
        );
    }

    #[test]
    fn test_encode_witness_padding() {
        let z = vec![FE::from(1), FE::from(2), FE::from(3)];
        let z_mle = encode_witness(&z);
        assert_eq!(z_mle.num_vars(), 2);
        assert_eq!(
            z_mle.evaluate(vec![FE::one(), FE::one()]).unwrap(),
            FE::zero()
        );
    }

    #[test]
    fn test_eq_poly_two_vars() {
        let a = FE::from(3);
        let b = FE::from(5);
        let tau = vec![a, b];
        let eq = eq_poly(&tau);

        assert_eq!(eq.num_vars(), 2);
        let vals = eq.evals();

        let one = FE::one();
        let one_minus_a = one - a;
        let one_minus_b = one - b;

        assert_eq!(vals[0], one_minus_a * one_minus_b);
        assert_eq!(vals[1], one_minus_a * b);
        assert_eq!(vals[2], a * one_minus_b);
        assert_eq!(vals[3], a * b);
    }

    #[test]
    fn test_eq_poly_at_boolean_point() {
        let tau = vec![FE::one(), FE::zero()];
        let eq = eq_poly(&tau);
        let val = eq.evaluate(tau.clone()).unwrap();
        assert_eq!(val, FE::one());

        let other = vec![FE::zero(), FE::zero()];
        let val_other = eq.evaluate(other).unwrap();
        assert_eq!(val_other, FE::zero());
    }

    #[test]
    fn test_matrix_vector_product_mle() {
        let zero = FE::zero();
        let one = FE::one();
        let two = FE::from(2u64);

        let a_dense = vec![vec![one, zero], vec![zero, two]];
        let a = SparseMatrix::from_dense(&a_dense);

        let r_x = vec![FE::zero()];
        let mz = matrix_vector_product_mle(&a, 2, 2, &r_x).unwrap();
        assert_eq!(mz.evaluate(vec![FE::zero()]).unwrap(), one);
        assert_eq!(mz.evaluate(vec![FE::one()]).unwrap(), zero);

        let r_x1 = vec![FE::one()];
        let mz1 = matrix_vector_product_mle(&a, 2, 2, &r_x1).unwrap();
        assert_eq!(mz1.evaluate(vec![FE::zero()]).unwrap(), zero);
        assert_eq!(mz1.evaluate(vec![FE::one()]).unwrap(), two);
    }

    #[test]
    fn test_mz_eval_correctness() {
        let zero = FE::zero();
        let one = FE::one();
        let two = FE::from(2u64);

        let a_dense = vec![vec![one, zero], vec![zero, two]];
        let a = SparseMatrix::from_dense(&a_dense);
        let z = vec![FE::from(3u64), FE::from(5u64)];

        let r_x = vec![FE::zero()];
        let val = mz_eval(&a, &z, 2, &r_x);
        assert_eq!(val, FE::from(3u64));

        let r_x1 = vec![FE::one()];
        let val1 = mz_eval(&a, &z, 2, &r_x1);
        assert_eq!(val1, FE::from(10u64));
    }
}
