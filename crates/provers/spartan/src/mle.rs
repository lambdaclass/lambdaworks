//! Multilinear extension (MLE) utilities for Spartan.
//!
//! Provides functions to encode R1CS matrices and witness vectors as MLEs,
//! and to compute key operations like eq_poly and matrix-vector product MLEs.

use lambdaworks_math::field::{element::FieldElement, traits::IsField};
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

use crate::errors::SpartanError;

/// Returns the smallest power of 2 that is >= n.
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        1
    } else if n.is_power_of_two() {
        n
    } else {
        n.next_power_of_two()
    }
}

/// Encodes a witness vector z as a dense multilinear polynomial (MLE).
///
/// The witness z is padded to the next power of 2 with zeros.
/// The resulting MLE has log₂(padded_len) variables.
///
/// The evaluation at boolean point (b₀, b₁, ..., b_{k-1}) where bi ∈ {0,1}
/// equals z[i] where i is the integer with binary representation b₀b₁...b_{k-1}.
pub fn encode_witness<F: IsField>(z: &[FieldElement<F>]) -> DenseMultilinearPolynomial<F>
where
    F::BaseType: Send + Sync,
{
    let n = z.len();
    // Ensure at least 2 evaluations so the MLE has at least 1 variable (required by sumcheck)
    let padded_len = next_power_of_two(n).max(2);
    let mut evals = z.to_vec();
    evals.resize(padded_len, FieldElement::zero());
    DenseMultilinearPolynomial::new(evals)
}

/// Encodes an m × n matrix M as a sparse-then-dense multilinear polynomial.
///
/// The MLE has log₂(num_rows_padded) + log₂(num_cols_padded) variables.
/// Entry M[i][j] maps to index (i * num_cols_padded + j) in the evaluation vector.
///
/// Returns a DenseMultilinearPolynomial for compatibility with the sumcheck crate.
pub fn encode_matrix_dense<F: IsField>(
    matrix: &[Vec<FieldElement<F>>],
    num_rows_padded: usize,
    num_cols_padded: usize,
) -> Result<DenseMultilinearPolynomial<F>, SpartanError>
where
    F::BaseType: Send + Sync,
{
    if !num_rows_padded.is_power_of_two() || !num_cols_padded.is_power_of_two() {
        return Err(SpartanError::MleError(
            "Matrix dimensions must be powers of two for MLE encoding".to_string(),
        ));
    }

    let total = num_rows_padded * num_cols_padded;
    let mut evals = vec![FieldElement::zero(); total];

    for (i, row) in matrix.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            if i < num_rows_padded && j < num_cols_padded {
                evals[i * num_cols_padded + j] = val.clone();
            }
        }
    }

    Ok(DenseMultilinearPolynomial::new(evals))
}

/// Computes eq(τ, ·) multilinear polynomial evaluations on {0,1}^s.
///
/// The equality polynomial is:
/// eq(τ, x) = ∏ᵢ (τᵢ·xᵢ + (1−τᵢ)·(1−xᵢ))
///
/// This evaluates to 1 when x = τ (over the boolean hypercube) and 0 otherwise.
///
/// The evaluations are computed efficiently using the "expansion" algorithm:
/// Start with [1], then for each τᵢ expand: new[2k] = old[k] * (1-τᵢ), new[2k+1] = old[k] * τᵢ
pub fn eq_poly<F: IsField>(tau: &[FieldElement<F>]) -> DenseMultilinearPolynomial<F>
where
    F::BaseType: Send + Sync,
{
    let s = tau.len();
    let n = 1 << s;
    let mut evals = vec![FieldElement::one(); 1];

    for tau_i in tau.iter() {
        let half = evals.len();
        let mut new_evals = vec![FieldElement::zero(); half * 2];
        for k in 0..half {
            let one_minus_tau_i = FieldElement::<F>::one() - tau_i;
            new_evals[2 * k] = &evals[k] * &one_minus_tau_i;
            new_evals[2 * k + 1] = &evals[k] * tau_i;
        }
        evals = new_evals;
    }

    debug_assert_eq!(evals.len(), n);
    DenseMultilinearPolynomial::new(evals)
}

/// Computes eq(r_x, binary(i)) for all i ∈ [0, num_rows).
///
/// This is the Lagrange basis evaluation of the equality polynomial at a fixed point r_x.
/// Returns a vector of size num_rows_padded where entry i = ∏ⱼ (r_x[j]·b_j + (1−r_x[j])·(1−b_j))
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

/// Computes the matrix-vector product MLE: M̃Z(r_x) as a function of column index.
///
/// Given matrix M (m × n), witness MLE z_mle (over log n variables),
/// and a fixed point r_x (of length log m), computes:
///
///   MZ(r_x)[j] = ∑_{i=0}^{m_padded - 1} M[i][j] · eq(r_x, binary(i))
///
/// Returns a DenseMultilinearPolynomial in log(n_padded) variables
/// whose evaluation at boolean point binary(j) equals MZ(r_x)[j].
///
/// This represents the "row-collapsed" version of M with the row dimension
/// fixed at r_x via the equality polynomial.
pub fn matrix_vector_product_mle<F: IsField>(
    matrix: &[Vec<FieldElement<F>>],
    num_rows_padded: usize,
    num_cols_padded: usize,
    r_x: &[FieldElement<F>],
) -> Result<DenseMultilinearPolynomial<F>, SpartanError>
where
    F::BaseType: Send + Sync,
{
    // Compute eq(r_x, binary(i)) for all i
    let eq_weights = eq_evals(r_x, num_rows_padded);

    // For each column j, compute ∑_i M[i][j] * eq_weights[i]
    let mut result = vec![FieldElement::zero(); num_cols_padded];

    for (i, row) in matrix.iter().enumerate() {
        if i >= num_rows_padded {
            break;
        }
        let w = &eq_weights[i];
        for (j, val) in row.iter().enumerate() {
            if j < num_cols_padded {
                result[j] = &result[j] + w * val;
            }
        }
    }

    Ok(DenseMultilinearPolynomial::new(result))
}

/// Computes AZ(r_x) = ∑_j A_r_x[j] * z_mle(binary(j)) where A_r_x[j] = ∑_i A[i][j]*eq(r_x, i).
///
/// This is the evaluation of the matrix-vector product MLE at a scalar point,
/// i.e., the sum ∑_j (M̃Z)(r_x, binary(j)) * z_mle(binary(j)).
///
/// More simply: (Az)_x = ∑_j (∑_i A[i][j]*eq(r_x,i)) * z[j]
pub fn mz_eval<F: IsField>(
    matrix: &[Vec<FieldElement<F>>],
    z: &[FieldElement<F>],
    num_rows_padded: usize,
    r_x: &[FieldElement<F>],
) -> FieldElement<F>
where
    F::BaseType: Send + Sync,
{
    let eq_weights = eq_evals(r_x, num_rows_padded);

    let mut result = FieldElement::zero();
    for (i, row) in matrix.iter().enumerate() {
        if i >= num_rows_padded {
            break;
        }
        // Compute <A[i], z>
        let az_i: FieldElement<F> = row
            .iter()
            .zip(z.iter())
            .map(|(a_ij, z_j)| a_ij * z_j)
            .fold(FieldElement::zero(), |acc, x| acc + x);

        result += &eq_weights[i] * &az_i;
    }

    result
}

/// Converts a witness index `i` into its boolean evaluation point for the witness MLE.
///
/// `DenseMultilinearPolynomial` uses MSB-first ordering: the bit representation of `i`
/// with `n` bits has the most significant bit as variable 0.
/// That is, `point[k] = (i >> (n-1-k)) & 1`.
///
/// Used to open the witness MLE at specific positions, e.g. to verify that
/// z̃(bits(i)) == public_inputs[i-1] for each public input index i.
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

/// Evaluates a matrix MLE at (r_x, r_y).
///
/// Computes ∑_{i,j} M[i][j] · eq(r_x, binary(i)) · eq(r_y, binary(j))
pub fn matrix_mle_eval<F: IsField>(
    matrix: &[Vec<FieldElement<F>>],
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
    for (i, row) in matrix.iter().enumerate() {
        if i >= num_rows_padded {
            break;
        }
        for (j, val) in row.iter().enumerate() {
            if j < num_cols_padded {
                result += &eq_x[i] * &eq_y[j] * val;
            }
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
        // z = [1, 7, 3, 5], 2 variables, so:
        // z_mle(0,0) = 1, z_mle(0,1) = 7, z_mle(1,0) = 3, z_mle(1,1) = 5
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
        // Odd-length witness should be padded
        let z = vec![FE::from(1), FE::from(2), FE::from(3)];
        let z_mle = encode_witness(&z);
        // Padded to length 4, so 2 variables
        assert_eq!(z_mle.num_vars(), 2);
        assert_eq!(
            z_mle.evaluate(vec![FE::one(), FE::one()]).unwrap(),
            FE::zero() // padded zero
        );
    }

    #[test]
    fn test_eq_poly_two_vars() {
        // tau = [a, b]
        // eq_poly(tau) evaluations:
        // (0,0) -> (1-a)(1-b)
        // (0,1) -> (1-a)*b
        // (1,0) -> a*(1-b)
        // (1,1) -> a*b
        let a = FE::from(3);
        let b = FE::from(5);
        let tau = vec![a, b];
        let eq = eq_poly(&tau);

        assert_eq!(eq.num_vars(), 2);
        let evals = eq.evals();

        let one = FE::one();
        let one_minus_a = one - a;
        let one_minus_b = one - b;

        assert_eq!(evals[0], one_minus_a * one_minus_b); // (0,0)
        assert_eq!(evals[1], one_minus_a * b); // (0,1)
        assert_eq!(evals[2], a * one_minus_b); // (1,0)
        assert_eq!(evals[3], a * b); // (1,1)
    }

    #[test]
    fn test_eq_poly_at_boolean_point() {
        // eq(tau, tau) should equal 1 when tau is a boolean point
        let tau = vec![FE::one(), FE::zero()];
        let eq = eq_poly(&tau);
        let val = eq.evaluate(tau.clone()).unwrap();
        assert_eq!(val, FE::one());

        // eq(tau, other) should equal 0 for other boolean points
        let other = vec![FE::zero(), FE::zero()];
        let val_other = eq.evaluate(other).unwrap();
        assert_eq!(val_other, FE::zero());
    }

    #[test]
    fn test_matrix_vector_product_mle() {
        // A = [[1,0],[0,2]], z = [3,5], Az = [3, 10]
        // At r_x = [0] (selects row 0): should give A[0][·]*z[·] sum weighted by eq([0], binary(i))
        // eq([0], 0) = 1, eq([0], 1) = 0
        // So MZ([0])[j] = A[0][j] for j=0,1
        let zero = FE::zero();
        let one = FE::one();
        let two = FE::from(2u64);

        let a = vec![vec![one, zero], vec![zero, two]];

        // r_x = [0] selects row 0
        let r_x = vec![FE::zero()];
        let mz = matrix_vector_product_mle(&a, 2, 2, &r_x).unwrap();

        // MZ([0])[0] = A[0][0]*eq([0],0) + A[1][0]*eq([0],1) = 1*1 + 0*0 = 1
        // MZ([0])[1] = A[0][1]*eq([0],0) + A[1][1]*eq([0],1) = 0*1 + 2*0 = 0
        assert_eq!(mz.evaluate(vec![FE::zero()]).unwrap(), one); // col 0
        assert_eq!(mz.evaluate(vec![FE::one()]).unwrap(), zero); // col 1

        // r_x = [1] selects row 1
        let r_x1 = vec![FE::one()];
        let mz1 = matrix_vector_product_mle(&a, 2, 2, &r_x1).unwrap();
        // MZ([1])[0] = A[0][0]*eq([1],0) + A[1][0]*eq([1],1) = 1*0 + 0*1 = 0
        // MZ([1])[1] = A[0][1]*eq([1],0) + A[1][1]*eq([1],1) = 0*0 + 2*1 = 2
        assert_eq!(mz1.evaluate(vec![FE::zero()]).unwrap(), zero); // col 0
        assert_eq!(mz1.evaluate(vec![FE::one()]).unwrap(), two); // col 1
    }

    #[test]
    fn test_mz_eval_correctness() {
        // A = [[1,0],[0,2]], z = [3,5], Az = [3, 10]
        // At r_x = [0]: ∑_j MZ([0])[j] * z[j] should give the "row 0 contribution"
        // Actually mz_eval computes ∑_i eq(r_x, i) * <A[i], z>
        // = eq([0],0)*<A[0],z> + eq([0],1)*<A[1],z>
        // = 1*3 + 0*10 = 3
        let zero = FE::zero();
        let one = FE::one();
        let two = FE::from(2u64);

        let a = vec![vec![one, zero], vec![zero, two]];
        let z = vec![FE::from(3u64), FE::from(5u64)];

        let r_x = vec![FE::zero()];
        let val = mz_eval(&a, &z, 2, &r_x);
        assert_eq!(val, FE::from(3u64)); // eq([0],0)*<A[0],z> = 1*3 = 3

        let r_x1 = vec![FE::one()];
        let val1 = mz_eval(&a, &z, 2, &r_x1);
        assert_eq!(val1, FE::from(10u64)); // eq([1],1)*<A[1],z> = 1*10 = 10
    }
}
