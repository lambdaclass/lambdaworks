//! Berlekamp-Welch algorithm for unique decoding of Reed-Solomon codes.
//!
//! # Overview
//!
//! The Berlekamp-Welch algorithm can correct up to t = floor((n-k)/2) errors
//! by finding two polynomials E(x) and N(x) such that:
//!
//! ```text
//! E(αᵢ) · yᵢ = N(αᵢ)  for all i ∈ [0, n)
//! ```
//!
//! where:
//! - E(x) is the error locator polynomial of degree t (with leading coeff 1)
//! - N(x) = E(x) · P(x) where P(x) is the original message polynomial
//! - αᵢ are the evaluation domain points
//! - yᵢ are the received (possibly corrupted) values
//!
//! # Algorithm
//!
//! 1. Set up a linear system from the key equation E(αᵢ) · yᵢ = N(αᵢ)
//! 2. Solve the system using Gaussian elimination
//! 3. Recover P(x) = N(x) / E(x)
//!
//! # Complexity
//!
//! - Time: O(n³) for naive Gaussian elimination
//! - Can be improved to O(n² log n) using structured linear algebra

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::Polynomial;

use crate::reed_solomon::ReedSolomonCode;

/// Errors that can occur during decoding.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodingError {
    /// Too many errors to correct
    TooManyErrors {
        found: usize,
        max_correctable: usize,
    },
    /// Linear system has no solution (inconsistent)
    NoSolution,
    /// Error locator doesn't divide N(x)
    DivisionFailed,
    /// Invalid input
    InvalidInput(String),
}

impl std::fmt::Display for DecodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodingError::TooManyErrors {
                found,
                max_correctable,
            } => {
                write!(
                    f,
                    "Too many errors: found {}, max correctable is {}",
                    found, max_correctable
                )
            }
            DecodingError::NoSolution => write!(f, "Linear system has no solution"),
            DecodingError::DivisionFailed => write!(f, "E(x) does not divide N(x)"),
            DecodingError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for DecodingError {}

/// Result of Berlekamp-Welch decoding.
#[derive(Debug, Clone)]
pub struct DecodingResult<F: IsField> {
    /// The recovered message polynomial
    pub polynomial: Polynomial<FieldElement<F>>,
    /// The error locator polynomial (roots are error positions)
    pub error_locator: Polynomial<FieldElement<F>>,
    /// Number of errors corrected
    pub num_errors: usize,
}

/// Decodes a received word using the Berlekamp-Welch algorithm.
///
/// # Arguments
///
/// * `code` - The Reed-Solomon code
/// * `received` - The received word (possibly with errors)
/// * `max_errors` - Maximum number of errors to try correcting (default: unique decoding radius)
///
/// # Returns
///
/// The decoded polynomial if successful, or an error if decoding fails.
///
/// # Example
///
/// ```
/// use reed_solomon_codes::reed_solomon::ReedSolomonCode;
/// use reed_solomon_codes::berlekamp_welch::decode;
/// use reed_solomon_codes::distance::introduce_errors_at_positions;
/// use reed_solomon_codes::FE;
///
/// let code = ReedSolomonCode::new(16, 8);
/// let message: Vec<FE> = (0..8).map(|i| FE::from(i as u64)).collect();
/// let codeword = code.encode(&message);
///
/// // Introduce 3 errors (max correctable is 4)
/// let corrupted = introduce_errors_at_positions(&codeword, &[2, 5, 11]);
///
/// let result = decode(&code, &corrupted, None).unwrap();
/// assert_eq!(result.polynomial.coefficients(), &message);
/// ```
pub fn decode<F: IsField + Clone>(
    code: &ReedSolomonCode<F>,
    received: &[FieldElement<F>],
    max_errors: Option<usize>,
) -> Result<DecodingResult<F>, DecodingError> {
    let n = code.code_length();

    if received.len() != n {
        return Err(DecodingError::InvalidInput(format!(
            "Received word length {} doesn't match code length {}",
            received.len(),
            n
        )));
    }

    let t = max_errors.unwrap_or_else(|| code.unique_decoding_radius());

    // Try decoding starting with 0 errors, then 1, etc. up to t
    // This ensures we find the minimum error count that produces a valid decoding
    for num_errors in 0..=t {
        if let Ok(result) = decode_with_error_count(code, received, num_errors) {
            return Ok(result);
        }
    }

    Err(DecodingError::TooManyErrors {
        found: t + 1,
        max_correctable: t,
    })
}

/// Attempts to decode assuming exactly `num_errors` errors.
fn decode_with_error_count<F: IsField + Clone>(
    code: &ReedSolomonCode<F>,
    received: &[FieldElement<F>],
    num_errors: usize,
) -> Result<DecodingResult<F>, DecodingError> {
    let n = code.code_length();
    let k = code.dimension();
    let domain = code.domain();

    // Special case: no errors
    if num_errors == 0 {
        // Just interpolate the received word
        let poly =
            Polynomial::interpolate(domain, received).map_err(|_| DecodingError::NoSolution)?;

        if poly.degree() >= k {
            return Err(DecodingError::NoSolution);
        }

        return Ok(DecodingResult {
            polynomial: poly,
            error_locator: Polynomial::new(&[FieldElement::<F>::one()]),
            num_errors: 0,
        });
    }

    // Set up the Berlekamp-Welch linear system
    // We're looking for:
    // - E(x) of degree t with leading coefficient 1: E(x) = x^t + e_{t-1}x^{t-1} + ... + e_0
    // - N(x) of degree t + k - 1: N(x) = n_{t+k-1}x^{t+k-1} + ... + n_0
    //
    // The key equation E(αᵢ) · yᵢ = N(αᵢ) gives us n linear equations.
    // Unknowns: e_0, ..., e_{t-1} (t unknowns) and n_0, ..., n_{t+k-1} (t+k unknowns)
    // Total unknowns: 2t + k
    // We have n equations, need n ≥ 2t + k, i.e., t ≤ (n-k)/2

    let t = num_errors;
    let num_n_coeffs = t + k; // degree of N(x) is at most t + k - 1
    let num_unknowns = t + num_n_coeffs; // e_0..e_{t-1} and n_0..n_{t+k-1}

    // Build the augmented matrix [A | b] for the system Ax = b
    // Each row corresponds to one evaluation point αᵢ
    // The equation is: (Σⱼ eⱼ αᵢʲ) · yᵢ + αᵢᵗ · yᵢ = Σⱼ nⱼ αᵢʲ
    // Rearranging: Σⱼ (yᵢ αᵢʲ) eⱼ - Σⱼ αᵢʲ nⱼ = -αᵢᵗ · yᵢ

    let mut matrix: Vec<Vec<FieldElement<F>>> = Vec::with_capacity(n);
    let mut rhs: Vec<FieldElement<F>> = Vec::with_capacity(n);

    for i in 0..n {
        let alpha = &domain[i];
        let y = &received[i];

        let mut row = Vec::with_capacity(num_unknowns);

        // Coefficients for e_0, ..., e_{t-1}: y · α^j for j = 0..t-1
        let mut alpha_power = FieldElement::<F>::one();
        for _ in 0..t {
            row.push(y * &alpha_power);
            alpha_power = &alpha_power * alpha;
        }

        // Coefficients for n_0, ..., n_{t+k-1}: -α^j for j = 0..t+k-1
        let mut alpha_power = FieldElement::<F>::one();
        for _ in 0..num_n_coeffs {
            row.push(-&alpha_power);
            alpha_power = &alpha_power * alpha;
        }

        matrix.push(row);

        // RHS: -y · α^t
        let mut alpha_t = FieldElement::<F>::one();
        for _ in 0..t {
            alpha_t = &alpha_t * alpha;
        }
        rhs.push(-(y * &alpha_t));
    }

    // Solve the linear system using Gaussian elimination
    let solution = gaussian_elimination(&mut matrix, &mut rhs)?;

    // Extract E(x) and N(x) from the solution
    // E(x) = x^t + e_{t-1}x^{t-1} + ... + e_0
    let mut e_coeffs: Vec<FieldElement<F>> = solution[0..t].to_vec();
    e_coeffs.push(FieldElement::<F>::one()); // leading coefficient is 1

    // N(x) = n_{t+k-1}x^{t+k-1} + ... + n_0
    let n_coeffs: Vec<FieldElement<F>> = solution[t..].to_vec();

    let error_locator = Polynomial::new(&e_coeffs);
    let n_poly = Polynomial::new(&n_coeffs);

    // Compute P(x) = N(x) / E(x)
    let Ok((quotient, remainder)) = n_poly.clone().long_division_with_remainder(&error_locator)
    else {
        return Err(DecodingError::DivisionFailed);
    };

    // Check that E(x) divides N(x) exactly
    if remainder != Polynomial::zero() {
        return Err(DecodingError::DivisionFailed);
    }

    // Verify the polynomial has the right degree
    if quotient.degree() >= k {
        return Err(DecodingError::NoSolution);
    }

    Ok(DecodingResult {
        polynomial: quotient,
        error_locator,
        num_errors: t,
    })
}

/// Solves a linear system using Gaussian elimination with partial pivoting.
///
/// Solves Ax = b where A is an n×m matrix and b is an n-vector.
/// Returns the solution x or an error if no solution exists.
fn gaussian_elimination<F: IsField + Clone>(
    matrix: &mut [Vec<FieldElement<F>>],
    rhs: &mut [FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, DecodingError> {
    let n = matrix.len(); // number of equations
    if n == 0 {
        return Err(DecodingError::NoSolution);
    }
    let m = matrix[0].len(); // number of unknowns

    if n < m {
        return Err(DecodingError::NoSolution);
    }

    // Forward elimination
    let mut pivot_col = 0;
    for pivot_row in 0..m {
        if pivot_col >= m {
            break;
        }

        // Find pivot (partial pivoting)
        let mut max_row = pivot_row;
        for (row, row_data) in matrix
            .iter()
            .enumerate()
            .skip(pivot_row + 1)
            .take(n - pivot_row - 1)
        {
            if row_data[pivot_col] != FieldElement::<F>::zero() {
                max_row = row;
                break;
            }
        }

        // If this column is all zeros, try next column
        if matrix[max_row][pivot_col] == FieldElement::<F>::zero() {
            pivot_col += 1;
            continue;
        }

        // Swap rows
        if max_row != pivot_row {
            matrix.swap(pivot_row, max_row);
            rhs.swap(pivot_row, max_row);
        }

        // Scale pivot row
        let pivot = matrix[pivot_row][pivot_col].clone();
        let pivot_inv = pivot.inv().map_err(|_| DecodingError::NoSolution)?;

        for elem in &mut matrix[pivot_row][pivot_col..m] {
            *elem = &*elem * &pivot_inv;
        }
        rhs[pivot_row] = &rhs[pivot_row] * &pivot_inv;

        // Eliminate column
        let pivot_row_data: Vec<_> = matrix[pivot_row][pivot_col..m].to_vec();
        for row in 0..n {
            if row != pivot_row && matrix[row][pivot_col] != FieldElement::<F>::zero() {
                let factor = matrix[row][pivot_col].clone();
                for (row_elem, pivot_elem) in
                    matrix[row][pivot_col..m].iter_mut().zip(&pivot_row_data)
                {
                    let sub = &factor * pivot_elem;
                    *row_elem = &*row_elem - &sub;
                }
                let sub = &factor * &rhs[pivot_row];
                rhs[row] = &rhs[row] - &sub;
            }
        }

        pivot_col += 1;
    }

    // Check for inconsistency (non-zero RHS with zero row)
    if rhs[m..n]
        .iter()
        .any(|elem| *elem != FieldElement::<F>::zero())
    {
        return Err(DecodingError::NoSolution);
    }

    // Back substitution (matrix should be in reduced row echelon form)
    let mut solution = vec![FieldElement::<F>::zero(); m];
    for row in (0..m.min(n)).rev() {
        // Find the pivot column for this row
        let pivot_col = matrix[row]
            .iter()
            .position(|elem| *elem == FieldElement::<F>::one());

        if let Some(col) = pivot_col {
            solution[col] = rhs[row].clone();
            for j in (col + 1)..m {
                let sub = &matrix[row][j] * &solution[j];
                solution[col] = &solution[col] - &sub;
            }
        }
    }

    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{hamming_distance, introduce_errors_at_positions};
    use crate::Babybear31PrimeField;

    type FE = FieldElement<Babybear31PrimeField>;

    #[test]
    fn test_decode_no_errors() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::new(16, 8);
        let message: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let codeword = code.encode(&message);

        let result = decode(&code, &codeword, None).unwrap();

        assert_eq!(result.polynomial.coefficients(), &message);
        assert_eq!(result.num_errors, 0);
    }

    #[test]
    fn test_decode_one_error() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::new(16, 8);
        let message: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let codeword = code.encode(&message);

        // Introduce 1 error at position 5
        let corrupted = introduce_errors_at_positions(&codeword, &[5]);

        assert_eq!(hamming_distance(&codeword, &corrupted), 1);

        let result = decode(&code, &corrupted, None).unwrap();

        assert_eq!(result.polynomial.coefficients(), &message);
    }

    #[test]
    fn test_decode_max_errors() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::new(16, 8);
        let t = code.unique_decoding_radius(); // 4 errors

        let message: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let codeword = code.encode(&message);

        // Introduce exactly t errors
        let error_positions: Vec<usize> = (0..t).collect();
        let corrupted = introduce_errors_at_positions(&codeword, &error_positions);

        assert_eq!(hamming_distance(&codeword, &corrupted), t);

        let result = decode(&code, &corrupted, None).unwrap();

        assert_eq!(result.polynomial.coefficients(), &message);
    }

    #[test]
    fn test_decode_consecutive_domain() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::with_consecutive_domain(16, 8);
        let message: Vec<FE> = (0..8).map(|i| FE::from((i * 3 + 1) as u64)).collect();
        let codeword = code.encode(&message);

        // Introduce 2 errors
        let corrupted = introduce_errors_at_positions(&codeword, &[3, 10]);

        let result = decode(&code, &corrupted, None).unwrap();

        assert_eq!(result.polynomial.coefficients(), &message);
    }

    #[test]
    fn test_decode_various_error_counts() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::new(32, 16);
        let t = code.unique_decoding_radius(); // 8 errors

        let message: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();
        let codeword = code.encode(&message);

        for num_errors in 0..=t {
            let error_positions: Vec<usize> = (0..num_errors).map(|i| i * 3).collect();
            let corrupted = introduce_errors_at_positions(&codeword, &error_positions);

            let result = decode(&code, &corrupted, None)
                .unwrap_or_else(|_| panic!("Should decode with {} errors", num_errors));

            assert_eq!(
                result.polynomial.coefficients(),
                &message,
                "Failed with {} errors",
                num_errors
            );
        }
    }

    #[test]
    fn test_error_locator_roots() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::with_consecutive_domain(16, 8);
        let message: Vec<FE> = (0..8).map(|i| FE::from(i as u64)).collect();
        let codeword = code.encode(&message);

        let error_positions = vec![2, 7, 11];
        let corrupted = introduce_errors_at_positions(&codeword, &error_positions);

        let result = decode(&code, &corrupted, None).unwrap();
        let domain = code.domain();

        // The error locator should have roots at the error positions
        for &pos in &error_positions {
            let eval = result.error_locator.evaluate(&domain[pos]);
            assert_eq!(
                eval,
                FE::zero(),
                "Error locator should be zero at error position {}",
                pos
            );
        }
    }
}
