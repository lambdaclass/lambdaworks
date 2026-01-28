//! Sudan's list decoding algorithm for Reed-Solomon codes.
//!
//! # Overview
//!
//! Sudan's algorithm (1997) was the first polynomial-time list decoding
//! algorithm for RS codes that could correct beyond the unique decoding radius.
//!
//! # Decoding Radius
//!
//! Sudan's algorithm can correct up to `t` errors where:
//!
//! ```text
//! t < n - sqrt(2 * n * k)
//! ```
//!
//! Compare this to unique decoding (Berlekamp-Welch) which corrects up to:
//!
//! ```text
//! t ≤ (n - k) / 2
//! ```
//!
//! # Algorithm
//!
//! 1. **Interpolation**: Find a non-zero bivariate polynomial Q(x, y) of low
//!    (1, k-1)-weighted degree that passes through all received points (αᵢ, yᵢ).
//!
//! 2. **Root Finding**: Find all polynomials f(x) of degree < k such that
//!    Q(x, f(x)) = 0. These are candidates for the original message.
//!
//! # Simplification
//!
//! The key insight is that for Sudan's algorithm (multiplicity 1), we can use
//! Q(x, y) = A(x) + B(x)·y (linear in y), which simplifies the root finding
//! step to polynomial division.

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::Polynomial;

use crate::polynomial_utils::{find_polynomial_roots, BivariatePolynomial};
use crate::reed_solomon::ReedSolomonCode;

/// Result of Sudan's list decoding.
#[derive(Debug, Clone)]
pub struct SudanListDecodingResult<F: IsField> {
    /// All polynomials f(x) that are consistent with the received word
    /// within the decoding radius
    pub candidates: Vec<Polynomial<FieldElement<F>>>,
    /// The interpolating polynomial Q(x, y) used
    pub interpolation_polynomial: BivariatePolynomial<F>,
    /// Number of errors the algorithm was configured to handle
    pub error_bound: usize,
}

/// Computes the maximum number of errors Sudan's algorithm can correct.
///
/// Returns the largest integer t such that t < n - sqrt(2nk).
pub fn sudan_decoding_radius(n: usize, k: usize) -> usize {
    let threshold = (n as f64) - (2.0 * (n as f64) * (k as f64)).sqrt();
    if threshold <= 0.0 {
        0
    } else {
        (threshold - 0.001).floor() as usize // -0.001 to handle floating point edge cases
    }
}

/// List decodes a received word using Sudan's algorithm.
///
/// # Arguments
///
/// * `code` - The Reed-Solomon code
/// * `received` - The received word (possibly with errors)
///
/// # Returns
///
/// A list of all polynomials of degree < k that agree with the received word
/// on at least `n - t` positions, where `t` is the Sudan decoding radius.
///
/// # Algorithm Details
///
/// Sudan's algorithm finds Q(x, y) = A(x) + B(x)·y such that:
/// - Q(αᵢ, yᵢ) = 0 for all i ∈ [0, n)
/// - deg(A) ≤ D, deg(B) ≤ D - k + 1 for appropriate D
///
/// The parameter D is chosen to ensure the linear system has a non-trivial solution.
pub fn sudan_list_decode<F: IsField + Clone>(
    code: &ReedSolomonCode<F>,
    received: &[FieldElement<F>],
) -> SudanListDecodingResult<F> {
    let n = code.code_length();
    let k = code.dimension();
    let domain = code.domain();

    assert_eq!(
        received.len(),
        n,
        "Received word length must equal code length"
    );

    // Compute the appropriate degree bound D
    // We need: (D + 1) + (D - k + 2) > n
    // i.e., 2D - k + 3 > n
    // i.e., D > (n + k - 3) / 2
    let d = ((n + k) / 2) + 1;

    // Build the interpolation polynomial Q(x, y) = A(x) + B(x)·y
    // where deg(A) ≤ D and deg(B) ≤ D - k + 1
    let q = interpolate_sudan(domain, received, d, k);

    // Find all polynomial roots
    let candidates = find_polynomial_roots(&q, k);

    // Filter candidates to those that actually agree with received on enough positions
    let error_bound = sudan_decoding_radius(n, k);
    let agreement_threshold = n - error_bound;

    let valid_candidates: Vec<_> = candidates
        .into_iter()
        .filter(|f| {
            let agreement = domain
                .iter()
                .zip(received.iter())
                .filter(|(alpha, y)| f.evaluate(alpha) == **y)
                .count();
            agreement >= agreement_threshold
        })
        .collect();

    SudanListDecodingResult {
        candidates: valid_candidates,
        interpolation_polynomial: q,
        error_bound,
    }
}

/// Performs the interpolation step of Sudan's algorithm.
///
/// Finds Q(x, y) = A(x) + B(x)·y passing through all points (αᵢ, yᵢ).
fn interpolate_sudan<F: IsField + Clone>(
    domain: &[FieldElement<F>],
    received: &[FieldElement<F>],
    d: usize,
    k: usize,
) -> BivariatePolynomial<F> {
    let n = domain.len();

    // Degree bounds
    let deg_a = d; // deg(A) ≤ D
    let deg_b = if d >= k - 1 { d - k + 1 } else { 0 }; // deg(B) ≤ D - k + 1

    let num_a_coeffs = deg_a + 1;
    let num_b_coeffs = deg_b + 1;
    let num_unknowns = num_a_coeffs + num_b_coeffs;

    // Build the linear system
    // Q(αᵢ, yᵢ) = A(αᵢ) + B(αᵢ)·yᵢ = 0
    // Σⱼ aⱼ αᵢʲ + yᵢ Σⱼ bⱼ αᵢʲ = 0

    let mut matrix: Vec<Vec<FieldElement<F>>> = Vec::with_capacity(n);

    for i in 0..n {
        let alpha = &domain[i];
        let y = &received[i];

        let mut row = Vec::with_capacity(num_unknowns);

        // Coefficients for a_0, a_1, ..., a_{deg_a}
        let mut alpha_power = FieldElement::<F>::one();
        for _ in 0..num_a_coeffs {
            row.push(alpha_power.clone());
            alpha_power = &alpha_power * alpha;
        }

        // Coefficients for b_0, b_1, ..., b_{deg_b}: multiply by yᵢ
        let mut alpha_power = FieldElement::<F>::one();
        for _ in 0..num_b_coeffs {
            row.push(y * &alpha_power);
            alpha_power = &alpha_power * alpha;
        }

        matrix.push(row);
    }

    // Find a non-trivial solution to the homogeneous system
    let solution = find_kernel_vector(&matrix, num_unknowns);

    // Extract A(x) and B(x) from the solution
    let a_coeffs: Vec<FieldElement<F>> = solution[0..num_a_coeffs].to_vec();
    let b_coeffs: Vec<FieldElement<F>> = solution[num_a_coeffs..].to_vec();

    let a = Polynomial::new(&a_coeffs);
    let b = Polynomial::new(&b_coeffs);

    BivariatePolynomial::new(vec![a, b])
}

/// Finds a non-trivial vector in the kernel of a matrix.
///
/// Given an m×n matrix A (with m < n), finds a non-zero vector x such that Ax = 0.
fn find_kernel_vector<F: IsField + Clone>(
    matrix: &[Vec<FieldElement<F>>],
    num_cols: usize,
) -> Vec<FieldElement<F>> {
    let m = matrix.len();
    let n = num_cols;

    // Make a copy of the matrix
    let mut mat: Vec<Vec<FieldElement<F>>> = matrix.to_vec();

    // Pad rows if needed
    for row in mat.iter_mut() {
        while row.len() < n {
            row.push(FieldElement::<F>::zero());
        }
    }

    // Gaussian elimination to find row echelon form
    let mut pivot_cols = Vec::new();
    let mut pivot_row = 0;

    for col in 0..n {
        if pivot_row >= m {
            break;
        }

        // Find non-zero entry in this column
        let mut found = false;
        for row in pivot_row..m {
            if mat[row][col] != FieldElement::<F>::zero() {
                mat.swap(pivot_row, row);
                found = true;
                break;
            }
        }

        if !found {
            continue;
        }

        pivot_cols.push(col);

        // Scale pivot row
        let pivot = mat[pivot_row][col].clone();
        let pivot_inv = pivot.inv().unwrap_or_else(|_| FieldElement::<F>::one());
        for elem in &mut mat[pivot_row][col..n] {
            *elem = &*elem * &pivot_inv;
        }

        // Eliminate other rows
        let pivot_row_data: Vec<_> = mat[pivot_row][col..n].to_vec();
        #[allow(clippy::needless_range_loop)]
        for row in 0..m {
            if row != pivot_row && mat[row][col] != FieldElement::<F>::zero() {
                let factor = mat[row][col].clone();
                for (row_elem, pivot_elem) in mat[row][col..n].iter_mut().zip(&pivot_row_data) {
                    let sub = &factor * pivot_elem;
                    *row_elem = &*row_elem - &sub;
                }
            }
        }

        pivot_row += 1;
    }

    // Find a free variable (column that's not a pivot)
    let mut free_col = None;
    for col in 0..n {
        if !pivot_cols.contains(&col) {
            free_col = Some(col);
            break;
        }
    }

    // Build the kernel vector
    let mut kernel = vec![FieldElement::<F>::zero(); n];

    if let Some(fc) = free_col {
        kernel[fc] = FieldElement::<F>::one();

        // Back-substitute to find other components
        for (row, &pc) in pivot_cols.iter().enumerate() {
            if row < m {
                kernel[pc] = -mat[row][fc].clone();
            }
        }
    } else {
        // No free variable found, use last column as free
        // This shouldn't happen if the system is underdetermined
        kernel[n - 1] = FieldElement::<F>::one();
    }

    kernel
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{agreement, introduce_errors_at_positions};
    use crate::Babybear31PrimeField;

    type FE = FieldElement<Babybear31PrimeField>;

    #[test]
    fn test_sudan_decoding_radius() {
        // RS[16, 8]
        // Unique decoding: floor((16-8)/2) = 4
        // Sudan: floor(16 - sqrt(2*16*8)) = floor(16 - sqrt(256)) = floor(16 - 16) = 0
        // Actually sqrt(256) = 16, so radius is floor(16 - 16) - 1 = -1, capped at 0
        // This shows Sudan doesn't help much for rate 1/2 codes
        let radius = sudan_decoding_radius(16, 8);
        assert!(radius <= 4); // Should be comparable to unique decoding

        // RS[32, 8] - lower rate code
        // Unique decoding: floor((32-8)/2) = 12
        // Sudan: floor(32 - sqrt(2*32*8)) = floor(32 - sqrt(512)) ≈ floor(32 - 22.6) = 9
        let radius32 = sudan_decoding_radius(32, 8);
        assert!(radius32 >= 8); // Should be better than 0
    }

    #[test]
    fn test_sudan_no_errors() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::new(16, 8);
        let message: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let codeword = code.encode(&message);

        let result = sudan_list_decode(&code, &codeword);

        // Should find exactly the original polynomial
        assert!(!result.candidates.is_empty());
        let original_poly = Polynomial::new(&message);
        assert!(
            result.candidates.contains(&original_poly),
            "Should contain original polynomial"
        );
    }

    #[test]
    fn test_sudan_few_errors() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::with_consecutive_domain(32, 8);
        let message: Vec<FE> = (0..8).map(|i| FE::from((i + 1) as u64)).collect();
        let codeword = code.encode(&message);

        // Introduce a few errors
        let corrupted = introduce_errors_at_positions(&codeword, &[3, 10, 20]);

        let result = sudan_list_decode(&code, &corrupted);

        let original_poly = Polynomial::new(&message);
        assert!(
            result.candidates.contains(&original_poly),
            "List should contain original polynomial"
        );
    }

    #[test]
    fn test_interpolation_passes_through_points() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::with_consecutive_domain(8, 4);
        let message: Vec<FE> = (0..4).map(|i| FE::from(i as u64)).collect();
        let codeword = code.encode(&message);
        let domain = code.domain();

        // Just test that interpolation works
        let d = 4;
        let q = interpolate_sudan(domain, &codeword, d, 4);

        // Q(αᵢ, yᵢ) should be zero for all i
        for (alpha, y) in domain.iter().zip(codeword.iter()) {
            let eval = q.evaluate(alpha, y);
            assert_eq!(
                eval,
                FE::zero(),
                "Q should pass through all received points"
            );
        }
    }

    #[test]
    fn test_list_contains_valid_codewords() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::with_consecutive_domain(16, 4);
        let message: Vec<FE> = vec![
            FE::from(1u64),
            FE::from(2u64),
            FE::from(3u64),
            FE::from(0u64),
        ];
        let codeword = code.encode(&message);

        let result = sudan_list_decode(&code, &codeword);

        // All candidates should have high agreement with the received word
        let domain = code.domain();
        for candidate in &result.candidates {
            let agr = agreement(&codeword, domain, candidate);
            assert!(
                agr >= code.code_length() - result.error_bound,
                "Candidate should have high agreement"
            );
        }
    }
}
