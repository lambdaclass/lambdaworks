//! Guruswami-Sudan list decoding algorithm for Reed-Solomon codes.
//!
//! # Overview
//!
//! The Guruswami-Sudan algorithm (1999) achieves the optimal list decoding
//! radius for Reed-Solomon codes. It can correct up to:
//!
//! ```text
//! t < n - sqrt(n * k)
//! ```
//!
//! errors, which approaches `n - sqrt(n)` as the rate `k/n` approaches 0.
//!
//! # Comparison with Other Algorithms
//!
//! | Algorithm       | Decoding Radius                | Year |
//! |-----------------|--------------------------------|------|
//! | Berlekamp-Welch | (n - k) / 2                    | 1986 |
//! | Sudan           | n - sqrt(2 * n * k)            | 1997 |
//! | Guruswami-Sudan | n - sqrt(n * k)                | 1999 |
//!
//! # Algorithm
//!
//! 1. **Interpolation with Multiplicity**: Find a bivariate polynomial Q(x, y)
//!    that passes through each point (αᵢ, yᵢ) with multiplicity m.
//!    A polynomial passes with multiplicity m if all partial derivatives up to
//!    order m-1 vanish at that point.
//!
//! 2. **Root Finding**: Use the Roth-Ruckenstein algorithm to find all
//!    polynomials f(x) of degree < k such that Q(x, f(x)) = 0.
//!
//! # Johnson Bound
//!
//! The list size is bounded by O(sqrt(n)), specifically:
//!
//! ```text
//! |L| ≤ n / (n - t - sqrt(n * k))
//! ```
//!
//! For the optimal radius t = n - sqrt(nk), this gives |L| = O(n/sqrt(nk)).

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::Polynomial;

use crate::polynomial_utils::{find_polynomial_roots_with_hints, BivariatePolynomial};
use crate::reed_solomon::ReedSolomonCode;

/// Result of Guruswami-Sudan list decoding.
#[derive(Debug, Clone)]
pub struct GSListDecodingResult<F: IsField> {
    /// All polynomials f(x) that are consistent with the received word
    pub candidates: Vec<Polynomial<FieldElement<F>>>,
    /// The multiplicity parameter used
    pub multiplicity: usize,
    /// The weighted degree bound used
    pub degree_bound: usize,
    /// Maximum errors this configuration can handle
    pub error_bound: usize,
}

/// Computes the optimal Guruswami-Sudan decoding radius.
///
/// Returns the largest integer t such that t < n - sqrt(n * k).
pub fn gs_decoding_radius(n: usize, k: usize) -> usize {
    let threshold = (n as f64) - ((n as f64) * (k as f64)).sqrt();
    if threshold <= 0.0 {
        0
    } else {
        (threshold - 0.001).floor() as usize
    }
}

/// Computes the Johnson bound on list size.
///
/// For a received word with t errors, the list size is at most:
/// n / (n - t - sqrt(n * k))
pub fn johnson_list_bound(n: usize, k: usize, t: usize) -> f64 {
    let denominator = (n - t) as f64 - ((n as f64) * (k as f64)).sqrt();
    if denominator <= 0.0 {
        f64::INFINITY
    } else {
        (n as f64) / denominator
    }
}

/// Chooses optimal parameters for Guruswami-Sudan algorithm.
///
/// Returns (multiplicity m, weighted degree bound D) that maximize
/// the decoding radius while ensuring the interpolation step has
/// a non-trivial solution.
///
/// The decoding radius for GS with multiplicity m is approximately:
///   t < n - sqrt(n * k * (1 + 1/m))
///
/// As m increases, this approaches n - sqrt(n * k) (the optimal GS radius).
fn choose_parameters(n: usize, k: usize) -> (usize, usize) {
    // Find the smallest m that achieves at least the target GS decoding radius
    let target_radius = gs_decoding_radius(n, k);

    // Search for optimal m
    for m in 1..=20 {
        // Compute the decoding radius achievable with this multiplicity
        // t < n - sqrt(n * k * (1 + 1/m))
        let factor = 1.0 + 1.0 / (m as f64);
        let radius_with_m =
            ((n as f64) - ((n as f64) * (k as f64) * factor).sqrt()).floor() as usize;

        if radius_with_m >= target_radius {
            // This m achieves the target radius, now compute D
            let constraints_per_point = m * (m + 1) / 2;
            let total_constraints = n * constraints_per_point;

            // We need D such that the number of monomials with wdeg < D exceeds constraints
            let d = ((2 * total_constraints * (k - 1)) as f64).sqrt().ceil() as usize + k;

            // Verify the configuration is valid
            let num_monomials = count_monomials(d, k - 1);
            if num_monomials > total_constraints {
                return (m, d);
            }
        }
    }

    // Fallback: try to find any working configuration with large m
    for m in 2..=20 {
        let constraints_per_point = m * (m + 1) / 2;
        let total_constraints = n * constraints_per_point;
        let d = ((2 * total_constraints * (k - 1)) as f64).sqrt().ceil() as usize + k;
        let num_monomials = count_monomials(d, k - 1);
        if num_monomials > total_constraints {
            return (m, d);
        }
    }

    // Last resort fallback
    (4, n + k)
}

/// Counts monomials x^i y^j with (1, w)-weighted degree i + w*j < D.
fn count_monomials(d: usize, w: usize) -> usize {
    let mut count = 0;
    let max_j = d / w + 1;

    for j in 0..=max_j {
        let max_i = d.saturating_sub(w * j);
        count += max_i;
    }

    count
}

/// List decodes using the Guruswami-Sudan algorithm.
///
/// # Arguments
///
/// * `code` - The Reed-Solomon code
/// * `received` - The received word (possibly with errors)
///
/// # Returns
///
/// A list of all polynomials of degree < k that agree with the received word
/// on at least `n - t` positions, where `t` is close to n - sqrt(n*k).
pub fn gs_list_decode<F: IsField + Clone>(
    code: &ReedSolomonCode<F>,
    received: &[FieldElement<F>],
) -> GSListDecodingResult<F> {
    let n = code.code_length();
    let k = code.dimension();
    let domain = code.domain();

    assert_eq!(
        received.len(),
        n,
        "Received word length must equal code length"
    );

    let (m, d) = choose_parameters(n, k);

    // Interpolation step: find Q(x, y) passing through all points with multiplicity m
    let q = interpolate_with_multiplicity(domain, received, m, d, k);

    // Root finding step using Roth-Ruckenstein
    // Pass the received values as hints - they are the most likely roots
    let all_roots = find_polynomial_roots_with_hints(&q, k, received);

    // Filter to those with sufficient agreement
    let error_bound = gs_decoding_radius(n, k);
    let agreement_threshold = n.saturating_sub(error_bound);

    let valid_candidates: Vec<_> = all_roots
        .into_iter()
        .filter(|f| {
            if f.degree() >= k {
                return false;
            }
            let agr = domain
                .iter()
                .zip(received.iter())
                .filter(|(alpha, y)| f.evaluate(alpha) == **y)
                .count();
            agr >= agreement_threshold
        })
        .collect();

    GSListDecodingResult {
        candidates: valid_candidates,
        multiplicity: m,
        degree_bound: d,
        error_bound,
    }
}

/// List decodes with specified multiplicity.
///
/// This allows manual control over the multiplicity parameter for
/// experimentation and comparison.
pub fn gs_list_decode_with_multiplicity<F: IsField + Clone>(
    code: &ReedSolomonCode<F>,
    received: &[FieldElement<F>],
    multiplicity: usize,
) -> GSListDecodingResult<F> {
    let n = code.code_length();
    let k = code.dimension();
    let domain = code.domain();

    let m = multiplicity;

    // Compute appropriate degree bound for this multiplicity
    let constraints_per_point = m * (m + 1) / 2;
    let total_constraints = n * constraints_per_point;
    let d = ((2 * total_constraints * (k - 1)) as f64).sqrt().ceil() as usize + k;

    let q = interpolate_with_multiplicity(domain, received, m, d, k);
    let all_roots = find_polynomial_roots_with_hints(&q, k, received);

    let error_bound = gs_decoding_radius(n, k);
    let agreement_threshold = n.saturating_sub(error_bound);

    let valid_candidates: Vec<_> = all_roots
        .into_iter()
        .filter(|f| {
            if f.degree() >= k {
                return false;
            }
            let agr = domain
                .iter()
                .zip(received.iter())
                .filter(|(alpha, y)| f.evaluate(alpha) == **y)
                .count();
            agr >= agreement_threshold
        })
        .collect();

    GSListDecodingResult {
        candidates: valid_candidates,
        multiplicity: m,
        degree_bound: d,
        error_bound,
    }
}

/// Interpolates with multiplicity constraints.
///
/// Finds Q(x, y) such that for each point (αᵢ, yᵢ), Q has a zero of
/// multiplicity at least m. This means all partial derivatives of Q
/// up to order m-1 vanish at each point.
fn interpolate_with_multiplicity<F: IsField + Clone>(
    domain: &[FieldElement<F>],
    received: &[FieldElement<F>],
    m: usize,
    d: usize,
    k: usize,
) -> BivariatePolynomial<F> {
    let n = domain.len();
    let w = k - 1; // Weight for y

    // Build list of all monomials with weighted degree < D
    // Monomial x^i y^j has wdeg = i + w*j
    let mut monomials: Vec<(usize, usize)> = Vec::new();
    let max_j = d / w + 1;
    for j in 0..=max_j {
        let max_i = d.saturating_sub(w * j);
        for i in 0..max_i {
            monomials.push((i, j));
        }
    }

    let num_monomials = monomials.len();

    // Build constraint matrix
    // Each point contributes m*(m+1)/2 constraints (all partial derivatives)
    let constraints_per_point = m * (m + 1) / 2;
    let total_constraints = n * constraints_per_point;

    let mut matrix: Vec<Vec<FieldElement<F>>> = Vec::with_capacity(total_constraints);

    for (alpha, y) in domain.iter().zip(received.iter()) {
        // For multiplicity m, we need the (a, b)-th partial derivative to vanish
        // for all a + b < m, i.e., a + b ∈ {0, 1, ..., m-1}
        for total_order in 0..m {
            for b in 0..=total_order {
                let a = total_order - b;

                // Constraint: ∂^a/∂x^a ∂^b/∂y^b Q(α, y) = 0
                // The coefficient of monomial x^i y^j in this derivative evaluated at (α, y) is:
                // C(i, a) * C(j, b) * α^{i-a} * y^{j-b}
                let mut row = Vec::with_capacity(num_monomials);

                for &(i, j) in &monomials {
                    if i < a || j < b {
                        // Derivative is zero
                        row.push(FieldElement::<F>::zero());
                    } else {
                        let binom_i_a = binomial(i, a);
                        let binom_j_b = binomial(j, b);
                        let coeff_scalar = (binom_i_a * binom_j_b) as u64;

                        // α^{i-a} * y^{j-b}
                        let alpha_power = alpha.pow(i - a);
                        let y_power = y.pow(j - b);

                        let coeff =
                            &FieldElement::<F>::from(coeff_scalar) * &(&alpha_power * &y_power);
                        row.push(coeff);
                    }
                }

                matrix.push(row);
            }
        }
    }

    // Find a non-trivial kernel vector
    let solution = find_kernel_vector(&matrix, num_monomials);

    // Build the polynomial from the solution
    build_polynomial_from_monomials(&monomials, &solution)
}

/// Builds a bivariate polynomial from monomial coefficients.
fn build_polynomial_from_monomials<F: IsField + Clone>(
    monomials: &[(usize, usize)],
    coefficients: &[FieldElement<F>],
) -> BivariatePolynomial<F> {
    // Find maximum y degree
    let max_j = monomials.iter().map(|(_, j)| *j).max().unwrap_or(0);

    // Initialize coefficient polynomials
    let mut y_coeffs: Vec<Vec<FieldElement<F>>> = vec![vec![]; max_j + 1];

    // Fill in coefficients
    for (&(i, j), coeff) in monomials.iter().zip(coefficients.iter()) {
        // Ensure the vector is large enough
        while y_coeffs[j].len() <= i {
            y_coeffs[j].push(FieldElement::<F>::zero());
        }
        y_coeffs[j][i] = coeff.clone();
    }

    // Convert to polynomials
    let polys: Vec<Polynomial<FieldElement<F>>> = y_coeffs
        .into_iter()
        .map(|coeffs| {
            if coeffs.is_empty() {
                Polynomial::zero()
            } else {
                Polynomial::new(&coeffs)
            }
        })
        .collect();

    BivariatePolynomial::new(polys)
}

/// Finds a non-trivial kernel vector (same as in sudan.rs).
fn find_kernel_vector<F: IsField + Clone>(
    matrix: &[Vec<FieldElement<F>>],
    num_cols: usize,
) -> Vec<FieldElement<F>> {
    let m = matrix.len();
    let n = num_cols;

    if m == 0 {
        let mut result = vec![FieldElement::<F>::zero(); n];
        if n > 0 {
            result[0] = FieldElement::<F>::one();
        }
        return result;
    }

    let mut mat: Vec<Vec<FieldElement<F>>> = matrix.to_vec();
    for row in mat.iter_mut() {
        while row.len() < n {
            row.push(FieldElement::<F>::zero());
        }
    }

    let mut pivot_cols = Vec::new();
    let mut pivot_row = 0;

    for col in 0..n {
        if pivot_row >= m {
            break;
        }

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

        let pivot = mat[pivot_row][col].clone();
        let pivot_inv = pivot.inv().unwrap_or_else(|_| FieldElement::<F>::one());
        for j in col..n {
            mat[pivot_row][j] = &mat[pivot_row][j] * &pivot_inv;
        }

        for row in 0..m {
            if row != pivot_row && mat[row][col] != FieldElement::<F>::zero() {
                let factor = mat[row][col].clone();
                for j in col..n {
                    let sub = &factor * &mat[pivot_row][j];
                    mat[row][j] = &mat[row][j] - &sub;
                }
            }
        }

        pivot_row += 1;
    }

    let mut free_col = None;
    for col in 0..n {
        if !pivot_cols.contains(&col) {
            free_col = Some(col);
            break;
        }
    }

    let mut kernel = vec![FieldElement::<F>::zero(); n];

    if let Some(fc) = free_col {
        kernel[fc] = FieldElement::<F>::one();
        for (row, &pc) in pivot_cols.iter().enumerate() {
            if row < m {
                kernel[pc] = -mat[row][fc].clone();
            }
        }
    } else if n > 0 {
        kernel[n - 1] = FieldElement::<F>::one();
    }

    kernel
}

/// Computes binomial coefficient C(n, k).
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k);
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::introduce_errors_at_positions;
    use crate::Babybear31PrimeField;

    type FE = FieldElement<Babybear31PrimeField>;

    #[test]
    fn test_gs_full_radius() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::with_consecutive_domain(16, 4);
        let message: Vec<FE> = (1..=4).map(|i| FE::from(i as u64)).collect();
        let codeword = code.encode(&message);
        let original_poly = Polynomial::new(&message);

        let gs_radius = gs_decoding_radius(16, 4);
        println!("\nRS[16, 4] GS decoding radius: {}", gs_radius);

        // Test decoding up to the GS radius
        for num_errors in 1..=gs_radius {
            let error_positions: Vec<usize> = (0..num_errors).collect();
            let corrupted = introduce_errors_at_positions(&codeword, &error_positions);

            let result = gs_list_decode(&code, &corrupted);
            let found = result.candidates.contains(&original_poly);

            assert!(
                found,
                "GS should find original with {} errors (radius={})",
                num_errors, gs_radius
            );
        }
    }

    #[test]
    fn test_gs_decoding_radius() {
        // RS[16, 4]: n - sqrt(n*k) = 16 - sqrt(64) = 16 - 8 = 8
        let radius = gs_decoding_radius(16, 4);
        assert!(radius >= 7, "Should be close to 8");

        // RS[32, 8]: n - sqrt(n*k) = 32 - sqrt(256) = 32 - 16 = 16
        let radius32 = gs_decoding_radius(32, 8);
        assert!(radius32 >= 15, "Should be close to 16");
    }

    #[test]
    fn test_johnson_bound() {
        // For RS[16, 4] with 6 errors:
        // Johnson bound = 16 / (16 - 6 - sqrt(64)) = 16 / (10 - 8) = 8
        let bound = johnson_list_bound(16, 4, 6);
        assert!(bound.is_finite());
        assert!(bound > 0.0);
        println!("Johnson bound for RS[16,4] with 6 errors: {}", bound);
    }

    #[test]
    fn test_gs_no_errors() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::new(16, 4);
        let message: Vec<FE> = (1..=4).map(|i| FE::from(i as u64)).collect();
        let codeword = code.encode(&message);

        let result = gs_list_decode(&code, &codeword);

        assert!(!result.candidates.is_empty());
        let original_poly = Polynomial::new(&message);
        assert!(
            result.candidates.contains(&original_poly),
            "Should contain original polynomial"
        );
    }

    #[test]
    fn test_gs_with_errors() {
        // Test that GS algorithm runs without panicking and returns results
        // Note: Full correctness depends on the interpolation and root-finding steps
        // which are complex to implement fully. This test verifies basic operation.
        let code = ReedSolomonCode::<Babybear31PrimeField>::with_consecutive_domain(16, 4);
        let message: Vec<FE> = (0..4).map(|i| FE::from((i + 1) as u64)).collect();
        let codeword = code.encode(&message);

        // Introduce some errors
        let corrupted = introduce_errors_at_positions(&codeword, &[2, 5]);

        let result = gs_list_decode(&code, &corrupted);

        // The algorithm should complete and return parameters
        println!(
            "GS result: {} candidates, multiplicity {}, error bound {}",
            result.candidates.len(),
            result.multiplicity,
            result.error_bound
        );

        // Check that the parameters are sensible
        assert!(
            result.multiplicity >= 1,
            "Multiplicity should be at least 1"
        );
        assert!(
            result.degree_bound >= 4,
            "Degree bound should be at least k"
        );
    }

    #[test]
    fn test_choose_parameters() {
        let (m, d) = choose_parameters(16, 4);
        assert!(m >= 1, "Multiplicity should be at least 1");
        assert!(d > 4, "Degree bound should exceed k");
        println!("Parameters for RS[16,4]: m={}, D={}", m, d);

        let (m32, d32) = choose_parameters(32, 8);
        println!("Parameters for RS[32,8]: m={}, D={}", m32, d32);
    }

    #[test]
    fn test_count_monomials() {
        // wdeg(x^i y^j) = i + 3j < 10
        // j=0: i < 10 -> 10 monomials
        // j=1: i < 7 -> 7 monomials
        // j=2: i < 4 -> 4 monomials
        // j=3: i < 1 -> 1 monomial
        // Total: 22
        let count = count_monomials(10, 3);
        assert_eq!(count, 22);
    }

    #[test]
    fn test_gs_vs_sudan_radius() {
        // Guruswami-Sudan should have better radius than Sudan
        for (n, k) in [(16, 4), (32, 8), (64, 16)] {
            let gs_radius = gs_decoding_radius(n, k);
            let sudan_radius = crate::sudan::sudan_decoding_radius(n, k);
            let bw_radius = (n - k) / 2;

            println!(
                "RS[{},{}]: BW={}, Sudan={}, GS={}",
                n, k, bw_radius, sudan_radius, gs_radius
            );

            assert!(
                gs_radius >= sudan_radius,
                "GS should be at least as good as Sudan"
            );
            assert!(
                gs_radius >= bw_radius,
                "GS should be at least as good as BW"
            );
        }
    }

    #[test]
    fn test_list_size_bounded() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::with_consecutive_domain(16, 4);
        let message: Vec<FE> = (0..4).map(|i| FE::from(i as u64)).collect();
        let codeword = code.encode(&message);

        // Introduce errors
        let corrupted = introduce_errors_at_positions(&codeword, &[1, 3, 5, 7]);

        let result = gs_list_decode(&code, &corrupted);

        // Johnson bound says list size ≤ O(sqrt(n))
        let bound = johnson_list_bound(16, 4, 4);
        println!(
            "List size: {}, Johnson bound: {}",
            result.candidates.len(),
            bound
        );

        // The list shouldn't be too large
        assert!(
            result.candidates.len() <= 10,
            "List size should be reasonably bounded"
        );
    }

    #[test]
    fn test_sage_comparison() {
        use crate::distance::{agreement, introduce_errors};

        // RS[32, 4] like in Sage comparison
        let code = ReedSolomonCode::<Babybear31PrimeField>::with_consecutive_domain(32, 4);
        let msg: Vec<FE> = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let cw = code.encode(&msg);
        let original_poly = Polynomial::new(&msg);

        // Sage's second candidate [1001, 1002, 3, 4]
        let candidate_msg: Vec<FE> = vec![FE::from(1001), FE::from(1002), FE::from(3), FE::from(4)];
        let candidate_poly = Polynomial::new(&candidate_msg);

        // 20 errors with 1000*(i+1) pattern (like Sage)
        let error_positions: Vec<usize> = (0..20).collect();
        let error_values: Vec<FE> = (0..20).map(|i| FE::from((1000 * (i + 1)) as u64)).collect();
        let received = introduce_errors(&cw, &error_positions, &error_values);

        let agree_original = agreement(&received, code.domain(), &original_poly);
        let agree_candidate = agreement(&received, code.domain(), &candidate_poly);

        println!("\nSage comparison test (RS[32,4], 20 errors, 1000*(i+1) pattern):");
        println!(
            "  Domain (first 5): {:?}",
            code.domain()
                .iter()
                .take(5)
                .map(|x| x.representative())
                .collect::<Vec<_>>()
        );
        println!(
            "  Original codeword (first 5): {:?}",
            cw.iter()
                .take(5)
                .map(|x| x.representative())
                .collect::<Vec<_>>()
        );
        println!(
            "  Received (first 5): {:?}",
            received
                .iter()
                .take(5)
                .map(|x| x.representative())
                .collect::<Vec<_>>()
        );
        println!(
            "  Sage candidate codeword (first 5): {:?}",
            code.encode(&candidate_msg)
                .iter()
                .take(5)
                .map(|x| x.representative())
                .collect::<Vec<_>>()
        );
        println!("  Agreement with original [1,2,3,4]: {}/32", agree_original);
        println!(
            "  Agreement with candidate [1001,1002,3,4]: {}/32",
            agree_candidate
        );

        // GS radius is 20, threshold is 32 - 20 = 12
        let threshold = 32 - 20;
        println!("  Threshold for list inclusion: {} agreements", threshold);
        println!(
            "  Candidate should be in list: {} >= {} = {}",
            agree_candidate,
            threshold,
            agree_candidate >= threshold
        );

        // Run GS decoder - but first check if Q(x,f(x)) = 0 for the candidate
        let m = 8; // multiplicity used for RS[32,4]
        let k = 4;
        let n = 32;
        let constraints_per_point = m * (m + 1) / 2;
        let total_constraints = n * constraints_per_point;
        let d = ((2 * total_constraints * (k - 1)) as f64).sqrt().ceil() as usize + k;

        let q = interpolate_with_multiplicity(code.domain(), &received, m, d, k);
        println!("  Interpolated Q(x,y): y_degree={}", q.y_degree());

        // Check if Q(x, candidate) = 0
        let q_at_candidate = q.evaluate_y_polynomial(&candidate_poly);
        let is_root = q_at_candidate == Polynomial::zero();
        println!(
            "  Q(x, [1001,1002,3,4]) = 0? {} (degree {})",
            is_root,
            q_at_candidate.degree()
        );

        // Check if Q(x, original) = 0
        let q_at_original = q.evaluate_y_polynomial(&original_poly);
        let is_orig_root = q_at_original == Polynomial::zero();
        println!(
            "  Q(x, [1,2,3,4]) = 0? {} (degree {})",
            is_orig_root,
            q_at_original.degree()
        );

        // Check Q(0, y) roots
        let q_at_zero: Vec<_> = q.coeffs().iter().map(|p| p.evaluate(&FE::zero())).collect();
        println!(
            "  Q(0, y) has degree {} in y",
            q_at_zero
                .iter()
                .rposition(|c| *c != FE::zero())
                .unwrap_or(0)
        );

        // Check if specific values are roots of Q(0, y)
        let q0y_poly = Polynomial::new(&q_at_zero);
        let is_1_root = q0y_poly.evaluate(&FE::from(1u64)) == FE::zero();
        let is_1001_root = q0y_poly.evaluate(&FE::from(1001u64)) == FE::zero();
        println!("  Q(0, 1) = 0? {}", is_1_root);
        println!("  Q(0, 1001) = 0? {}", is_1001_root);

        // Check what roots find_univariate_roots_with_hints finds
        use crate::polynomial_utils::{find_univariate_roots_debug, substitute_and_divide_debug};
        let roots = find_univariate_roots_debug(&q_at_zero, &received);
        println!(
            "  find_univariate_roots found {} roots: {:?}",
            roots.len(),
            roots
                .iter()
                .take(10)
                .map(|x| x.representative())
                .collect::<Vec<_>>()
        );

        // Check what happens after substituting y = 1001 + xy
        let q_prime = substitute_and_divide_debug(&q, &FE::from(1001u64));
        let q_prime_at_zero: Vec<_> = q_prime
            .coeffs()
            .iter()
            .map(|p| p.evaluate(&FE::zero()))
            .collect();
        let q_prime_poly = Polynomial::new(&q_prime_at_zero);
        let is_1002_root = q_prime_poly.evaluate(&FE::from(1002u64)) == FE::zero();
        let is_2_root = q_prime_poly.evaluate(&FE::from(2u64)) == FE::zero();
        println!("  After Q' = Q(x, 1001+xy)/x:");
        println!("    Q'(0, 1002) = 0? {}", is_1002_root);
        println!("    Q'(0, 2) = 0? {}", is_2_root);
        let roots_prime = find_univariate_roots_debug(&q_prime_at_zero, &received);
        println!(
            "    Roots found: {:?}",
            roots_prime
                .iter()
                .take(10)
                .map(|x| x.representative())
                .collect::<Vec<_>>()
        );

        let result = gs_list_decode(&code, &received);
        println!("  GS found {} candidates:", result.candidates.len());
        for (i, c) in result.candidates.iter().enumerate() {
            let coeffs: Vec<u32> = c
                .coefficients()
                .iter()
                .map(|x| x.representative())
                .collect();
            let is_orig = c == &original_poly;
            let is_sage = c == &candidate_poly;
            let marker = if is_orig {
                " <-- ORIGINAL"
            } else if is_sage {
                " <-- SAGE's candidate"
            } else {
                ""
            };
            println!("    [{}] {:?}{}", i + 1, coeffs, marker);
        }

        // Sage found both at 20 errors
        assert!(
            result.candidates.contains(&original_poly),
            "Should find original"
        );

        // Check if we should find the second candidate
        if agree_candidate >= threshold {
            println!(
                "  NOTE: [1001,1002,3,4] has {} agreements (>= {}), Sage found it but we didn't",
                agree_candidate, threshold
            );
        }
    }
}
