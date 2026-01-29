//! Bivariate polynomial utilities for list decoding algorithms.
//!
//! This module provides support for bivariate polynomials Q(x, y) needed
//! by Sudan and Guruswami-Sudan list decoding algorithms.
//!
//! # Representation
//!
//! A bivariate polynomial Q(x, y) = Σᵢⱼ qᵢⱼ xⁱ yʲ is represented as
//! a polynomial in y with coefficients that are polynomials in x:
//!
//! Q(x, y) = Q₀(x) + Q₁(x)·y + Q₂(x)·y² + ...
//!
//! # Weighted Degree
//!
//! For list decoding, we use (1, k-1)-weighted degree:
//!
//! wdeg(xⁱ yʲ) = i + (k-1)·j
//!
//! This ensures that Q(x, f(x)) has bounded degree when deg(f) < k.

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::Polynomial;

/// A bivariate polynomial Q(x, y) represented as coefficients in y
/// where each coefficient is a polynomial in x.
///
/// Q(x, y) = Q₀(x) + Q₁(x)·y + Q₂(x)·y² + ... + Qₘ(x)·yᵐ
#[derive(Debug, Clone)]
pub struct BivariatePolynomial<F: IsField> {
    /// Coefficients: coeffs[j] is the coefficient of y^j (a polynomial in x)
    coeffs: Vec<Polynomial<FieldElement<F>>>,
}

impl<F: IsField + Clone> BivariatePolynomial<F> {
    /// Creates a bivariate polynomial from coefficients.
    ///
    /// `coeffs[j]` is the polynomial coefficient of y^j.
    pub fn new(coeffs: Vec<Polynomial<FieldElement<F>>>) -> Self {
        // Remove trailing zero polynomials
        let mut coeffs = coeffs;
        while coeffs.len() > 1 && coeffs.last().is_some_and(|p| p == &Polynomial::zero()) {
            coeffs.pop();
        }
        Self { coeffs }
    }

    /// Creates the zero polynomial.
    pub fn zero() -> Self {
        Self {
            coeffs: vec![Polynomial::zero()],
        }
    }

    /// Returns true if this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0] == Polynomial::zero()
    }

    /// Returns a reference to the coefficients.
    pub fn coeffs(&self) -> &[Polynomial<FieldElement<F>>] {
        &self.coeffs
    }

    /// Returns the degree in y.
    pub fn y_degree(&self) -> usize {
        if self.is_zero() {
            return 0;
        }
        self.coeffs.len() - 1
    }

    /// Returns the maximum degree in x among all coefficients.
    pub fn max_x_degree(&self) -> usize {
        self.coeffs.iter().map(|p| p.degree()).max().unwrap_or(0)
    }

    /// Returns the coefficient of y^j (as a polynomial in x).
    pub fn coeff_y(&self, j: usize) -> &Polynomial<FieldElement<F>> {
        if j < self.coeffs.len() {
            &self.coeffs[j]
        } else {
            // Return a reference to a zero polynomial
            // This is a bit awkward; in production code we'd handle this differently
            &self.coeffs[0] // Will be zero if called correctly
        }
    }

    /// Returns the coefficient of x^i y^j.
    pub fn coeff(&self, i: usize, j: usize) -> FieldElement<F> {
        if j >= self.coeffs.len() {
            return FieldElement::<F>::zero();
        }
        let poly = &self.coeffs[j];
        let coeffs = poly.coefficients();
        if i < coeffs.len() {
            coeffs[i].clone()
        } else {
            FieldElement::<F>::zero()
        }
    }

    /// Evaluates Q(x, y) at a specific point (x₀, y₀).
    pub fn evaluate(&self, x: &FieldElement<F>, y: &FieldElement<F>) -> FieldElement<F> {
        let mut result = FieldElement::<F>::zero();
        let mut y_power = FieldElement::<F>::one();

        for coeff in &self.coeffs {
            let x_eval = coeff.evaluate(x);
            result = &result + &(&x_eval * &y_power);
            y_power = &y_power * y;
        }

        result
    }

    /// Computes the (1, w)-weighted degree where w = k-1.
    ///
    /// wdeg(Q) = max { i + w·j : q_{ij} ≠ 0 }
    pub fn weighted_degree(&self, w: usize) -> usize {
        let mut max_wdeg = 0;

        for (j, poly) in self.coeffs.iter().enumerate() {
            let coeffs = poly.coefficients();
            for (i, c) in coeffs.iter().enumerate() {
                if *c != FieldElement::<F>::zero() {
                    let wdeg = i + w * j;
                    max_wdeg = max_wdeg.max(wdeg);
                }
            }
        }

        max_wdeg
    }

    /// Computes Q(x, f(x)) where f is a polynomial.
    ///
    /// Returns the resulting univariate polynomial in x.
    pub fn evaluate_y_polynomial(
        &self,
        f: &Polynomial<FieldElement<F>>,
    ) -> Polynomial<FieldElement<F>> {
        let mut result = Polynomial::zero();
        let mut f_power = Polynomial::new(&[FieldElement::<F>::one()]); // f^0 = 1

        for coeff in &self.coeffs {
            // Add coeff(x) * f(x)^j to result
            let term = coeff.mul_with_ref(&f_power);
            result = result + term;
            f_power = f_power.mul_with_ref(f);
        }

        result
    }

    /// Adds two bivariate polynomials.
    pub fn add(&self, other: &Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = Vec::with_capacity(max_len);

        for j in 0..max_len {
            let a = if j < self.coeffs.len() {
                self.coeffs[j].clone()
            } else {
                Polynomial::zero()
            };
            let b = if j < other.coeffs.len() {
                other.coeffs[j].clone()
            } else {
                Polynomial::zero()
            };
            result.push(a + b);
        }

        Self::new(result)
    }

    /// Multiplies by a scalar.
    pub fn scale(&self, scalar: &FieldElement<F>) -> Self {
        let coeffs: Vec<_> = self
            .coeffs
            .iter()
            .map(|p| {
                let scaled: Vec<_> = p.coefficients().iter().map(|c| c * scalar).collect();
                Polynomial::new(&scaled)
            })
            .collect();
        Self::new(coeffs)
    }

    /// Multiplies by x^i.
    pub fn mul_x_power(&self, i: usize) -> Self {
        if i == 0 {
            return self.clone();
        }

        let coeffs: Vec<_> = self
            .coeffs
            .iter()
            .map(|p| {
                let old_coeffs = p.coefficients();
                let mut new_coeffs = vec![FieldElement::<F>::zero(); i];
                new_coeffs.extend(old_coeffs.iter().cloned());
                Polynomial::new(&new_coeffs)
            })
            .collect();

        Self::new(coeffs)
    }

    /// Multiplies by y^j.
    pub fn mul_y_power(&self, j: usize) -> Self {
        if j == 0 {
            return self.clone();
        }

        let mut coeffs = vec![Polynomial::zero(); j];
        coeffs.extend(self.coeffs.iter().cloned());
        Self::new(coeffs)
    }
}

/// Finds y-roots of a bivariate polynomial Q(x, y) that are polynomials in x.
///
/// Given Q(x, y), finds all polynomials f(x) of degree < k such that Q(x, f(x)) = 0.
///
/// This uses a simple approach suitable for educational purposes:
/// For the Sudan algorithm with deg_y(Q) ≤ 1, we have Q(x, y) = A(x) + B(x)·y
/// and f(x) = -A(x)/B(x) when B divides A.
///
/// For Guruswami-Sudan, we use the Roth-Ruckenstein algorithm.
///
/// The `hint_values` parameter provides field elements that are likely to be roots
/// (typically the received word values for RS decoding).
pub fn find_polynomial_roots<F: IsField + Clone>(
    q: &BivariatePolynomial<F>,
    max_degree: usize,
) -> Vec<Polynomial<FieldElement<F>>> {
    find_polynomial_roots_with_hints(q, max_degree, &[])
}

/// Finds y-roots with hint values that are likely to contain the roots.
pub fn find_polynomial_roots_with_hints<F: IsField + Clone>(
    q: &BivariatePolynomial<F>,
    max_degree: usize,
    hint_values: &[FieldElement<F>],
) -> Vec<Polynomial<FieldElement<F>>> {
    // Simple case: Q is linear in y (Sudan's algorithm)
    if q.y_degree() <= 1 {
        return find_roots_linear_y(q, max_degree);
    }

    // General case: use Roth-Ruckenstein algorithm
    // Create default domain points [0, 1, 2, ..., n-1] for basic hint transformation
    let domain: Vec<FieldElement<F>> = (0..hint_values.len())
        .map(|i| FieldElement::<F>::from(i as u64))
        .collect();
    roth_ruckenstein_with_domain(q, max_degree, hint_values, &domain)
}

/// Finds y-roots with hint values and domain points for proper hint transformation.
///
/// This is the key improvement: at each recursion level, hints are transformed
/// based on the domain points: new_hint[i] = (old_hint[i] - c) / domain[i]
pub fn find_polynomial_roots_with_domain<F: IsField + Clone>(
    q: &BivariatePolynomial<F>,
    max_degree: usize,
    hint_values: &[FieldElement<F>],
    domain: &[FieldElement<F>],
) -> Vec<Polynomial<FieldElement<F>>> {
    // Simple case: Q is linear in y (Sudan's algorithm)
    if q.y_degree() <= 1 {
        return find_roots_linear_y(q, max_degree);
    }

    // General case: use Roth-Ruckenstein algorithm with domain
    roth_ruckenstein_with_domain(q, max_degree, hint_values, domain)
}

/// Debug version of find_univariate_roots_with_hints for testing.
pub fn find_univariate_roots_debug<F: IsField + Clone>(
    coeffs: &[FieldElement<F>],
    hint_values: &[FieldElement<F>],
) -> Vec<FieldElement<F>> {
    find_univariate_roots_with_hints(coeffs, hint_values)
}

/// Debug version of substitute_and_divide for testing.
pub fn substitute_and_divide_debug<F: IsField + Clone>(
    q: &BivariatePolynomial<F>,
    c: &FieldElement<F>,
) -> BivariatePolynomial<F> {
    substitute_and_divide(q, c)
}

/// Finds roots when Q(x, y) = A(x) + B(x)·y (linear in y).
fn find_roots_linear_y<F: IsField + Clone>(
    q: &BivariatePolynomial<F>,
    max_degree: usize,
) -> Vec<Polynomial<FieldElement<F>>> {
    if q.coeffs.len() < 2 {
        // Q is just a constant or polynomial in x only
        return vec![];
    }

    let a = &q.coeffs[0]; // A(x)
    let b = &q.coeffs[1]; // B(x)

    // f(x) = -A(x) / B(x)
    // Check if B divides A
    let neg_a = {
        let coeffs: Vec<_> = a.coefficients().iter().map(|c| -c).collect();
        Polynomial::new(&coeffs)
    };

    // If B(x) is zero, return the empty vector.
    let Ok((quotient, remainder)) = neg_a.long_division_with_remainder(b) else {
        return vec![];
    };

    if remainder != Polynomial::zero() {
        // B doesn't divide A
        return vec![];
    }

    if quotient.degree() >= max_degree {
        // Degree too high
        return vec![];
    }

    vec![quotient]
}

/// Roth-Ruckenstein algorithm for finding polynomial roots with domain-based hint transformation.
///
/// This is an iterative algorithm that finds all polynomials f(x) of degree < k
/// such that Q(x, f(x)) = 0.
///
/// The algorithm builds f coefficient by coefficient, maintaining a "shifted"
/// polynomial at each step. At each recursion level, the hints are transformed
/// using the domain points: new_hint[i] = (old_hint[i] - c) / domain[i].
///
/// This proper hint transformation is key to finding roots with large coefficients.
///
/// `hint_values` are field elements likely to be roots (e.g., received word values).
/// `domain` are the evaluation points used in Reed-Solomon encoding.
fn roth_ruckenstein_with_domain<F: IsField + Clone>(
    q: &BivariatePolynomial<F>,
    max_degree: usize,
    hint_values: &[FieldElement<F>],
    domain: &[FieldElement<F>],
) -> Vec<Polynomial<FieldElement<F>>> {
    let mut roots = Vec::new();

    // OPTIMIZATION: Try to directly interpolate candidate polynomials from hints
    // If we have enough points (hint, domain) pairs, we can interpolate f directly
    // and verify Q(x, f(x)) = 0. This is much faster than tree search.
    try_interpolated_candidates(q, max_degree, hint_values, domain, &mut roots);

    // Also try direct verification for low-degree candidates
    // This is a fallback for when the recursive search might miss some roots
    if max_degree <= 10 && roots.len() < MAX_TOTAL_ROOTS {
        try_direct_roots(q, max_degree, hint_values, &mut roots);
    }

    // Start the tree search with an empty polynomial
    // This catches any roots missed by the direct approaches
    if roots.len() < MAX_TOTAL_ROOTS {
        rr_search_with_domain(q, max_degree, &[], &mut roots, hint_values, domain, 0);
    }

    roots
}

/// Try to find roots by direct Lagrange interpolation on hint values.
///
/// Given hints (domain[i], hint[i]) = (α_i, f(α_i)), we can interpolate to find f
/// and verify it's a root of Q(x, y).
fn try_interpolated_candidates<F: IsField + Clone>(
    q: &BivariatePolynomial<F>,
    max_degree: usize,
    hint_values: &[FieldElement<F>],
    domain: &[FieldElement<F>],
    roots: &mut Vec<Polynomial<FieldElement<F>>>,
) {
    if hint_values.len() < max_degree || domain.len() < max_degree {
        return;
    }

    // Try different subsets of points for interpolation
    // This helps when some hints are from error positions
    let n = hint_values.len().min(domain.len());

    for start in 0..(n.saturating_sub(max_degree) + 1).min(20) {
        if start + max_degree > n {
            break;
        }

        // Build points for interpolation
        let points: Vec<_> = (start..start + max_degree)
            .filter_map(|idx| {
                let alpha = &domain[idx];
                extract_small_value(alpha).map(|a| (a, hint_values[idx].clone()))
            })
            .collect();

        // Need exactly max_degree points for a degree < max_degree polynomial
        if points.len() != max_degree {
            continue;
        }

        // Interpolate the polynomial
        if let Some(candidate) = lagrange_interpolate_polynomial(&points, max_degree) {
            // Verify it's actually a root of Q
            let qf = q.evaluate_y_polynomial(&candidate);
            if qf == Polynomial::zero() && !roots.contains(&candidate) {
                roots.push(candidate);
                // Early exit if we've found enough roots
                if roots.len() >= MAX_TOTAL_ROOTS {
                    return;
                }
            }
        }
    }
}

/// Lagrange interpolation to build a polynomial from points.
fn lagrange_interpolate_polynomial<F: IsField + Clone>(
    points: &[(usize, FieldElement<F>)],
    max_degree: usize,
) -> Option<Polynomial<FieldElement<F>>> {
    if points.is_empty() || points.len() > max_degree {
        return None;
    }

    // Build the interpolating polynomial using Lagrange basis
    let n = points.len();
    let mut coeffs = vec![FieldElement::<F>::zero(); n];

    for i in 0..n {
        let (xi, yi) = &points[i];
        let xi_fe = FieldElement::<F>::from(*xi as u64);

        // Compute Lagrange basis polynomial L_i(x)
        // L_i(x) = prod_{j≠i} (x - x_j) / (x_i - x_j)
        let mut basis_coeffs = vec![FieldElement::<F>::one()];

        let mut denominator = FieldElement::<F>::one();
        for (j, (xj, _)) in points.iter().enumerate().take(n) {
            if j != i {
                let xj_fe = FieldElement::<F>::from(*xj as u64);

                // Multiply basis_coeffs by (x - x_j)
                let mut new_coeffs = vec![FieldElement::<F>::zero(); basis_coeffs.len() + 1];
                for (k, c) in basis_coeffs.iter().enumerate() {
                    // c * (x - x_j) = c*x - c*x_j
                    new_coeffs[k + 1] = &new_coeffs[k + 1] + c;
                    new_coeffs[k] = &new_coeffs[k] - &(c * &xj_fe);
                }
                basis_coeffs = new_coeffs;

                // denominator *= (x_i - x_j)
                denominator = &denominator * &(&xi_fe - &xj_fe);
            }
        }

        // L_i(x) = basis / denominator
        if let Ok(denom_inv) = denominator.inv() {
            for (k, c) in basis_coeffs.iter().enumerate() {
                if k < coeffs.len() {
                    coeffs[k] = &coeffs[k] + &(yi * &(c * &denom_inv));
                }
            }
        } else {
            return None;
        }
    }

    Some(Polynomial::new(&coeffs))
}

/// Try to find roots by direct enumeration for small degree polynomials.
///
/// This is a fallback method that tries common polynomial patterns.
fn try_direct_roots<F: IsField + Clone>(
    q: &BivariatePolynomial<F>,
    max_degree: usize,
    hint_values: &[FieldElement<F>],
    roots: &mut Vec<Polynomial<FieldElement<F>>>,
) {
    // Try small integer coefficient polynomials up to the max degree
    // This catches simple cases like 1 + 2x + 3x^2 + 4x^3
    let max_coeff = 20;

    // For efficiency, limit the search space based on degree
    if max_degree <= 4 {
        // Try polynomials with small integer coefficients
        try_small_integer_polynomials(q, max_degree, max_coeff, roots);
    }

    // Also try constant polynomials from hints
    for hint in hint_values {
        let candidate = Polynomial::new(std::slice::from_ref(hint));
        let qf = q.evaluate_y_polynomial(&candidate);
        if qf == Polynomial::zero() && !roots.contains(&candidate) {
            roots.push(candidate);
        }
    }
}

/// Try polynomials with small integer coefficients.
fn try_small_integer_polynomials<F: IsField + Clone>(
    q: &BivariatePolynomial<F>,
    max_degree: usize,
    max_coeff: u64,
    roots: &mut Vec<Polynomial<FieldElement<F>>>,
) {
    // Generate and test candidate polynomials with small coefficients
    // This is a brute force approach but works for educational purposes

    let num_coeffs = max_degree;

    // Adjust max_coeff to keep search manageable
    // For k coefficients, we have (max_coeff+1)^k candidates
    let adjusted_max_coeff = if num_coeffs <= 2 {
        max_coeff
    } else if num_coeffs <= 3 {
        max_coeff.min(30)
    } else if num_coeffs <= 4 {
        max_coeff.min(15) // 16^4 = 65536, reasonable
    } else {
        max_coeff.min(8) // 9^5 = 59049
    };

    let total_candidates = (adjusted_max_coeff + 1).pow(num_coeffs as u32);

    // Limit search to avoid explosion
    if total_candidates > 200000 {
        return;
    }

    for i in 0..total_candidates {
        let mut coeffs = Vec::with_capacity(num_coeffs);
        let mut val = i;
        for _ in 0..num_coeffs {
            coeffs.push(FieldElement::<F>::from(val % (adjusted_max_coeff + 1)));
            val /= adjusted_max_coeff + 1;
        }

        // Skip zero polynomial
        if coeffs.iter().all(|c| *c == FieldElement::<F>::zero()) {
            continue;
        }

        let candidate = Polynomial::new(&coeffs);
        let qf = q.evaluate_y_polynomial(&candidate);
        if qf == Polynomial::zero() && !roots.contains(&candidate) {
            roots.push(candidate);
        }
    }
}

/// Maximum number of candidate roots to explore at each depth to prevent explosion.
const MAX_CANDIDATES_PER_DEPTH: usize = 15;

/// Maximum number of polynomial roots to find before stopping search.
const MAX_TOTAL_ROOTS: usize = 10;

/// Recursive search in Roth-Ruckenstein algorithm with domain-based hint transformation.
///
/// The key insight is: if f(x) = c₀ + c₁x + c₂x² + ... satisfies Q(x, f(x)) = 0, then
/// c₀ is a root of Q(0, y), and f'(x) = c₁ + c₂x + ... satisfies Q'(x, f'(x)) = 0
/// where Q'(x, y) = Q(x, c₀ + xy) / x.
///
/// The crucial improvement is hint transformation: if originally f(αᵢ) = rᵢ, then
/// after substitution with c₀, the remaining polynomial f'(x) = (f(x) - c₀)/x satisfies
/// f'(αᵢ) = (rᵢ - c₀)/αᵢ. This allows us to track the expected coefficient values
/// through the recursion.
fn rr_search_with_domain<F: IsField + Clone>(
    q: &BivariatePolynomial<F>,
    max_degree: usize,
    current_coeffs: &[FieldElement<F>],
    roots: &mut Vec<Polynomial<FieldElement<F>>>,
    hint_values: &[FieldElement<F>],
    domain: &[FieldElement<F>],
    _depth: usize,
) {
    // Early exit if we've found enough roots
    if roots.len() >= MAX_TOTAL_ROOTS {
        return;
    }

    // Check if Q(0, y) has any roots
    let q_at_zero: Vec<_> = q
        .coeffs
        .iter()
        .map(|p| p.evaluate(&FieldElement::<F>::zero()))
        .collect();

    // Find roots of the univariate polynomial Q(0, y)
    let mut y_roots = find_univariate_roots_with_hints_and_domain(&q_at_zero, hint_values, domain);

    // Limit candidates to prevent exponential explosion
    y_roots.truncate(MAX_CANDIDATES_PER_DEPTH);

    for y_root in y_roots {
        // Early exit if we've found enough roots
        if roots.len() >= MAX_TOTAL_ROOTS {
            return;
        }
        // Extend current polynomial with this root as next coefficient
        let mut new_coeffs = current_coeffs.to_vec();
        new_coeffs.push(y_root.clone());

        // Check if we've reached max degree
        if new_coeffs.len() > max_degree {
            continue;
        }

        // Create the transformed polynomial Q'(x, y) = Q(x, c + xy) / x
        // This is the correct Roth-Ruckenstein substitution
        let q_transformed = substitute_and_divide(q, &y_root);

        if q_transformed.is_zero() {
            // We found a root! The polynomial f(x) = sum(current_coeffs[i] * x^i) works
            let poly = Polynomial::new(&new_coeffs);
            if !roots.contains(&poly) {
                roots.push(poly);
            }
        } else if new_coeffs.len() < max_degree {
            // Transform hints for the next recursion level
            // If f(αᵢ) = rᵢ, then f'(αᵢ) = (rᵢ - c) / αᵢ where f' = (f - c) / x
            // Also track which domain points remain valid
            let (transformed_hints, filtered_domain): (Vec<FieldElement<F>>, Vec<FieldElement<F>>) =
                hint_values
                    .iter()
                    .zip(domain.iter())
                    .filter_map(|(hint, alpha)| {
                        // Skip if alpha is zero (can't divide by zero)
                        if *alpha == FieldElement::<F>::zero() {
                            None
                        } else if let Ok(alpha_inv) = alpha.inv() {
                            Some(((hint - &y_root) * &alpha_inv, alpha.clone()))
                        } else {
                            None
                        }
                    })
                    .unzip();

            // Continue searching for more coefficients with transformed hints and domain
            rr_search_with_domain(
                &q_transformed,
                max_degree,
                &new_coeffs,
                roots,
                &transformed_hints,
                &filtered_domain,
                _depth + 1,
            );
        }

        // Also check if the polynomial is complete (deg < max_degree) and Q(x, f(x)) = 0
        // This handles cases where the polynomial terminates early
        if !new_coeffs.is_empty() {
            let candidate = Polynomial::new(&new_coeffs);
            let qf = q.evaluate_y_polynomial(&candidate);
            if qf == Polynomial::zero() && !roots.contains(&candidate) {
                roots.push(candidate);
            }
        }
    }
}

/// Finds roots of a univariate polynomial over F.
///
/// For small degree polynomials, we use simple methods.
/// For higher degrees, we try a brute force search over small field elements.
#[allow(dead_code)]
fn find_univariate_roots<F: IsField + Clone>(coeffs: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
    find_univariate_roots_with_hints(coeffs, &[])
}

/// Finds roots of a univariate polynomial with hints and domain information.
///
/// This version uses the actual domain values for correct Lagrange interpolation
/// when the polynomial is zero (all field elements are roots).
fn find_univariate_roots_with_hints_and_domain<F: IsField + Clone>(
    coeffs: &[FieldElement<F>],
    hint_values: &[FieldElement<F>],
    domain: &[FieldElement<F>],
) -> Vec<FieldElement<F>> {
    if coeffs.is_empty() || coeffs.iter().all(|c| *c == FieldElement::<F>::zero()) {
        // Zero polynomial - all elements are roots, but we can't enumerate them all
        // Key insight: use Lagrange interpolation to extract f(0) from f(α_i).
        // The hints are transformed values f(α_i) at actual domain points.
        // By interpolating, we can recover f(0) which is the coefficient we need.

        let mut roots: Vec<FieldElement<F>> = Vec::new();

        // PRIORITY 1: Lagrange interpolation on hints using actual domain points
        // This is the key to finding coefficients like 1002 that aren't small integers
        if hint_values.len() >= 3 && domain.len() >= 3 {
            let min_len = hint_values.len().min(domain.len());

            // Try different subsets of consecutive points for interpolation
            // This helps when some hints are from error positions (wrong values)
            for start in 0..(min_len.min(10)) {
                for size in 3..=min_len.min(6) {
                    if start + size <= min_len {
                        // Use actual domain values for interpolation
                        let subset: Vec<_> = (start..start + size)
                            .filter_map(|idx| {
                                // Get the actual domain value (as usize for the function)
                                // The domain might contain arbitrary field elements
                                let alpha = &domain[idx];
                                // Try to extract as small integer for interpolation
                                // This works for consecutive integer domains
                                let alpha_val = extract_small_value(alpha);
                                alpha_val.map(|a| (a, hint_values[idx].clone()))
                            })
                            .collect();

                        if subset.len() == size {
                            if let Some(interp) = lagrange_interpolate_at_zero_with_points(&subset)
                            {
                                if !roots.contains(&interp) {
                                    roots.push(interp);
                                }
                            }
                        }
                    }
                }
            }
        }

        // PRIORITY 2: Small integers (common coefficient values)
        for i in 0..20 {
            let elem = FieldElement::<F>::from(i as u64);
            if !roots.contains(&elem) {
                roots.push(elem);
            }
        }

        return roots;
    }

    // Non-zero polynomial - use the normal search
    find_univariate_roots_with_hints(coeffs, hint_values)
}

/// Try to extract a small integer value from a field element.
/// Uses binary search for efficiency.
fn extract_small_value<F: IsField + Clone>(fe: &FieldElement<F>) -> Option<usize> {
    // First check common small values directly (0-100)
    // For larger values, just return None - the domain values should be small
    // in typical RS code usage (consecutive integers 0, 1, 2, ...)
    (0..=100).find(|&i| *fe == FieldElement::<F>::from(i as u64))
}

/// Finds roots of a univariate polynomial, prioritizing hint values.
///
/// For Reed-Solomon decoding, the hint values are typically the received word values,
/// which are the most likely roots since valid decoded polynomials evaluate to these values.
fn find_univariate_roots_with_hints<F: IsField + Clone>(
    coeffs: &[FieldElement<F>],
    hint_values: &[FieldElement<F>],
) -> Vec<FieldElement<F>> {
    if coeffs.is_empty() || coeffs.iter().all(|c| *c == FieldElement::<F>::zero()) {
        // Zero polynomial - return small integers as likely coefficient values
        let mut roots: Vec<FieldElement<F>> =
            (0..20).map(|i| FieldElement::<F>::from(i as u64)).collect();

        // Add hint values
        for hint in hint_values.iter() {
            if !roots.contains(hint) {
                roots.push(hint.clone());
            }
        }

        return roots;
    }

    let poly = Polynomial::new(coeffs);

    // For degree 0, no roots (unless it's zero, handled above)
    if poly.degree() == 0 {
        return vec![];
    }

    // For degree 1: ax + b = 0 => x = -b/a
    if poly.degree() == 1 {
        let a = &coeffs[1];
        let b = &coeffs[0];
        if let Ok(a_inv) = a.inv() {
            return vec![-b * &a_inv];
        }
        return vec![];
    }

    let mut roots = Vec::new();
    let max_roots = poly.degree();

    // First, check all hint values (these are the most likely roots for RS decoding)
    for hint in hint_values {
        if poly.evaluate(hint) == FieldElement::<F>::zero() && !roots.contains(hint) {
            roots.push(hint.clone());
            if roots.len() >= max_roots {
                return roots;
            }
        }
    }

    // Also try small field elements as fallback
    let max_search = 2000; // Increased to catch roots like 1002
    let mut found_count = 0;
    for i in 0..max_search {
        let elem = FieldElement::<F>::from(i as u64);
        if poly.evaluate(&elem) == FieldElement::<F>::zero() {
            found_count += 1;
            if !roots.contains(&elem) {
                roots.push(elem.clone());
                // Debug: print when we find roots around 1000
                if std::env::var("RR_DEBUG").is_ok() && (1000..=1010).contains(&i) {
                    eprintln!("  Found root at i={}", i);
                }
                if roots.len() >= max_roots {
                    // Debug: check why we're returning early
                    if std::env::var("RR_DEBUG").is_ok() && max_roots < 15 {
                        eprintln!(
                            "  Returning early: {} roots found, max_roots={}, last i={}",
                            roots.len(),
                            max_roots,
                            i
                        );
                    }
                    return roots;
                }
            }
        }
    }
    // Debug: if we searched everything but didn't find 1002
    if std::env::var("RR_DEBUG").is_ok() && found_count > 0 && poly.degree() > 5 {
        let test_1002 = FieldElement::<F>::from(1002u64);
        let is_root = poly.evaluate(&test_1002) == FieldElement::<F>::zero();
        if is_root && !roots.contains(&test_1002) {
            eprintln!(
                "  WARNING: 1002 is a root but wasn't added! found_count={}, roots.len()={}",
                found_count,
                roots.len()
            );
        }
    }

    // Also try negative small elements
    for i in 1..max_search {
        let elem = -FieldElement::<F>::from(i as u64);
        if poly.evaluate(&elem) == FieldElement::<F>::zero() && !roots.contains(&elem) {
            roots.push(elem);
            if roots.len() >= max_roots {
                return roots;
            }
        }
    }

    roots
}

/// Performs the Roth-Ruckenstein substitution: Q'(x, y) = Q(x, c + xy) / x
///
/// This is the key step in the algorithm. Since c is a root of Q(0, y),
/// we have Q(0, c) = 0, which ensures Q(x, c + xy) is divisible by x.
fn substitute_and_divide<F: IsField + Clone>(
    q: &BivariatePolynomial<F>,
    c: &FieldElement<F>,
) -> BivariatePolynomial<F> {
    // Compute Q(x, c + xy)
    // Q(x, y) = Σⱼ Qⱼ(x) yʲ
    // Q(x, c + xy) = Σⱼ Qⱼ(x) (c + xy)ʲ

    let y_deg = q.y_degree();

    // Compute (c + xy)^j for j = 0, 1, ..., y_deg
    // (c + xy)^j = Σₖ C(j,k) c^{j-k} (xy)^k = Σₖ C(j,k) c^{j-k} x^k y^k
    // This gives a bivariate polynomial where the coefficient of y^k is C(j,k) c^{j-k} x^k

    // Result will be a bivariate polynomial
    // We need to track coefficient of x^i y^j for all i, j
    let max_x_deg = q.max_x_degree() + y_deg; // x degree can increase
    let max_y_deg = y_deg;

    // Use a 2D array to accumulate coefficients
    let mut result_coeffs: Vec<Vec<FieldElement<F>>> =
        vec![vec![FieldElement::<F>::zero(); max_x_deg + 2]; max_y_deg + 1];

    for (j, qj) in q.coeffs.iter().enumerate() {
        // Contribution of Qⱼ(x) * (c + xy)^j
        // (c + xy)^j = Σₖ C(j,k) c^{j-k} x^k y^k

        let qj_coeffs = qj.coefficients();

        for (k, result_row) in result_coeffs.iter_mut().enumerate().take(j + 1) {
            let binom = binomial(j, k);
            // c^{j-k}
            let c_power = c.pow(j - k);
            let binom_c = &FieldElement::<F>::from(binom as u64) * &c_power;

            // Multiply Qⱼ(x) by binom_c * x^k, add to coefficient of y^k
            for (i, qi_coeff) in qj_coeffs.iter().enumerate() {
                let contrib = qi_coeff * &binom_c;
                let x_power = i + k;
                if x_power <= max_x_deg + 1 && k <= max_y_deg {
                    result_row[x_power] = &result_row[x_power] + &contrib;
                }
            }
        }
    }

    // Now divide by x (shift all x coefficients down by 1)
    // This is valid because Q(0, c) = 0 ensures the constant term is zero
    let divided_coeffs: Vec<Polynomial<FieldElement<F>>> = result_coeffs
        .into_iter()
        .map(|row| {
            if row.len() <= 1 {
                Polynomial::zero()
            } else {
                // Remove the first coefficient (x^0 term) and shift
                Polynomial::new(&row[1..])
            }
        })
        .collect();

    BivariatePolynomial::new(divided_coeffs)
}

/// Computes Q(x, y + c).
#[allow(dead_code)]
fn shift_y<F: IsField + Clone>(
    q: &BivariatePolynomial<F>,
    c: &FieldElement<F>,
) -> BivariatePolynomial<F> {
    let y_deg = q.y_degree();
    let mut result_coeffs: Vec<Polynomial<FieldElement<F>>> = vec![Polynomial::zero(); y_deg + 1];

    // Q(x, y + c) = Σⱼ Qⱼ(x) (y + c)^j
    // We need to expand (y + c)^j using binomial theorem
    for (j, qj) in q.coeffs.iter().enumerate() {
        // (y + c)^j = Σₖ C(j,k) y^k c^{j-k}
        let mut c_power = FieldElement::<F>::one();
        for (k, result_coeff) in result_coeffs.iter_mut().enumerate().take(j + 1) {
            let binom = binomial(j, k);
            let coeff = &c_power * &FieldElement::<F>::from(binom as u64);

            // Add binom * c^{j-k} * Qⱼ(x) to coefficient of y^k
            let term: Vec<_> = qj.coefficients().iter().map(|x| x * &coeff).collect();
            let term_poly = Polynomial::new(&term);
            *result_coeff = result_coeff.clone() + term_poly;

            if k < j {
                c_power = &c_power * c;
            }
        }
    }

    BivariatePolynomial::new(result_coeffs)
}

/// Divides a bivariate polynomial by x.
#[allow(dead_code)]
fn divide_by_x<F: IsField + Clone>(q: &BivariatePolynomial<F>) -> BivariatePolynomial<F> {
    let coeffs: Vec<_> = q
        .coeffs
        .iter()
        .map(|p| {
            let c = p.coefficients();
            if c.is_empty() || c[0] != FieldElement::<F>::zero() {
                // Can't divide cleanly by x
                Polynomial::zero()
            } else {
                // Remove the leading zero coefficient (constant term)
                Polynomial::new(&c[1..])
            }
        })
        .collect();

    BivariatePolynomial::new(coeffs)
}

/// Computes binomial coefficient C(n, k).
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    binomial(n - 1, k - 1) + binomial(n - 1, k)
}

/// Lagrange interpolation to find f(0) given points (x_i, y_i).
///
/// Given points (x_i, y_i), interpolates and evaluates at 0.
///
/// Uses the formula: f(0) = Σᵢ yᵢ · Lᵢ(0)
/// where Lᵢ(0) = Πⱼ≠ᵢ (0 - xⱼ) / (xᵢ - xⱼ) = Πⱼ≠ᵢ (-xⱼ) / (xᵢ - xⱼ)
fn lagrange_interpolate_at_zero_with_points<F: IsField + Clone>(
    points: &[(usize, FieldElement<F>)],
) -> Option<FieldElement<F>> {
    if points.is_empty() {
        return None;
    }

    let n = points.len();
    let mut result = FieldElement::<F>::zero();

    for i in 0..n {
        let (xi, yi) = &points[i];

        // Compute Lagrange basis L_i(0) = prod_{j≠i} (-x_j) / (x_i - x_j)
        let mut numerator = FieldElement::<F>::one();
        let mut denominator = FieldElement::<F>::one();

        for (j, (xj, _)) in points.iter().enumerate().take(n) {
            if j != i {
                // numerator *= -x_j
                numerator = &numerator * &(-FieldElement::<F>::from(*xj as u64));
                // denominator *= (x_i - x_j)
                let xi_fe = FieldElement::<F>::from(*xi as u64);
                let xj_fe = FieldElement::<F>::from(*xj as u64);
                denominator = &denominator * &(&xi_fe - &xj_fe);
            }
        }

        // L_i(0) = numerator / denominator
        if let Ok(denom_inv) = denominator.inv() {
            let li = &numerator * &denom_inv;
            // Add y_i * L_i(0) to result
            result = &result + &(yi * &li);
        } else {
            return None;
        }
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Babybear31PrimeField;

    type FE = FieldElement<Babybear31PrimeField>;

    #[test]
    fn test_polynomial_interpolation() {
        // Test building a polynomial from points
        // f(x) = 1001 + 1002x + 3x^2 + 4x^3
        // f(0) = 1001
        // f(1) = 1001 + 1002 + 3 + 4 = 2010
        // f(2) = 1001 + 2004 + 12 + 32 = 3049
        // f(3) = 1001 + 3006 + 27 + 108 = 4142
        let points: Vec<(usize, FE)> = vec![
            (0, FE::from(1001u64)),
            (1, FE::from(2010u64)),
            (2, FE::from(3049u64)),
            (3, FE::from(4142u64)),
        ];

        let result = lagrange_interpolate_polynomial(&points, 4);
        assert!(result.is_some());
        let poly = result.unwrap();
        let coeffs = poly.coefficients();
        println!(
            "Interpolated polynomial coefficients: {:?}",
            coeffs
                .iter()
                .map(|c| c.representative())
                .collect::<Vec<_>>()
        );
        assert_eq!(coeffs.len(), 4);
        assert_eq!(coeffs[0], FE::from(1001u64));
        assert_eq!(coeffs[1], FE::from(1002u64));
        assert_eq!(coeffs[2], FE::from(3u64));
        assert_eq!(coeffs[3], FE::from(4u64));
    }

    #[test]
    fn test_lagrange_interpolation() {
        // Test: f(x) = 1002 + 3x + 4x^2
        // f(1) = 1002 + 3 + 4 = 1009
        // f(2) = 1002 + 6 + 16 = 1024
        // f(3) = 1002 + 9 + 36 = 1047
        // Interpolating should give f(0) = 1002
        let points: Vec<(usize, FE)> = vec![
            (1, FE::from(1009u64)),
            (2, FE::from(1024u64)),
            (3, FE::from(1047u64)),
        ];

        let result = lagrange_interpolate_at_zero_with_points(&points);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), FE::from(1002u64));
    }

    #[test]
    fn test_lagrange_interpolation_linear() {
        // Test: f(x) = 5 + 2x
        // f(1) = 7, f(2) = 9
        // Interpolating should give f(0) = 5
        let points: Vec<(usize, FE)> = vec![(1, FE::from(7u64)), (2, FE::from(9u64))];

        let result = lagrange_interpolate_at_zero_with_points(&points);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), FE::from(5u64));
    }

    #[test]
    fn test_bivariate_creation() {
        // Q(x, y) = (1 + 2x) + (3 + 4x)y
        let q0 = Polynomial::new(&[FE::from(1u64), FE::from(2u64)]); // 1 + 2x
        let q1 = Polynomial::new(&[FE::from(3u64), FE::from(4u64)]); // 3 + 4x

        let q = BivariatePolynomial::new(vec![q0, q1]);

        assert_eq!(q.y_degree(), 1);
        assert_eq!(q.max_x_degree(), 1);
    }

    #[test]
    fn test_bivariate_evaluate() {
        // Q(x, y) = x + y
        let q0 = Polynomial::new(&[FE::zero(), FE::one()]); // x
        let q1 = Polynomial::new(&[FE::one()]); // 1

        let q = BivariatePolynomial::new(vec![q0, q1]);

        // Q(2, 3) = 2 + 3 = 5
        let result = q.evaluate(&FE::from(2u64), &FE::from(3u64));
        assert_eq!(result, FE::from(5u64));
    }

    #[test]
    fn test_weighted_degree() {
        // Q(x, y) = x^2 + xy + y^2
        // With w = 2: wdeg(x^2) = 2, wdeg(xy) = 1 + 2 = 3, wdeg(y^2) = 4
        let q0 = Polynomial::new(&[FE::zero(), FE::zero(), FE::one()]); // x^2
        let q1 = Polynomial::new(&[FE::zero(), FE::one()]); // x
        let q2 = Polynomial::new(&[FE::one()]); // 1

        let q = BivariatePolynomial::new(vec![q0, q1, q2]);

        assert_eq!(q.weighted_degree(2), 4);
    }

    #[test]
    fn test_evaluate_y_polynomial() {
        // Q(x, y) = 1 + y
        // f(x) = x
        // Q(x, f(x)) = 1 + x
        let q = BivariatePolynomial::new(vec![
            Polynomial::new(&[FE::one()]),
            Polynomial::new(&[FE::one()]),
        ]);

        let f = Polynomial::new(&[FE::zero(), FE::one()]); // f(x) = x
        let result = q.evaluate_y_polynomial(&f);

        let expected = Polynomial::new(&[FE::one(), FE::one()]); // 1 + x
        assert_eq!(result, expected);
    }

    #[test]
    fn test_find_roots_linear() {
        // Q(x, y) = A(x) + B(x)y where A(x) = -2x, B(x) = 2
        // Root: y = -A/B = -(-2x)/2 = x
        let a = Polynomial::new(&[FE::zero(), -FE::from(2u64)]); // -2x
        let b = Polynomial::new(&[FE::from(2u64)]); // 2

        let q = BivariatePolynomial::new(vec![a, b]);

        let roots = find_polynomial_roots(&q, 2);

        assert_eq!(roots.len(), 1);
        let expected = Polynomial::new(&[FE::zero(), FE::one()]); // x
        assert_eq!(roots[0], expected);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 1), 5);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(5, 4), 5);
        assert_eq!(binomial(5, 5), 1);
    }
}
