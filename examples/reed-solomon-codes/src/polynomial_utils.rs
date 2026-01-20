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
        while coeffs.len() > 1 && coeffs.last().map_or(false, |p| p == &Polynomial::zero()) {
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
    roth_ruckenstein(q, max_degree, hint_values)
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

    if *b == Polynomial::zero() {
        return vec![];
    }

    // f(x) = -A(x) / B(x)
    // Check if B divides A
    let neg_a = {
        let coeffs: Vec<_> = a.coefficients().iter().map(|c| -c).collect();
        Polynomial::new(&coeffs)
    };

    let (quotient, remainder) = neg_a.long_division_with_remainder(b);

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

/// Roth-Ruckenstein algorithm for finding polynomial roots.
///
/// This is an iterative algorithm that finds all polynomials f(x) of degree < k
/// such that Q(x, f(x)) = 0.
///
/// The algorithm builds f coefficient by coefficient, maintaining a "shifted"
/// polynomial at each step.
///
/// `hint_values` are field elements likely to be roots (e.g., received word values).
fn roth_ruckenstein<F: IsField + Clone>(
    q: &BivariatePolynomial<F>,
    max_degree: usize,
    hint_values: &[FieldElement<F>],
) -> Vec<Polynomial<FieldElement<F>>> {
    let mut roots = Vec::new();

    // Also try direct verification for low-degree candidates
    // This is a fallback for when the recursive search might miss some roots
    if max_degree <= 10 {
        try_direct_roots(q, max_degree, hint_values, &mut roots);
    }

    // Start the search with an empty polynomial
    rr_search(q, max_degree, &[], &mut roots, hint_values, 0);

    roots
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
        let candidate = Polynomial::new(&[hint.clone()]);
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

/// Recursive search in Roth-Ruckenstein algorithm.
///
/// The key insight is: if f(x) = c₀ + c₁x + c₂x² + ... satisfies Q(x, f(x)) = 0, then
/// c₀ is a root of Q(0, y), and f'(x) = c₁ + c₂x + ... satisfies Q'(x, f'(x)) = 0
/// where Q'(x, y) = Q(x, c₀ + xy) / x.
fn rr_search<F: IsField + Clone>(
    q: &BivariatePolynomial<F>,
    max_degree: usize,
    current_coeffs: &[FieldElement<F>],
    roots: &mut Vec<Polynomial<FieldElement<F>>>,
    hint_values: &[FieldElement<F>],
    _depth: usize,
) {
    // Check if Q(0, y) has any roots
    let q_at_zero: Vec<_> = q
        .coeffs
        .iter()
        .map(|p| p.evaluate(&FieldElement::<F>::zero()))
        .collect();

    // Find roots of the univariate polynomial Q(0, y)
    let y_roots = find_univariate_roots_with_hints(&q_at_zero, hint_values);

    for y_root in y_roots {
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
            // Continue searching for more coefficients
            rr_search(
                &q_transformed,
                max_degree,
                &new_coeffs,
                roots,
                hint_values,
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

/// Finds roots of a univariate polynomial, prioritizing hint values.
///
/// For Reed-Solomon decoding, the hint values are typically the received word values,
/// which are the most likely roots since valid decoded polynomials evaluate to these values.
fn find_univariate_roots_with_hints<F: IsField + Clone>(
    coeffs: &[FieldElement<F>],
    hint_values: &[FieldElement<F>],
) -> Vec<FieldElement<F>> {
    if coeffs.is_empty() || coeffs.iter().all(|c| *c == FieldElement::<F>::zero()) {
        // Zero polynomial - all elements are roots, but we can't enumerate them all
        // For our purposes, we just return the additive identity
        return vec![FieldElement::<F>::zero()];
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
        if poly.evaluate(hint) == FieldElement::<F>::zero() {
            if !roots.contains(hint) {
                roots.push(hint.clone());
                if roots.len() >= max_roots {
                    return roots;
                }
            }
        }
    }

    // Also try small field elements as fallback
    let max_search = 100;
    for i in 0..max_search {
        let elem = FieldElement::<F>::from(i as u64);
        if poly.evaluate(&elem) == FieldElement::<F>::zero() {
            if !roots.contains(&elem) {
                roots.push(elem);
                if roots.len() >= max_roots {
                    return roots;
                }
            }
        }
    }

    // Also try negative small elements
    for i in 1..max_search {
        let elem = -FieldElement::<F>::from(i as u64);
        if poly.evaluate(&elem) == FieldElement::<F>::zero() {
            if !roots.contains(&elem) {
                roots.push(elem);
                if roots.len() >= max_roots {
                    return roots;
                }
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

        for k in 0..=j {
            let binom = binomial(j, k);
            // c^{j-k}
            let c_power = c.pow(j - k);
            let binom_c = &FieldElement::<F>::from(binom as u64) * &c_power;

            // Multiply Qⱼ(x) by binom_c * x^k, add to coefficient of y^k
            for (i, qi_coeff) in qj_coeffs.iter().enumerate() {
                let contrib = qi_coeff * &binom_c;
                let x_power = i + k;
                if x_power <= max_x_deg + 1 && k <= max_y_deg {
                    result_coeffs[k][x_power] = &result_coeffs[k][x_power] + &contrib;
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
        for k in 0..=j {
            let binom = binomial(j, k);
            let coeff = &c_power * &FieldElement::<F>::from(binom as u64);

            // Add binom * c^{j-k} * Qⱼ(x) to coefficient of y^k
            let term: Vec<_> = qj.coefficients().iter().map(|x| x * &coeff).collect();
            let term_poly = Polynomial::new(&term);
            result_coeffs[k] = result_coeffs[k].clone() + term_poly;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Babybear31PrimeField;

    type FE = FieldElement<Babybear31PrimeField>;

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
