//! Constraint helper functions for ZK Mastermind circuit
//!
//! # Educational Example - Runtime Helpers Only
//!
//! **WARNING**: The functions in this module perform **runtime computations**,
//! NOT algebraic constraint generation. They are used for:
//! - Testing and validation
//! - Trace table generation (computing witness values)
//! - Educational demonstration of Mastermind logic
//!
//! These functions do NOT generate ZK-compatible algebraic constraints.
//! For a production ZK circuit, you would need to:
//! 1. Replace branching logic with polynomial arithmetic
//! 2. Use auxiliary witness columns for intermediate values
//! 3. Express all operations as low-degree polynomial constraints
//!
//! For example, `is_zero(x)` in a real ZK circuit requires:
//! - An auxiliary witness `inv_x` (inverse of x, or 0 if x=0)
//! - Constraints: `x * inv_x = 1 - is_zero` and `x * is_zero = 0`

use lambdaworks_math::field::{element::FieldElement, traits::IsFFTField};

/// Check if a value is zero, returns 1 if zero, 0 otherwise.
///
/// **Note**: This is a runtime computation, not an algebraic constraint.
/// In a real ZK circuit, this would require auxiliary witness columns
/// and polynomial constraints as described in the module documentation.
pub fn is_zero<F: IsFFTField>(value: &FieldElement<F>) -> FieldElement<F> {
    if *value == FieldElement::zero() {
        FieldElement::one()
    } else {
        FieldElement::zero()
    }
}

/// Check if two values are equal using the is_zero function
pub fn is_equal<F: IsFFTField>(a: &FieldElement<F>, b: &FieldElement<F>) -> FieldElement<F> {
    is_zero(&(a - b))
}

/// Range check: verify that value is in range [0, max_value - 1]
/// Returns a value that should be 0 if the constraint is satisfied
///
/// The constraint is: value * (value - 1) * ... * (value - max_value + 1) = 0
pub fn range_check<F: IsFFTField>(value: &FieldElement<F>, max_value: u64) -> FieldElement<F> {
    let mut product = FieldElement::<F>::one();
    for i in 0..max_value {
        product *= value - FieldElement::<F>::from(i);
    }
    product
}

/// Compute the exact match count between secret and guess
/// Returns the number of positions where secret[i] == guess[i]
pub fn compute_exact_matches<F: IsFFTField>(
    secret: &[FieldElement<F>; 4],
    guess: &[FieldElement<F>; 4],
) -> FieldElement<F> {
    let mut sum = FieldElement::zero();
    for i in 0..4 {
        let eq = is_equal(&secret[i], &guess[i]);
        sum += eq;
    }
    sum
}

/// Compute color counts for partial match calculation
/// Returns arrays of length 6 with counts for each color
pub fn compute_color_counts<F: IsFFTField>(code: &[FieldElement<F>; 4]) -> [FieldElement<F>; 6] {
    let mut counts = [
        FieldElement::zero(),
        FieldElement::zero(),
        FieldElement::zero(),
        FieldElement::zero(),
        FieldElement::zero(),
        FieldElement::zero(),
    ];

    for item in code.iter() {
        // Get the color value as u64
        // We use a simple approach: convert to u64 by iterating
        for color in 0..6u64 {
            let color_fe = FieldElement::<F>::from(color);
            if *item == color_fe {
                counts[color as usize] = counts[color as usize].clone() + FieldElement::one();
                break;
            }
        }
    }

    counts
}

/// Check if a <= b for small field element values (0-10)
/// This works by comparing against known small values
fn is_less_than_or_equal<F: IsFFTField>(a: &FieldElement<F>, b: &FieldElement<F>) -> bool {
    // For small values, we can check by iterating
    for i in 0..10u64 {
        let val = FieldElement::from(i);
        if *a == val {
            // Found a, now check if b >= i
            for j in i..10u64 {
                if *b == FieldElement::from(j) {
                    return true;
                }
            }
            return false;
        }
    }
    // Default: compare by subtraction (works if we know values are small)
    let diff = b - a;
    // Check if diff corresponds to a small non-negative number
    for i in 0..10u64 {
        if diff == FieldElement::from(i) {
            return true;
        }
    }
    false
}

/// Compute partial matches
/// partial = sum(min(secret_count[c], guess_count[c])) - exact
pub fn compute_partial_matches<F: IsFFTField>(
    secret: &[FieldElement<F>; 4],
    guess: &[FieldElement<F>; 4],
    exact: &FieldElement<F>,
) -> FieldElement<F> {
    let secret_counts = compute_color_counts(secret);
    let guess_counts = compute_color_counts(guess);

    // Sum of min counts
    let mut sum_mins = FieldElement::<F>::zero();
    for c in 0..6 {
        // Compare field elements directly for small values (0-4)
        let s = &secret_counts[c];
        let g = &guess_counts[c];

        // Determine min by comparing with small values
        let min_val = if is_less_than_or_equal(s, g) {
            s.clone()
        } else {
            g.clone()
        };
        sum_mins += min_val;
    }

    // partial = sum_mins - exact
    sum_mins - exact
}

/// Verify that the computed feedback matches the expected feedback
pub fn verify_feedback_constraint<F: IsFFTField>(
    secret: &[FieldElement<F>; 4],
    guess: &[FieldElement<F>; 4],
    expected_exact: &FieldElement<F>,
    expected_partial: &FieldElement<F>,
) -> (FieldElement<F>, FieldElement<F>) {
    let computed_exact = compute_exact_matches(secret, guess);
    let computed_partial = compute_partial_matches(secret, guess, &computed_exact);

    let exact_diff = computed_exact - expected_exact;
    let partial_diff = computed_partial - expected_partial;

    (exact_diff, partial_diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

    type F = Stark252PrimeField;

    #[test]
    fn test_is_zero() {
        let zero = FieldElement::<F>::zero();
        let one = FieldElement::<F>::one();

        assert_eq!(is_zero(&zero), FieldElement::one());
        assert_eq!(is_zero(&one), FieldElement::zero());
    }

    #[test]
    fn test_is_equal() {
        let a = FieldElement::<F>::from(5u64);
        let b = FieldElement::<F>::from(5u64);
        let c = FieldElement::<F>::from(3u64);

        assert_eq!(is_equal(&a, &b), FieldElement::one());
        assert_eq!(is_equal(&a, &c), FieldElement::zero());
    }

    #[test]
    fn test_range_check() {
        // Values 0-5 should pass (result = 0)
        for i in 0..6 {
            let val = FieldElement::<F>::from(i);
            let result = range_check(&val, 6);
            assert_eq!(
                result,
                FieldElement::zero(),
                "Value {} should pass range check",
                i
            );
        }

        // Value 6 should fail (result != 0)
        let val = FieldElement::<F>::from(6u64);
        let result = range_check(&val, 6);
        assert_ne!(
            result,
            FieldElement::zero(),
            "Value 6 should fail range check"
        );
    }

    #[test]
    fn test_compute_exact_matches() {
        let secret = [
            FieldElement::<F>::from(0u64), // Red
            FieldElement::<F>::from(1u64), // Blue
            FieldElement::<F>::from(2u64), // Green
            FieldElement::<F>::from(3u64), // Yellow
        ];

        // Exact match at positions 0 and 1
        let guess = [
            FieldElement::<F>::from(0u64), // Red (exact)
            FieldElement::<F>::from(1u64), // Blue (exact)
            FieldElement::<F>::from(4u64), // Orange (wrong)
            FieldElement::<F>::from(5u64), // Purple (wrong)
        ];

        let exact = compute_exact_matches(&secret, &guess);
        assert_eq!(exact, FieldElement::from(2u64));
    }

    #[test]
    fn test_compute_color_counts() {
        let code = [
            FieldElement::<F>::from(0u64), // Red
            FieldElement::<F>::from(0u64), // Red
            FieldElement::<F>::from(1u64), // Blue
            FieldElement::<F>::from(2u64), // Green
        ];

        let counts = compute_color_counts(&code);
        assert_eq!(counts[0], FieldElement::from(2u64)); // 2 Reds
        assert_eq!(counts[1], FieldElement::from(1u64)); // 1 Blue
        assert_eq!(counts[2], FieldElement::from(1u64)); // 1 Green
        assert_eq!(counts[3], FieldElement::from(0u64)); // 0 Yellow
    }
}
