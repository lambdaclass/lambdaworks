//! Hamming distance and related utilities for error-correcting codes.
//!
//! # Definitions
//!
//! - **Hamming weight** w(c): Number of non-zero positions in a codeword
//! - **Hamming distance** d(a, b): Number of positions where a and b differ
//!
//! # Properties
//!
//! - d(a, b) = w(a - b) for linear codes
//! - Minimum distance of a code: d_min = min { d(a, b) : a ≠ b, a, b ∈ C }
//! - For linear codes: d_min = min { w(c) : c ≠ 0, c ∈ C }
//!
//! # Error Correction
//!
//! A code with minimum distance d can:
//! - Detect up to d - 1 errors
//! - Correct up to floor((d - 1) / 2) errors

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

/// Computes the Hamming weight of a vector.
///
/// The Hamming weight is the number of non-zero entries.
///
/// # Example
///
/// ```
/// use reed_solomon_codes::distance::hamming_weight;
/// use reed_solomon_codes::FE;
///
/// let v = vec![FE::from(0u64), FE::from(5u64), FE::from(0u64), FE::from(3u64)];
/// assert_eq!(hamming_weight(&v), 2);
/// ```
pub fn hamming_weight<F: IsField>(v: &[FieldElement<F>]) -> usize {
    v.iter()
        .filter(|x| **x != FieldElement::<F>::zero())
        .count()
}

/// Computes the Hamming distance between two vectors.
///
/// The Hamming distance is the number of positions where the vectors differ.
///
/// # Panics
///
/// Panics if the vectors have different lengths.
///
/// # Example
///
/// ```
/// use reed_solomon_codes::distance::hamming_distance;
/// use reed_solomon_codes::FE;
///
/// let a = vec![FE::from(1u64), FE::from(2u64), FE::from(3u64), FE::from(4u64)];
/// let b = vec![FE::from(1u64), FE::from(5u64), FE::from(3u64), FE::from(6u64)];
/// assert_eq!(hamming_distance(&a, &b), 2); // differ at positions 1 and 3
/// ```
pub fn hamming_distance<F: IsField>(a: &[FieldElement<F>], b: &[FieldElement<F>]) -> usize {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    a.iter().zip(b.iter()).filter(|(x, y)| *x != *y).count()
}

/// Computes the relative Hamming distance (normalized by length).
///
/// Returns a value in [0, 1] representing the fraction of differing positions.
pub fn relative_hamming_distance<F: IsField>(a: &[FieldElement<F>], b: &[FieldElement<F>]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    hamming_distance(a, b) as f64 / a.len() as f64
}

/// Introduces errors into a codeword at specified positions.
///
/// # Arguments
///
/// * `codeword` - The original codeword
/// * `error_positions` - Indices where errors should be introduced
/// * `error_values` - Values to add at error positions
///
/// # Returns
///
/// A new vector with errors introduced (codeword + error vector).
///
/// # Panics
///
/// Panics if error_positions and error_values have different lengths,
/// or if any position is out of bounds.
pub fn introduce_errors<F: IsField + Clone>(
    codeword: &[FieldElement<F>],
    error_positions: &[usize],
    error_values: &[FieldElement<F>],
) -> Vec<FieldElement<F>> {
    assert_eq!(
        error_positions.len(),
        error_values.len(),
        "Error positions and values must have same length"
    );

    let mut corrupted = codeword.to_vec();
    for (&pos, error) in error_positions.iter().zip(error_values.iter()) {
        assert!(pos < codeword.len(), "Error position {} out of bounds", pos);
        corrupted[pos] = &corrupted[pos] + error;
    }

    corrupted
}

/// Introduces random-like errors at specified positions.
///
/// Uses a simple deterministic pattern for reproducibility in tests.
pub fn introduce_errors_at_positions<F: IsField + Clone>(
    codeword: &[FieldElement<F>],
    error_positions: &[usize],
) -> Vec<FieldElement<F>> {
    let error_values: Vec<FieldElement<F>> = error_positions
        .iter()
        .enumerate()
        .map(|(i, _)| FieldElement::<F>::from((i + 1) as u64))
        .collect();

    introduce_errors(codeword, error_positions, &error_values)
}

/// Counts the number of errors between received word and original codeword.
pub fn count_errors<F: IsField>(
    received: &[FieldElement<F>],
    original: &[FieldElement<F>],
) -> usize {
    hamming_distance(received, original)
}

/// Finds the positions where two vectors differ.
pub fn error_positions<F: IsField>(
    received: &[FieldElement<F>],
    original: &[FieldElement<F>],
) -> Vec<usize> {
    assert_eq!(
        received.len(),
        original.len(),
        "Vectors must have same length"
    );

    received
        .iter()
        .zip(original.iter())
        .enumerate()
        .filter(|(_, (r, o))| r != o)
        .map(|(i, _)| i)
        .collect()
}

/// Computes the agreement between a received word and a polynomial.
///
/// Returns the number of positions where the polynomial evaluation matches
/// the received value. This is n - d(received, codeword).
pub fn agreement<F: IsField + Clone>(
    received: &[FieldElement<F>],
    domain: &[FieldElement<F>],
    poly: &lambdaworks_math::polynomial::Polynomial<FieldElement<F>>,
) -> usize {
    assert_eq!(
        received.len(),
        domain.len(),
        "Received word and domain must have same length"
    );

    received
        .iter()
        .zip(domain.iter())
        .filter(|(r, alpha)| **r == poly.evaluate(alpha))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Babybear31PrimeField;

    type FE = FieldElement<Babybear31PrimeField>;

    #[test]
    fn test_hamming_weight() {
        let zero_vec = vec![FE::zero(); 5];
        assert_eq!(hamming_weight(&zero_vec), 0);

        let one_nonzero = vec![FE::zero(), FE::from(1u64), FE::zero()];
        assert_eq!(hamming_weight(&one_nonzero), 1);

        let all_nonzero = vec![FE::from(1u64), FE::from(2u64), FE::from(3u64)];
        assert_eq!(hamming_weight(&all_nonzero), 3);
    }

    #[test]
    fn test_hamming_distance() {
        let a = vec![FE::from(1u64), FE::from(2u64), FE::from(3u64)];
        let b = vec![FE::from(1u64), FE::from(2u64), FE::from(3u64)];
        assert_eq!(hamming_distance(&a, &b), 0);

        let c = vec![FE::from(1u64), FE::from(5u64), FE::from(3u64)];
        assert_eq!(hamming_distance(&a, &c), 1);

        let d = vec![FE::from(4u64), FE::from(5u64), FE::from(6u64)];
        assert_eq!(hamming_distance(&a, &d), 3);
    }

    #[test]
    fn test_hamming_distance_equals_weight_of_difference() {
        // For linear codes: d(a, b) = w(a - b)
        let a = vec![FE::from(1u64), FE::from(2u64), FE::from(3u64)];
        let b = vec![FE::from(1u64), FE::from(5u64), FE::from(6u64)];

        let diff: Vec<FE> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();

        assert_eq!(hamming_distance(&a, &b), hamming_weight(&diff));
    }

    #[test]
    fn test_introduce_errors() {
        let codeword = vec![
            FE::from(1u64),
            FE::from(2u64),
            FE::from(3u64),
            FE::from(4u64),
        ];
        let positions = vec![1, 3];
        let errors = vec![FE::from(10u64), FE::from(20u64)];

        let corrupted = introduce_errors(&codeword, &positions, &errors);

        assert_eq!(corrupted[0], FE::from(1u64)); // unchanged
        assert_eq!(corrupted[1], FE::from(12u64)); // 2 + 10
        assert_eq!(corrupted[2], FE::from(3u64)); // unchanged
        assert_eq!(corrupted[3], FE::from(24u64)); // 4 + 20
    }

    #[test]
    fn test_error_positions() {
        let original = vec![
            FE::from(1u64),
            FE::from(2u64),
            FE::from(3u64),
            FE::from(4u64),
        ];
        let received = vec![
            FE::from(1u64),
            FE::from(5u64),
            FE::from(3u64),
            FE::from(7u64),
        ];

        let positions = error_positions(&received, &original);
        assert_eq!(positions, vec![1, 3]);
    }

    #[test]
    fn test_agreement() {
        use lambdaworks_math::polynomial::Polynomial;

        // p(x) = 1 + x
        let poly = Polynomial::new(&[FE::from(1u64), FE::from(1u64)]);
        let domain = vec![
            FE::from(0u64),
            FE::from(1u64),
            FE::from(2u64),
            FE::from(3u64),
        ];

        // Correct evaluations: [1, 2, 3, 4]
        let correct_received = vec![
            FE::from(1u64),
            FE::from(2u64),
            FE::from(3u64),
            FE::from(4u64),
        ];
        assert_eq!(agreement(&correct_received, &domain, &poly), 4);

        // One error: [1, 2, 3, 100]
        let one_error = vec![
            FE::from(1u64),
            FE::from(2u64),
            FE::from(3u64),
            FE::from(100u64),
        ];
        assert_eq!(agreement(&one_error, &domain, &poly), 3);
    }

    #[test]
    fn test_relative_hamming_distance() {
        let a = vec![
            FE::from(1u64),
            FE::from(2u64),
            FE::from(3u64),
            FE::from(4u64),
        ];
        let b = vec![
            FE::from(1u64),
            FE::from(5u64),
            FE::from(3u64),
            FE::from(6u64),
        ];

        let rel_dist = relative_hamming_distance(&a, &b);
        assert!((rel_dist - 0.5).abs() < 1e-10); // 2 out of 4 differ
    }
}
