//! Reed-Solomon code structure and encoding.
//!
//! A Reed-Solomon code RS[n, k, d] over a finite field F_q encodes messages
//! of length k into codewords of length n by interpreting the message as
//! coefficients of a polynomial and evaluating it at n distinct points.
//!
//! # Parameters
//!
//! - **n** (code length): Number of evaluation points in the domain
//! - **k** (dimension): Maximum degree + 1 of message polynomials
//! - **d** (minimum distance): n - k + 1 (achieves Singleton bound)
//!
//! # Example
//!
//! ```
//! use reed_solomon_codes::reed_solomon::ReedSolomonCode;
//! use reed_solomon_codes::FE;
//!
//! // Create an RS[16, 8, 9] code (can correct up to 4 errors)
//! let code = ReedSolomonCode::new(16, 8);
//!
//! // Encode a message
//! let message: Vec<FE> = (0..8).map(|i| FE::from(i as u64)).collect();
//! let codeword = code.encode(&message);
//!
//! assert_eq!(codeword.len(), 16);
//! ```

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsFFTField, IsField, IsPrimeField};
use lambdaworks_math::polynomial::Polynomial;

/// A Reed-Solomon code defined by its parameters and evaluation domain.
///
/// The code RS[n, k] has:
/// - Code length n
/// - Dimension k (messages are polynomials of degree < k)
/// - Minimum distance d = n - k + 1
/// - Error correction capability t = floor((n-k)/2)
#[derive(Debug, Clone)]
pub struct ReedSolomonCode<F: IsField + Clone> {
    /// Code length (number of evaluation points)
    n: usize,
    /// Dimension (maximum degree + 1 of message polynomials)
    k: usize,
    /// Evaluation domain: n distinct field elements
    evaluation_domain: Vec<FieldElement<F>>,
}

impl<F: IsField + Clone> ReedSolomonCode<F> {
    /// Creates a new Reed-Solomon code with an arbitrary evaluation domain.
    ///
    /// # Arguments
    ///
    /// * `domain` - A vector of n distinct field elements
    /// * `k` - The dimension (messages are polynomials of degree < k)
    ///
    /// # Panics
    ///
    /// Panics if k > n or if domain contains duplicates.
    pub fn with_domain(domain: Vec<FieldElement<F>>, k: usize) -> Self {
        let n = domain.len();
        assert!(k <= n, "Dimension k must be at most code length n");
        assert!(k > 0, "Dimension k must be positive");

        // Verify all domain elements are distinct
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(
                    domain[i] != domain[j],
                    "Evaluation domain must contain distinct elements"
                );
            }
        }

        Self {
            n,
            k,
            evaluation_domain: domain,
        }
    }

    /// Returns the code length n.
    pub fn code_length(&self) -> usize {
        self.n
    }

    /// Returns the dimension k.
    pub fn dimension(&self) -> usize {
        self.k
    }

    /// Returns the minimum distance d = n - k + 1.
    ///
    /// This is the Singleton bound, which RS codes achieve with equality
    /// (making them Maximum Distance Separable codes).
    pub fn minimum_distance(&self) -> usize {
        self.n - self.k + 1
    }

    /// Returns the unique decoding radius: floor((d-1)/2) = floor((n-k)/2).
    ///
    /// This is the maximum number of errors that can be corrected by
    /// unique decoding algorithms like Berlekamp-Welch.
    pub fn unique_decoding_radius(&self) -> usize {
        (self.n - self.k) / 2
    }

    /// Returns the code rate k/n.
    pub fn rate(&self) -> f64 {
        self.k as f64 / self.n as f64
    }

    /// Returns a reference to the evaluation domain.
    pub fn domain(&self) -> &[FieldElement<F>] {
        &self.evaluation_domain
    }

    /// Encodes a message (polynomial coefficients) into a codeword.
    ///
    /// The message is interpreted as coefficients of a polynomial p(x) of
    /// degree < k, and the codeword is [p(α₀), p(α₁), ..., p(αₙ₋₁)] where
    /// αᵢ are the evaluation domain points.
    ///
    /// # Arguments
    ///
    /// * `message` - Polynomial coefficients [c₀, c₁, ..., cₖ₋₁] representing
    ///   p(x) = c₀ + c₁x + c₂x² + ... + cₖ₋₁xᵏ⁻¹
    ///
    /// # Panics
    ///
    /// Panics if message length exceeds k.
    pub fn encode(&self, message: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
        assert!(
            message.len() <= self.k,
            "Message length {} exceeds dimension k = {}",
            message.len(),
            self.k
        );

        // Construct the message polynomial
        let poly = Polynomial::new(message);

        // Evaluate at all domain points
        self.evaluation_domain
            .iter()
            .map(|alpha| poly.evaluate(alpha))
            .collect()
    }

    /// Encodes a polynomial directly into a codeword.
    ///
    /// # Panics
    ///
    /// Panics if polynomial degree >= k.
    pub fn encode_polynomial(&self, poly: &Polynomial<FieldElement<F>>) -> Vec<FieldElement<F>> {
        assert!(
            poly.degree() < self.k,
            "Polynomial degree {} must be less than k = {}",
            poly.degree(),
            self.k
        );

        self.evaluation_domain
            .iter()
            .map(|alpha| poly.evaluate(alpha))
            .collect()
    }
}

impl<F: IsFFTField + Clone> ReedSolomonCode<F>
where
    F::BaseType: Clone,
{
    /// Creates a Reed-Solomon code using a multiplicative subgroup as domain.
    ///
    /// This uses the n-th roots of unity as evaluation points, which enables
    /// FFT-based encoding and is common in cryptographic applications.
    ///
    /// # Arguments
    ///
    /// * `n` - Code length (must be a power of 2 and ≤ 2^TWO_ADICITY)
    /// * `k` - Dimension (messages are polynomials of degree < k)
    ///
    /// # Panics
    ///
    /// Panics if n is not a power of 2 or exceeds the field's two-adicity.
    pub fn new(n: usize, k: usize) -> Self {
        assert!(n.is_power_of_two(), "Code length n must be a power of 2");
        assert!(k <= n, "Dimension k must be at most code length n");
        assert!(k > 0, "Dimension k must be positive");

        let log_n = n.trailing_zeros() as u64;
        assert!(
            log_n <= F::TWO_ADICITY,
            "Code length n = 2^{} exceeds field's two-adicity {}",
            log_n,
            F::TWO_ADICITY
        );

        // Get the n-th primitive root of unity
        let omega = F::get_primitive_root_of_unity(log_n).unwrap();

        // Generate domain: [1, ω, ω², ..., ωⁿ⁻¹]
        let mut domain = Vec::with_capacity(n);
        let mut current = FieldElement::<F>::one();
        for _ in 0..n {
            domain.push(current.clone());
            current = &current * &omega;
        }

        Self {
            n,
            k,
            evaluation_domain: domain,
        }
    }
}

impl<F: IsPrimeField + Clone> ReedSolomonCode<F> {
    /// Creates a Reed-Solomon code using consecutive integers as domain.
    ///
    /// The evaluation domain is [0, 1, 2, ..., n-1] ⊂ F_p.
    /// This is simple but doesn't enable FFT-based encoding.
    ///
    /// # Arguments
    ///
    /// * `n` - Code length (must be less than field characteristic)
    /// * `k` - Dimension
    pub fn with_consecutive_domain(n: usize, k: usize) -> Self {
        assert!(k <= n, "Dimension k must be at most code length n");
        assert!(k > 0, "Dimension k must be positive");

        let domain: Vec<FieldElement<F>> =
            (0..n).map(|i| FieldElement::<F>::from(i as u64)).collect();

        Self {
            n,
            k,
            evaluation_domain: domain,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Babybear31PrimeField;

    type FE = FieldElement<Babybear31PrimeField>;

    #[test]
    fn test_code_parameters() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::new(16, 8);

        assert_eq!(code.code_length(), 16);
        assert_eq!(code.dimension(), 8);
        assert_eq!(code.minimum_distance(), 9); // n - k + 1 = 16 - 8 + 1 = 9
        assert_eq!(code.unique_decoding_radius(), 4); // floor((n-k)/2) = 4
        assert!((code.rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_domain_is_roots_of_unity() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::new(8, 4);
        let domain = code.domain();

        // Check that domain elements are distinct
        for i in 0..8 {
            for j in (i + 1)..8 {
                assert_ne!(domain[i], domain[j], "Domain elements must be distinct");
            }
        }

        // Check that ω^n = 1
        let omega = &domain[1];
        let mut omega_n = omega.clone();
        for _ in 1..8 {
            omega_n = &omega_n * omega;
        }
        assert_eq!(omega_n, FE::one(), "ω^n should equal 1");
    }

    #[test]
    fn test_encode_constant_polynomial() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::new(8, 4);

        // Constant polynomial p(x) = 42 should give codeword [42, 42, ..., 42]
        let message = vec![FE::from(42u64)];
        let codeword = code.encode(&message);

        assert_eq!(codeword.len(), 8);
        for eval in &codeword {
            assert_eq!(*eval, FE::from(42u64));
        }
    }

    #[test]
    fn test_encode_linear_polynomial() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::with_consecutive_domain(8, 4);

        // p(x) = 1 + 2x evaluated at [0, 1, 2, ..., 7]
        // should give [1, 3, 5, 7, 9, 11, 13, 15]
        let message = vec![FE::from(1u64), FE::from(2u64)];
        let codeword = code.encode(&message);

        assert_eq!(codeword.len(), 8);
        for (i, eval) in codeword.iter().enumerate() {
            let expected = FE::from((1 + 2 * i) as u64);
            assert_eq!(*eval, expected, "Mismatch at position {}", i);
        }
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::new(16, 8);

        // Create a random-ish message
        let message: Vec<FE> = (0..8).map(|i| FE::from((i * 17 + 5) as u64)).collect();
        let codeword = code.encode(&message);

        // The codeword should be the evaluations of the polynomial
        let poly = Polynomial::new(&message);
        for (i, alpha) in code.domain().iter().enumerate() {
            assert_eq!(
                codeword[i],
                poly.evaluate(alpha),
                "Mismatch at position {}",
                i
            );
        }
    }

    #[test]
    fn test_singleton_bound() {
        // Test various code parameters
        for (n, k) in [(8, 4), (16, 8), (32, 16), (16, 4), (16, 12)] {
            let code = ReedSolomonCode::<Babybear31PrimeField>::new(n, k);

            // Singleton bound: d ≤ n - k + 1
            // RS codes achieve equality (MDS property)
            assert_eq!(
                code.minimum_distance(),
                n - k + 1,
                "RS code should achieve Singleton bound"
            );
        }
    }

    #[test]
    fn test_with_domain() {
        // Create a custom domain
        let domain: Vec<FE> = vec![
            FE::from(1u64),
            FE::from(2u64),
            FE::from(3u64),
            FE::from(5u64),
        ];
        let code = ReedSolomonCode::<Babybear31PrimeField>::with_domain(domain.clone(), 2);

        assert_eq!(code.code_length(), 4);
        assert_eq!(code.dimension(), 2);
        assert_eq!(code.minimum_distance(), 3);
    }

    #[test]
    #[should_panic(expected = "distinct elements")]
    fn test_duplicate_domain_elements_panic() {
        let domain: Vec<FE> = vec![
            FE::from(1u64),
            FE::from(2u64),
            FE::from(1u64), // duplicate!
            FE::from(3u64),
        ];
        let _ = ReedSolomonCode::<Babybear31PrimeField>::with_domain(domain, 2);
    }

    #[test]
    #[should_panic(expected = "exceeds dimension")]
    fn test_message_too_long_panic() {
        let code = ReedSolomonCode::<Babybear31PrimeField>::new(8, 4);
        let message: Vec<FE> = (0..5).map(|i| FE::from(i as u64)).collect(); // 5 > 4
        let _ = code.encode(&message);
    }
}
