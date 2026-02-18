/// Polynomial quotient ring Zq[X]/(X^n + 1).
///
/// Elements are polynomials of degree < n with coefficients in a field F.
/// Multiplication is performed modulo X^n + 1, using the identity X^n ≡ -1.
///
/// This module is designed for lattice-based cryptography (Dilithium, Kyber)
/// where the ring Rq = Zq[X]/(X^n + 1) is fundamental.
use crate::fft::errors::FFTError;
use crate::field::element::FieldElement;
use crate::field::traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf};
use crate::polynomial::Polynomial;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::ops::{Add, Neg, Sub};

/// An element of the quotient ring Zq[X]/(X^n + 1).
///
/// Internally stores a polynomial of degree < N. All arithmetic
/// operations automatically reduce modulo X^N + 1.
#[derive(Clone, PartialEq, Eq)]
pub struct PolynomialRingElement<F: IsField, const N: usize> {
    poly: Polynomial<FieldElement<F>>,
}

impl<F: IsField, const N: usize> fmt::Debug for PolynomialRingElement<F, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PolynomialRingElement")
            .field("n", &N)
            .field("degree", &self.poly.degree())
            .field("coefficients", &self.poly.coefficients)
            .finish()
    }
}

impl<F: IsField, const N: usize> PolynomialRingElement<F, N> {
    /// Creates a new ring element from coefficients, reducing modulo X^N + 1.
    ///
    /// If more than N coefficients are provided, the polynomial is reduced.
    /// Coefficients are in order [c_0, c_1, ..., c_{n-1}] representing
    /// c_0 + c_1*X + ... + c_{n-1}*X^{n-1}.
    pub fn new(coefficients: &[FieldElement<F>]) -> Self {
        let poly = Polynomial::new(coefficients);
        Self::from_poly(poly)
    }

    /// Creates a ring element from an existing polynomial, reducing modulo X^N + 1.
    pub fn from_poly(poly: Polynomial<FieldElement<F>>) -> Self {
        let reduced = reduce_mod_xn_plus_1::<F, N>(poly);
        Self { poly: reduced }
    }

    /// Returns the zero element of the ring.
    pub fn zero() -> Self {
        Self {
            poly: Polynomial::zero(),
        }
    }

    /// Returns the multiplicative identity (constant polynomial 1).
    pub fn one() -> Self {
        Self {
            poly: Polynomial::new(&[FieldElement::one()]),
        }
    }

    /// Returns true if this is the zero element.
    pub fn is_zero(&self) -> bool {
        self.poly.is_zero()
    }

    /// Returns a reference to the underlying polynomial.
    pub fn poly(&self) -> &Polynomial<FieldElement<F>> {
        &self.poly
    }

    /// Returns the coefficients of the polynomial.
    /// The result may have fewer than N elements if trailing coefficients are zero.
    pub fn coefficients(&self) -> &[FieldElement<F>] {
        self.poly.coefficients()
    }

    /// Returns the coefficient at position `i`, or zero if `i >= degree + 1`.
    pub fn coefficient(&self, i: usize) -> FieldElement<F> {
        if i < self.poly.coefficients.len() {
            self.poly.coefficients[i].clone()
        } else {
            FieldElement::zero()
        }
    }

    /// Returns all N coefficients, padding with zeros as needed.
    pub fn padded_coefficients(&self) -> Vec<FieldElement<F>> {
        let mut coeffs = self.poly.coefficients.clone();
        coeffs.resize(N, FieldElement::zero());
        coeffs
    }

    /// Multiplies by a scalar field element.
    pub fn scalar_mul(&self, scalar: &FieldElement<F>) -> Self {
        Self {
            poly: self.poly.scale_coeffs(scalar),
        }
    }

    /// Schoolbook multiplication: multiply polynomials then reduce modulo X^N + 1.
    pub fn mul_schoolbook(&self, other: &Self) -> Self {
        let product = self.poly.mul_with_ref(&other.poly);
        Self::from_poly(product)
    }
}

/// Precomputed twist factors for negacyclic NTT.
///
/// Stores powers of psi (a primitive 2N-th root of unity) and their inverses,
/// so they can be reused across multiple multiplications.
pub struct NegacyclicTwistFactors<F: IsField, const N: usize> {
    /// psi_powers[i] = psi^i for i in 0..N
    pub psi_powers: Vec<FieldElement<F>>,
    /// psi_inv_powers[i] = psi^(-i) for i in 0..N
    pub psi_inv_powers: Vec<FieldElement<F>>,
}

impl<F: IsFFTField + IsSubFieldOf<F>, const N: usize> NegacyclicTwistFactors<F, N> {
    /// Computes twist factors for the negacyclic NTT of size N.
    ///
    /// Returns `None` if the field lacks a primitive 2N-th root of unity.
    pub fn new() -> Option<Self> {
        let order = (2 * N).trailing_zeros() as u64;
        let psi: FieldElement<F> = F::get_primitive_root_of_unity(order).ok()?;
        let psi_inv = psi.inv().ok()?;

        let mut psi_powers = Vec::with_capacity(N);
        let mut psi_inv_powers = Vec::with_capacity(N);
        let mut psi_pow = FieldElement::one();
        let mut psi_inv_pow = FieldElement::one();
        for i in 0..N {
            psi_powers.push(psi_pow.clone());
            psi_inv_powers.push(psi_inv_pow.clone());
            if i < N - 1 {
                psi_pow = &psi_pow * &psi;
                psi_inv_pow = &psi_inv_pow * &psi_inv;
            }
        }

        Some(Self {
            psi_powers,
            psi_inv_powers,
        })
    }
}

impl<F: IsFFTField + IsSubFieldOf<F>, const N: usize> PolynomialRingElement<F, N> {
    /// FFT-based multiplication using the negacyclic NTT.
    ///
    /// Uses a size-N NTT (instead of size-2N) by exploiting the X^N ≡ -1
    /// structure of the quotient ring. Falls back to schoolbook multiplication
    /// if the field lacks sufficient two-adicity.
    pub fn mul_ntt(&self, other: &Self) -> Self {
        match Self::mul_negacyclic_ntt_inner(self, other) {
            Ok(product) => product,
            Err(_) => self.mul_schoolbook(other),
        }
    }

    /// Negacyclic NTT multiplication with precomputed twist factors.
    ///
    /// When performing many multiplications, precompute the twist factors once
    /// with [`NegacyclicTwistFactors::new`] and pass them here to avoid
    /// recomputing psi powers each time.
    pub fn mul_ntt_with_factors(
        &self,
        other: &Self,
        factors: &NegacyclicTwistFactors<F, N>,
    ) -> Self {
        match Self::mul_negacyclic_with_factors_inner(self, other, factors) {
            Ok(product) => product,
            Err(_) => self.mul_schoolbook(other),
        }
    }

    fn mul_negacyclic_ntt_inner(&self, other: &Self) -> Result<Self, FFTError> {
        let order = (2 * N).trailing_zeros() as u64;
        let psi: FieldElement<F> =
            F::get_primitive_root_of_unity(order).map_err(|_| FFTError::RootOfUnityError(order))?;
        let psi_inv = psi.inv().map_err(|_| FFTError::InverseOfZero)?;

        // Pad to N coefficients
        let a_coeffs = self.padded_coefficients();
        let b_coeffs = other.padded_coefficients();

        // Twist: multiply a[i] and b[i] by psi^i
        let mut a_twisted = Vec::with_capacity(N);
        let mut b_twisted = Vec::with_capacity(N);
        let mut psi_pow = FieldElement::one();
        for i in 0..N {
            a_twisted.push(&a_coeffs[i] * &psi_pow);
            b_twisted.push(&b_coeffs[i] * &psi_pow);
            if i < N - 1 {
                psi_pow = &psi_pow * &psi;
            }
        }

        // Forward NTT of size N, pointwise multiply, inverse NTT
        let c_poly = Self::ntt_pointwise_mul(&a_twisted, &b_twisted)?;

        // Untwist: multiply c[i] by psi^(-i)
        let c_coeffs = c_poly.coefficients();
        let mut result = Vec::with_capacity(N);
        let mut psi_inv_pow = FieldElement::one();
        for i in 0..N {
            if i < c_coeffs.len() {
                result.push(&c_coeffs[i] * &psi_inv_pow);
            } else {
                result.push(FieldElement::zero());
            }
            if i < N - 1 {
                psi_inv_pow = &psi_inv_pow * &psi_inv;
            }
        }

        Ok(Self {
            poly: Polynomial::new(&result),
        })
    }

    fn mul_negacyclic_with_factors_inner(
        &self,
        other: &Self,
        factors: &NegacyclicTwistFactors<F, N>,
    ) -> Result<Self, FFTError> {
        let a_coeffs = self.padded_coefficients();
        let b_coeffs = other.padded_coefficients();

        // Twist using precomputed psi powers
        let a_twisted: Vec<_> = a_coeffs
            .iter()
            .zip(factors.psi_powers.iter())
            .map(|(c, p)| c * p)
            .collect();
        let b_twisted: Vec<_> = b_coeffs
            .iter()
            .zip(factors.psi_powers.iter())
            .map(|(c, p)| c * p)
            .collect();

        // Forward NTT of size N, pointwise multiply, inverse NTT
        let c_poly = Self::ntt_pointwise_mul(&a_twisted, &b_twisted)?;

        // Untwist using precomputed psi inverse powers
        let c_coeffs = c_poly.coefficients();
        let result: Vec<_> = (0..N)
            .map(|i| {
                if i < c_coeffs.len() {
                    &c_coeffs[i] * &factors.psi_inv_powers[i]
                } else {
                    FieldElement::zero()
                }
            })
            .collect();

        Ok(Self {
            poly: Polynomial::new(&result),
        })
    }

    /// Forward NTT, pointwise multiply, inverse NTT (shared by both paths).
    fn ntt_pointwise_mul(
        a_twisted: &[FieldElement<F>],
        b_twisted: &[FieldElement<F>],
    ) -> Result<Polynomial<FieldElement<F>>, FFTError> {
        let a_poly = Polynomial::new(a_twisted);
        let b_poly = Polynomial::new(b_twisted);
        let a_evals = Polynomial::evaluate_fft::<F>(&a_poly, 1, Some(N))?;
        let b_evals = Polynomial::evaluate_fft::<F>(&b_poly, 1, Some(N))?;

        let c_evals: Vec<_> = a_evals
            .iter()
            .zip(b_evals.iter())
            .map(|(a, b)| a * b)
            .collect();

        Polynomial::interpolate_fft::<F>(&c_evals)
    }
}

impl<F: IsPrimeField, const N: usize> PolynomialRingElement<F, N>
where
    F::CanonicalType: Into<u64>,
{
    /// Converts a field element to its centered representation in [-(q-1)/2, (q-1)/2].
    ///
    /// For a field element in [0, q-1], this maps values > (q-1)/2 to their
    /// negative equivalents.
    pub fn centered_coefficient(&self, i: usize) -> i64 {
        centered_mod::<F>(&self.coefficient(i))
    }

    /// Returns the infinity norm: max |c_i| in centered representation.
    ///
    /// The infinity norm is the maximum absolute value of any coefficient
    /// when interpreted in the centered range [-(q-1)/2, (q-1)/2].
    pub fn infinity_norm(&self) -> u64 {
        infinity_norm::<F>(self.coefficients())
    }

    /// Returns true if all coefficients in centered representation have
    /// absolute value at most `bound`.
    pub fn is_small(&self, bound: u64) -> bool {
        self.coefficients()
            .iter()
            .all(|c| (centered_mod::<F>(c).unsigned_abs()) <= bound)
    }
}

/// Interpret a field element in its centered representation.
///
/// For a prime field with modulus q, maps [0, q-1] to [-(q-1)/2, (q-1)/2]:
/// - Values in [0, (q-1)/2] map to themselves
/// - Values in ((q-1)/2, q-1] map to value - q
pub fn centered_mod<F: IsPrimeField>(a: &FieldElement<F>) -> i64
where
    F::CanonicalType: Into<u64>,
{
    let val: u64 = a.canonical().into();
    let q: u64 = F::modulus_minus_one().into();
    let q = q + 1; // actual modulus
    let half = (q - 1) / 2;
    if val <= half {
        val as i64
    } else {
        val as i64 - q as i64
    }
}

/// Computes the infinity norm of a slice of field elements.
///
/// Returns the maximum absolute value of any element when interpreted
/// in the centered representation [-(q-1)/2, (q-1)/2].
pub fn infinity_norm<F: IsPrimeField>(coefficients: &[FieldElement<F>]) -> u64
where
    F::CanonicalType: Into<u64>,
{
    coefficients
        .iter()
        .map(|c| centered_mod::<F>(c).unsigned_abs())
        .max()
        .unwrap_or(0)
}

/// Reduces a polynomial modulo X^N + 1.
///
/// Since X^N ≡ -1 in the quotient ring, for each coefficient at position k >= N,
/// we subtract it from position k - N (because X^k = X^{k-N} * X^N = -X^{k-N}).
/// This generalizes for higher degrees by repeated application.
fn reduce_mod_xn_plus_1<F: IsField, const N: usize>(
    poly: Polynomial<FieldElement<F>>,
) -> Polynomial<FieldElement<F>> {
    let coeffs = poly.coefficients();
    if coeffs.len() <= N {
        return poly;
    }

    let mut result = vec![FieldElement::<F>::zero(); N];

    for (i, c) in coeffs.iter().enumerate() {
        let pos = i % N;
        // Each full wraparound through N flips the sign.
        // X^N = -1, X^{2N} = 1, X^{3N} = -1, ...
        let wraps = i / N;
        if wraps.is_multiple_of(2) {
            result[pos] = &result[pos] + c;
        } else {
            result[pos] = &result[pos] - c;
        }
    }

    Polynomial::new(&result)
}

// Operator implementations

impl<F: IsField, const N: usize> Add for &PolynomialRingElement<F, N> {
    type Output = PolynomialRingElement<F, N>;

    fn add(self, other: &PolynomialRingElement<F, N>) -> PolynomialRingElement<F, N> {
        // Addition of polynomials with degree < N stays < N, no reduction needed
        PolynomialRingElement {
            poly: &self.poly + &other.poly,
        }
    }
}

impl<F: IsField, const N: usize> Add for PolynomialRingElement<F, N> {
    type Output = PolynomialRingElement<F, N>;

    fn add(self, other: PolynomialRingElement<F, N>) -> PolynomialRingElement<F, N> {
        &self + &other
    }
}

impl<F: IsField, const N: usize> Sub for &PolynomialRingElement<F, N> {
    type Output = PolynomialRingElement<F, N>;

    fn sub(self, other: &PolynomialRingElement<F, N>) -> PolynomialRingElement<F, N> {
        PolynomialRingElement {
            poly: &self.poly - &other.poly,
        }
    }
}

impl<F: IsField, const N: usize> Sub for PolynomialRingElement<F, N> {
    type Output = PolynomialRingElement<F, N>;

    fn sub(self, other: PolynomialRingElement<F, N>) -> PolynomialRingElement<F, N> {
        &self - &other
    }
}

impl<F: IsField, const N: usize> Neg for &PolynomialRingElement<F, N> {
    type Output = PolynomialRingElement<F, N>;

    fn neg(self) -> PolynomialRingElement<F, N> {
        PolynomialRingElement { poly: -&self.poly }
    }
}

impl<F: IsField, const N: usize> Neg for PolynomialRingElement<F, N> {
    type Output = PolynomialRingElement<F, N>;

    fn neg(self) -> PolynomialRingElement<F, N> {
        -&self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fields::fft_friendly::dilithium_prime::DilithiumField;

    type FE = FieldElement<DilithiumField>;
    type R256 = PolynomialRingElement<DilithiumField, 256>;

    const Q: u64 = 8380417;

    fn fe(v: u64) -> FE {
        FE::from(v)
    }

    #[test]
    fn zero_element() {
        let z = R256::zero();
        assert!(z.is_zero());
        assert_eq!(z.infinity_norm(), 0);
    }

    #[test]
    fn one_element() {
        let one = R256::one();
        assert!(!one.is_zero());
        assert_eq!(one.coefficient(0), fe(1));
        assert_eq!(one.coefficient(1), fe(0));
    }

    #[test]
    fn add_two_elements() {
        let a = R256::new(&[fe(1), fe(2), fe(3)]);
        let b = R256::new(&[fe(4), fe(5), fe(6)]);
        let c = &a + &b;
        assert_eq!(c.coefficient(0), fe(5));
        assert_eq!(c.coefficient(1), fe(7));
        assert_eq!(c.coefficient(2), fe(9));
    }

    #[test]
    fn sub_two_elements() {
        let a = R256::new(&[fe(5), fe(7), fe(9)]);
        let b = R256::new(&[fe(1), fe(2), fe(3)]);
        let c = &a - &b;
        assert_eq!(c.coefficient(0), fe(4));
        assert_eq!(c.coefficient(1), fe(5));
        assert_eq!(c.coefficient(2), fe(6));
    }

    #[test]
    fn neg_element() {
        let a = R256::new(&[fe(1), fe(2), fe(3)]);
        let neg_a = -&a;
        let sum = &a + &neg_a;
        assert!(sum.is_zero());
    }

    #[test]
    fn x_n_reduces_to_minus_one() {
        // X^256 should become -1 in the ring
        let mut coeffs = vec![fe(0); 257];
        coeffs[256] = fe(1); // X^256
        let elem = R256::new(&coeffs);
        // X^256 ≡ -1, so coefficient 0 should be q-1 (= -1 mod q)
        assert_eq!(elem.coefficient(0), fe(Q - 1));
        assert_eq!(elem.coefficients().len(), 1); // only constant term after reduction + trimming
    }

    #[test]
    fn x_2n_reduces_to_one() {
        // X^512 should become +1 in the ring
        let mut coeffs = vec![fe(0); 513];
        coeffs[512] = fe(1); // X^512
        let elem = R256::new(&coeffs);
        assert_eq!(elem.coefficient(0), fe(1));
    }

    #[test]
    fn schoolbook_mul_basic() {
        // (1 + X) * (1 + X) = 1 + 2X + X^2
        let a = R256::new(&[fe(1), fe(1)]);
        let b = a.mul_schoolbook(&a);
        assert_eq!(b.coefficient(0), fe(1));
        assert_eq!(b.coefficient(1), fe(2));
        assert_eq!(b.coefficient(2), fe(1));
    }

    #[test]
    fn schoolbook_mul_wraps() {
        // Multiply X^{255} * X = X^{256} ≡ -1
        let mut coeffs_a = vec![fe(0); 256];
        coeffs_a[255] = fe(1);
        let a = R256::new(&coeffs_a);

        let b = R256::new(&[fe(0), fe(1)]); // X

        let product = a.mul_schoolbook(&b);
        // X^256 ≡ -1, so constant term should be -1
        assert_eq!(product.coefficient(0), fe(Q - 1));
    }

    #[test]
    fn ntt_mul_matches_schoolbook() {
        // Small polynomials: (2 + 3X + X^2) * (1 + 4X)
        let a = R256::new(&[fe(2), fe(3), fe(1)]);
        let b = R256::new(&[fe(1), fe(4)]);

        let school = a.mul_schoolbook(&b);
        let ntt = a.mul_ntt(&b);
        assert_eq!(school, ntt);
    }

    #[test]
    fn scalar_mul() {
        let a = R256::new(&[fe(1), fe(2), fe(3)]);
        let scaled = a.scalar_mul(&fe(5));
        assert_eq!(scaled.coefficient(0), fe(5));
        assert_eq!(scaled.coefficient(1), fe(10));
        assert_eq!(scaled.coefficient(2), fe(15));
    }

    #[test]
    fn centered_mod_positive() {
        // Values in [0, (q-1)/2] should stay positive
        let half = (Q - 1) / 2; // 4190208
        assert_eq!(centered_mod::<DilithiumField>(&fe(0)), 0);
        assert_eq!(centered_mod::<DilithiumField>(&fe(42)), 42);
        assert_eq!(centered_mod::<DilithiumField>(&fe(half)), half as i64);
    }

    #[test]
    fn centered_mod_negative() {
        // Values in ((q-1)/2, q-1] should map to negative
        assert_eq!(centered_mod::<DilithiumField>(&fe(Q - 1)), -1);
        assert_eq!(centered_mod::<DilithiumField>(&fe(Q - 42)), -42);
    }

    #[test]
    fn infinity_norm_basic() {
        // Polynomial with coefficients 1, -2, 3 (where -2 is represented as q-2)
        let a = R256::new(&[fe(1), fe(Q - 2), fe(3)]);
        assert_eq!(a.infinity_norm(), 3);
    }

    #[test]
    fn is_small_check() {
        let a = R256::new(&[fe(1), fe(Q - 2), fe(3)]);
        assert!(a.is_small(3));
        assert!(!a.is_small(2));
    }

    #[test]
    fn padded_coefficients_length() {
        let a = R256::new(&[fe(1), fe(2)]);
        let padded = a.padded_coefficients();
        assert_eq!(padded.len(), 256);
        assert_eq!(padded[0], fe(1));
        assert_eq!(padded[1], fe(2));
        assert_eq!(padded[255], fe(0));
    }

    #[test]
    fn mul_by_zero_is_zero() {
        let a = R256::new(&[fe(1), fe(2), fe(3)]);
        let z = R256::zero();
        assert!(a.mul_schoolbook(&z).is_zero());
    }

    #[test]
    fn mul_by_one_is_identity() {
        let a = R256::new(&[fe(1), fe(2), fe(3)]);
        let one = R256::one();
        assert_eq!(a.mul_schoolbook(&one), a);
    }

    #[test]
    fn distributivity() {
        // (a + b) * c == a*c + b*c
        let a = R256::new(&[fe(1), fe(2), fe(3)]);
        let b = R256::new(&[fe(4), fe(5)]);
        let c = R256::new(&[fe(7), fe(8), fe(9), fe(10)]);

        let lhs = (&a + &b).mul_schoolbook(&c);
        let rhs = &a.mul_schoolbook(&c) + &b.mul_schoolbook(&c);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn ntt_mul_matches_schoolbook_random_like() {
        // Use deterministic but varied coefficients
        let a_coeffs: Vec<FE> = (0..64).map(|i| fe((i * 7 + 13) % Q)).collect();
        let b_coeffs: Vec<FE> = (0..64).map(|i| fe((i * 11 + 37) % Q)).collect();

        let a = R256::new(&a_coeffs);
        let b = R256::new(&b_coeffs);

        let school = a.mul_schoolbook(&b);
        let ntt = a.mul_ntt(&b);
        assert_eq!(school, ntt);
    }

    #[test]
    fn ntt_with_factors_matches_schoolbook() {
        let factors = NegacyclicTwistFactors::<DilithiumField, 256>::new().unwrap();

        // Small polynomials: (2 + 3X + X²) * (1 + 4X)
        let a = R256::new(&[fe(2), fe(3), fe(1)]);
        let b = R256::new(&[fe(1), fe(4)]);

        let school = a.mul_schoolbook(&b);
        let with_factors = a.mul_ntt_with_factors(&b, &factors);
        assert_eq!(school, with_factors);
    }

    #[test]
    fn ntt_with_factors_matches_ntt() {
        let factors = NegacyclicTwistFactors::<DilithiumField, 256>::new().unwrap();

        // 128-coeff deterministic polynomials
        let a_coeffs: Vec<FE> = (0..128).map(|i| fe((i * 13 + 5) % Q)).collect();
        let b_coeffs: Vec<FE> = (0..128).map(|i| fe((i * 17 + 11) % Q)).collect();

        let a = R256::new(&a_coeffs);
        let b = R256::new(&b_coeffs);

        let ntt = a.mul_ntt(&b);
        let with_factors = a.mul_ntt_with_factors(&b, &factors);
        assert_eq!(ntt, with_factors);
    }

    #[cfg(feature = "std")]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        prop_compose! {
            fn ring_element()(coeffs in proptest::collection::vec(0..8380417u64, 1..=256)) -> R256 {
                let fes: Vec<FE> = coeffs.into_iter().map(fe).collect();
                R256::new(&fes)
            }
        }

        proptest! {
            #[test]
            fn prop_distributivity(a in ring_element(), b in ring_element(), c in ring_element()) {
                let lhs = (&a + &b).mul_schoolbook(&c);
                let rhs = &a.mul_schoolbook(&c) + &b.mul_schoolbook(&c);
                prop_assert_eq!(lhs, rhs);
            }

            #[test]
            fn prop_ntt_matches_schoolbook(a in ring_element(), b in ring_element()) {
                let school = a.mul_schoolbook(&b);
                let ntt = a.mul_ntt(&b);
                prop_assert_eq!(school, ntt);
            }

            #[test]
            fn prop_ntt_with_factors_matches_schoolbook(a in ring_element(), b in ring_element()) {
                let factors = NegacyclicTwistFactors::<DilithiumField, 256>::new().unwrap();
                let school = a.mul_schoolbook(&b);
                let with_factors = a.mul_ntt_with_factors(&b, &factors);
                prop_assert_eq!(school, with_factors);
            }

            #[test]
            fn prop_add_commutative(a in ring_element(), b in ring_element()) {
                prop_assert_eq!(&a + &b, &b + &a);
            }

            #[test]
            fn prop_add_sub_inverse(a in ring_element(), b in ring_element()) {
                let sum = &a + &b;
                let diff = &sum - &b;
                prop_assert_eq!(diff, a);
            }
        }
    }
}
