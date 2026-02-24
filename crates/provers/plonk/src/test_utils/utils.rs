use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_crypto::commitments::kzg::StructuredReferenceString;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::pairing::BLS12381AtePairing;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve, field_extension::BLS12381PrimeField, twist::BLS12381TwistCurve,
        },
        traits::IsEllipticCurve,
    },
    field::element::FieldElement,
    traits::IsRandomFieldElementGenerator,
};

pub type Curve = BLS12381Curve;
pub type TwistedCurve = BLS12381TwistCurve;
pub type FpField = BLS12381PrimeField;
pub type FpElement = FieldElement<FpField>;
pub type Pairing = BLS12381AtePairing;
pub type KZG = KateZaveruchaGoldberg<FrField, Pairing>;
pub const ORDER_R_MINUS_1_ROOT_UNITY: FrElement = FrElement::from_hex_unchecked("7");

pub type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
pub type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;

/// Generates a test SRS for the BLS12381 curve
/// n is the number of constraints in the system.
pub fn test_srs(n: usize) -> StructuredReferenceString<G1Point, G2Point> {
    let s = FrElement::from(2);
    let g1 = <BLS12381Curve as IsEllipticCurve>::generator();
    let g2 = <BLS12381TwistCurve as IsEllipticCurve>::generator();

    // Use iterative multiplication instead of pow() for efficiency
    let powers_main_group: Vec<G1Point> = (0..n + 3)
        .scan(FrElement::one(), |s_power, _| {
            let result = g1.operate_with_self(s_power.canonical());
            *s_power = &*s_power * &s;
            Some(result)
        })
        .collect();
    let powers_secondary_group = [g2.clone(), g2.operate_with_self(s.canonical())];

    StructuredReferenceString::new(&powers_main_group, &powers_secondary_group)
}

/// Generates a domain to interpolate: 1, omega, omega², ..., omega^(size-1)
pub fn generate_domain<F: IsField>(omega: &FieldElement<F>, size: usize) -> Vec<FieldElement<F>> {
    (0..size)
        .scan(FieldElement::one(), |power, _| {
            let result = power.clone();
            *power = &*power * omega;
            Some(result)
        })
        .collect()
}

/// Generates the permutation coefficients for the copy constraints.
/// polynomials S1, S2, S3.
pub fn generate_permutation_coefficients<F: IsField>(
    omega: &FieldElement<F>,
    n: usize,
    permutation: &[usize],
    order_r_minus_1_root_unity: &FieldElement<F>,
) -> Vec<FieldElement<F>> {
    let identity = identity_permutation(omega, n, order_r_minus_1_root_unity);
    permutation
        .iter()
        .take(n * 3)
        .map(|&i| identity[i].clone())
        .collect()
}

/// The identity permutation, auxiliary function to generate the copy constraints.
fn identity_permutation<F: IsField>(
    w: &FieldElement<F>,
    n: usize,
    order_r_minus_1_root_unity: &FieldElement<F>,
) -> Vec<FieldElement<F>> {
    let u = order_r_minus_1_root_unity;
    let u_sq = u * u;

    // Precompute w^i for i in 0..n using iterative multiplication
    let w_powers: Vec<_> = (0..n)
        .scan(FieldElement::one(), |w_power, _| {
            let result = w_power.clone();
            *w_power = &*w_power * w;
            Some(result)
        })
        .collect();

    // Build result: [w^i * u^0, w^i * u^1, w^i * u^2] for each column
    let mut result = Vec::with_capacity(3 * n);
    for w_i in &w_powers {
        result.push(w_i.clone());
    }
    for w_i in &w_powers {
        result.push(w_i * u);
    }
    for w_i in &w_powers {
        result.push(w_i * &u_sq);
    }
    result
}

/// A mock random number generator for deterministic tests.
/// Returns zero, which disables blinding (no zero-knowledge in tests).
/// **Warning**: Do NOT use in production - use `SecureRandomFieldGenerator` instead.
#[derive(Clone)]
pub struct TestRandomFieldGenerator;
impl IsRandomFieldElementGenerator<FrField> for TestRandomFieldGenerator {
    fn generate(&self) -> FrElement {
        FrElement::zero()
    }
}

/// A cryptographically secure random field element generator.
///
/// Uses the operating system's random number generator (via `rand::rngs::OsRng`)
/// to generate random field elements for polynomial blinding. This ensures the
/// zero-knowledge property of the PLONK proof system.
///
/// # Usage
///
/// ```ignore
/// use lambdaworks_plonk::test_utils::utils::SecureRandomFieldGenerator;
///
/// let rng = SecureRandomFieldGenerator;
/// let prover = Prover::new(kzg, rng);
/// ```
///
/// # Security
///
/// This generator is suitable for production use. Each call to `generate()`
/// produces an independent, uniformly distributed random field element.
#[derive(Clone, Default)]
pub struct SecureRandomFieldGenerator;

impl IsRandomFieldElementGenerator<FrField> for SecureRandomFieldGenerator {
    fn generate(&self) -> FrElement {
        use rand::Rng;

        // Generate random bytes and reduce modulo the field order
        let mut rng = rand::rngs::OsRng;
        let random_bytes: [u8; 32] = rng.gen();

        // Convert bytes to field element (automatically reduces modulo field order)
        FrElement::from_bytes_be(&random_bytes).unwrap_or_else(|_| {
            // Fallback: generate another random element
            // This should be extremely rare
            let random_bytes2: [u8; 32] = rng.gen();
            FrElement::from_bytes_be(&random_bytes2)
                .expect("Failed to generate random field element")
        })
    }
}

#[cfg(test)]
mod secure_rng_tests {
    use super::*;

    #[test]
    fn test_secure_rng_generates_different_values() {
        let rng = SecureRandomFieldGenerator;
        let v1 = rng.generate();
        let v2 = rng.generate();
        let v3 = rng.generate();

        // With overwhelming probability, these should all be different
        assert_ne!(v1, v2);
        assert_ne!(v2, v3);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_secure_rng_generates_nonzero() {
        let rng = SecureRandomFieldGenerator;
        // Generate 100 values and verify none are zero
        // (probability of getting zero is negligible: ~2^-254)
        for _ in 0..100 {
            let v = rng.generate();
            assert_ne!(v, FrElement::zero());
        }
    }

    #[test]
    fn test_test_rng_returns_zero() {
        let rng = TestRandomFieldGenerator;
        assert_eq!(rng.generate(), FrElement::zero());
        assert_eq!(rng.generate(), FrElement::zero());
    }
}
