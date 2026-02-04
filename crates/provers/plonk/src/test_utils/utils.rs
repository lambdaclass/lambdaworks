use lambdaworks_crypto::pcs::kzg::{
    KZGAdapter, KZGCommitment, KZGCommitterKey, KZGPublicParams, KZGVerifierKey,
    StructuredReferenceString, KZG as KZGImpl,
};
use lambdaworks_crypto::pcs::PolynomialCommitmentScheme;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::pairing::BLS12381AtePairing;
use lambdaworks_math::field::traits::IsField;
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

/// Type alias for the KZG adapter that implements `IsCommitmentScheme`.
/// This provides backward compatibility with the PLONK prover.
pub type KZG = KZGAdapter<Pairing>;
pub type TestKZG = KZGImpl<FrField, Pairing>;
pub type TestCommitment = KZGCommitment<Pairing>;
pub type TestCommitterKey = KZGCommitterKey<Pairing>;
pub type TestVerifierKey = KZGVerifierKey<Pairing>;
pub type TestPublicParams = KZGPublicParams<Pairing>;

pub const ORDER_R_MINUS_1_ROOT_UNITY: FrElement = FrElement::from_hex_unchecked("7");

pub type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
pub type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;

/// Generates a test SRS for the BLS12381 curve
/// n is the number of constraints in the system.
pub fn test_srs(n: usize) -> StructuredReferenceString<G1Point, G2Point> {
    let s = FrElement::from(2);
    let g1 = <BLS12381Curve as IsEllipticCurve>::generator();
    let g2 = <BLS12381TwistCurve as IsEllipticCurve>::generator();

    let powers_main_group: Vec<G1Point> = (0..n + 3)
        .map(|exp| g1.operate_with_self(s.pow(exp as u64).canonical()))
        .collect();
    let powers_secondary_group = [g2.clone(), g2.operate_with_self(s.canonical())];

    StructuredReferenceString::new(&powers_main_group, &powers_secondary_group)
}

/// Create test public parameters from an SRS
pub fn test_public_params(n: usize) -> TestPublicParams {
    KZGPublicParams::from_srs(test_srs(n))
}

/// Create test committer and verifier keys
pub fn test_keys(n: usize) -> (TestCommitterKey, TestVerifierKey) {
    let pp = test_public_params(n);
    TestKZG::trim(&pp, n + 2).expect("trim should succeed for valid degree")
}

/// Generates a domain to interpolate: 1, omega, omega², ..., omega^size
pub fn generate_domain<F: IsField>(omega: &FieldElement<F>, size: usize) -> Vec<FieldElement<F>> {
    (1..size).fold(vec![FieldElement::one()], |mut acc, _| {
        acc.push(acc.last().unwrap() * omega);
        acc
    })
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
    let permuted: Vec<FieldElement<F>> = (0..n * 3)
        .map(|i| identity[permutation[i]].clone())
        .collect();
    permuted
}

/// The identity permutation, auxiliary function to generate the copy constraints.
fn identity_permutation<F: IsField>(
    w: &FieldElement<F>,
    n: usize,
    order_r_minus_1_root_unity: &FieldElement<F>,
) -> Vec<FieldElement<F>> {
    let u = order_r_minus_1_root_unity;
    let mut result: Vec<FieldElement<F>> = vec![];
    for index_column in 0..=2 {
        for index_row in 0..n {
            result.push(w.pow(index_row) * u.pow(index_column as u64));
        }
    }
    result
}

/// A mock of a random number generator, to have deterministic tests.
/// When set to zero, there is no zero knowledge applied, because it is used
/// to get random numbers to blind polynomials.
#[derive(Clone)]
pub struct TestRandomFieldGenerator;
impl IsRandomFieldElementGenerator<FrField> for TestRandomFieldGenerator {
    fn generate(&self) -> FrElement {
        FrElement::zero()
    }
}
