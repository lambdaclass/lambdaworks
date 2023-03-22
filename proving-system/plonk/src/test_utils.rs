use crate::setup::{Circuit, CommonPreprocessedInput, Witness};
use lambdaworks_crypto::commitments::kzg::StructuredReferenceString;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve, field_extension::BLS12381PrimeField, twist::BLS12381TwistCurve,
        },
        traits::IsEllipticCurve,
    },
    field::{element::FieldElement, fields::montgomery_backed_prime_fields::U256PrimeField},
    polynomial::Polynomial,
};
// TODO: Generalize

use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::pairing::BLS12381AtePairing,
    field::fields::montgomery_backed_prime_fields::IsMontgomeryConfiguration,
    unsigned_integer::element::U256,
};

pub const ORDER_R: U256 =
    U256::from("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");

#[derive(Clone, Debug)]
pub struct FrConfig;
impl IsMontgomeryConfiguration<4> for FrConfig {
    const MODULUS: U256 = ORDER_R;
}

pub type Curve = BLS12381Curve;
pub type TwistedCurve = BLS12381TwistCurve;
pub type FrField = U256PrimeField<FrConfig>;
pub type FpField = BLS12381PrimeField;
pub type FrElement = FieldElement<FrField>;
pub type FpElement = FieldElement<FpField>;
pub type Pairing = BLS12381AtePairing;
pub const MAXIMUM_DEGREE: usize = 10;
pub type KZG = KateZaveruchaGoldberg<MAXIMUM_DEGREE, FrField, Pairing>;
pub const NUMBER_CONSTRAINTS: usize = 4;
pub const ORDER_4_ROOT_UNITY: FrElement =
    FrElement::from_hex("8d51ccce760304d0ec030002760300000001000000000000"); // order 4
pub const ORDER_R_MINUS_1_ROOT_UNITY: FrElement = FrElement::from_hex("7");

type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;

pub fn test_srs_1() -> StructuredReferenceString<10, G1Point, G2Point> {
    let s = FrElement::from(2);
    let g1 = <BLS12381Curve as IsEllipticCurve>::generator();
    let g2 = <BLS12381TwistCurve as IsEllipticCurve>::generator();

    let powers_main_group: Vec<G1Point> = (0..24)
        .map(|exp| g1.operate_with_self(s.pow(exp as u64).representative()))
        .collect();
    let powers_secondary_group = [g2.clone(), g2.operate_with_self(s.representative())];

    StructuredReferenceString::new(&powers_main_group, &powers_secondary_group)
}

pub fn test_circuit_1() -> Circuit {
    Circuit {
        number_public_inputs: 2,
        number_private_inputs: 1,
        number_internal_variables: 1,
    }
}

pub fn test_common_preprocessed_input_1() -> CommonPreprocessedInput<FrField> {
    let w = ORDER_4_ROOT_UNITY;
    let u = ORDER_R_MINUS_1_ROOT_UNITY;
    let domain = (1..4).fold(vec![FieldElement::one()], |mut acc, _| {
        acc.push(acc.last().unwrap() * ORDER_4_ROOT_UNITY);
        acc
    });

    let s1_lagrange = vec![
        u.pow(2_u64) * w.pow(3_u64),
        u.pow(0_u64) * w.pow(3_u64),
        u.pow(0_u64) * w.pow(0_u64),
        u.pow(0_u64) * w.pow(1_u64),
    ];
    let s2_lagrange = vec![
        u.pow(0_u64) * w.pow(2_u64),
        u.pow(1_u64) * w.pow(0_u64),
        u.pow(1_u64) * w.pow(2_u64),
        u.pow(2_u64) * w.pow(2_u64),
    ];
    let s3_lagrange = vec![
        u.pow(1_u64) * w.pow(1_u64),
        u.pow(2_u64) * w.pow(0_u64),
        u.pow(1_u64) * w.pow(3_u64),
        u.pow(2_u64) * w.pow(1_u64),
    ];

    CommonPreprocessedInput {
        n: NUMBER_CONSTRAINTS,
        omega: ORDER_4_ROOT_UNITY,
        k1: ORDER_R_MINUS_1_ROOT_UNITY,
        domain: domain.clone(),

        ql: Polynomial::interpolate(
            &domain,
            &[
                -FieldElement::one(),
                -FieldElement::one(),
                FieldElement::zero(),
                FieldElement::one(),
            ],
        ),
        qr: Polynomial::interpolate(
            &domain,
            &[
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
                -FieldElement::one(),
            ],
        ),
        qo: Polynomial::interpolate(
            &domain,
            &[
                FieldElement::zero(),
                FieldElement::zero(),
                -FieldElement::one(),
                FieldElement::zero(),
            ],
        ),
        qm: Polynomial::interpolate(
            &domain,
            &[
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::one(),
                FieldElement::zero(),
            ],
        ),
        qc: Polynomial::interpolate(
            &domain,
            &[
                FieldElement::from(0_u64), // TODO: this should be filled in by the prover
                FieldElement::from(0_u64), // TODO: this should be filled in by the prover
                FieldElement::zero(),
                FieldElement::zero(),
            ],
        ),

        s1: Polynomial::interpolate(&domain, &s1_lagrange),
        s2: Polynomial::interpolate(&domain, &s2_lagrange),
        s3: Polynomial::interpolate(&domain, &s3_lagrange),

        s1_lagrange,
        s2_lagrange,
        s3_lagrange,
    }
}

pub fn test_witness_1() -> Witness<FrField> {
    Witness {
        a: vec![
            FrElement::from(2_u64),
            FrElement::from(4_u64),
            FrElement::from(2_u64),
            FrElement::from(4_u64),
        ],
        b: vec![
            FrElement::from(2_u64),
            FrElement::from(2_u64),
            FrElement::from(2_u64),
            FrElement::from(4_u64),
        ],
        c: vec![
            FrElement::from(2_u64),
            FrElement::from(2_u64),
            FrElement::from(4_u64),
            FrElement::from(2_u64),
        ],
    }
}