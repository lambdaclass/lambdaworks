use crate::setup::{CommonPreprocessedInput, Witness};
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
    traits::IsRandomFieldElementGenerator,
};
use serde::{Deserialize, Serialize};
// TODO: Generalize

use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::pairing::BLS12381AtePairing,
    field::fields::montgomery_backed_prime_fields::IsModulus, unsigned_integer::element::U256,
};

pub const ORDER_R: U256 =
    U256::from("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");

#[derive(Clone, Debug)]
pub struct FrConfig;
impl IsModulus<U256> for FrConfig {
    const MODULUS: U256 = ORDER_R;
}

pub type Curve = BLS12381Curve;
pub type TwistedCurve = BLS12381TwistCurve;
pub type FrField = U256PrimeField<FrConfig>;
pub type FpField = BLS12381PrimeField;
pub type FrElement = FieldElement<FrField>;
pub type FpElement = FieldElement<FpField>;
pub type Pairing = BLS12381AtePairing;
pub type KZG = KateZaveruchaGoldberg<FrField, Pairing>;
pub const NUMBER_CONSTRAINTS: usize = 4;
pub const ORDER_4_ROOT_UNITY: FrElement =
    FrElement::from_hex("8d51ccce760304d0ec030002760300000001000000000000"); // order 4
pub const ORDER_8_ROOT_UNITY: FrElement =
    FrElement::from_hex("345766f603fa66e78c0625cd70d77ce2b38b21c28713b7007228fd3397743f7a"); // order 8
pub const ORDER_R_MINUS_1_ROOT_UNITY: FrElement = FrElement::from_hex("7");

type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;

pub fn test_srs(n: usize) -> StructuredReferenceString<G1Point, G2Point> {
    let s = FrElement::from(2);
    let g1 = <BLS12381Curve as IsEllipticCurve>::generator();
    let g2 = <BLS12381TwistCurve as IsEllipticCurve>::generator();

    let powers_main_group: Vec<G1Point> = (0..n + 3)
        .map(|exp| g1.operate_with_self(s.pow(exp as u64).representative()))
        .collect();
    let powers_secondary_group = [g2.clone(), g2.operate_with_self(s.representative())];

    StructuredReferenceString::new(&powers_main_group, &powers_secondary_group)
}

pub fn identity_permutation(w: FrElement, n: u64) -> Vec<FrElement> {
    let u = ORDER_R_MINUS_1_ROOT_UNITY;
    let mut result: Vec<FrElement> = vec![];
    for index_column in 0..=2 {
        for index_row in 0..n {
            result.push(w.pow(index_row) * u.pow(index_column as u64));
        }
    }
    result
}

pub fn test_common_preprocessed_input_1() -> CommonPreprocessedInput<FrField> {
    let w = ORDER_4_ROOT_UNITY;
    let n = 4;
    let domain = (1..n).fold(vec![FieldElement::one()], |mut acc, _| {
        acc.push(acc.last().unwrap() * &w);
        acc
    });

    let permutation = vec![11, 3, 0, 1, 2, 4, 6, 10, 5, 8, 7, 9];
    let identity = identity_permutation(w, n);
    let permuted: Vec<FrElement> = (0..12)
        .map(|i| identity[permutation[i as usize]].clone())
        .collect();

    let s1_lagrange: Vec<FrElement> = permuted[..4].to_vec();
    let s2_lagrange: Vec<FrElement> = permuted[4..8].to_vec();
    let s3_lagrange: Vec<FrElement> = permuted[8..].to_vec();

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

pub fn test_witness_1(x: FrElement, e: FrElement) -> Witness<FrField> {
    let y = &x * &e;
    let empty = x.clone();
    Witness {
        a: vec![
            x.clone(), // Public input
            y.clone(), // Public input
            x.clone(), // LHS for multiplication
            y,         // LHS for ==
        ],
        b: vec![
            empty.clone(),
            empty.clone(),
            e.clone(), // RHS for multiplication
            &x * &e,   // RHS for ==
        ],
        c: vec![
            empty.clone(),
            empty.clone(),
            &x * &e, // Output of multiplication
            empty,
        ],
    }
}

pub fn test_common_preprocessed_input_2() -> CommonPreprocessedInput<FrField> {
    let w = ORDER_8_ROOT_UNITY;
    let n = 8;
    let domain = (1..n).fold(vec![FieldElement::one()], |mut acc, _| {
        acc.push(acc.last().unwrap() * &w);
        acc
    });

    let permutation = vec![
        23, 4, 0, 18, 1, 2, 5, 6, 7, 8, 10, 9, 19, 11, 13, 14, 15, 16, 3, 12, 17, 20, 21, 22,
    ];
    let identity = identity_permutation(w, n);
    let permuted: Vec<FrElement> = (0..24)
        .map(|i| identity[permutation[i as usize]].clone())
        .collect();

    let s1_lagrange: Vec<FrElement> = permuted[..8].to_vec();
    let s2_lagrange: Vec<FrElement> = permuted[8..16].to_vec();
    let s3_lagrange: Vec<FrElement> = permuted[16..].to_vec();

    CommonPreprocessedInput {
        n: n as usize,
        omega: ORDER_8_ROOT_UNITY,
        k1: ORDER_R_MINUS_1_ROOT_UNITY,
        domain: domain.clone(),

        ql: Polynomial::interpolate(
            &domain,
            &[
                -FieldElement::one(),
                -FieldElement::one(),
                FieldElement::zero(),
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
            ],
        ),
        qr: Polynomial::interpolate(
            &domain,
            &[
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
                -FieldElement::one(),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
            ],
        ),
        qo: Polynomial::interpolate(
            &domain,
            &[
                FieldElement::zero(),
                FieldElement::zero(),
                -FieldElement::one(),
                -FieldElement::one(),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
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
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
            ],
        ),
        qc: Polynomial::interpolate(
            &domain,
            &[
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::from(5_u64),
                FieldElement::zero(),
                FieldElement::zero(),
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

pub fn test_witness_2(x: FrElement, e: FrElement) -> Witness<FrField> {
    Witness {
        a: vec![
            x.clone(),
            &x * &e + FieldElement::from(5_u64),
            x.clone(),
            &x * &e,
            &x * &e + FieldElement::from(5_u64),
            x.clone(),
            x.clone(),
            x.clone(),
        ],
        b: vec![
            x.clone(),
            x.clone(),
            e.clone(),
            x.clone(),
            &x * &e + FieldElement::from(5_u64),
            x.clone(),
            x.clone(),
            x.clone(),
        ],
        c: vec![
            x.clone(),
            x.clone(),
            &x * &e,
            &x * &e + FieldElement::from(5_u64),
            x.clone(),
            x.clone(),
            x.clone(),
            x,
        ],
    }
}

pub struct TestRandomFieldGenerator;
impl IsRandomFieldElementGenerator<FrField> for TestRandomFieldGenerator {
    fn generate(&self) -> FrElement {
        FrElement::zero()
    }
}

#[allow(non_snake_case)]
#[derive(Serialize, Deserialize)]
struct JsonPlonkCircuit {
    N: u64,
    Omega: String,
    Input: Vec<String>,
    Ql: Vec<String>,
    Qr: Vec<String>,
    Qm: Vec<String>,
    Qo: Vec<String>,
    Qc: Vec<String>,
    A: Vec<String>,
    B: Vec<String>,
    C: Vec<String>,
    Permutation: Vec<u64>,
}

pub fn common_preprocessed_input_from_json(
    json_string: &str,
) -> (
    Witness<FrField>,
    CommonPreprocessedInput<FrField>,
    Vec<FrElement>,
) {
    let json_input: JsonPlonkCircuit = serde_json::from_str(json_string).unwrap();
    let str2frelement = |ss: Vec<String>| {
        ss.iter()
            .map(|s| FrElement::from_hex(s))
            .collect::<Vec<FrElement>>()
    };
    let permutation = json_input
        .Permutation
        .iter()
        .map(|x| *x as usize)
        .collect::<Vec<usize>>();
    let n: usize = json_input.N as usize;
    let omega = FrElement::from_hex(&json_input.Omega);
    let domain = (1..n).fold(vec![FieldElement::one()], |mut acc, _| {
        acc.push(acc.last().unwrap() * &omega);
        acc
    });

    let identity = identity_permutation(omega.clone(), n as u64);
    let permuted: Vec<FrElement> = (0..n * 3)
        .map(|i| identity[permutation[i]].clone())
        .collect();

    let s1_lagrange: Vec<FrElement> = permuted[..n].to_vec();
    let s2_lagrange: Vec<FrElement> = permuted[n..2 * n].to_vec();
    let s3_lagrange: Vec<FrElement> = permuted[2 * n..].to_vec();
    (
        Witness {
            a: str2frelement(json_input.A),
            b: str2frelement(json_input.B),
            c: str2frelement(json_input.C),
        },
        CommonPreprocessedInput {
            n,
            domain: domain.clone(),
            omega,
            k1: ORDER_R_MINUS_1_ROOT_UNITY,
            ql: Polynomial::interpolate(&domain, &str2frelement(json_input.Ql)),
            qr: Polynomial::interpolate(&domain, &str2frelement(json_input.Qr)),
            qo: Polynomial::interpolate(&domain, &str2frelement(json_input.Qo)),
            qm: Polynomial::interpolate(&domain, &str2frelement(json_input.Qm)),
            qc: Polynomial::interpolate(&domain, &str2frelement(json_input.Qc)),
            s1: Polynomial::interpolate(&domain, &s1_lagrange),
            s2: Polynomial::interpolate(&domain, &s2_lagrange),
            s3: Polynomial::interpolate(&domain, &s3_lagrange),
            s1_lagrange,
            s2_lagrange,
            s3_lagrange,
        },
        str2frelement(json_input.Input),
    )
}

#[cfg(test)]
mod tests {
    use super::common_preprocessed_input_from_json;

    #[test]
    fn test_import_gnark_circuit_from_json() {
        common_preprocessed_input_from_json(
            r#"{
 "N": 4,
 "Omega": "8d51ccce760304d0ec030002760300000001000000000000",
  "Input": [
  "2",
  "4"
 ],
 "Ql": [
  "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000000",
  "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000000",
  "0",
  "1"
 ],
 "Qr": [
  "0",
  "0",
  "0",
  "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000000"
 ],
 "Qm": [
  "0",
  "0",
  "1",
  "0"
 ],
 "Qo": [
  "0",
  "0",
  "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000000",
  "0"
 ],
 "Qc": [
  "0",
  "0",
  "0",
  "0"
 ],
 "A": [
  "2",
  "4",
  "2",
  "4"
 ],
 "B": [
  "2",
  "2",
  "2",
  "4"
 ],
 "C": [
  "2",
  "2",
  "4",
  "2"
 ],
 "Permutation": [
  11,
  3,
  2,
  1,
  0,
  4,
  5,
  10,
  6,
  8,
  7,
  9
 ]
}"#,
        );
    }
}
