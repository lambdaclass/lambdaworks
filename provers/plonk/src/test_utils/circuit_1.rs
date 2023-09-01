use super::utils::{
    generate_domain, generate_permutation_coefficients, ORDER_R_MINUS_1_ROOT_UNITY,
};
use crate::setup::{CommonPreprocessedInput, Witness};
use lambdaworks_math::fft::polynomial::FFTPoly;
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::default_types::{FrElement, FrField},
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};

pub const ORDER_4_ROOT_UNITY: FrElement =
    FrElement::from_hex_unchecked("8d51ccce760304d0ec030002760300000001000000000000"); // order 4

/*  Test circuit for the program:
    public input x
    public input y
    private input e
    z = x * e
    assert y == z
*/
pub fn test_common_preprocessed_input_1() -> CommonPreprocessedInput<FrField> {
    let n = 4;
    let omega = FrField::get_primitive_root_of_unity(2).unwrap();
    let domain = generate_domain(&omega, n);
    let permuted = generate_permutation_coefficients(
        &omega,
        n,
        &[11, 3, 0, 1, 2, 4, 6, 10, 5, 8, 7, 9],
        &ORDER_R_MINUS_1_ROOT_UNITY,
    );

    let s1_lagrange: Vec<FrElement> = permuted[..4].to_vec();
    let s2_lagrange: Vec<FrElement> = permuted[4..8].to_vec();
    let s3_lagrange: Vec<FrElement> = permuted[8..].to_vec();

    CommonPreprocessedInput {
        n,
        omega,
        domain,
        k1: ORDER_R_MINUS_1_ROOT_UNITY,
        // domain: domain.clone(),
        ql: Polynomial::interpolate_fft(&[
            -FieldElement::one(),
            -FieldElement::one(),
            FieldElement::zero(),
            FieldElement::one(),
        ])
        .unwrap(),

        qr: Polynomial::interpolate_fft(&[
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            -FieldElement::one(),
        ])
        .unwrap(),

        qo: Polynomial::interpolate_fft(&[
            FieldElement::zero(),
            FieldElement::zero(),
            -FieldElement::one(),
            FieldElement::zero(),
        ])
        .unwrap(),

        qm: Polynomial::interpolate_fft(&[
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::zero(),
        ])
        .unwrap(),

        qc: Polynomial::interpolate_fft(&[
            FieldElement::from(0_u64),
            FieldElement::from(0_u64),
            FieldElement::zero(),
            FieldElement::zero(),
        ])
        .unwrap(),

        s1: Polynomial::interpolate_fft(&s1_lagrange).unwrap(),
        s2: Polynomial::interpolate_fft(&s2_lagrange).unwrap(),
        s3: Polynomial::interpolate_fft(&s3_lagrange).unwrap(),

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
