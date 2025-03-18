use super::utils::{
    generate_domain, generate_permutation_coefficients, ORDER_R_MINUS_1_ROOT_UNITY,
};
use crate::setup::{CommonPreprocessedInput, Witness};
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::default_types::{FrElement, FrField},
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};

pub const ORDER_8_ROOT_UNITY: FrElement = FrElement::from_hex_unchecked(
    "345766f603fa66e78c0625cd70d77ce2b38b21c28713b7007228fd3397743f7a",
); // order 8

/*  Test circuit for the program:
    public input x
    public input y
    private input e
    z1 = x * e
    z2 = z1 + 5
    assert y == z2
*/
pub fn test_common_preprocessed_input_2() -> CommonPreprocessedInput<FrField> {
    let n: usize = 8;
    let omega = FrField::get_primitive_root_of_unity(3).unwrap();
    let domain = generate_domain(&omega, n);
    let permutation = &[
        23, 4, 0, 18, 1, 2, 5, 6, 7, 8, 10, 9, 19, 11, 13, 14, 15, 16, 3, 12, 17, 20, 21, 22,
    ];
    let permuted =
        generate_permutation_coefficients(&omega, n, permutation, &ORDER_R_MINUS_1_ROOT_UNITY);

    let s1_lagrange: Vec<FrElement> = permuted[..8].to_vec();
    let s2_lagrange: Vec<FrElement> = permuted[8..16].to_vec();
    let s3_lagrange: Vec<FrElement> = permuted[16..].to_vec();

    CommonPreprocessedInput {
        n,
        omega,
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
        )
        .unwrap(),
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
        )
        .unwrap(),
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
        )
        .unwrap(),
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
        )
        .unwrap(),
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
        )
        .unwrap(),

        s1: Polynomial::interpolate(&domain, &s1_lagrange).unwrap(),
        s2: Polynomial::interpolate(&domain, &s2_lagrange).unwrap(),
        s3: Polynomial::interpolate(&domain, &s3_lagrange).unwrap(),

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
