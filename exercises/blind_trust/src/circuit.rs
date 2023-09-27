use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::default_types::{FrElement, FrField},
    field::element::FieldElement,
    polynomial::Polynomial,
};
use lambdaworks_plonk::setup::{CommonPreprocessedInput, Witness};

use crate::sith_generate_proof::{ORDER_8_ROOT_UNITY, ORDER_R_MINUS_1_ROOT_UNITY};

/// Generates a domain to interpolate: 1, omega, omega², ..., omega^size
pub fn generate_domain(omega: &FrElement, size: usize) -> Vec<FrElement> {
    (1..size).fold(vec![FieldElement::one()], |mut acc, _| {
        acc.push(acc.last().unwrap() * omega);
        acc
    })
}

/// The identity permutation, auxiliary function to generate the copy constraints.
fn identity_permutation(w: &FrElement, n: usize) -> Vec<FrElement> {
    let u = ORDER_R_MINUS_1_ROOT_UNITY;
    let mut result: Vec<FrElement> = vec![];
    for index_column in 0..=2 {
        for index_row in 0..n {
            result.push(w.pow(index_row) * u.pow(index_column as u64));
        }
    }
    result
}

/// Generates the permutation coefficients for the copy constraints.
/// polynomials S1, S2, S3.
pub fn generate_permutation_coefficients(
    omega: &FrElement,
    n: usize,
    permutation: &[usize],
) -> Vec<FrElement> {
    let identity = identity_permutation(omega, n);
    let permuted: Vec<FrElement> = (0..n * 3)
        .map(|i| identity[permutation[i]].clone())
        .collect();
    permuted
}

/// Witness generator for the circuit `ASSERT y == x * h + b`
pub fn circuit_witness(
    b: &FrElement,
    y: &FrElement,
    h: &FrElement,
    x: &FrElement,
) -> Witness<FrField> {
    let z = x * h;
    let w = &z + b;
    let empty = b.clone();
    Witness {
        a: vec![
            b.clone(),
            y.clone(),
            x.clone(),
            b.clone(),
            w.clone(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
        ],
        b: vec![
            empty.clone(),
            empty.clone(),
            h.clone(),
            z.clone(),
            y.clone(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
        ],
        c: vec![
            empty.clone(),
            empty.clone(),
            z.clone(),
            w.clone(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
        ],
    }
}

/// Common preprocessed input for the circuit `ASSERT y == x * h + b`
pub fn circuit_common_preprocessed_input() -> CommonPreprocessedInput<FrField> {
    let n: usize = 8;
    let omega = ORDER_8_ROOT_UNITY;
    let domain = generate_domain(&omega, n);
    let permutation = &[
        23, 12, 2, 0, 19, 3, 5, 6, 7, 8, 10, 18, 1, 9, 13, 14, 15, 16, 11, 4, 17, 20, 21, 22,
    ];
    let permuted = generate_permutation_coefficients(&omega, n, permutation);

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
                FieldElement::one(),
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
                FieldElement::zero(),
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
