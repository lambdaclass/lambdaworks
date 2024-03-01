use std::ops::Neg;

use baby_snark::{
    self, common::FrElement, scs::SquareConstraintSystem, setup, ssp::SquareSpanProgram, verify,
    Prover,
};

#[test]
fn test_simplest_circuit() {
    let u = vec![i64_vec_to_field(&[1, 0]), i64_vec_to_field(&[0, 1])];
    let witness = i64_vec_to_field(&[1, 1]);
    let public = i64_vec_to_field(&[]);

    test_integration(u, witness, public)
}

#[test]
fn size_not_pow2() {
    let u: &[&[i64]] = &[
        &[1, 3, 2, 4, 5],
        &[-1, -2, 3, 4, -2],
        &[1, 2, 3, 2, 2],
        &[-3, -2, 0, 0, 0],
        &[0, 9, 2, -1, 3],
    ];
    let input: &[i64] = &[1, 2, 3, 4, 5];
    let witness = i64_vec_to_field(&[3, 4, 5]);
    let public = i64_vec_to_field(&[1, 2]);
    let input_field = i64_vec_to_field(input);
    let u_field = normalize(i64_matrix_to_field(u), &input_field);

    test_integration(u_field, witness, public);
}

#[test]
fn test_simple_circuit() {
    let u = vec![
        i64_vec_to_field(&[-1, 2, 0, 0]),
        i64_vec_to_field(&[-1, 0, 2, 0]),
        i64_vec_to_field(&[-1, 0, 0, 2]),
        i64_vec_to_field(&[-1, 2, 2, -4]),
    ];
    let witness = i64_vec_to_field(&[1, 1, 1]);
    let public = i64_vec_to_field(&[1]);

    test_integration(u, witness, public)
}

fn test_integration(u: Vec<Vec<FrElement>>, witness: Vec<FrElement>, public: Vec<FrElement>) {
    let mut input = public.clone();
    input.extend(witness.clone());

    let ssp = SquareSpanProgram::from_scs(SquareConstraintSystem::from_matrix(u, public.len()));
    let (proving_key, verifying_key) = setup(&ssp);

    let proof = Prover::prove(&input, &ssp, &proving_key);
    let verified = verify(&verifying_key, &proof, &public);

    assert!(verified);
}

fn i64_to_field(element: &i64) -> FrElement {
    let mut fr_element = FrElement::from(element.unsigned_abs());
    if element.is_negative() {
        fr_element = fr_element.neg()
    }

    fr_element
}

fn i64_vec_to_field(elements: &[i64]) -> Vec<FrElement> {
    elements.iter().map(i64_to_field).collect()
}

fn i64_matrix_to_field(elements: &[&[i64]]) -> Vec<Vec<FrElement>> {
    let mut matrix = Vec::new();
    for f in elements {
        matrix.push(i64_vec_to_field(f));
    }
    matrix
}

fn normalize(matrix: Vec<Vec<FrElement>>, input: &Vec<FrElement>) -> Vec<Vec<FrElement>> {
    let mut new_matrix = Vec::new();

    for row in matrix {
        let coef = row
            .iter()
            .zip(input)
            .map(|(a, b)| a * b)
            .reduce(|a, b| a + b)
            .unwrap();
        let new_row: Vec<FrElement> = row.iter().map(|x| x * coef.inv().unwrap()).collect();
        new_matrix.push(new_row);
    }

    new_matrix
}
