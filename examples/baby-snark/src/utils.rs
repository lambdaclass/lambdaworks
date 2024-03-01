use std::ops::Neg;

use crate::{
    common::FrElement, scs::SquareConstraintSystem, setup, ssp::SquareSpanProgram, verify, Prover,
};

pub fn test_integration(
    u: Vec<Vec<FrElement>>,
    witness: Vec<FrElement>,
    public: Vec<FrElement>,
    pass: bool,
) {
    let mut input = public.clone();
    input.extend(witness.clone());

    let ssp = SquareSpanProgram::from_scs(SquareConstraintSystem::from_matrix(u, public.len()));
    let (proving_key, verifying_key) = setup(&ssp);

    let proof = Prover::prove(&input, &ssp, &proving_key);
    let verified = verify(&verifying_key, &proof, &public);
    if pass {
        assert!(verified);
    } else {
        assert!(!verified);
    }
}

pub fn i64_to_field(element: &i64) -> FrElement {
    let mut fr_element = FrElement::from(element.unsigned_abs());
    if element.is_negative() {
        fr_element = fr_element.neg()
    }

    fr_element
}

pub fn i64_vec_to_field(elements: &[i64]) -> Vec<FrElement> {
    elements.iter().map(i64_to_field).collect()
}

pub fn i64_matrix_to_field(elements: &[&[i64]]) -> Vec<Vec<FrElement>> {
    let mut matrix = Vec::new();
    for f in elements {
        matrix.push(i64_vec_to_field(f));
    }
    matrix
}

pub fn normalize(matrix: &mut Vec<Vec<FrElement>>, input: &Vec<FrElement>) {
    for i in 0..matrix.len() {
        let coef = matrix[i]
            .iter()
            .zip(input)
            .map(|(a, b)| a * b)
            .reduce(|a, b| a + b)
            .unwrap();
        matrix[i] = matrix[i].iter().map(|x| x * coef.inv().unwrap()).collect();
    }
}
