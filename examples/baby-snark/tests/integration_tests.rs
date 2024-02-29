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
    let mut input = public.clone();
    input.extend(witness.clone());

    let ssp = SquareSpanProgram::from_scs(SquareConstraintSystem::from_matrices(u, public.len()));
    let (pk, vk) = setup(&ssp);

    let proof = Prover::prove(&input, &ssp, &pk);
    let verified = verify(&vk, &proof, &public);

    assert!(verified);
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
    let mut input = public.clone();
    input.extend(witness.clone());

    let ssp = SquareSpanProgram::from_scs(SquareConstraintSystem::from_matrices(u, 1));
    let (pk, vk) = setup(&ssp);

    let proof = Prover::prove(&input, &ssp, &pk);
    let verified = verify(&vk, &proof, &public);

    assert!(verified);
}

fn i64_to_field(element: &i64) -> FrElement {
    let mut fr_element = FrElement::from(element.abs() as u64);
    if element.is_negative() {
        fr_element = fr_element.neg()
    }

    fr_element
}

fn i64_vec_to_field(elements: &[i64]) -> Vec<FrElement> {
    elements.iter().map(i64_to_field).collect()
}
