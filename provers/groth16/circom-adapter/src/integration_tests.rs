use std::fs;

use crate::*;
use lambdaworks_groth16::*;

const TEST_DIR: &str = "src/test_files/";

// Proves & verifies a Poseidon circuit with 1 input and 2 outputs. The input is decimal 100.
#[test]
fn poseidon_parse_prove_verify() {
    let test_dir = format!("{TEST_DIR}poseidon/");

    let (qap, w) = circom_to_lambda(
        &fs::read_to_string(format!("{test_dir}test.r1cs.json")).expect("Error reading the file"),
        &fs::read_to_string(format!("{test_dir}witness.json")).expect("Error reading the file"),
    );

    let (pk, vk) = setup(&qap);

    let accept = verify(
        &vk,
        &Prover::prove(&w, &qap, &pk),
        &w[..qap.num_of_public_inputs],
    );
    assert!(accept);
}
