use std::fs;

use lambdaworks_circom_adapter::circom_to_lambda;

// Proves & verifies a Poseidon circuit with 1 input and 2 outputs. The input is decimal 100.
#[test]
fn poseidon_parse_prove_verify() {
    let (qap, w) = circom_to_lambda(
        &fs::read_to_string(format!("./tests/poseidon/test.r1cs.json"))
            .expect("Error reading the file"),
        &fs::read_to_string(format!("./tests/poseidon/witness.json"))
            .expect("Error reading the file"),
    );

    let (pk, vk) = lambdaworks_groth16::setup(&qap);

    let accept = lambdaworks_groth16::verify(
        &vk,
        &lambdaworks_groth16::Prover::prove(&w, &qap, &pk),
        &w[..qap.num_of_public_inputs],
    );
    assert!(accept);
}
