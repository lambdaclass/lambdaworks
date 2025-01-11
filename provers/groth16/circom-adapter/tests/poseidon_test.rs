use lambdaworks_circom_adapter::{circom_to_lambda, read_circom_r1cs, read_circom_witness};

// Proves & verifies a Poseidon circuit with 1 input and 2 outputs. The input is decimal 100.
#[test]
fn poseidon_parse_prove_verify() {
    let r1cs = read_circom_r1cs("./tests/poseidon/test.r1cs.json").expect("could not read r1cs");
    let wtns =
        read_circom_witness("./tests/poseidon/witness.json").expect("could not read witness");

    let (qap, w) = circom_to_lambda(r1cs, wtns);

    let (pk, vk) = lambdaworks_groth16::setup(&qap);

    let accept = lambdaworks_groth16::verify(
        &vk,
        &lambdaworks_groth16::Prover::prove(&w, &qap, &pk),
        &w[..qap.num_of_public_inputs],
    );
    assert!(accept);
}
