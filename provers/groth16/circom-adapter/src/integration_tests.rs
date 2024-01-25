use crate::*;
use lambdaworks_groth16::*;

#[test]
fn init() {
    let (w, qap) = circom_r1cs_to_lambda_qap("test.r1cs.json", "witness.json");
    let (pk, vk) = setup(&qap);

    let accept = verify(
        &vk,
        &Prover::prove(&w, &qap, &pk),
        &w[..qap.num_of_public_inputs],
    );
    assert!(accept);
}
