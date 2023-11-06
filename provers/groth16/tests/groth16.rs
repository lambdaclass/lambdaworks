use lambdaworks_groth16::{common::*, qap_example_circuit_1, setup, verify, Proof, Prover};

#[test]
fn test_groth16_1() {
    let qap = qap_example_circuit_1(); // x^3 + x + 5 = 35

    let (pk, vk) = setup(&qap);

    let w = ["0x1", "0x3", "0x23", "0x9", "0x1b", "0x1e"] // x = 3
        .map(FrElement::from_hex_unchecked)
        .to_vec();

    let serialized_proof = Prover::prove(&w, &qap, &pk).serialize();
    let deserialized_proof = Proof::deserialize(&serialized_proof).unwrap();

    let accept = verify(&vk, &deserialized_proof, &w[..qap.num_of_public_inputs]);
    assert!(accept);
}

#[test]
fn test_groth16_2() {
    let qap = qap_example_circuit_1(); // x^3 + x + 5 = 35

    let (pk, vk) = setup(&qap);

    let w = ["0x1", "0x1", "0x7", "0x1", "0x1", "0x2"] // x = 1
        .map(FrElement::from_hex_unchecked)
        .to_vec();

    let serialized_proof = Prover::prove(&w, &qap, &pk).serialize();
    let deserialized_proof = Proof::deserialize(&serialized_proof).unwrap();

    let accept = verify(&vk, &deserialized_proof, &w[..qap.num_of_public_inputs]);
    assert!(accept);
}
