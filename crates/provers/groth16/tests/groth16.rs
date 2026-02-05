use lambdaworks_groth16::{common::*, setup, verify, Proof, Prover};

mod test_circuits;
use test_circuits::*;

#[test]
fn vitalik() {
    let qap = test_circuits::vitalik_qap(); // x^3 + x + 5 = 35

    let (pk, vk) = setup(&qap).expect("setup should succeed");

    for w in [
        ["0x1", "0x3", "0x23", "0x9", "0x1b", "0x1e"],
        ["0x1", "0x1", "0x7", "0x1", "0x1", "0x2"],
    ] {
        let w = w // x = 3
            .map(FrElement::from_hex_unchecked)
            .to_vec();

        let serialized_proof = Prover::prove(&w, &qap, &pk).unwrap().serialize();
        let deserialized_proof = Proof::deserialize(&serialized_proof).unwrap();

        let accept = verify(&vk, &deserialized_proof, &w[..qap.num_of_public_inputs])
            .expect("verification should succeed");
        assert!(accept);
    }
}

#[test]
fn example() {
    let qap = test_qap_2();
    let (pk, vk) = setup(&qap).expect("setup should succeed");

    // 1, x, y, ~out, sym_1, sym_2, sym_3, sym_4
    let w = ["0x1", "0x5", "0x3", "0x0", "0x19", "0x9", "0x0", "0x0"] // x = 5, y = 3
        .map(FrElement::from_hex_unchecked)
        .to_vec();

    let serialized_proof = Prover::prove(&w, &qap, &pk).unwrap().serialize();
    let deserialized_proof = Proof::deserialize(&serialized_proof).unwrap();

    let accept = verify(&vk, &deserialized_proof, &w[..qap.num_of_public_inputs])
        .expect("verification should succeed");
    assert!(accept);
}

#[test]
fn invalid_proof_rejected() {
    let qap = test_circuits::vitalik_qap();
    let (pk, vk) = setup(&qap).expect("setup should succeed");

    let w = ["0x1", "0x3", "0x23", "0x9", "0x1b", "0x1e"]
        .map(FrElement::from_hex_unchecked)
        .to_vec();

    let proof = Prover::prove(&w, &qap, &pk).unwrap();

    // Modify the public input to make the proof invalid
    let wrong_public_inputs = [FrElement::from_hex_unchecked("0x99")];

    let accept = verify(&vk, &proof, &wrong_public_inputs).expect("verification should succeed");
    assert!(!accept, "proof with wrong public inputs should be rejected");
}
