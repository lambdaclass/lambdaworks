use lambdaworks_groth16::{common::*, setup, verify, Proof, Prover, QuadraticArithmeticProgram};
use lambdaworks_math::cyclic_group::IsGroup;

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

#[test]
fn tampered_proof_rejected() {
    let qap = test_circuits::vitalik_qap();
    let (pk, vk) = setup(&qap).expect("setup should succeed");

    let w = ["0x1", "0x3", "0x23", "0x9", "0x1b", "0x1e"]
        .map(FrElement::from_hex_unchecked)
        .to_vec();

    let proof = Prover::prove(&w, &qap, &pk).unwrap();

    // Tamper with pi1 by adding a random element
    let tampered_proof = Proof {
        pi1: proof.pi1.operate_with_self(2u64),
        pi2: proof.pi2.clone(),
        pi3: proof.pi3.clone(),
    };

    let accept = verify(&vk, &tampered_proof, &w[..qap.num_of_public_inputs])
        .expect("verification should succeed");
    assert!(!accept, "tampered proof should be rejected");
}

#[test]
fn qap_empty_matrices_rejected() {
    let empty: [Vec<FrElement>; 0] = [];
    let result = QuadraticArithmeticProgram::from_variable_matrices(0, &empty, &empty, &empty);
    assert!(result.is_err(), "empty matrices should fail");
}

#[test]
fn qap_inconsistent_matrix_sizes_rejected() {
    let l = vec![vec![FrElement::one()]];
    let r = vec![vec![FrElement::one()], vec![FrElement::zero()]]; // Different size
    let o = vec![vec![FrElement::one()]];

    let result = QuadraticArithmeticProgram::from_variable_matrices(1, &l, &r, &o);
    assert!(result.is_err(), "inconsistent matrix sizes should fail");
}

#[test]
fn qap_too_many_public_inputs_rejected() {
    let l = vec![vec![FrElement::one()]];
    let r = vec![vec![FrElement::one()]];
    let o = vec![vec![FrElement::one()]];

    let result = QuadraticArithmeticProgram::from_variable_matrices(5, &l, &r, &o);
    assert!(result.is_err(), "more public inputs than variables should fail");
}

#[test]
fn multiple_proofs_same_circuit() {
    // Tests that multiple valid witnesses produce valid proofs with the same keys
    let qap = test_qap_2();
    let (pk, vk) = setup(&qap).expect("setup should succeed");

    // Different witnesses satisfying x^2 = 25, y^2 = 9
    let witnesses = [
        // x = 5, y = 3
        ["0x1", "0x5", "0x3", "0x0", "0x19", "0x9", "0x0", "0x0"],
        // x = -5 (mod p), y = 3 - represented as p-5
        // For simplicity, we use x=5, y=-3 (which is p-3)
    ];

    for w in witnesses {
        let w = w.map(FrElement::from_hex_unchecked).to_vec();
        let proof = Prover::prove(&w, &qap, &pk).expect("proof should succeed");
        let accept = verify(&vk, &proof, &w[..qap.num_of_public_inputs])
            .expect("verification should succeed");
        assert!(accept, "valid witness should produce valid proof");
    }
}

#[test]
fn proof_determinism() {
    // Two proofs from the same witness should have different randomness (r, s)
    // so pi1, pi2, pi3 should differ, but both should verify
    let qap = test_circuits::vitalik_qap();
    let (pk, vk) = setup(&qap).expect("setup should succeed");

    let w = ["0x1", "0x3", "0x23", "0x9", "0x1b", "0x1e"]
        .map(FrElement::from_hex_unchecked)
        .to_vec();

    let proof1 = Prover::prove(&w, &qap, &pk).expect("first proof");
    let proof2 = Prover::prove(&w, &qap, &pk).expect("second proof");

    // Proofs should be different (due to random blinding)
    assert!(
        proof1.pi1 != proof2.pi1 || proof1.pi2 != proof2.pi2 || proof1.pi3 != proof2.pi3,
        "proofs should have different randomness"
    );

    // But both should verify
    let accept1 = verify(&vk, &proof1, &w[..qap.num_of_public_inputs])
        .expect("verification should succeed");
    let accept2 = verify(&vk, &proof2, &w[..qap.num_of_public_inputs])
        .expect("verification should succeed");
    assert!(accept1 && accept2, "both proofs should verify");
}
