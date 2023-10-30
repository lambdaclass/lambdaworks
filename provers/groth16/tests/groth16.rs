use lambdaworks_groth16::{
    common::*, prover::generate_proof, setup::setup, test_utils::circuit_1::qap_example_circuit_1,
    verifier::verify,
};

#[test]
fn test_groth16() {
    /*
        sym_1 = x * x
        y = sym_1 * x
        sym_2 = y + x
        ~out = sym_2 + 5
    */
    let qap = qap_example_circuit_1();

    let (pk, vk) = setup(&qap);

    let w = ["0x1", "0x3", "0x23", "0x9", "0x1b", "0x1e"]
        .map(|e| FrElement::from_hex_unchecked(e))
        .to_vec();

    let proof = generate_proof(&w, &qap, &pk);

    let accept = verify(&vk, &proof, &w[..qap.num_of_public_inputs]);
    assert!(accept);
}
