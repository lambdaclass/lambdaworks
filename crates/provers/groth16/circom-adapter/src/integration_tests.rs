use std::fs;

use crate::*;
use lambdaworks_groth16::*;

const TEST_DIR: &str = "test_files";

// Proves & verifies a Poseidon circuit with 1 input and 2 outputs. The input is decimal 100.
#[test]
fn poseidon_parse_prove_verify() {
    let test_dir = format!("{TEST_DIR}/poseidon");

    let (qap, w) = circom_to_lambda(
        &fs::read_to_string(format!("{test_dir}/test.r1cs.json")).expect("Error reading the file"),
        &fs::read_to_string(format!("{test_dir}/witness.json")).expect("Error reading the file"),
    );

    let (pk, vk) = setup(&qap);

    let accept = verify(
        &vk,
        &Prover::prove(&w, &qap, &pk),
        &w[..qap.num_of_public_inputs],
    );
    assert!(accept);
}

// Converts following Circom circuit and inputs into Lambdaworks-compatible QAP and witness assignments
//
// template Test() {
//	signal input x;
//	signal output out;
//
//	signal sym_1;
//	signal y;
//
//	sym_1 <== x * x;
//	out <== (sym_1 * x) + (x + 5);
// }
//
// { "x": 3 }
#[test]
fn vitalik_w_and_qap() {
    let test_dir = format!("{TEST_DIR}/vitalik_example");

    let (qap, w) = circom_to_lambda(
        &fs::read_to_string(format!("{test_dir}/test.r1cs.json")).expect("Error reading the file"),
        &fs::read_to_string(format!("{test_dir}/witness.json")).expect("Error reading the file"),
    );

    // Circom witness contains outputs before circuit inputs where Lambdaworks puts inputs before the output. Freshly generated
    // witness assignment "w" must be in form ["1", "x", "~out", "sym_1"]
    assert_eq!(
        w,
        ["1", "3", "23", "9"]
            .map(FrElement::from_hex_unchecked)
            .to_vec()
    );

    // Regarding QAP, we expect 2 constraints in following form
    // -sym_1 = -x * x
    // x + 5 - ~out = -sym_1 * x
    //
    // Same ordering difference exists for variable matrices, too. Circom adapter changes the
    // order of the rows in the same way it rearranges the witness ordering.

    const BLS12381_MINUS_ONE_STR: &str =
        "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000000";
    const BLS12381_ZERO_STR: &str = "0x0";
    const BLS12381_ONE_STR: &str = "0x1";

    let expected_num_of_public_inputs = 1;
    let [temp_l, temp_r, temp_o] = [
        [
            [BLS12381_ZERO_STR, BLS12381_ZERO_STR],      // 1
            [BLS12381_MINUS_ONE_STR, BLS12381_ZERO_STR], // x
            [BLS12381_ZERO_STR, BLS12381_ZERO_STR],      // ~out
            [BLS12381_ZERO_STR, BLS12381_MINUS_ONE_STR], // sym_1
        ],
        [
            [BLS12381_ZERO_STR, BLS12381_ZERO_STR], // 1
            [BLS12381_ONE_STR, BLS12381_ONE_STR],   // x
            [BLS12381_ZERO_STR, BLS12381_ZERO_STR], // ~out
            [BLS12381_ZERO_STR, BLS12381_ZERO_STR], // sym_1
        ],
        [
            [BLS12381_ZERO_STR, "5"],                    // 1
            [BLS12381_ZERO_STR, BLS12381_ONE_STR],       // x
            [BLS12381_ZERO_STR, BLS12381_MINUS_ONE_STR], // ~out
            [BLS12381_MINUS_ONE_STR, BLS12381_ZERO_STR], // sym_1
        ],
    ]
    .map(|matrix| matrix.map(|row| row.map(FrElement::from_hex_unchecked).to_vec()));
    let expected_qap =
        QAP::from_variable_matrices(expected_num_of_public_inputs, &temp_l, &temp_r, &temp_o);

    let expected_l = expected_qap.l;
    let expected_r = expected_qap.r;
    let expected_o = expected_qap.o;

    assert_eq!(qap.l, expected_l);
    assert_eq!(qap.r, expected_r);
    assert_eq!(qap.o, expected_o);
}
