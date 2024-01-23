use crate::*;
use lambdaworks_groth16::*;

#[test]
fn init() {
    let qap = circom_r1cs_to_lambda_qap("test.r1cs.json");
    let (pk, vk) = setup(&qap);

    let witness = read_circom_witness("witness.json");

    let accept = verify(
        &vk,
        &Prover::prove(&witness, &qap, &pk),
        &witness[..qap.num_of_public_inputs],
    );
    assert!(accept);
}

#[test]
fn vitalik_example() {
    /*
        pub out ~out
        sig x, sym_1, y, sym_2

        x * x = sym_1;
        sym_1 * x = y;
        (y + x) * 1 = sym_2
        (sym_2 + 5) * 1 = ~out
    */

    // let lambda_cs = arkworks_cs_to_lambda_cs(&cs);

    // let qap = QuadraticArithmeticProgram::from_r1cs(lambda_cs.constraints);

    // let (pk, vk) = setup(&qap);

    // let accept = verify(
    //     &vk,
    //     &Prover::prove(&lambda_cs.witness, &qap, &pk),
    //     &lambda_cs.witness[..qap.num_of_public_inputs],
    // );
    // assert!(accept);
}
