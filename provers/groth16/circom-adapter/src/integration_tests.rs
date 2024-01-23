use lambdaworks_groth16::common::FrField;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;

pub type U256 = UnsignedInteger<4>;

use crate::r1cs::*;
use lambdaworks_groth16::*;

#[test]
fn init() {
    read_circom_r1cs("src/circuit.r1cs.json");
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
