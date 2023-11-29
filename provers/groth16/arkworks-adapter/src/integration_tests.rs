use crate::{io_and_witness_from_arkworks_cs, pinocchio_r1cs_from_arkworks_cs};
use ark_bls12_381::Fr;

use ark_relations::{lc, r1cs::ConstraintSystem};
use lambdaworks_groth16::{setup, verify, Proof, Prover, QuadraticArithmeticProgram};

#[test]
fn create_proof_from_arkworks_and_verify_it() {
    /*
        c * d = e
        (a + b) + e = result

        e is witness
    */

    let cs = ConstraintSystem::<Fr>::new_ref();

    let _a = Fr::from(1555);
    let _b = Fr::from(25555);
    let _c = Fr::from(35555);
    let _d = Fr::from(45555);

    let a = cs.new_input_variable(|| Ok(_a)).unwrap();
    let b = cs.new_input_variable(|| Ok(_b)).unwrap();
    let c = cs.new_input_variable(|| Ok(_c)).unwrap();
    let d = cs.new_input_variable(|| Ok(_d)).unwrap();

    let e = cs.new_witness_variable(|| Ok(_c * _d)).unwrap();
    cs.enforce_constraint(lc!() + c, lc!() + d, lc!() + e)
        .unwrap();

    let result = cs.new_input_variable(|| Ok(_c * _d * (_a + _b))).unwrap();

    cs.enforce_constraint(lc!() + a + b, lc!() + e, lc!() + result)
        .unwrap();

    let r1cs = pinocchio_r1cs_from_arkworks_cs(&cs);
    let w = io_and_witness_from_arkworks_cs(&cs);

    let qap = QuadraticArithmeticProgram::from_r1cs(r1cs);

    println!("---------------------------- witness");
    println!("----------------------------");
    w.iter().for_each(|e| {
        println!("{}", e.to_string());
    });

    println!("---------------------------- num_pub_ins");
    println!("----------------------------");
    println!("{:?}", qap.num_of_public_inputs);
    qap.l.iter().for_each(|row| {
        println!("{:?}", row);
    });
    qap.r.iter().for_each(|row| {
        println!("{:?}", row);
    });
    qap.o.iter().for_each(|row| {
        println!("{:?}", row);
    });

    let (pk, vk) = setup(&qap);

    let serialized_proof = Prover::prove(&w, &qap, &pk).serialize();
    let deserialized_proof = Proof::deserialize(&serialized_proof).unwrap();

    let accept = verify(&vk, &deserialized_proof, &w[..qap.num_of_public_inputs]);
    assert!(accept);
}
