use lambdaworks_circom_adapter::*;
use lambdaworks_groth16::*;

fn main() {
    // TODO: path error here
    println!("\nReading input files");
    let circom_r1cs = read_circom_r1cs("./input_files/test.r1cs.json").unwrap();
    let circom_witness = read_circom_witness("./input_files/witness.json").unwrap();

    println!("\nConverting to Lambdaworks-compatible QAP and witness assignments");
    let (qap, w) = circom_to_lambda(circom_r1cs, circom_witness);

    println!("\nPerforming trusted setup");
    let (pk, vk) = setup(&qap);

    println!("\nProving");
    let proof = Prover::prove(&w, &qap, &pk);

    println!("\nVerifying");
    let accept = verify(&vk, &proof, &w[..qap.num_of_public_inputs]);

    assert!(accept);
    println!("Proof verified!");
}
