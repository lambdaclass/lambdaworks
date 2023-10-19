use groth16::prover::prove;
use groth16::setup::setup;
use groth16::verifier;
use verifier::Verifier;

fn main() {
    println!("start");
    // setup
    let s = setup();
    // prove
    // todo! define types correctly
    let r1cs_constraint_system = 0;
    let witness = 0;
    let proof = prove(r1cs_constraint_system, s.proving_key, witness).unwrap();
    // verify
    let verifier = Verifier::new();
    let is_proof_valid = Verifier::verify(&verifier, &proof, &s.verifying_key);
    println!("proof is valid {:?}", &is_proof_valid);
    println!("finish");
}
