use lambdaworks_math::{
    cyclic_group::IsGroup,
    msm::pippenger::msm,
    elliptic_curve::traits::{IsEllipticCurve, IsPairing},
};

use pinocchio::common::{
    sample_fr_elem, Curve, G1Point, G2Point, Pairing, TwistedCurve, FE
};
use pinocchio::qap::QuadraticArithmeticProgram;
use pinocchio::setup::{setup, EvaluationKey, VerificationKey,ToxicWaste};
use pinocchio::prover::generate_proof;
use pinocchio::verifier::verify;

use pinocchio::test_utils::{new_test_r1cs, test_qap_solver};

fn test_pinocchio(toxic_waste: ToxicWaste) {
    println!("Running Pinocchio test...");

    let test_qap = new_test_r1cs().into();
    
    // Setup
    let (evaluation_key, verification_key) = setup(&test_qap, toxic_waste);

    // Define inputs
    let inputs = [FE::from(1), FE::from(2), FE::from(3), FE::from(4)];

    // Execute the QAP
    let (c_mid, c_output) = test_qap_solver(inputs.clone());

    // Construct the full witness vector
    let mut c_vector = inputs.to_vec();
    c_vector.push(c_mid);
    c_vector.push(c_output.clone());

    // Generate proof
    let proof = generate_proof(&evaluation_key, &test_qap, &c_vector);

    let mut c_io_vector = inputs.to_vec();
    c_io_vector.push(c_output);

    // Verify the proof
    let accepted = verify(&verification_key, &proof, &c_io_vector);

    if accepted {
        println!("Proof verified successfully!");
    } else {
        println!("Proof verification failed.");
    }
}
#[cfg(test)]
   #[test]
    fn test_pinocchio_random_toxic_waste() {
        let toxic_waste = ToxicWaste::sample();
        test_pinocchio(toxic_waste);
    }