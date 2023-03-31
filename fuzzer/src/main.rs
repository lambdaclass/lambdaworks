#[macro_use]
extern crate honggfuzz;
use crate::gnark_backend_wrapper::c_go_structures::{GoString, KeyPair};
use crate::gnark_backend_wrapper::errors::GnarkBackendError;
use std::ffi::{CStr, CString};
use std::num::TryFromIntError;
use std::os::raw::{c_char, c_uchar};

// This lists all the functions in the foreign interface along with their type signature.
extern "C" {
    fn CreateCircuit(public_input_x: GoString, public_input_y: GoString, private_variable: GoString) -> c_uchar;
    fn NewWitness(circuit: GoString, field: GoString) -> c_uchar;
    fn VerityWithCircuitWitness(circuit: GoString, witness: GoString) -> c_uchar;
}

fn main() {
    loop {
        fuzz!(|data1: u64, data2: u64| {

            // This is the circuit for x * e == y
            let common_preprocesed_input = test_common_preprocessed_input_1();
            let srs = fuzzer_srs();

            // Public input
            let x = FieldElement::from(data1);
            let go_x = GoString::try_from(data1);
            let y = FieldElement::from(data1 * data2);
            let go_y = GoString::try_from(data1 * data2);

            // Private variable
            let e = FieldElement::from(data2);
            let go_e = GoString::try_from(data2);

            let public_input = vec![x.clone(), y];
            let witness = fuzzer_witness(x, e);

            let kzg = KZG::new(srs);
            let verifying_key = setup(&common_preprocesed_input, &kzg);
            let random_generator = TestRandomFieldGenerator {};

            let prover = Prover::new(kzg.clone(), random_generator);

            let circuit = unsafe { CreateCircuit(go_x , go_y, go_e) };
            let go_circuit = GoString::try_from(circuit);

            let  witness = unsafe { NewWitness(go_circuit, go_e) };
            let go_witness = GoString::try_from(witness);
            let  verify_with_gnark = unsafe { VerityWithCircuitWitness(go_circuit, go_witness) };
            
            let proof = prover.prove(
                &witness,
                &public_input,
                &common_preprocesed_input,
                &verifying_key,
            );

            let verifier = Verifier::new(kzg);
            assert_eq!(verifier.verify(
                &proof,
                &public_input,
                &common_preprocesed_input,
                &verifying_key
            ), verify_with_gnark);        
        });
    }
}

fn fuzzer_witness(x: FrElement, e: FrElement) -> Witness<FrField> {
    let y = &x * &e;
    let empty = x.clone();
    Witness {
        a: vec![
            x.clone(), // Public input
            y.clone(), // Public input
            x.clone(), // LHS for multiplication
            y,         // LHS for ==
        ],
        b: vec![
            empty.clone(),
            empty.clone(),
            e.clone(), // RHS for multiplication
            &x * &e,   // RHS for ==
        ],
        c: vec![
            empty.clone(),
            empty.clone(),
            &x * &e, // Output of multiplication
            empty,
        ],
    }
}

fn fuzzer_srs() -> StructuredReferenceString<G1Point, G2Point> {
    let s = FrElement::from(2);
    let g1 = <BLS12381Curve as IsEllipticCurve>::generator();
    let g2 = <BLS12381TwistCurve as IsEllipticCurve>::generator();

    let powers_main_group: Vec<G1Point> = (0..40)
        .map(|exp| g1.operate_with_self(s.pow(exp as u64).representative()))
        .collect();
    let powers_secondary_group = [g2.clone(), g2.operate_with_self(s.representative())];

    StructuredReferenceString::new(&powers_main_group, &powers_secondary_group)
}


