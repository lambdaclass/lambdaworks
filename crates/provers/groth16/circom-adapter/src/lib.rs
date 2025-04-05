use lambdaworks_groth16::{common::FrElement, QuadraticArithmeticProgram};

mod readers;
pub use readers::*;

/// Given a [`CircomR1CS`] and [`CircomWitness`] it returns a [QAP](QuadraticArithmeticProgram),
/// witness, and public signals; all compatible with Lambdaworks.
pub fn circom_to_lambda(
    circom_r1cs: CircomR1CS,
    witness: CircomWitness,
) -> (QuadraticArithmeticProgram, Vec<FrElement>, Vec<FrElement>) {
    let num_of_outputs = circom_r1cs.num_outputs;
    let num_of_pub_inputs = circom_r1cs.num_pub_inputs;

    // we could get a slice using the QAP but the QAP does not keep track of the number of private inputs;
    // so instead we get the public signals here
    let public_inputs = witness[..num_of_pub_inputs + num_of_outputs + 1].to_vec();

    // get L,R,O matrices from R1CS
    let [l, r, o] = build_lro_from_circom_r1cs(circom_r1cs);
    let qap = QuadraticArithmeticProgram::from_variable_matrices(public_inputs.len(), &l, &r, &o);

    (qap, witness, public_inputs)
}

/// Takes as input circom.r1cs.json file and outputs LRO matrices
#[inline]
fn build_lro_from_circom_r1cs(circom_r1cs: CircomR1CS) -> [Vec<Vec<FrElement>>; 3] {
    let mut l = vec![vec![FrElement::zero(); circom_r1cs.num_constraints]; circom_r1cs.num_vars];
    let mut r = l.clone(); // same initial value as above
    let mut o = l.clone(); // same initial value as above

    // assign each constraint from the R1CS hash-maps to LRO matrices
    for (constraint_idx, constraint) in circom_r1cs.constraints.into_iter().enumerate() {
        // destructuring here to avoid clones
        let [lc, rc, oc] = constraint;

        for (var_idx, val) in lc {
            l[var_idx][constraint_idx] = val;
        }
        for (var_idx, val) in rc {
            r[var_idx][constraint_idx] = val;
        }
        for (var_idx, val) in oc {
            o[var_idx][constraint_idx] = val;
        }
    }

    [l, r, o]
}
