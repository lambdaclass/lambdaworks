use std::time::Instant;

use lambdaworks_winterfell_adapter::{
    adapter::{public_inputs::AirAdapterPublicInputs, QuadFeltTranscript},
    examples::miden_vm::MidenVMQuadFeltAir,
};
use miden_air::{ProvingOptions, PublicInputs};
use miden_assembly::Assembler;
use miden_core::{Felt, FieldElement, StackInputs, StarkField};
use miden_processor::{self as processor};
use processor::DefaultHost;
use stark_platinum_prover::{
    proof::options::{ProofOptions, SecurityLevel},
    prover::{IsStarkProver, Prover},
    verifier::{IsStarkVerifier, Verifier},
};
use winter_prover::Trace;

fn compute_fibonacci(n: usize) -> Felt {
    let mut t0 = Felt::ZERO;
    let mut t1 = Felt::ONE;

    for _ in 0..n {
        t1 = t0 + t1;
        core::mem::swap(&mut t0, &mut t1);
    }
    t0
}

fn main() {
    let fibonacci_number = 16;

    let program = format!(
        "begin
            repeat.{}
                swap dup.1 add
            end
        end",
        fibonacci_number - 1
    );

    println!("\nCompiling miden fibonacci program");

    let program = Assembler::default().compile(program).unwrap();
    let expected_result = vec![compute_fibonacci(fibonacci_number).as_int()];
    let stack_inputs = StackInputs::try_from_values([0, 1]).unwrap();

    let mut lambda_proof_options = ProofOptions::new_secure(SecurityLevel::Conjecturable100Bits, 3);
    lambda_proof_options.blowup_factor = 8;

    println!("\nExecuting program in Miden VM");
    let winter_trace = processor::execute(
        &program,
        stack_inputs.clone(),
        DefaultHost::default(),
        *ProvingOptions::default().execution_options(),
    )
    .unwrap();
    let program_info = winter_trace.program_info().clone();
    let stack_outputs = winter_trace.stack_outputs().clone();

    let pub_inputs = PublicInputs::new(program_info, stack_inputs, stack_outputs.clone());

    assert_eq!(
        expected_result,
        stack_outputs.clone().stack_truncated(1),
        "Program result was computed incorrectly"
    );

    let pub_inputs = AirAdapterPublicInputs::new(
        pub_inputs,
        vec![2; 182],
        vec![0, 1],
        winter_trace.get_info(),
        winter_trace.clone().into(),
    );

    println!("\nImporting trace to lambdaworks");
    let trace =
        MidenVMQuadFeltAir::convert_winterfell_trace_table(winter_trace.main_segment().clone());

    println!("\nProving ");

    let timer0 = Instant::now();
    let proof = Prover::<MidenVMQuadFeltAir>::prove(
        &trace,
        &pub_inputs,
        &lambda_proof_options,
        QuadFeltTranscript::new(&[]),
    )
    .unwrap();
    let elapsed0 = timer0.elapsed();
    println!("Total time spent proving: {:?}", elapsed0);

    println!("\nVerifying ");
    let timer0 = Instant::now();
    assert!(Verifier::<MidenVMQuadFeltAir>::verify(
        &proof,
        &pub_inputs,
        &lambda_proof_options,
        QuadFeltTranscript::new(&[]),
    ));
    let elapsed0 = timer0.elapsed();
    println!("Total time spent verifying: {:?}", elapsed0);

    println!("\nDone!");
}
