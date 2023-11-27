use criterion::{black_box, criterion_group, criterion_main, Criterion};
use miden_air::{HashFunction, ProcessorAir, ProvingOptions, PublicInputs};
use miden_assembly::Assembler;
use miden_core::{Felt, Program, StackInputs};
use miden_processor::DefaultHost;
use miden_processor::{self as processor};
use miden_prover::prove;
use processor::ExecutionTrace;
use stark_platinum_prover::{proof::options::ProofOptions, prover::IsStarkProver};
use winter_air::FieldExtension;
use winter_prover::Trace;
use winterfell_adapter::adapter::air::AirAdapter;
use winterfell_adapter::adapter::public_inputs::AirAdapterPublicInputs;
use winterfell_adapter::examples::miden_vm::{
    ExecutionTraceMetadata, MidenProver, MidenProverTranscript,
};

struct BenchInstance {
    program: Program,
    stack_inputs: StackInputs,
    lambda_proof_options: ProofOptions,
}

fn create_bench_instance(fibonacci_number: usize) -> BenchInstance {
    let program = format!(
        "begin
            repeat.{}
                swap dup.1 add
            end
        end",
        fibonacci_number - 1
    );
    let program = Assembler::default().compile(program).unwrap();
    let stack_inputs = StackInputs::try_from_values([0, 1]).unwrap();
    let mut lambda_proof_options = ProofOptions::default_test_options();
    lambda_proof_options.blowup_factor = 8;

    BenchInstance {
        program,
        stack_inputs,
        lambda_proof_options,
    }
}

pub fn bench_prove_miden_fibonacci(c: &mut Criterion) {
    let instance = create_bench_instance(100);

    c.bench_function("winterfell_prover", |b| {
        b.iter(|| {
            let proving_options = ProvingOptions::new(
                instance.lambda_proof_options.fri_number_of_queries,
                instance.lambda_proof_options.blowup_factor as usize,
                instance.lambda_proof_options.grinding_factor as u32,
                FieldExtension::None,
                2,
                0,
                HashFunction::Blake3_192,
            );

            let (_outputs, _proof) = black_box(
                prove(
                    &instance.program,
                    instance.stack_inputs.clone(),
                    DefaultHost::default(),
                    proving_options,
                )
                .unwrap(),
            );
        })
    });

    c.bench_function("lambda_prover", |b| {
        b.iter(|| {
            // This is here because the only pub method in miden
            // is a prove function that executes AND proves.
            // This makes the benchmark a more fair
            // in the case that the program execution takes
            // too long.
            let winter_trace = processor::execute(
                &instance.program,
                instance.stack_inputs.clone(),
                DefaultHost::default(),
                *ProvingOptions::default().execution_options(),
            )
            .unwrap();

            let program_info = winter_trace.program_info().clone();
            let stack_outputs = winter_trace.stack_outputs().clone();
            let pub_inputs = AirAdapterPublicInputs::<ProcessorAir, ExecutionTraceMetadata>::new(
                PublicInputs::new(
                    program_info,
                    instance.stack_inputs.clone(),
                    stack_outputs.clone(),
                ),
                vec![2; 182],
                vec![0, 1],
                winter_trace.get_info(),
                winter_trace.clone().into(),
            );

            let trace =
                AirAdapter::<ProcessorAir, ExecutionTrace, Felt, _>::convert_winterfell_trace_table(
                    winter_trace.main_segment().clone(),
                );

            let _proof = black_box(
                MidenProver::prove::<AirAdapter<ProcessorAir, ExecutionTrace, Felt, _>>(
                    &trace,
                    &pub_inputs,
                    &instance.lambda_proof_options,
                    MidenProverTranscript::new(&[]),
                )
                .unwrap(),
            );
        })
    });
}

criterion_group!(benches, bench_prove_miden_fibonacci);
criterion_main!(benches);
