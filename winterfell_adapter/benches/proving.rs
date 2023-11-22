use criterion::{criterion_group, criterion_main, Criterion};
use miden_air::{ProcessorAir, ProvingOptions, PublicInputs};
use miden_assembly::Assembler;
use miden_core::{Felt, Program, StackInputs};
use miden_processor::DefaultHost;
use miden_processor::{self as processor};
use miden_prover::prove;
use processor::ExecutionTrace;
use stark_platinum_prover::prover::MidenProver;
use stark_platinum_prover::trace::TraceTable;
use stark_platinum_prover::transcript::MidenProverTranscript;
use stark_platinum_prover::{proof::options::ProofOptions, prover::IsStarkProver};
use winter_math::{FieldElement, StarkField};
use winter_prover::Trace;
use winterfell_adapter::adapter::air::AirAdapter;
use winterfell_adapter::adapter::public_inputs::AirAdapterPublicInputs;

struct BenchInstance {
    program: Program,
    stack_inputs: StackInputs,
    trace: TraceTable<Felt>,
    pub_inputs: AirAdapterPublicInputs<ProcessorAir, ExecutionTrace, Felt>,
    lambda_proof_options: ProofOptions,
    transcript: MidenProverTranscript,
}

fn create_bench_instance() -> BenchInstance {
    let fibonacci_number = 16;
    let program = format!(
        "begin
            repeat.{}
                swap dup.1 add
            end
        end",
        fibonacci_number - 1
    );
    let program = Assembler::default().compile(&program).unwrap();
    let stack_inputs = StackInputs::try_from_values([0, 1]).unwrap();

    let mut lambda_proof_options = ProofOptions::default_test_options();
    lambda_proof_options.blowup_factor = 8;

    let winter_trace = processor::execute(
        &program,
        stack_inputs.clone(),
        DefaultHost::default(),
        *ProvingOptions::default().execution_options(),
    )
    .unwrap();
    let program_info = winter_trace.program_info().clone();
    let stack_outputs = winter_trace.stack_outputs().clone();

    let pub_inputs = PublicInputs::new(program_info, stack_inputs.clone(), stack_outputs.clone());

    let pub_inputs = AirAdapterPublicInputs::<ProcessorAir, ExecutionTrace, Felt>::new(
        pub_inputs,
        vec![0; 182], // Not used, but still has to have 182 things because of zip's.
        vec![2; 182],
        vec![0, 1],
        winter_trace.clone(),
        winter_trace.get_info(),
        1,
    );

    let trace = AirAdapter::<ProcessorAir, ExecutionTrace, Felt>::convert_winterfell_trace_table(
        winter_trace.main_segment().clone(),
    );

    let transcript = MidenProverTranscript::new(&[]);

    BenchInstance {
        program,
        stack_inputs,
        trace,
        pub_inputs,
        lambda_proof_options,
        transcript,
    }
}

pub fn bench_prove_miden_fibonacci(c: &mut Criterion) {
    let instance = create_bench_instance();

    c.bench_function("winterfell_prover", |b| {
        b.iter(|| {
            let (mut outputs, proof) = prove(
                &instance.program,
                instance.stack_inputs.clone(),
                DefaultHost::default(),
                ProvingOptions::default(),
            )
            .unwrap();
        })
    });

    c.bench_function("lambda_prover", |b| {
        b.iter(|| {
            let proof = MidenProver::prove::<AirAdapter<ProcessorAir, ExecutionTrace, Felt>>(
                &instance.trace,
                &instance.pub_inputs,
                &instance.lambda_proof_options,
                MidenProverTranscript::new(&[]),
            )
            .unwrap();
        })
    });
}

criterion_group!(benches, bench_prove_miden_fibonacci);
criterion_main!(benches);
