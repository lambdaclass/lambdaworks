use lambdaworks_math::field::fields::u64_prime_field::FE17;
use lambdaworks_stark::{
    air::{
        context::{AirContext, ProofOptions},
        example::cairo,
        trace::TraceTable,
        TraceInfo, TraceLayout,
    },
    cairo_vm::{
        cairo_mem::CairoMemory, cairo_trace::CairoTrace,
        execution_trace::build_cairo_execution_trace,
    },
    prover::prove,
    verifier::verify,
};

use lambdaworks_stark::air::example::{
    fibonacci_2_columns, fibonacci_f17, quadratic_air, simple_fibonacci,
};

use crate::util::FE;

pub fn prove_fib(trace_length: usize) {
    let trace = simple_fibonacci::fibonacci_trace([FE::from(1), FE::from(1)], trace_length);
    let trace_table = TraceTable::new_from_cols(&trace, None);

    let layout = TraceLayout {
        main_segment_width: 1,
        aux_segments_info: None,
    };

    let trace_info = TraceInfo {
        layout,
        trace_length,
    };

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_info,
        transition_degrees: vec![1],
        aux_transition_degrees: None,
        transition_exemptions: vec![2],
        transition_offsets: vec![0, 1, 2],
        aux_transition_offsets: None,
        num_transition_constraints: 1,
        num_aux_transition_constraints: 0,
    };

    let fibonacci_air = simple_fibonacci::FibonacciAIR::new(context);

    let result = prove(&trace_table, &fibonacci_air);
    verify(&result, &fibonacci_air);
}

pub fn prove_fib_2_cols() {
    let trace_columns =
        fibonacci_2_columns::fibonacci_trace_2_columns([FE::from(1), FE::from(1)], 16);

    let trace_table = TraceTable::new_from_cols(&trace_columns, None);
    let trace_length = trace_table.n_rows();

    let layout = TraceLayout {
        main_segment_width: 2,
        aux_segments_info: None,
    };

    let trace_info = TraceInfo {
        layout,
        trace_length,
    };

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_info,
        transition_degrees: vec![1, 1],
        aux_transition_degrees: None,
        transition_exemptions: vec![1, 1],
        transition_offsets: vec![0, 1],
        aux_transition_offsets: None,
        num_transition_constraints: 2,
        num_aux_transition_constraints: 0,
    };

    let fibonacci_air = fibonacci_2_columns::Fibonacci2ColsAIR::from(context);

    let result = prove(&trace_table, &fibonacci_air);
    verify(&result, &fibonacci_air);
}

pub fn prove_fib17() {
    let trace = simple_fibonacci::fibonacci_trace([FE17::from(1), FE17::from(1)], 4);
    let trace_table = TraceTable::new_from_cols(&trace, None);
    let trace_length = trace_table.n_rows();

    let layout = TraceLayout {
        main_segment_width: 1,
        aux_segments_info: None,
    };

    let trace_info = TraceInfo {
        layout,
        trace_length,
    };

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_info,
        transition_degrees: vec![1],
        aux_transition_degrees: None,
        transition_exemptions: vec![2],
        transition_offsets: vec![0, 1, 2],
        aux_transition_offsets: None,
        num_transition_constraints: 1,
        num_aux_transition_constraints: 0,
    };

    let fibonacci_air = fibonacci_f17::Fibonacci17AIR::from(context);

    let result = prove(&trace_table, &fibonacci_air);
    verify(&result, &fibonacci_air);
}

pub fn prove_quadratic() {
    let trace = quadratic_air::quadratic_trace(FE::from(3), 16);
    let trace_table = TraceTable {
        main_segment: trace,
        main_segment_width: 1,
        aux_segments: None,
    };

    let trace_length = trace_table.n_rows();

    let layout = TraceLayout {
        main_segment_width: 1,
        aux_segments_info: None,
    };

    let trace_info = TraceInfo {
        layout,
        trace_length,
    };

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_info,
        transition_degrees: vec![2],
        aux_transition_degrees: None,
        transition_exemptions: vec![1],
        transition_offsets: vec![0, 1],
        aux_transition_offsets: None,
        num_transition_constraints: 1,
        num_aux_transition_constraints: 0,
    };

    let quadratic_air = quadratic_air::QuadraticAIR::from(context);

    let result = prove(&trace_table, &quadratic_air);
    verify(&result, &quadratic_air);
}

// We added an attribute to disable the `dead_code` lint because clippy doesn't take into account
// functions used by criterion.

#[allow(dead_code)]
pub fn prove_cairo_fibonacci_5() {
    let base_dir = env!("CARGO_MANIFEST_DIR");
    let dir_trace = base_dir.to_owned() + "/src/cairo_vm/test_data/fibonacci_5.trace";
    let dir_memory = base_dir.to_owned() + "/src/cairo_vm/test_data/fibonacci_5.memory";

    let raw_trace = CairoTrace::from_file(&dir_trace).expect("Cairo trace binary file not found");
    let memory = CairoMemory::from_file(&dir_memory).expect("Cairo memory binary file not found");

    let execution_trace = build_cairo_execution_trace(&raw_trace, &memory);

    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 5,
        coset_offset: 3,
    };

    let cairo_air = cairo::CairoAIR::new(proof_options, &execution_trace);

    prove(&execution_trace, &cairo_air);
}

#[allow(dead_code)]
pub fn prove_cairo_fibonacci_10() {
    let base_dir = env!("CARGO_MANIFEST_DIR");
    let dir_trace = base_dir.to_owned() + "/src/cairo_vm/test_data/fibonacci_10.trace";
    let dir_memory = base_dir.to_owned() + "/src/cairo_vm/test_data/fibonacci_10.memory";

    let raw_trace = CairoTrace::from_file(&dir_trace).expect("Cairo trace binary file not found");
    let memory = CairoMemory::from_file(&dir_memory).expect("Cairo memory binary file not found");

    let execution_trace = build_cairo_execution_trace(&raw_trace, &memory);

    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 5,
        coset_offset: 3,
    };

    let cairo_air = cairo::CairoAIR::new(proof_options, &execution_trace);

    prove(&execution_trace, &cairo_air);
}

#[allow(dead_code)]
pub fn prove_cairo_fibonacci_30() {
    let base_dir = env!("CARGO_MANIFEST_DIR");
    let dir_trace = base_dir.to_owned() + "/src/cairo_vm/test_data/fibonacci_30.trace";
    let dir_memory = base_dir.to_owned() + "/src/cairo_vm/test_data/fibonacci_30.memory";

    let raw_trace = CairoTrace::from_file(&dir_trace).expect("Cairo trace binary file not found");
    let memory = CairoMemory::from_file(&dir_memory).expect("Cairo memory binary file not found");

    let execution_trace = build_cairo_execution_trace(&raw_trace, &memory);

    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 5,
        coset_offset: 3,
    };

    let cairo_air = cairo::CairoAIR::new(proof_options, &execution_trace);

    prove(&execution_trace, &cairo_air);
}

#[allow(dead_code)]
pub fn prove_cairo_fibonacci_50() {
    let base_dir = env!("CARGO_MANIFEST_DIR");
    let dir_trace = base_dir.to_owned() + "/src/cairo_vm/test_data/fibonacci_50.trace";
    let dir_memory = base_dir.to_owned() + "/src/cairo_vm/test_data/fibonacci_50.memory";

    let raw_trace = CairoTrace::from_file(&dir_trace).expect("Cairo trace binary file not found");
    let memory = CairoMemory::from_file(&dir_memory).expect("Cairo memory binary file not found");

    let execution_trace = build_cairo_execution_trace(&raw_trace, &memory);

    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 5,
        coset_offset: 3,
    };

    let cairo_air = cairo::CairoAIR::new(proof_options, &execution_trace);

    prove(&execution_trace, &cairo_air);
}

#[allow(dead_code)]
pub fn prove_cairo_fibonacci_100() {
    let base_dir = env!("CARGO_MANIFEST_DIR");
    let dir_trace = base_dir.to_owned() + "/src/cairo_vm/test_data/fibonacci_100.trace";
    let dir_memory = base_dir.to_owned() + "/src/cairo_vm/test_data/fibonacci_100.memory";

    let raw_trace = CairoTrace::from_file(&dir_trace).expect("Cairo trace binary file not found");
    let memory = CairoMemory::from_file(&dir_memory).expect("Cairo memory binary file not found");

    let execution_trace = build_cairo_execution_trace(&raw_trace, &memory);

    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 5,
        coset_offset: 3,
    };

    let cairo_air = cairo::CairoAIR::new(proof_options, &execution_trace);

    prove(&execution_trace, &cairo_air);
}
