use lambdaworks_math::field::fields::{
    fft_friendly::stark_252_prime_field::MontgomeryConfigStark252PrimeField,
    montgomery_backed_prime_fields::MontgomeryBackendPrimeField, u64_prime_field::FE17,
};
use lambdaworks_stark::{
    air::{
        context::{AirContext, ProofOptions},
        example::{
            fibonacci_2_columns::Fibonacci2ColsAIR, fibonacci_f17::Fibonacci17AIR,
            quadratic_air::QuadraticAIR, simple_fibonacci::FibonacciAIR,
        },
    },
    cairo_run::run::run_program,
    cairo_vm::{cairo_mem::CairoMemory, cairo_trace::CairoTrace},
    fri::{FieldElement, U64PrimeField},
};

use lambdaworks_stark::air::example::{
    fibonacci_2_columns, fibonacci_f17, quadratic_air, simple_fibonacci,
};

use crate::util::FE;

#[allow(dead_code)]
pub fn generate_fib_proof_params(
    trace_length: usize,
) -> (
    Vec<Vec<FieldElement<MontgomeryBackendPrimeField<MontgomeryConfigStark252PrimeField, 4>>>>,
    FibonacciAIR,
) {
    let trace = simple_fibonacci::fibonacci_trace([FE::from(1), FE::from(1)], trace_length);
    let trace_length = trace[0].len();

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_length,
        trace_columns: trace.len(),
        transition_degrees: vec![1],
        transition_exemptions: vec![2],
        transition_offsets: vec![0, 1, 2],
        num_transition_constraints: 1,
    };

    let fibonacci_air = simple_fibonacci::FibonacciAIR::from(context);

    (trace, fibonacci_air)
}

#[allow(dead_code)]
pub fn generate_fib_2_cols_proof_params(
    trace_length: usize,
) -> (
    Vec<Vec<FieldElement<MontgomeryBackendPrimeField<MontgomeryConfigStark252PrimeField, 4>>>>,
    Fibonacci2ColsAIR,
) {
    let trace =
        fibonacci_2_columns::fibonacci_trace_2_columns([FE::from(1), FE::from(1)], trace_length);

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_length: trace.len(),
        transition_degrees: vec![1, 1],
        transition_exemptions: vec![1, 1],
        transition_offsets: vec![0, 1],
        num_transition_constraints: 2,
        trace_columns: 2,
    };

    let fibonacci_air = fibonacci_2_columns::Fibonacci2ColsAIR::from(context);

    (trace, fibonacci_air)
}

#[allow(dead_code)]
pub fn generate_fib17_proof_params(
    trace_length: usize,
) -> (Vec<Vec<FieldElement<U64PrimeField<17>>>>, Fibonacci17AIR) {
    let trace = simple_fibonacci::fibonacci_trace([FE17::from(1), FE17::from(1)], trace_length);

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_length: trace.len(),
        trace_columns: trace[0].len(),
        transition_degrees: vec![1],
        transition_exemptions: vec![2],
        transition_offsets: vec![0, 1, 2],
        num_transition_constraints: 1,
    };

    let fibonacci_air = fibonacci_f17::Fibonacci17AIR::from(context);

    (trace, fibonacci_air)
}

#[allow(dead_code)]
pub fn generate_quadratic_proof_params(
    trace_length: usize,
) -> (
    Vec<FieldElement<MontgomeryBackendPrimeField<MontgomeryConfigStark252PrimeField, 4>>>,
    QuadraticAIR,
) {
    let trace = quadratic_air::quadratic_trace(FE::from(3), trace_length);

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_length: trace.len(),
        trace_columns: trace.len(),
        transition_degrees: vec![2],
        transition_exemptions: vec![1],
        transition_offsets: vec![0, 1],
        num_transition_constraints: 1,
    };

    let quadratic_air = quadratic_air::QuadraticAIR::from(context);

    (trace, quadratic_air)
}

// We added an attribute to disable the `dead_code` lint because clippy doesn't take into account
// functions used by criterion.
#[allow(dead_code)]
pub fn generate_cairo_trace(filename: &str, layout: &str) -> (CairoTrace, CairoMemory) {
    let base_dir = format!("{}/src/cairo_vm/test_data/", env!("CARGO_MANIFEST_DIR"));

    let contract = format!("{base_dir}/{filename}.json");
    let trace_path = format!("{base_dir}/{filename}.trace");
    let memory_path = format!("{base_dir}/{filename}.memory");

    run_program(None, layout, &contract, &trace_path, &memory_path).unwrap();

    let raw_trace = CairoTrace::from_file(&trace_path).expect("Cairo trace binary file not found");
    let memory = CairoMemory::from_file(&memory_path).expect("Cairo memory binary file not found");

    (raw_trace, memory)
}
