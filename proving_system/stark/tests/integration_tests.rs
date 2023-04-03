pub mod air;

use lambdaworks_math::field::fields::{
    fft_friendly::stark_252_prime_field::Stark252PrimeField, u64_prime_field::FE17,
};

use lambdaworks_stark::{
    air::{
        context::{AirContext, ProofOptions},
        trace::TraceTable,
        AIR,
    },
    fri::FieldElement,
    prover::prove,
    verifier::verify,
};

use crate::air::{
    fibonacci_2_columns::{fibonacci_trace_2_columns, Fibonacci2ColsAIR},
    fibonacci_f17::Fibonacci17AIR,
    simple_fibonacci::{fibonacci_trace, FibonacciAIR},
};

pub type FE = FieldElement<Stark252PrimeField>;

#[test]
fn test_prove_fib() {
    let trace = fibonacci_trace([FE::from(1), FE::from(1)], 8);
    let trace_length = trace[0].len();
    let trace_table = TraceTable::new_from_cols(&trace);

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_length,
        trace_columns: trace_table.n_cols,
        transition_degrees: vec![1],
        transition_exemptions: vec![2],
        transition_offsets: vec![0, 1, 2],
        num_transition_constraints: 1,
    };

    let fibonacci_air = FibonacciAIR::new(context);

    let result = prove(&trace_table, &fibonacci_air);
    assert!(verify(&result, &fibonacci_air));
}

#[test]
fn test_prove_fib17() {
    let trace = fibonacci_trace([FE17::from(1), FE17::from(1)], 4);
    let trace_table = TraceTable::new_from_cols(&trace);

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_length: trace_table.n_rows(),
        trace_columns: trace_table.n_cols,
        transition_degrees: vec![1],
        transition_exemptions: vec![2],
        transition_offsets: vec![0, 1, 2],
        num_transition_constraints: 1,
    };

    let fibonacci_air = Fibonacci17AIR::new(context);

    let result = prove(&trace_table, &fibonacci_air);
    assert!(verify(&result, &fibonacci_air));
}

#[test]
fn test_prove_fib_2_cols() {
    let trace_columns = fibonacci_trace_2_columns([FE::from(1), FE::from(1)], 16);

    let trace_table = TraceTable::new_from_cols(&trace_columns);

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_length: trace_table.n_rows(),
        transition_degrees: vec![1, 1],
        transition_exemptions: vec![1, 1],
        transition_offsets: vec![0, 1],
        num_transition_constraints: 2,
        trace_columns: 2,
    };

    let fibonacci_air = Fibonacci2ColsAIR::new(context);

    let result = prove(&trace_table, &fibonacci_air);
    assert!(verify(&result, &fibonacci_air));
}
