use lambdaworks_math::field::fields::{
    fft_friendly::stark_252_prime_field::Stark252PrimeField, u64_prime_field::FE17,
};

use lambdaworks_stark::{
    air::{
        context::{AirContext, ProofOptions},
        example::{
            cairo, fibonacci_2_columns::fibonacci_trace_2_columns, simple_fibonacci::FibonacciAIR,
        },
        trace::TraceTable,
    },
    fri::FieldElement,
    prover::prove,
    verifier::verify,
};

use lambdaworks_stark::air::example::{
    fibonacci_2_columns, fibonacci_f17, quadratic_air, simple_fibonacci,
};

pub type FE = FieldElement<Stark252PrimeField>;

#[test]
fn test_prove_fib() {
    let trace = simple_fibonacci::fibonacci_trace([FE::from(1), FE::from(1)], 8);
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
    let trace = simple_fibonacci::fibonacci_trace([FE17::from(1), FE17::from(1)], 4);
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

    let fibonacci_air = fibonacci_f17::Fibonacci17AIR::new(context);

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

    let fibonacci_air = fibonacci_2_columns::Fibonacci2ColsAIR::new(context);

    let result = prove(&trace_table, &fibonacci_air);
    assert!(verify(&result, &fibonacci_air));
}

#[test]
fn test_prove_quadratic() {
    let trace = quadratic_air::quadratic_trace(FE::from(3), 4);
    let trace_table = TraceTable {
        table: trace.clone(),
        n_cols: 1,
    };

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_length: trace.len(),
        trace_columns: trace_table.n_cols,
        transition_degrees: vec![2],
        transition_exemptions: vec![1],
        transition_offsets: vec![0, 1],
        num_transition_constraints: 1,
    };

    let fibonacci_air = quadratic_air::QuadraticAIR::new(context);

    let result = prove(&trace_table, &fibonacci_air);
    assert!(verify(&result, &fibonacci_air));
}

#[test]
#[ignore = "we need a valid trace for this test"]
fn test() {
    // This trace is obtained from Giza when running the prover for the mentioned program.
    let trace_table = TraceTable::new_from_cols(&vec![
        // col 0
        vec![FE::zero(), FE::zero(), FE::one(), FE::zero()],
        // col 1
        vec![FE::one(), FE::one(), FE::one(), FE::zero()],
        // col 2
        vec![FE::one(), FE::one(), FE::zero(), FE::zero()],
        // col 3
        vec![FE::zero(), FE::zero(), FE::one(), FE::zero()],
        // col 4
        vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()],
        // col 5
        vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()],
        // col 6
        vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()],
        // col 7
        vec![FE::zero(), FE::zero(), FE::one(), FE::zero()],
        // col 8
        vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()],
        // col 9
        vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()],
        // col 10
        vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()],
        // col 11
        vec![FE::one(), FE::zero(), FE::zero(), FE::zero()],
        // col 12
        vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()],
        // col 13
        vec![FE::zero(), FE::zero(), FE::one(), FE::zero()],
        // col 14
        vec![FE::one(), FE::one(), FE::zero(), FE::zero()],
        // col 15
        vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()],
        // col 16
        vec![FE::from(3), FE::from(3), FE::from(9), FE::zero()],
        // col 17
        vec![FE::from(8), FE::from(9), FE::from(9), FE::zero()],
        // col 18
        vec![FE::from(8), FE::from(8), FE::from(8), FE::zero()],
        // col 19
        vec![FE::from(1), FE::from(3), FE::from(5), FE::zero()],
        // col 20
        vec![FE::from(8), FE::from(8), FE::from(6), FE::zero()],
        // col 21
        vec![FE::from(7), FE::from(7), FE::from(7), FE::zero()],
        // col 22
        vec![FE::from(2), FE::from(4), FE::from(7), FE::zero()],
        // col 23
        vec![
            FE::from(0x480680017fff8000),
            FE::from(0x400680017fff7fff),
            FE::from(0x208b7fff7fff7ffe),
            FE::zero(),
        ],
        // col 24
        vec![FE::from(3), FE::from(3), FE::from(9), FE::zero()],
        // col 25
        vec![FE::from(9), FE::from(9), FE::from(9), FE::zero()],
        // col 26
        vec![FE::from(3), FE::from(3), FE::from(9), FE::zero()],
        // col 27
        vec![
            FE::from(0x8000),
            FE::from(0x7fff),
            FE::from(0x7ffe),
            FE::zero(),
        ],
        // col 28
        vec![
            FE::from(0x7fff),
            FE::from(0x7fff),
            FE::from(0x7fff),
            FE::zero(),
        ],
        // col 29
        vec![
            FE::from(0x8001),
            FE::from(0x8001),
            FE::from(0x7fff),
            FE::zero(),
        ],
        // col 30
        vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()],
        // col 31
        vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()],
        // col 32
        vec![FE::from(0x1b), FE::from(0x1b), FE::from(0x51), FE::zero()],
        // col 33
        vec![FE::one(), FE::one(), FE::zero(), FE::zero()],
    ]);

    let some_cairo_air = cairo::CairoAIR::new(&trace_table);

    let result = prove(&trace_table, &some_cairo_air);
    assert!(verify(&result, &some_cairo_air));
}
