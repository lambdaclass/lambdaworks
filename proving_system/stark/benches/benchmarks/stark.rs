use criterion::{criterion_group, criterion_main, Criterion};
use lambdaworks_crypto::merkle_tree::test_merkle::TestHasher;
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
    prover::Prover,
    verifier::Verifier,
};

use lambdaworks_stark::air::example::{
    fibonacci_2_columns, fibonacci_f17, quadratic_air, simple_fibonacci,
};

pub type FE = FieldElement<Stark252PrimeField>;

pub fn proof_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("starks");
    group.sample_size(10);

    group.bench_function("simple_fibonacci", |bench| {
        bench.iter(prove_fib);
    });

    group.bench_function("fibonacci_2_columns", |bench| {
        bench.iter(prove_fib_2_cols);
    });

    group.bench_function("fibonacci_f17", |bench| {
        bench.iter(prove_fib17);
    });

    group.bench_function("quadratic_air", |bench| {
        bench.iter(prove_quadratic);
    });
}

criterion_group!(benches, proof_benchmark);
criterion_main!(benches);

fn prove_fib() {
    let trace = simple_fibonacci::fibonacci_trace([FE::from(1), FE::from(1)], 16);
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

    let fibonacci_air = simple_fibonacci::FibonacciAIR::new(context);
    let mut prover = Prover::new(&fibonacci_air);
    let mut verifier = Verifier::new(&fibonacci_air);

    let result = prover.prove::<TestHasher>(&trace_table);
    verifier.verify(&result);
}

fn prove_fib_2_cols() {
    let trace_columns =
        fibonacci_2_columns::fibonacci_trace_2_columns([FE::from(1), FE::from(1)], 16);

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
    let mut prover = Prover::new(&fibonacci_air);
    let mut verifier = Verifier::new(&fibonacci_air);

    let result = prover.prove::<TestHasher>(&trace_table);
    verifier.verify(&result);
}

fn prove_fib17() {
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
    let mut prover = Prover::new(&fibonacci_air);
    let mut verifier = Verifier::new(&fibonacci_air);

    let result = prover.prove::<TestHasher>(&trace_table);
    verifier.verify(&result);
}

fn prove_quadratic() {
    let trace = quadratic_air::quadratic_trace(FE::from(3), 16);
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
    let mut prover = Prover::new(&fibonacci_air);
    let mut verifier = Verifier::new(&fibonacci_air);

    let result = prover.prove::<TestHasher>(&trace_table);
    verifier.verify(&result);
}
