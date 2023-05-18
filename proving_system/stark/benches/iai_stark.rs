use iai::black_box;
use lambdaworks_stark::{
    air::{context::ProofOptions, example::cairo},
    prover::prove,
    verifier::verify,
};

mod functions;
mod util;

#[inline(never)]
fn simple_fibonacci_benchmarks() {
    let (trace, fibonacci_air) = functions::stark::generate_fib_proof_params(8);

    let proof = black_box(prove(black_box(&trace), black_box(&fibonacci_air)));

    let ok = black_box(verify(black_box(&proof), black_box(&fibonacci_air)));

    assert!(ok);
}

#[inline(never)]
fn two_col_fibonacci_benchmarks() {
    let (trace, fibonacci_air) = functions::stark::generate_fib_2_cols_proof_params(16);

    let proof = black_box(prove(black_box(&trace), black_box(&fibonacci_air)));

    let ok = black_box(verify(black_box(&proof), black_box(&fibonacci_air)));

    assert!(ok);
}

#[inline(never)]
fn cairo_fibonacci_benchmarks() {
    let trace = functions::stark::generate_cairo_trace("fibonacci_10");

    let proof_options = ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 32,
        coset_offset: 3,
    };
    let cairo_air = cairo::CairoAIR::new(proof_options, &trace.0);

    let proof = black_box(prove(black_box(&trace), black_box(&cairo_air)));

    let ok = black_box(verify(black_box(&proof), black_box(&cairo_air)));

    assert!(ok);
}

#[inline(never)]
fn cairo_factorial_benchmarks() {
    let trace = functions::stark::generate_cairo_trace("factorial_8");

    let proof_options = ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 32,
        coset_offset: 3,
    };
    let cairo_air = cairo::CairoAIR::new(proof_options, &trace.0);

    let proof = black_box(prove(black_box(&trace), black_box(&cairo_air)));

    let ok = black_box(verify(black_box(&proof), black_box(&cairo_air)));

    assert!(ok);
}

iai::main!(
    simple_fibonacci_benchmarks,
    two_col_fibonacci_benchmarks,
    cairo_fibonacci_benchmarks,
    cairo_factorial_benchmarks,
);
