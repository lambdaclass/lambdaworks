use iai_callgrind::black_box;
use lambdaworks_stark::{
    air::{context::ProofOptions, example::cairo},
    fri::FieldElement,
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
    let mut cairo_air = cairo::CairoAIR::new(proof_options, &trace.0);

    // PC FINAL AND AP FINAL are not computed correctly since they are extracted after padding to
    // power of two and therefore are zero
    cairo_air.pub_inputs.ap_final = FieldElement::zero();
    cairo_air.pub_inputs.pc_final = FieldElement::zero();

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
    let mut cairo_air = cairo::CairoAIR::new(proof_options, &trace.0);

    // PC FINAL AND AP FINAL are not computed correctly since they are extracted after padding to
    // power of two and therefore are zero
    cairo_air.pub_inputs.ap_final = FieldElement::zero();
    cairo_air.pub_inputs.pc_final = FieldElement::zero();

    let proof = black_box(prove(black_box(&trace), black_box(&cairo_air)));

    let ok = black_box(verify(black_box(&proof), black_box(&cairo_air)));

    assert!(ok);
}

iai_callgrind::main!(
    callgrind_args = "toggle-collect=util::*";
    functions = simple_fibonacci_benchmarks,
                two_col_fibonacci_benchmarks,
                cairo_fibonacci_benchmarks,
                cairo_factorial_benchmarks,
);
