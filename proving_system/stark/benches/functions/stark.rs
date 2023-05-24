use lambdaworks_math::field::fields::u64_prime_field::FE17;
use lambdaworks_stark::air::example::{
    fibonacci_2_columns, fibonacci_f17, quadratic_air, simple_fibonacci,
};
use lambdaworks_stark::{
    air::{
        context::{AirContext, ProofOptions},
        example::cairo::{self, PublicInputs},
    },
    cairo_vm::{cairo_mem::CairoMemory, cairo_trace::CairoTrace},
    prover::prove,
    verifier::verify,
};

use crate::util::FE;

pub fn load_cairo_trace_and_memory(program_name: &str) -> (CairoTrace, CairoMemory) {
    let base_dir = env!("CARGO_MANIFEST_DIR");
    let dir_trace = format!("{}/src/cairo_vm/test_data/{}.trace", base_dir, program_name);
    let dir_memory = format!(
        "{}/src/cairo_vm/test_data/{}.memory",
        base_dir, program_name
    );

    let raw_trace = CairoTrace::from_file(&dir_trace).unwrap();
    let memory = CairoMemory::from_file(&dir_memory).unwrap();

    (raw_trace, memory)
}

pub fn prove_fib(trace_length: usize) {
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

    let result = prove(&trace, &fibonacci_air, &mut ()).unwrap();
    verify(&result, &fibonacci_air, &());
}

pub fn prove_fib_2_cols() {
    let trace_columns =
        fibonacci_2_columns::fibonacci_trace_2_columns([FE::from(1), FE::from(1)], 16);

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_length: trace_columns.len(),
        transition_degrees: vec![1, 1],
        transition_exemptions: vec![1, 1],
        transition_offsets: vec![0, 1],
        num_transition_constraints: 2,
        trace_columns: 2,
    };

    let fibonacci_air = fibonacci_2_columns::Fibonacci2ColsAIR::from(context);

    let result = prove(&trace_columns, &fibonacci_air, &mut ()).unwrap();
    verify(&result, &fibonacci_air, &());
}

#[allow(dead_code)]
pub fn prove_fib17() {
    let trace = simple_fibonacci::fibonacci_trace([FE17::from(1), FE17::from(1)], 4);

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

    let result = prove(&trace, &fibonacci_air, &mut ()).unwrap();
    verify(&result, &fibonacci_air, &());
}

#[allow(dead_code)]
pub fn prove_quadratic() {
    let trace = quadratic_air::quadratic_trace(FE::from(3), 16);

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

    let result = prove(&trace, &quadratic_air, &mut ()).unwrap();
    verify(&result, &quadratic_air, &());
}

// We added an attribute to disable the `dead_code` lint because clippy doesn't take into account
// functions used by criterion.

#[allow(dead_code)]
pub fn prove_cairo_fibonacci_5() {
    let (raw_trace, memory) = load_cairo_trace_and_memory("fibonacci_5");

    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 5,
        coset_offset: 3,
    };

    let cairo_air = cairo::CairoAIR::new(proof_options, 100, raw_trace.steps());
    let mut public_input = PublicInputs {
        pc_init: FE::from(raw_trace.rows[0].pc),
        ap_init: FE::from(raw_trace.rows[0].ap),
        fp_init: FE::from(raw_trace.rows[0].fp),
        pc_final: FE::zero(),
        ap_final: FE::zero(),
        num_steps: raw_trace.steps(),
        program: Vec::new(),
        range_check_min: None,
        range_check_max: None,
    };

    prove(&(raw_trace, memory), &cairo_air, &mut public_input).unwrap();
}

#[allow(dead_code)]
pub fn prove_cairo_fibonacci_10() {
    let (raw_trace, memory) = load_cairo_trace_and_memory("fibonacci_10");

    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 5,
        coset_offset: 3,
    };

    let cairo_air = cairo::CairoAIR::new(proof_options, 100, raw_trace.steps());

    let mut public_input = PublicInputs {
        pc_init: FE::from(raw_trace.rows[0].pc),
        ap_init: FE::from(raw_trace.rows[0].ap),
        fp_init: FE::from(raw_trace.rows[0].fp),
        pc_final: FE::zero(),
        ap_final: FE::zero(),
        num_steps: raw_trace.steps(),
        program: Vec::new(),
        range_check_min: None,
        range_check_max: None,
    };

    prove(&(raw_trace, memory), &cairo_air, &mut public_input).unwrap();
}

#[allow(dead_code)]
pub fn prove_cairo_fibonacci_30() {
    let (raw_trace, memory) = load_cairo_trace_and_memory("fibonacci_30");
    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 5,
        coset_offset: 3,
    };

    let cairo_air = cairo::CairoAIR::new(proof_options, 100, raw_trace.steps());
    let mut public_input = PublicInputs {
        pc_init: FE::from(raw_trace.rows[0].pc),
        ap_init: FE::from(raw_trace.rows[0].ap),
        fp_init: FE::from(raw_trace.rows[0].fp),
        pc_final: FE::zero(),
        ap_final: FE::zero(),
        num_steps: raw_trace.steps(),
        program: Vec::new(),
        range_check_min: None,
        range_check_max: None,
    };

    prove(&(raw_trace, memory), &cairo_air, &mut public_input).unwrap();
}

#[allow(dead_code)]
pub fn prove_cairo_fibonacci_50() {
    let (raw_trace, memory) = load_cairo_trace_and_memory("fibonacci_50");
    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 5,
        coset_offset: 3,
    };

    let cairo_air = cairo::CairoAIR::new(proof_options, 100, raw_trace.steps());
    let mut public_input = PublicInputs {
        pc_init: FE::from(raw_trace.rows[0].pc),
        ap_init: FE::from(raw_trace.rows[0].ap),
        fp_init: FE::from(raw_trace.rows[0].fp),
        pc_final: FE::zero(),
        ap_final: FE::zero(),
        num_steps: raw_trace.steps(),
        program: Vec::new(),
        range_check_min: None,
        range_check_max: None,
    };

    prove(&(raw_trace, memory), &cairo_air, &mut public_input).unwrap();
}

#[allow(dead_code)]
pub fn prove_cairo_fibonacci_100() {
    let (raw_trace, memory) = load_cairo_trace_and_memory("fibonacci_100");
    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 5,
        coset_offset: 3,
    };

    let cairo_air = cairo::CairoAIR::new(proof_options, 100, raw_trace.steps());
    let mut public_input = PublicInputs {
        pc_init: FE::from(raw_trace.rows[0].pc),
        ap_init: FE::from(raw_trace.rows[0].ap),
        fp_init: FE::from(raw_trace.rows[0].fp),
        pc_final: FE::zero(),
        ap_final: FE::zero(),
        num_steps: raw_trace.steps(),
        program: Vec::new(),
        range_check_min: None,
        range_check_max: None,
    };

    prove(&(raw_trace, memory), &cairo_air, &mut public_input).unwrap();
}

