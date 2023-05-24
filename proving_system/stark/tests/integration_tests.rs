use lambdaworks_math::field::fields::{
    fft_friendly::stark_252_prime_field::Stark252PrimeField, u64_prime_field::FE17,
};
use lambdaworks_math::helpers::resize_to_next_power_of_two;
use lambdaworks_stark::air::example::cairo::PublicInputs;
use lambdaworks_stark::air::example::fibonacci_rap::{fibonacci_rap_trace, FibonacciRAP};
use lambdaworks_stark::air::example::{
    cairo, dummy_air, fibonacci_2_columns, fibonacci_f17, quadratic_air, simple_fibonacci,
};
use lambdaworks_stark::cairo_vm::cairo_mem::CairoMemory;
use lambdaworks_stark::cairo_vm::cairo_trace::CairoTrace;
use lambdaworks_stark::{
    air::context::{AirContext, ProofOptions},
    fri::FieldElement,
    prover::prove,
    verifier::verify,
};

pub type FE = FieldElement<Stark252PrimeField>;

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

#[test_log::test]
fn test_prove_fib() {
    let trace = simple_fibonacci::fibonacci_trace([FE::from(1), FE::from(1)], 8);
    let trace_length = trace[0].len();

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_length,
        trace_columns: 1,
        transition_degrees: vec![1],
        transition_exemptions: vec![2],
        transition_offsets: vec![0, 1, 2],
        num_transition_constraints: 1,
    };

    let fibonacci_air = simple_fibonacci::FibonacciAIR::from(context);

    let result = prove(&trace, &fibonacci_air, &mut ()).unwrap();
    assert!(verify(&result, &fibonacci_air, &()));
}

#[test_log::test]
fn test_prove_fib17() {
    let trace = simple_fibonacci::fibonacci_trace([FE17::from(1), FE17::from(1)], 4);

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_length: trace[0].len(),
        trace_columns: 1,
        transition_degrees: vec![1],
        transition_exemptions: vec![2],
        transition_offsets: vec![0, 1, 2],
        num_transition_constraints: 1,
    };

    let fibonacci_air = fibonacci_f17::Fibonacci17AIR::from(context);

    let result = prove(&trace, &fibonacci_air, &mut ()).unwrap();
    assert!(verify(&result, &fibonacci_air, &()));
}

#[test_log::test]
fn test_prove_fib_2_cols() {
    let trace_columns =
        fibonacci_2_columns::fibonacci_trace_2_columns([FE::from(1), FE::from(1)], 16);

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 7,
            coset_offset: 3,
        },
        trace_length: trace_columns[0].len(),
        transition_degrees: vec![1, 1],
        transition_exemptions: vec![1, 1],
        transition_offsets: vec![0, 1],
        num_transition_constraints: 2,
        trace_columns: 2,
    };

    let fibonacci_air = fibonacci_2_columns::Fibonacci2ColsAIR::from(context);

    let result = prove(&trace_columns, &fibonacci_air, &mut ()).unwrap();
    assert!(verify(&result, &fibonacci_air, &()));
}

#[test_log::test]
fn test_prove_quadratic() {
    let trace = quadratic_air::quadratic_trace(FE::from(3), 4);

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_length: trace.len(),
        trace_columns: 1,
        transition_degrees: vec![2],
        transition_exemptions: vec![1],
        transition_offsets: vec![0, 1],
        num_transition_constraints: 1,
    };

    let quadratic_air = quadratic_air::QuadraticAIR::from(context);

    let result = prove(&trace, &quadratic_air, &mut ()).unwrap();
    assert!(verify(&result, &quadratic_air, &()));
}

#[ignore = "metal"]
#[test_log::test]
fn test_prove_cairo_simple_program() {
    /*
    Cairo program used in the test:

    ```
    func main() {
        let x = 1;
        let y = 2;
        assert x + y = 3;
        return ();
    }

    ```
    */
    let (raw_trace, memory) = load_cairo_trace_and_memory("simple_program");
    let proof_options = ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 1,
        coset_offset: 3,
    };

    let program_size = 5;
    let mut program = vec![];
    for i in 1..=program_size as u64 {
        program.push(memory.get(&i).unwrap().clone());
    }

    let cairo_air = cairo::CairoAIR::new(proof_options, 16, raw_trace.steps());

    let first_step = &raw_trace.rows[0];
    let last_step = &raw_trace.rows[raw_trace.steps() - 1];
    let mut public_input = PublicInputs {
        pc_init: FE::from(first_step.pc),
        ap_init: FE::from(first_step.ap),
        fp_init: FE::from(first_step.fp),
        pc_final: FE::from(last_step.pc),
        ap_final: FE::from(last_step.ap),
        program,
        range_check_min: None,
        range_check_max: None,
        num_steps: raw_trace.steps(),
    };

    let result = prove(&(raw_trace, memory), &cairo_air, &mut public_input).unwrap();
    assert!(verify(&result, &cairo_air, &public_input));
}

#[test_log::test]
fn test_prove_cairo_call_func() {
    /*
    Cairo program used in the test:

    ```
    func mul(x: felt, y: felt) -> (res: felt) {
        return (res = x * y);
    }

    func main() {
        let x = 2;
        let y = 3;

        let (res) = mul(x, y);
        assert res = 6;

        return ();
    }

    ```
    */
    let (raw_trace, memory) = load_cairo_trace_and_memory("call_func");
    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
    };

    let program_size = 11;
    let mut program = vec![];

    for i in 1..=program_size as u64 {
        program.push(memory.get(&i).unwrap().clone());
    }

    let cairo_air = cairo::CairoAIR::new(proof_options, 128, raw_trace.steps());
    let last_step = &raw_trace.rows[raw_trace.steps() - 1];
    let mut public_input = PublicInputs {
        pc_init: FE::from(raw_trace.rows[0].pc),
        ap_init: FE::from(raw_trace.rows[0].ap),
        fp_init: FE::from(raw_trace.rows[0].fp),
        pc_final: FieldElement::from(last_step.pc),
        ap_final: FieldElement::from(last_step.ap),
        range_check_min: None,
        range_check_max: None,
        program,
        num_steps: raw_trace.steps(),
    };

    let result = prove(&(raw_trace, memory), &cairo_air, &mut public_input).unwrap();
    assert!(verify(&result, &cairo_air, &public_input));
}

fn test_prove_cairo_fibonacci(file_name: &str, trace_length: usize) {
    let (raw_trace, memory) = load_cairo_trace_and_memory(file_name);

    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 5,
        coset_offset: 3,
    };

    let program_size = 24;
    let mut program = vec![];

    for i in 1..=program_size as u64 {
        program.push(memory.get(&i).unwrap().clone());
    }

    let cairo_air = cairo::CairoAIR::new(proof_options, trace_length, raw_trace.steps());

    let last_step = &raw_trace.rows[raw_trace.steps() - 1];
    let mut public_input = PublicInputs {
        pc_init: FE::from(raw_trace.rows[0].pc),
        ap_init: FE::from(raw_trace.rows[0].ap),
        fp_init: FE::from(raw_trace.rows[0].fp),
        pc_final: FieldElement::from(last_step.pc),
        ap_final: FieldElement::from(last_step.ap),
        range_check_min: None,
        range_check_max: None,
        program,
        num_steps: raw_trace.steps(),
    };

    let result = prove(&(raw_trace, memory), &cairo_air, &mut public_input).unwrap();
    assert!(verify(&result, &cairo_air, &public_input));
}

#[test_log::test]
fn test_prove_cairo_fibonacci_5() {
    test_prove_cairo_fibonacci("fibonacci_5", 64);
}

#[test_log::test]
fn test_prove_cairo_fibonacci_10() {
    test_prove_cairo_fibonacci("fibonacci_10", 128);
}

#[test_log::test]
fn test_prove_cairo_fibonacci_30() {
    test_prove_cairo_fibonacci("fibonacci_30", 256);
}

#[test_log::test]
fn test_prove_cairo_fibonacci_50() {
    test_prove_cairo_fibonacci("fibonacci_50", 512);
}

#[test_log::test]
fn test_prove_cairo_fibonacci_100() {
    test_prove_cairo_fibonacci("fibonacci_100", 1024);
}

#[test_log::test]
fn test_prove_rap_fib() {
    let trace_length = 16;
    let trace = fibonacci_rap_trace([FE::from(1), FE::from(1)], trace_length);
    let mut trace_cols = vec![trace[0].clone(), trace[1].clone()];
    resize_to_next_power_of_two(&mut trace_cols);
    let power_of_two_len = trace_cols[0].len();
    let exemptions = 3 + power_of_two_len - trace_length - 1;

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_columns: 3,
        trace_length: trace_cols[0].len(),
        transition_degrees: vec![1, 2],
        transition_offsets: vec![0, 1, 2],
        transition_exemptions: vec![exemptions, 1],
        num_transition_constraints: 2,
    };

    let fibonacci_rap = FibonacciRAP::new(context);

    let result = prove(&trace_cols, &fibonacci_rap, &mut ()).unwrap();
    assert!(verify(&result, &fibonacci_rap, &()));
}

#[test_log::test]
fn test_prove_dummy() {
    let trace_length = 16;
    let trace = dummy_air::dummy_trace(trace_length);

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_length,
        trace_columns: 2,
        transition_degrees: vec![2, 1],
        transition_exemptions: vec![0, 2],
        transition_offsets: vec![0, 1, 2],
        num_transition_constraints: 2,
    };

    let dummy_air = dummy_air::DummyAIR::from(context);

    let result = prove(&trace, &dummy_air, &mut ()).unwrap();
    assert!(verify(&result, &dummy_air, &()));
}

#[test_log::test]
fn test_verifier_rejects_proof_of_a_slightly_different_program() {
    // The prover generates a proof for a program that
    // is different from the one that the verifier
    // expects.
    let (program_1_raw_trace, program_1_memory) = load_cairo_trace_and_memory("simple_program");
    let proof_options = ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 1,
        coset_offset: 3,
    };

    let program_size = 5;
    let mut program_1 = vec![];
    for i in 1..=program_size as u64 {
        program_1.push(program_1_memory.get(&i).unwrap().clone());
    }

    let mut program_2 = program_1.clone();
    program_2[1] = FieldElement::from(5);
    program_2[3] = FieldElement::from(5);

    let cairo_air = cairo::CairoAIR::new(proof_options, 16, program_1_raw_trace.steps());

    let first_step = &program_1_raw_trace.rows[0];
    let last_step = &program_1_raw_trace.rows[program_1_raw_trace.steps() - 1];

    let mut public_input = PublicInputs {
        pc_init: FE::from(first_step.pc),
        ap_init: FE::from(first_step.ap),
        fp_init: FE::from(first_step.fp),
        pc_final: FE::from(last_step.pc),
        ap_final: FE::from(last_step.ap),
        program: program_1,
        range_check_min: None,
        range_check_max: None,
        num_steps: program_1_raw_trace.steps(),
    };

    let result = prove(
        &(program_1_raw_trace, program_1_memory),
        &cairo_air,
        &mut public_input,
    )
    .unwrap();

    // Here we change program 1 to program 2 in the public inputs.
    public_input.program = program_2;
    assert!(!verify(&result, &cairo_air, &public_input));
}

#[test_log::test]
fn test_verifier_rejects_proof_with_different_range_bounds() {
    // The verifier should reject when the range checks bounds
    // are different from those of the executed program.
    let (raw_trace, memory) = load_cairo_trace_and_memory("simple_program");

    let proof_options = ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 1,
        coset_offset: 3,
    };

    let program_size = 5;
    let mut program = vec![];
    for i in 1..=program_size as u64 {
        program.push(memory.get(&i).unwrap().clone());
    }

    let cairo_air = cairo::CairoAIR::new(proof_options, 16, raw_trace.steps());

    let first_step = &raw_trace.rows[0];
    let last_step = &raw_trace.rows[raw_trace.steps() - 1];

    let mut public_input = PublicInputs {
        pc_init: FE::from(first_step.pc),
        ap_init: FE::from(first_step.ap),
        fp_init: FE::from(first_step.fp),
        pc_final: FE::from(last_step.pc),
        ap_final: FE::from(last_step.ap),
        program: program,
        range_check_min: None,
        range_check_max: None,
        num_steps: raw_trace.steps(),
    };

    let result = prove(&(raw_trace, memory), &cairo_air, &mut public_input).unwrap();

    public_input.range_check_min = Some(public_input.range_check_min.unwrap() + 1);
    assert!(!verify(&result, &cairo_air, &public_input));

    public_input.range_check_min = Some(public_input.range_check_min.unwrap() - 1);
    public_input.range_check_max = Some(public_input.range_check_max.unwrap() - 1);
    assert!(!verify(&result, &cairo_air, &public_input));
}
