use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

use crate::{
    examples::{
        dummy_air::{self, DummyAIR},
        fibonacci_2_cols_shifted::{self, Fibonacci2ColsShifted},
        fibonacci_2_columns::{self, Fibonacci2ColsAIR},
        fibonacci_rap::{fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs},
        quadratic_air::{self, QuadraticAIR, QuadraticPublicInputs},
        simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs},
    },
    proof::options::ProofOptions,
    prover::prove,
    transcript::StoneProverTranscript,
    verifier::verify,
    Felt252,
};

#[test_log::test]
fn test_prove_fib() {
    let trace = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 8);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let proof = prove::<Stark252PrimeField, FibonacciAIR<Stark252PrimeField>>(
        &trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(
        verify::<Stark252PrimeField, FibonacciAIR<Stark252PrimeField>>(
            &proof,
            &pub_inputs,
            &proof_options,
            StoneProverTranscript::new(&[]),
        )
    );
}

#[test_log::test]
fn test_prove_fib17() {
    type FE = FieldElement<Stark252PrimeField>;
    let trace = simple_fibonacci::fibonacci_trace([FE::from(1), FE::from(1)], 4);

    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 7,
        coset_offset: 3,
        grinding_factor: 1,
    };

    let pub_inputs = FibonacciPublicInputs {
        a0: FE::one(),
        a1: FE::one(),
    };

    let proof = prove::<_, FibonacciAIR<_>>(
        &trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(verify::<_, FibonacciAIR<_>>(
        &proof,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_fib_2_cols() {
    let trace = fibonacci_2_columns::compute_trace([Felt252::from(1), Felt252::from(1)], 16);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let proof = prove::<Stark252PrimeField, Fibonacci2ColsAIR<Stark252PrimeField>>(
        &trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(verify::<
        Stark252PrimeField,
        Fibonacci2ColsAIR<Stark252PrimeField>,
    >(
        &proof,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_fib_2_cols_shifted() {
    let trace = fibonacci_2_cols_shifted::compute_trace(FieldElement::one(), 16);

    let claimed_index = 14;
    let claimed_value = trace.get_row(claimed_index)[0];
    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = fibonacci_2_cols_shifted::PublicInputs {
        claimed_value,
        claimed_index,
    };

    let proof = prove::<Stark252PrimeField, Fibonacci2ColsShifted<_>>(
        &trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(verify::<Stark252PrimeField, Fibonacci2ColsShifted<_>>(
        &proof,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_quadratic() {
    let trace = quadratic_air::quadratic_trace(Felt252::from(3), 4);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = QuadraticPublicInputs {
        a0: Felt252::from(3),
    };

    let proof = prove::<Stark252PrimeField, QuadraticAIR<Stark252PrimeField>>(
        &trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(
        verify::<Stark252PrimeField, QuadraticAIR<Stark252PrimeField>>(
            &proof,
            &pub_inputs,
            &proof_options,
            StoneProverTranscript::new(&[])
        )
    );
}

#[test_log::test]
fn test_prove_rap_fib() {
    let steps = 16;
    let trace = fibonacci_rap_trace([Felt252::from(1), Felt252::from(1)], steps);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciRAPPublicInputs {
        steps,
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let proof = prove::<Stark252PrimeField, FibonacciRAP<Stark252PrimeField>>(
        &trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(
        verify::<Stark252PrimeField, FibonacciRAP<Stark252PrimeField>>(
            &proof,
            &pub_inputs,
            &proof_options,
            StoneProverTranscript::new(&[])
        )
    );
}

#[test_log::test]
fn test_prove_dummy() {
    let trace_length = 16;
    let trace = dummy_air::dummy_trace(trace_length);

    let proof_options = ProofOptions::default_test_options();

    let proof = prove::<Stark252PrimeField, DummyAIR>(
        &trace,
        &(),
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(verify::<Stark252PrimeField, DummyAIR>(
        &proof,
        &(),
        &proof_options,
        StoneProverTranscript::new(&[])
    ));
}
