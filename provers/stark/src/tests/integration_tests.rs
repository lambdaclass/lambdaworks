use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

use crate::{
    examples::{
        bit_flags::{self, BitFlagsAIR},
        dummy_air::{self, DummyAIR},
        fibonacci_2_cols_shifted::{self, Fibonacci2ColsShifted},
        fibonacci_2_columns::{self, Fibonacci2ColsAIR},
        fibonacci_rap::{fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs},
        quadratic_air::{self, QuadraticAIR, QuadraticPublicInputs},
        simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs},
        simple_periodic_cols::{self, SimplePeriodicAIR, SimplePeriodicPublicInputs},
    },
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    transcript::StoneProverTranscript,
    verifier::{IsStarkVerifier, Verifier},
    Felt252,
};

#[test_log::test]
fn test_prove_fib() {
    let trace = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 1024);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let proof = Prover::<FibonacciAIR<Stark252PrimeField>>::prove(
        &trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(Verifier::<FibonacciAIR<Stark252PrimeField>>::verify(
        &proof,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    ));
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

    let proof = Prover::<FibonacciAIR<_>>::prove(
        &trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(Verifier::<FibonacciAIR<_>>::verify(
        &proof,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_simple_periodic_8() {
    let trace = simple_periodic_cols::simple_periodic_trace::<Stark252PrimeField>(8);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = SimplePeriodicPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::from(8),
    };

    let proof = Prover::<SimplePeriodicAIR<Stark252PrimeField>>::prove(
        &trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(Verifier::<SimplePeriodicAIR<Stark252PrimeField>>::verify(
        &proof,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_simple_periodic_32() {
    let trace = simple_periodic_cols::simple_periodic_trace::<Stark252PrimeField>(32);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = SimplePeriodicPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::from(32768),
    };

    let proof = Prover::<SimplePeriodicAIR<Stark252PrimeField>>::prove(
        &trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(Verifier::<SimplePeriodicAIR<Stark252PrimeField>>::verify(
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

    let proof = Prover::<Fibonacci2ColsAIR<Stark252PrimeField>>::prove(
        &trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();

    assert!(Verifier::<Fibonacci2ColsAIR<Stark252PrimeField>>::verify(
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

    let proof = Prover::<Fibonacci2ColsShifted<_>>::prove(
        &trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(Verifier::<Fibonacci2ColsShifted<_>>::verify(
        &proof,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_quadratic() {
    let trace = quadratic_air::quadratic_trace(Felt252::from(3), 32);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = QuadraticPublicInputs {
        a0: Felt252::from(3),
    };

    let proof = Prover::<QuadraticAIR<Stark252PrimeField>>::prove(
        &trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(Verifier::<QuadraticAIR<Stark252PrimeField>>::verify(
        &proof,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[])
    ));
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

    let proof = Prover::<FibonacciRAP<Stark252PrimeField>>::prove(
        &trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(Verifier::<FibonacciRAP<Stark252PrimeField>>::verify(
        &proof,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_dummy() {
    let trace_length = 16;
    let trace = dummy_air::dummy_trace(trace_length);

    let proof_options = ProofOptions::default_test_options();

    let proof =
        Prover::<DummyAIR>::prove(&trace, &(), &proof_options, StoneProverTranscript::new(&[]))
            .unwrap();

    assert!(Verifier::<DummyAIR>::verify(
        &proof,
        &(),
        &proof_options,
        StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_bit_flags() {
    let trace = bit_flags::bit_prefix_flag_trace(32);
    let proof_options = ProofOptions::default_test_options();

    let proof =
        Prover::<BitFlagsAIR>::prove(&trace, &(), &proof_options, StoneProverTranscript::new(&[]))
            .unwrap();

    assert!(Verifier::<BitFlagsAIR>::verify(
        &proof,
        &(),
        &proof_options,
        StoneProverTranscript::new(&[]),
    ));
}
