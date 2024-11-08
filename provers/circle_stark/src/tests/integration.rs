use lambdaworks_math::field::{
    element::FieldElement,
    fields::{
        fft_friendly::stark_252_prime_field::Stark252PrimeField, mersenne31::field::Mersenne31Field,
    },
};

use crate::{
    examples::simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs},
    prover::{IsStarkProver, Prover},
    // proof::options::ProofOptions,
    // prover::{IsStarkProver, Prover},
};

#[test_log::test]
fn test_prove_fib() {
    type FE = FieldElement<Mersenne31Field>;

    let trace = simple_fibonacci::fibonacci_trace([FE::one(), FE::one()], 16);

    let pub_inputs = FibonacciPublicInputs {
        a0: FE::one(),
        a1: FE::one(),
    };

    let proof = Prover::<FibonacciAIR>::prove(&trace, &pub_inputs);
    //     .unwrap();
    //     assert!(Verifier::<FibonacciAIR<Stark252PrimeField>>::verify(
    //         &proof,
    //         &pub_inputs,
    //         &proof_options,
    //         StoneProverTranscript::new(&[]),
    //     ));
}
