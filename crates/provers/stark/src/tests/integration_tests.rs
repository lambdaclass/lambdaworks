use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

use crate::examples::add::AddAir;
use crate::examples::cpu::CPUAir;
use crate::examples::mul::MulAir;
use crate::traits::AIR;
use crate::{
    examples::{
        bit_flags::{self, BitFlagsAIR},
        dummy_air::{self, DummyAIR},
        fibonacci_2_cols_shifted::{self, Fibonacci2ColsShifted},
        fibonacci_2_columns::{self, Fibonacci2ColsAIR},
        fibonacci_rap::{fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs},
        quadratic_air::{self, QuadraticAIR, QuadraticPublicInputs},
        read_only_memory::{sort_rap_trace, ReadOnlyPublicInputs, ReadOnlyRAP},
        simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs},
        simple_periodic_cols::{self, SimplePeriodicAIR, SimplePeriodicPublicInputs}, //         simple_periodic_cols::{self, SimplePeriodicAIR, SimplePeriodicPublicInputs},
    },
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    transcript::StoneProverTranscript,
    verifier::{IsStarkVerifier, Verifier},
    Felt252,
};
use lambdaworks_math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};

use crate::examples::read_only_memory_logup::{
    read_only_logup_trace, LogReadOnlyPublicInputs, LogReadOnlyRAP,
};

#[test_log::test]
fn test_prove_fib() {
    let mut trace = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 8);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let air =
        FibonacciAIR::<Stark252PrimeField>::new(trace.num_rows(), &pub_inputs, &proof_options);

    let proof =
        Prover::prove_single(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();
    assert!(Verifier::verify_single(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_simple_periodic_8() {
    let mut trace = simple_periodic_cols::simple_periodic_trace::<Stark252PrimeField>(8);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = SimplePeriodicPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::from(8),
    };

    let air =
        SimplePeriodicAIR::<Stark252PrimeField>::new(trace.num_rows(), &pub_inputs, &proof_options);

    let proof =
        Prover::prove_single(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();
    assert!(Verifier::verify_single(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_simple_periodic_32() {
    let mut trace = simple_periodic_cols::simple_periodic_trace::<Stark252PrimeField>(32);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = SimplePeriodicPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::from(32768),
    };

    let air =
        SimplePeriodicAIR::<Stark252PrimeField>::new(trace.num_rows(), &pub_inputs, &proof_options);

    let proof =
        Prover::prove_single(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify_single(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_fib_2_cols() {
    let mut trace = fibonacci_2_columns::compute_trace([Felt252::from(1), Felt252::from(1)], 16);

    let proof_options = ProofOptions::default_test_options();
    let pub_inputs = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let air =
        Fibonacci2ColsAIR::<Stark252PrimeField>::new(trace.num_rows(), &pub_inputs, &proof_options);

    let proof =
        Prover::prove_single(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify_single(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_fib_2_cols_shifted() {
    let mut trace = fibonacci_2_cols_shifted::compute_trace(FieldElement::one(), 16);

    let claimed_index = 14;
    let claimed_value = trace.main_table.get_row(claimed_index)[0];
    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = fibonacci_2_cols_shifted::PublicInputs {
        claimed_value,
        claimed_index,
    };

    let air = Fibonacci2ColsShifted::<Stark252PrimeField>::new(
        trace.num_rows(),
        &pub_inputs,
        &proof_options,
    );

    let proof =
        Prover::prove_single(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify_single(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_quadratic() {
    let mut trace = quadratic_air::quadratic_trace(Felt252::from(3), 32);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = QuadraticPublicInputs {
        a0: Felt252::from(3),
    };

    let air =
        QuadraticAIR::<Stark252PrimeField>::new(trace.num_rows(), &pub_inputs, &proof_options);

    let proof =
        Prover::prove_single(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify_single(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_rap_fib() {
    let steps = 16;
    let mut trace = fibonacci_rap_trace([Felt252::from(1), Felt252::from(1)], steps);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciRAPPublicInputs {
        steps,
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let air =
        FibonacciRAP::<Stark252PrimeField>::new(trace.num_rows(), &pub_inputs, &proof_options);

    let proof =
        Prover::prove_single(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify_single(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_dummy() {
    let trace_length = 16;
    let mut trace = dummy_air::dummy_trace(trace_length);

    let proof_options = ProofOptions::default_test_options();

    let air = DummyAIR::new(trace.num_rows(), &(), &proof_options);

    let proof =
        Prover::prove_single(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify_single(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_bit_flags() {
    let mut trace = bit_flags::bit_prefix_flag_trace(32);
    let proof_options = ProofOptions::default_test_options();

    let air = BitFlagsAIR::new(trace.num_rows(), &(), &proof_options);

    let proof =
        Prover::prove_single(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify_single(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[]),
    ));
}

#[test_log::test]
fn test_prove_read_only_memory() {
    let address_col = vec![
        FieldElement::<Stark252PrimeField>::from(3), // a0
        FieldElement::<Stark252PrimeField>::from(2), // a1
        FieldElement::<Stark252PrimeField>::from(2), // a2
        FieldElement::<Stark252PrimeField>::from(3), // a3
        FieldElement::<Stark252PrimeField>::from(4), // a4
        FieldElement::<Stark252PrimeField>::from(5), // a5
        FieldElement::<Stark252PrimeField>::from(1), // a6
        FieldElement::<Stark252PrimeField>::from(3), // a7
    ];
    let value_col = vec![
        FieldElement::<Stark252PrimeField>::from(10), // v0
        FieldElement::<Stark252PrimeField>::from(5),  // v1
        FieldElement::<Stark252PrimeField>::from(5),  // v2
        FieldElement::<Stark252PrimeField>::from(10), // v3
        FieldElement::<Stark252PrimeField>::from(25), // v4
        FieldElement::<Stark252PrimeField>::from(25), // v5
        FieldElement::<Stark252PrimeField>::from(7),  // v6
        FieldElement::<Stark252PrimeField>::from(10), // v7
    ];

    let pub_inputs = ReadOnlyPublicInputs {
        a0: FieldElement::<Stark252PrimeField>::from(3),
        v0: FieldElement::<Stark252PrimeField>::from(10),
        a_sorted0: FieldElement::<Stark252PrimeField>::from(1), // a6
        v_sorted0: FieldElement::<Stark252PrimeField>::from(7), // v6
    };
    let mut trace = sort_rap_trace(address_col, value_col);
    let proof_options = ProofOptions::default_test_options();

    let air = ReadOnlyRAP::<Stark252PrimeField>::new(trace.num_rows(), &pub_inputs, &proof_options);

    let proof =
        Prover::prove_single(&air, &mut trace, &mut StoneProverTranscript::new(&[])).unwrap();

    assert!(Verifier::verify_single(
        &proof,
        &air,
        &mut StoneProverTranscript::new(&[])
    ));
}

#[test_log::test]
fn test_prove_log_read_only_memory() {
    let address_col = vec![
        FieldElement::<Babybear31PrimeField>::from(3), // a0
        FieldElement::<Babybear31PrimeField>::from(2), // a1
        FieldElement::<Babybear31PrimeField>::from(2), // a2
        FieldElement::<Babybear31PrimeField>::from(3), // a3
        FieldElement::<Babybear31PrimeField>::from(4), // a4
        FieldElement::<Babybear31PrimeField>::from(5), // a5
        FieldElement::<Babybear31PrimeField>::from(1), // a6
        FieldElement::<Babybear31PrimeField>::from(3), // a7
    ];
    let value_col = vec![
        FieldElement::<Babybear31PrimeField>::from(30), // v0
        FieldElement::<Babybear31PrimeField>::from(20), // v1
        FieldElement::<Babybear31PrimeField>::from(20), // v2
        FieldElement::<Babybear31PrimeField>::from(30), // v3
        FieldElement::<Babybear31PrimeField>::from(40), // v4
        FieldElement::<Babybear31PrimeField>::from(50), // v5
        FieldElement::<Babybear31PrimeField>::from(10), // v6
        FieldElement::<Babybear31PrimeField>::from(30), // v7
    ];

    let pub_inputs = LogReadOnlyPublicInputs {
        a0: FieldElement::<Babybear31PrimeField>::from(3),
        v0: FieldElement::<Babybear31PrimeField>::from(30),
        a_sorted_0: FieldElement::<Babybear31PrimeField>::from(1),
        v_sorted_0: FieldElement::<Babybear31PrimeField>::from(10),
        m0: FieldElement::<Babybear31PrimeField>::from(1),
    };
    let mut trace = read_only_logup_trace(address_col, value_col);
    let proof_options = ProofOptions::default_test_options();

    let air = LogReadOnlyRAP::<Babybear31PrimeField, Degree4BabyBearExtensionField>::new(
        trace.num_rows(),
        &pub_inputs,
        &proof_options,
    );

    let proof = Prover::prove_single(
        &air,
        &mut trace,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    )
    .unwrap();

    assert!(Verifier::verify_single(
        &proof,
        &air,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    ));
}

#[test_log::test]
fn test_multi_prove_fib_3_tables() {
    let mut trace_1 = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 8);
    let mut trace_2 = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 16);
    let mut trace_3 = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 32);
    let proof_options = ProofOptions::default_test_options();

    let pub_inputs_1 = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };
    let pub_inputs_2 = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };
    let pub_inputs_3 = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let air_1 = FibonacciAIR::new(8, &pub_inputs_1, &proof_options);
    let air_2 = FibonacciAIR::new(16, &pub_inputs_2, &proof_options);
    let air_3 = FibonacciAIR::new(32, &pub_inputs_3, &proof_options);

    let mut airs: Vec<(
        &dyn AIR<
            Field = Stark252PrimeField,
            FieldExtension = Stark252PrimeField,
            PublicInputs = FibonacciPublicInputs<Stark252PrimeField>,
        >,
        &mut _,
    )> = vec![
        (&air_1, &mut trace_1),
        (&air_2, &mut trace_2),
        (&air_3, &mut trace_3),
    ];
    let proofs = Prover::prove(&mut airs, &mut StoneProverTranscript::new(&[])).unwrap();

    let airs_and_proofs: Vec<(
        &dyn AIR<
            Field = Stark252PrimeField,
            FieldExtension = Stark252PrimeField,
            PublicInputs = FibonacciPublicInputs<Stark252PrimeField>,
        >,
        &_,
    )> = vec![
        (&air_1, &proofs[0]),
        (&air_2, &proofs[1]),
        (&air_3, &proofs[2]),
    ];

    assert!(Verifier::verify(
        &airs_and_proofs,
        &mut StoneProverTranscript::new(&[]),
    ));
}

#[test_log::test]
fn test_multi_prove_2_tables_small_field() {
    let address_col_1 = vec![
        FieldElement::<Babybear31PrimeField>::from(3), // a0
        FieldElement::<Babybear31PrimeField>::from(2), // a1
        FieldElement::<Babybear31PrimeField>::from(2), // a2
        FieldElement::<Babybear31PrimeField>::from(3), // a3
        FieldElement::<Babybear31PrimeField>::from(4), // a4
        FieldElement::<Babybear31PrimeField>::from(5), // a5
        FieldElement::<Babybear31PrimeField>::from(1), // a6
        FieldElement::<Babybear31PrimeField>::from(3), // a7
    ];
    let value_col_1 = vec![
        FieldElement::<Babybear31PrimeField>::from(30), // v0
        FieldElement::<Babybear31PrimeField>::from(20), // v1
        FieldElement::<Babybear31PrimeField>::from(20), // v2
        FieldElement::<Babybear31PrimeField>::from(30), // v3
        FieldElement::<Babybear31PrimeField>::from(40), // v4
        FieldElement::<Babybear31PrimeField>::from(50), // v5
        FieldElement::<Babybear31PrimeField>::from(10), // v6
        FieldElement::<Babybear31PrimeField>::from(30), // v7
    ];

    let address_col_2 = vec![
        FieldElement::<Babybear31PrimeField>::from(15), // a0
        FieldElement::<Babybear31PrimeField>::from(12), // a1
        FieldElement::<Babybear31PrimeField>::from(17), // a2
        FieldElement::<Babybear31PrimeField>::from(10), // a3
        FieldElement::<Babybear31PrimeField>::from(14), // a4
        FieldElement::<Babybear31PrimeField>::from(11), // a5
        FieldElement::<Babybear31PrimeField>::from(16), // a6
        FieldElement::<Babybear31PrimeField>::from(13), // a7
    ];
    let value_col_2 = vec![
        FieldElement::<Babybear31PrimeField>::from(150), // v0
        FieldElement::<Babybear31PrimeField>::from(120), // v1
        FieldElement::<Babybear31PrimeField>::from(170), // v2
        FieldElement::<Babybear31PrimeField>::from(100), // v3
        FieldElement::<Babybear31PrimeField>::from(140), // v4
        FieldElement::<Babybear31PrimeField>::from(110), // v5
        FieldElement::<Babybear31PrimeField>::from(160), // v6
        FieldElement::<Babybear31PrimeField>::from(130), // v7
    ];

    let pub_inputs_1 = LogReadOnlyPublicInputs {
        a0: FieldElement::<Babybear31PrimeField>::from(3),
        v0: FieldElement::<Babybear31PrimeField>::from(30),
        a_sorted_0: FieldElement::<Babybear31PrimeField>::from(1),
        v_sorted_0: FieldElement::<Babybear31PrimeField>::from(10),
        m0: FieldElement::<Babybear31PrimeField>::from(1),
    };

    let pub_inputs_2 = LogReadOnlyPublicInputs {
        a0: FieldElement::<Babybear31PrimeField>::from(15),
        v0: FieldElement::<Babybear31PrimeField>::from(150),
        a_sorted_0: FieldElement::<Babybear31PrimeField>::from(10),
        v_sorted_0: FieldElement::<Babybear31PrimeField>::from(100),
        m0: FieldElement::<Babybear31PrimeField>::from(1),
    };

    let mut trace_1 = read_only_logup_trace(address_col_1, value_col_1);
    let mut trace_2 = read_only_logup_trace(address_col_2, value_col_2);
    let proof_options = ProofOptions::default_test_options();

    let air_1 = LogReadOnlyRAP::<Babybear31PrimeField, Degree4BabyBearExtensionField>::new(
        trace_1.num_rows(),
        &pub_inputs_1,
        &proof_options,
    );
    let air_2 = LogReadOnlyRAP::<Babybear31PrimeField, Degree4BabyBearExtensionField>::new(
        trace_2.num_rows(),
        &pub_inputs_2,
        &proof_options,
    );

    let mut airs: Vec<(
        &dyn AIR<
            Field = Babybear31PrimeField,
            FieldExtension = Degree4BabyBearExtensionField,
            PublicInputs = LogReadOnlyPublicInputs<Babybear31PrimeField>,
        >,
        &mut _,
    )> = vec![(&air_1, &mut trace_1), (&air_2, &mut trace_2)];

    let proofs = Prover::prove(
        &mut airs,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    )
    .unwrap();

    let airs_and_proofs: Vec<(
        &dyn AIR<
            Field = Babybear31PrimeField,
            FieldExtension = Degree4BabyBearExtensionField,
            PublicInputs = LogReadOnlyPublicInputs<Babybear31PrimeField>,
        >,
        &_,
    )> = vec![(&air_1, &proofs[0]), (&air_2, &proofs[1])];

    assert!(Verifier::verify(
        &airs_and_proofs,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    ));
}

#[test_log::test]
fn test_multi_prove_different_airs() {
    let mut trace_1 = dummy_air::dummy_trace(16);
    let mut trace_2 = bit_flags::bit_prefix_flag_trace(32);
    let proof_options = ProofOptions::default_test_options();

    let air_1 = DummyAIR::new(trace_1.num_rows(), &(), &proof_options);
    let air_2 = BitFlagsAIR::new(trace_2.num_rows(), &(), &proof_options);

    let mut airs: Vec<(
        &dyn AIR<
            Field = Stark252PrimeField,
            FieldExtension = Stark252PrimeField,
            PublicInputs = (),
        >,
        &mut _,
    )> = vec![(&air_1, &mut trace_1), (&air_2, &mut trace_2)];

    let proofs = Prover::prove(&mut airs, &mut StoneProverTranscript::new(&[])).unwrap();

    let airs_and_proofs: Vec<(
        &dyn AIR<
            Field = Stark252PrimeField,
            FieldExtension = Stark252PrimeField,
            PublicInputs = (),
        >,
        &_,
    )> = vec![(&air_1, &proofs[0]), (&air_2, &proofs[1])];

    assert!(Verifier::verify(
        &airs_and_proofs,
        &mut StoneProverTranscript::new(&[]),
    ));
}
use crate::trace::TraceTable;

type FE = FieldElement<Babybear31PrimeField>;
type ExtFE = FieldElement<Degree4BabyBearExtensionField>;

#[test_log::test]
fn test_multi_airs_log_up() {
    // CPU Trace
    // ADD | MUL | a | b  | c   | aux add | aux mul | aux total ?
    // 1   | 0   | 1 | 10 | 11  | 0       | 0       | 0
    // 0   | 1   | 2 | 20 | 40  | 0       | 0       | 0
    // 1   | 0   | 3 | 30 | 33  | 0       | 0       | 0
    // 0   | 1   | 4 | 40 | 160 | 0       | 0       | 0
    // 1   | 0   | 5 | 50 | 55  | 0       | 0       | 0
    // 1   | 0   | 6 | 60 | 66  | 0       | 0       | 0
    // 0   | 1   | 7 | 70 | 490 | 0       | 0       | 0
    // 0   | 1   | 8 | 80 | 640 | 0       | 0       | 0
    let add_column = vec![
        FE::one(),
        FE::zero(),
        FE::one(),
        FE::zero(),
        FE::one(),
        FE::one(),
        FE::zero(),
        FE::zero(),
    ];
    let mul_column = vec![
        FE::zero(),
        FE::one(),
        FE::zero(),
        FE::one(),
        FE::zero(),
        FE::zero(),
        FE::one(),
        FE::one(),
    ];
    let a_column = vec![
        FE::from(1),
        FE::from(2),
        FE::from(3),
        FE::from(4),
        FE::from(5),
        FE::from(6),
        FE::from(7),
        FE::from(8),
    ];
    let b_column = vec![
        FE::from(10),
        FE::from(20),
        FE::from(30),
        FE::from(40),
        FE::from(50),
        FE::from(60),
        FE::from(70),
        FE::from(80),
    ];
    let c_column = vec![
        FE::from(11),
        FE::from(40),
        FE::from(33),
        FE::from(160),
        FE::from(55),
        FE::from(66),
        FE::from(490),
        FE::from(640),
    ];
    let main_columns = vec![add_column, mul_column, a_column, b_column, c_column];
    let aux_columns = vec![
        vec![ExtFE::zero(); 8],
        vec![ExtFE::zero(); 8],
        vec![ExtFE::zero(); 8],
    ];
    let mut cpu_trace = TraceTable::from_columns(main_columns, aux_columns, 1);

    // ADD Trace
    // a | b  | c  | aux cpu | aux total
    // 1 | 10 | 11 | 0       | 0
    // 3 | 30 | 33 | 0       | 0
    // 5 | 50 | 55 | 0       | 0
    // 6 | 60 | 66 | 0       | 0
    let a_column = vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)];
    let b_column = vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)];
    let c_column = vec![FE::from(11), FE::from(33), FE::from(55), FE::from(66)];
    let mut add_trace = TraceTable::from_columns(
        vec![a_column, b_column, c_column],
        vec![vec![ExtFE::zero(); 4], vec![ExtFE::zero(); 4]],
        1,
    );

    // MUL Trace
    // a | b  | c   | aux cpu | aux total
    // 2 | 20 | 40  | 0       | 0
    // 4 | 40 | 160 | 0       | 0
    // 7 | 70 | 490 | 0       | 0
    // 8 | 80 | 640 | 0       | 0
    let a_column = vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)];
    let b_column = vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)];
    let c_column = vec![FE::from(40), FE::from(160), FE::from(490), FE::from(640)];
    let mut mul_trace = TraceTable::from_columns(
        vec![a_column, b_column, c_column],
        vec![vec![ExtFE::zero(); 4], vec![ExtFE::zero(); 4]],
        1,
    );

    let proof_options = ProofOptions::default_test_options();

    let public_inputs: Vec<Babybear31PrimeField> = Vec::new();

    let cpu_air = CPUAir::new(cpu_trace.num_rows(), &public_inputs, &proof_options);
    let add_air = AddAir::new(add_trace.num_rows(), &public_inputs, &proof_options);
    let mul_air = MulAir::new(mul_trace.num_rows(), &public_inputs, &proof_options);

    let mut airs: Vec<(
        &dyn AIR<
            Field = Babybear31PrimeField,
            FieldExtension = Degree4BabyBearExtensionField,
            PublicInputs = Vec<Babybear31PrimeField>,
        >,
        &mut TraceTable<Babybear31PrimeField, Degree4BabyBearExtensionField>,
    )> = vec![
        (&cpu_air, &mut cpu_trace),
        (&add_air, &mut add_trace),
        (&mul_air, &mut mul_trace),
    ];

    let proofs = Prover::prove(
        &mut airs,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    )
    .unwrap();

    // TODO: This should be done by the prover.
    let cpu_look_up_value = cpu_trace.aux_table.get(7, 2);
    let add_look_up_value = cpu_trace.aux_table.get(3, 1);
    let mul_look_up_value = cpu_trace.aux_table.get(3, 1);

    let look_up_values: Vec<&FieldElement<Degree4BabyBearExtensionField>> =
        vec![cpu_look_up_value, add_look_up_value, mul_look_up_value];

    let airs_and_proofs: Vec<(
        &dyn AIR<
            Field = Babybear31PrimeField,
            FieldExtension = Degree4BabyBearExtensionField,
            PublicInputs = Vec<Babybear31PrimeField>,
        >,
        &_,
    )> = vec![
        (&cpu_air, &proofs[0]),
        (&add_air, &proofs[1]),
        (&mul_air, &proofs[2]),
    ];

    assert!(Verifier::look_up_verify(
        &look_up_values,
        &airs_and_proofs,
        &mut DefaultTranscript::<Degree4BabyBearExtensionField>::new(&[]),
    ));
}
