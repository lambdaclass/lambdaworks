use crate::{
    examples::cpu_table::{self, CPUTableAIR},
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    verifier::{IsStarkVerifier, Verifier},
};
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::{
        babybear_u32::Babybear31PrimeField, quartic_babybear_u32::Degree4BabyBearU32ExtensionField,
    },
};
type FE = FieldElement<Babybear31PrimeField>;

pub fn build_cpu_columns_example() -> Vec<Vec<FE>> {
    let mut columns = Vec::new();
    // Timestamp: A word2L column containing the values 4 * i for i = 1,...
    // Column index: 0
    let timestamp_1 = vec![FE::zero(); 4];
    // Column index: 1
    let timestamp_2 = vec![
        FE::from(&4u32),
        FE::from(&8u32),
        FE::from(&12u32),
        FE::from(&16u32),
    ];
    columns.push(timestamp_1);
    columns.push(timestamp_2);

    // ----- 30 uncompressed decode columns -----

    // Word2L pc.
    // Column index: 2
    let pc_1 = vec![FE::zero(); 4];
    // Column index: 3
    let pc_2 = vec![
        FE::from(&4u32),
        FE::from(&8u32),
        FE::from(&12u32),
        FE::from(&16u32),
    ];
    columns.push(pc_1);
    columns.push(pc_2);

    // Index of source register 1.
    // Column index: 4
    let rs_1 = vec![FE::from(&1), FE::from(&2), FE::from(&3), FE::from(&4)];
    columns.push(rs_1);

    // Index of source register 2.
    // Column index: 5
    let rs_2 = vec![FE::from(&5), FE::from(&6), FE::from(&7), FE::from(&8)];
    columns.push(rs_2);

    // Index of destination register.
    // Column index: 6
    let rd = vec![FE::from(&9), FE::from(&10), FE::from(&11), FE::from(&12)];
    columns.push(rd);

    // Should the result be written.
    // Flag.
    // Column index: 7
    let write_register = vec![FE::one(), FE::zero(), FE::one(), FE::zero()];
    columns.push(write_register);

    // Does the memory access (read or write) touch at least 2 bytes.
    // Flag.
    // Column index: 8
    let memory_2_bytes = vec![FE::one(), FE::zero(), FE::one(), FE::zero()];
    columns.push(memory_2_bytes);

    // Does the memory access (read or write) touch 4 bytes.
    // Flag.
    // Column index: 9
    let memory_4_bytes = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(memory_4_bytes);

    // check this
    // Column index: 10
    let imm_1 = vec![FE::zero(); 4];
    // Column index: 11
    let imm_2 = vec![
        FE::from(&0u32),
        FE::from(&1u32),
        FE::from(&2u32),
        FE::from(&3u32),
    ];
    columns.push(imm_1);
    columns.push(imm_2);

    // check this two
    // Flag.
    // Column index: 12
    let signed = vec![FE::one(), FE::zero(), FE::one(), FE::zero()];
    columns.push(signed);

    // Flag.
    // Column index: 13
    let signed_2 = vec![FE::one(), FE::zero(), FE::one(), FE::zero()];
    columns.push(signed_2);

    // Flag that selects output of MUL or DIV.
    // TODO: chequear cuál es cero y cuál es 1 (de MUL y DIV).
    // Column index: 14
    let muldiv_selector = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(muldiv_selector);

    // One-hot 17 columns of flags:
    // Instructions in this example: add, mul, or, and.
    // Column index: 15
    let add = vec![FE::one(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(add);
    // Column index: 16
    let sub = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(sub);
    // Column index: 17
    let slt = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(slt);
    // Column index: 18
    let and = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(and);
    // Column index: 19
    let or = vec![FE::zero(), FE::zero(), FE::one(), FE::zero()];
    columns.push(or);
    // Column index: 20
    let xor = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(xor);
    // Column index: 21
    let sl = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(sl);
    // Column index: 22
    let sr = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(sr);
    // Column index: 23
    let jalr = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(jalr);
    // Column index: 24
    let beq = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(beq);
    // Column index: 25
    let blt = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(blt);
    // Column index: 26
    let load = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(load);
    // Column index: 27
    let store = vec![FE::zero(), FE::zero(), FE::zero(), FE::one()];
    columns.push(store);
    // Column index: 28
    let mul = vec![FE::zero(), FE::one(), FE::zero(), FE::zero()];
    columns.push(mul);
    // Column index: 29
    let divrem = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(divrem);
    // Column index: 30
    let ecall = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(ecall);
    // Column index: 31
    let ebreak = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(ebreak);

    // ------------------------------------

    // Column index: 32
    let next_pc_1 = vec![FE::zero(); 4];
    // Column index: 33
    let next_pc_2 = vec![FE::from(8), FE::from(12), FE::from(16), FE::from(20)];
    columns.push(next_pc_1);
    columns.push(next_pc_2);

    // rv1 Word4L
    // Column index: 34
    let rv1_1 = vec![FE::zero(); 4];
    // Column index: 35
    let rv1_2 = vec![FE::zero(); 4];
    // Column index: 36
    let rv1_3 = vec![FE::zero(); 4];
    // Column index: 37
    let rv1_4 = vec![
        FE::from(&10u32),
        FE::from(&20u32),
        FE::from(&30u32),
        FE::from(&40u32),
    ];
    columns.push(rv1_1);
    columns.push(rv1_2);
    columns.push(rv1_3);
    columns.push(rv1_4);

    // rv2 Word4L
    // Column index: 38
    let rv2_1 = vec![FE::zero(); 4];
    // Column index: 39
    let rv2_2 = vec![FE::zero(); 4];
    // Column index: 40
    let rv2_3 = vec![FE::zero(); 4];
    // Column index: 41
    let rv2_4 = vec![
        FE::from(&10u32),
        FE::from(&20u32),
        FE::from(&30u32),
        FE::from(&40u32),
    ];
    columns.push(rv2_1);
    columns.push(rv2_2);
    columns.push(rv2_3);
    columns.push(rv2_4);

    // rvd Word2L
    // Column index: 42
    let rvd_1 = vec![FE::zero(); 4];
    // Column index: 43
    let rvd_2 = vec![
        FE::from(&15u32),
        FE::from(&35u32),
        FE::from(&55u32),
        FE::from(&75u32),
    ];
    columns.push(rvd_1);
    columns.push(rvd_2);

    // The second argument of the (ALU) operation being performed.
    // Definition: (1 - STORE - LOAD)·rv2 + (1 - BEQ - BLT)·imm
    // Column index: 44
    let arg2_1 = vec![FE::zero(); 4];
    // Column index: 45
    let arg2_2 = vec![FE::zero(); 4];
    // Column index: 46
    let arg2_3 = vec![FE::zero(); 4];
    // Column index: 47
    let arg2_4 = vec![
        FE::from(&10u32),
        FE::from(&20u32),
        FE::from(&30u32),
        FE::from(&4032),
    ];
    columns.push(arg2_1);
    columns.push(arg2_2);
    columns.push(arg2_3);
    columns.push(arg2_4);

    // The word2L ALU result.
    // Column index: 48
    let res_1 = vec![FE::zero(); 4];
    // Column index: 49
    let res_2 = vec![FE::zero(); 4];
    // Column index: 50
    let res_3 = vec![FE::zero(); 4];
    // Column index: 51
    let res_4 = vec![
        FE::from(&20u32),
        FE::from(&400u32),
        FE::from(&30u32),
        FE::from(&40u32),
    ];
    columns.push(res_1);
    columns.push(res_2);
    columns.push(res_3);
    columns.push(res_4);

    // Wether rv1 and arg2 are equal.
    // Flag.
    // Column index: 52
    let is_equal = vec![FE::one(), FE::one(), FE::one(), FE::one()];
    columns.push(is_equal);

    // Whether a branch is taken.
    // Flag.
    // Column index: 53
    let branch_cond = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(branch_cond);

    columns
}

#[test]
fn test_prove_cpu_table() {
    let columns = build_cpu_columns_example();
    let mut trace = cpu_table::build_cpu_trace(columns);
    let proof_options = ProofOptions::default_test_options();

    let proof = Prover::<CPUTableAIR>::prove(
        &mut trace,
        &(),
        &proof_options,
        DefaultTranscript::<Degree4BabyBearU32ExtensionField>::new(&[]),
    )
    .unwrap();

    assert!(Verifier::<CPUTableAIR>::verify(
        &proof,
        &(),
        &proof_options,
        DefaultTranscript::<Degree4BabyBearU32ExtensionField>::new(&[]),
    ));
}
