use crate::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        transition::TransitionConstraint,
    },
    context::AirContext,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::{TransitionEvaluationContext, AIR},
};
use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::babybear_u32::Babybear31PrimeField,
    traits::IsFFTField,
};

type FE = FieldElement<Babybear31PrimeField>;

type CPUTraceTable = TraceTable<Babybear31PrimeField, Babybear31PrimeField>;

#[derive(Clone)]
pub struct BitConstraint {
    column_idx: usize,
    constraint_idx: usize,
}

impl BitConstraint {
    fn new(column_idx: usize, constraint_idx: usize) -> Self {
        Self {
            column_idx,
            constraint_idx,
        }
    }
}

impl TransitionConstraint<Babybear31PrimeField, Babybear31PrimeField> for BitConstraint {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn exemptions_period(&self) -> Option<usize> {
        None
    }

    fn periodic_exemptions_offset(&self) -> Option<usize> {
        None
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<
            Babybear31PrimeField,
            Babybear31PrimeField,
        >,
        transition_evaluations: &mut [FieldElement<Babybear31PrimeField>],
    ) {
        let (frame, _periodic_values, _rap_challenges) = match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values,
                rap_challenges,
            }
            | TransitionEvaluationContext::Verifier {
                frame,
                periodic_values,
                rap_challenges,
            } => (frame, periodic_values, rap_challenges),
        };

        let step = frame.get_evaluation_step(0);

        let flag = step.get_main_evaluation_element(0, self.column_idx);

        let one = FieldElement::<Babybear31PrimeField>::one();

        let bit_constraint = flag * (flag - one);

        transition_evaluations[self.constraint_idx()] = bit_constraint;
    }
}

pub struct CPUTableAIR {
    context: AirContext,
    constraints: Vec<Box<dyn TransitionConstraint<Babybear31PrimeField, Babybear31PrimeField>>>,
    trace_length: usize,
}

impl AIR for CPUTableAIR {
    type Field = Babybear31PrimeField;
    type FieldExtension = Babybear31PrimeField;
    type PublicInputs = ();

    const STEP_SIZE: usize = 1;

    fn new(
        trace_length: usize,
        _pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        // Constraint IS_BIT[f[i]] where:
        // f = [write_register, memory_2bytes, memory_4bytes, signed, signed2, muldiv_selector, ADD, SUB, SLT, AND, OR, XOR, SL, SR, JALR, BEQ, BLT, LOAD, STORE, MUL, DIVREM, ECALL, EBREAK]
        let columns_index_to_constraint = [
            7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 13,
        ];
        let constraints: Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> =
            columns_index_to_constraint
                .iter()
                .enumerate()
                .map(|(i, &column_idx)| {
                    Box::new(BitConstraint::new(column_idx, i))
                        as Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>
                })
                .collect();
        let num_transition_constraints = constraints.len();

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 54,
            transition_offsets: vec![0],
            num_transition_constraints,
        };

        Self {
            context,
            trace_length,
            constraints,
        }
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> {
        &self.constraints
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> BoundaryConstraints<Self::FieldExtension> {
        BoundaryConstraints::from_constraints(vec![])
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length * 2
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn trace_layout(&self) -> (usize, usize) {
        (1, 0)
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &()
    }
}

pub fn cpu_trace() -> CPUTraceTable {
    let mut columns = Vec::new();
    // Timestamp: A word2L column containing the values 2^{2*i} for i = 1,...
    // 2^{2 * 1}, 2^{2 * 2}, 2^{2 * 3}, 2^{2 * 4} = 4, 16, 64, 256
    // TODO. Chequear si esto es así. Como tenemos timestamp chicos, suponemos que no necesitamos una segunda word.
    // Column index: 0
    let timestamp_1 = vec![FE::zero(); 4];
    // Column index: 1
    let timestamp_2 = vec![
        FE::from(&4u32),
        FE::from(&16u32),
        FE::from(&64u32),
        FE::from(&256u32),
    ];
    columns.push(timestamp_1);
    columns.push(timestamp_2);

    // ----- 30 uncompressed decode columns -----

    // Word2L pc.
    // Column index: 2
    let pc_1 = vec![FE::zero(); 4];
    // Column index: 3
    let pc_2 = vec![
        FE::from(&0u32),
        FE::from(&1u32),
        FE::from(&2u32),
        FE::from(&3u32),
    ];
    columns.push(pc_1);
    columns.push(pc_2);

    // Index of source register 1.
    // Column index: 4
    let rs_1 = vec![FE::from(&1), FE::from(&2), FE::from(&3), FE::from(&4)];
    columns.push(rs_1);

    // Index of source register 2
    // Column index: 5
    let rs_2 = vec![FE::from(&5), FE::from(&6), FE::from(&7), FE::from(&8)];
    columns.push(rs_2);

    // Index of destination register
    // Column index: 6
    let rd = vec![FE::from(&9), FE::from(&10), FE::from(&11), FE::from(&12)];
    columns.push(rd);

    // Should the result be written
    // Column index: 7
    let write_register = vec![FE::one(), FE::zero(), FE::one(), FE::zero()];
    columns.push(write_register);

    // Does the memory access (read or write) touch at least 2 bytes
    // Column index: 8
    let memory_2_bytes = vec![FE::one(), FE::zero(), FE::one(), FE::zero()];
    columns.push(memory_2_bytes);

    // Does the memory access (read or write) touch 4 bytes
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
    // Flag
    // Column index: 12
    let signed = vec![FE::one(), FE::zero(), FE::one(), FE::zero()];
    columns.push(signed);

    // Flag
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
    let next_pc_2 = vec![FE::from(4), FE::from(5), FE::from(6), FE::from(7)];
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
    // Column index: 52
    let is_equal = vec![FE::one(), FE::one(), FE::one(), FE::one()];
    columns.push(is_equal);

    // Whether a branch is taken
    // Column index: 53
    let branch_cond = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(branch_cond);

    TraceTable::from_columns_main(columns, 1)
}
