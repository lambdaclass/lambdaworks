use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

use crate::{
    air::{
        constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
        context::AirContext,
        frame::Frame,
        AIR,
    },
    FE,
};

const ROW_LENGTH: usize = 33;

/// Main constraint identifiers
const INST: usize = 16;
const DST_ADDR: usize = 17;
const OP0_ADDR: usize = 18;
const OP1_ADDR: usize = 19;
const NEXT_AP: usize = 20;
const NEXT_FP: usize = 21;
const NEXT_PC_1: usize = 22;
const NEXT_PC_2: usize = 23;
const T0: usize = 24;
const T1: usize = 25;
const MUL_1: usize = 26;
const MUL_2: usize = 27;
const CALL_1: usize = 28;
const CALL_2: usize = 29;
const ASSERT_EQ: usize = 30;

// Frame row identifiers
//  - Flags
const F_DST_FP: usize = 0;
const F_OP_0_FP: usize = 1;
const F_OP_1_VAL: usize = 2;
const F_OP_1_FP: usize = 3;
const F_OP_1_AP: usize = 4;
const F_RES_ADD: usize = 5;
const F_RES_MUL: usize = 6;
const F_PC_ABS: usize = 7;
const F_PC_REL: usize = 8;
const F_PC_JNZ: usize = 9;
const F_AP_ADD: usize = 10;
const F_AP_ONE: usize = 11;
const F_OPC_CALL: usize = 12;
const F_OPC_RET: usize = 13;
const F_OPC_AEQ: usize = 14;
//  - Others
const FRAME_RES: usize = 16;
const FRAME_AP: usize = 17;
const FRAME_FP: usize = 18;
const FRAME_PC: usize = 19;
const FRAME_DST_ADDR: usize = 20;
const FRAME_OP0_ADDR: usize = 21;
const FRAME_OP1_ADDR: usize = 22;
const FRAME_INST: usize = 23;
const FRAME_DST: usize = 24;
const FRAME_OP0: usize = 25;
const FRAME_OP1: usize = 26;
const OFF_DST: usize = 27;
const OFF_OP0: usize = 28;
const OFF_OP1: usize = 29;
const FRAME_T0: usize = 30;
const FRAME_T1: usize = 31;
const FRAME_MUL: usize = 32;
const FRAME_SELECTOR: usize = 33;

// Trace layout
pub const MEM_P_TRACE_OFFSET: usize = 17;
pub const MEM_A_TRACE_OFFSET: usize = 19;

#[derive(Clone)]
pub struct PublicInputs {
    pub pc_init: FE,
    pub ap_init: FE,
    pub fp_init: FE,
    pub pc_final: FE,
    pub ap_final: FE,
    pub fp_final: FE,
    pub rc_min: u16, // minimum range check value (0 < rc_min < rc_max < 2^16)
    pub rc_max: u16, // maximum range check value
    pub mem: (Vec<u64>, Vec<Option<FE>>), // public memory
    pub num_steps: usize, // number of execution steps
                     // pub builtins: Vec<Builtin>, // list of builtins
}

#[derive(Clone)]
pub struct CairoAIR {
    context: AirContext,
    pub_inputs: PublicInputs,
}

impl AIR for CairoAIR {
    type Field = Stark252PrimeField;

    fn compute_transition(&self, frame: &Frame<Self::Field>) -> Vec<FieldElement<Self::Field>> {
        let mut constraints: Vec<FieldElement<Self::Field>> = Vec::with_capacity(ROW_LENGTH);

        generate_instr_constraints(&mut constraints, frame);
        generate_operand_constraints(&mut constraints, frame);
        generate_register_constraints(&mut constraints, frame);
        generate_opcode_constraints(&mut constraints, frame);
        enforce_selector(&mut constraints, frame);

        constraints
    }

    fn boundary_constraints(&self) -> BoundaryConstraints<Self::Field> {
        let last_step = self.context.trace_length - 1;
        let constraints = vec![
            // Initial 'pc' register
            BoundaryConstraint::new(MEM_A_TRACE_OFFSET, 0, self.pub_inputs.pc_init.clone()),
            // Initial 'ap' register
            BoundaryConstraint::new(MEM_P_TRACE_OFFSET, 0, self.pub_inputs.ap_init.clone()),
            // Final 'pc' register
            BoundaryConstraint::new(
                MEM_A_TRACE_OFFSET,
                last_step,
                self.pub_inputs.pc_final.clone(),
            ),
            // Final 'ap' register
            BoundaryConstraint::new(
                MEM_P_TRACE_OFFSET,
                last_step,
                self.pub_inputs.ap_final.clone(),
            ),
        ];

        BoundaryConstraints::from_constraints(constraints)
    }

    fn context(&self) -> AirContext {
        self.context.clone()
    }
}

fn generate_instr_constraints(
    constraints: &mut [FE],
    frame: &Frame<Stark252PrimeField>,
) {
    let curr = frame.get_row(0);
    // Bit constraints
    for (i, flag) in curr[0..16].iter().enumerate() {
        constraints[i] = match i {
            0..=14 => flag * (flag - FieldElement::one()),
            15 => flag.clone(),
            _ => panic!("Unknown flag offset"),
        };
    }
    let two = FieldElement::from(2);
    // Instruction unpacking
    let b15 = two.pow(15u32);
    let b16 = two.pow(16u32);
    let b32 = two.pow(32u32);
    let b48 = two.pow(48u32);
    let a = &curr[0..14]
        .iter()
        .enumerate()
        .fold(FieldElement::zero(), |acc, (n, flag)| {
            acc + FieldElement::from(2).pow(n) * flag
        });

    constraints[INST] = (&curr[OFF_DST] + &b15)
        + b16 * (&curr[OFF_OP0] + &b15)
        + b32 * (&curr[OFF_OP1] + &b15)
        + b48 * a
        - &curr[FRAME_INST];
}

fn generate_operand_constraints(
    constraints: &mut [FE],
    frame: &Frame<Stark252PrimeField>,
) {
    let curr = frame.get_row(0);
    let ap = &curr[FRAME_AP];
    let fp = &curr[FRAME_FP];
    let pc = &curr[FRAME_PC];
    let one = FieldElement::one();

    constraints[DST_ADDR] = &curr[F_DST_FP] * fp + (&one - &curr[F_DST_FP]) * ap + &curr[OFF_DST]
        - &curr[FRAME_DST_ADDR];
    constraints[OP0_ADDR] = &curr[F_OP_0_FP] * fp + (&one - &curr[F_OP_0_FP]) * ap + &curr[OFF_OP0]
        - &curr[FRAME_OP0_ADDR];
    constraints[OP1_ADDR] = &curr[F_OP_1_VAL] * pc
        + &curr[F_OP_1_AP] * ap
        + &curr[F_OP_1_FP] * fp
        + (&one - &curr[F_OP_1_VAL] - &curr[F_OP_1_AP] - &curr[F_OP_1_FP]) * &curr[FRAME_OP0]
        + &curr[OFF_OP1]
        - &curr[FRAME_OP1_ADDR];
}

fn generate_register_constraints(
    constraints: &mut [FE],
    frame: &Frame<Stark252PrimeField>,
) {
    let curr = frame.get_row(0);
    let next = frame.get_row(1);
    let one = FieldElement::one();
    let two = FieldElement::from(2);

    // ap and fp constraints
    constraints[NEXT_AP] = &curr[FRAME_AP]
        + &curr[F_AP_ADD] * &curr[FRAME_RES]
        + &curr[F_AP_ONE]
        + &curr[F_OPC_CALL] * &two
        - &next[FRAME_AP];
    constraints[NEXT_FP] = &curr[F_OPC_RET] * &curr[FRAME_DST]
        + &curr[F_OPC_CALL] * (&curr[FRAME_AP] + &two)
        + (&one - &curr[F_OPC_RET] - &curr[F_OPC_CALL]) * &curr[FRAME_AP]
        - &next[FRAME_AP];

    // pc constraints
    constraints[NEXT_PC_1] = (&curr[FRAME_T1] - &curr[F_PC_JNZ])
        * (&next[FRAME_PC] - (&curr[FRAME_PC] + frame_inst_size(curr)));
    constraints[NEXT_PC_2] = &curr[FRAME_T0]
        * (&next[FRAME_PC] - (&curr[FRAME_PC] + &curr[FRAME_OP1]))
        + (&one - &curr[F_PC_JNZ]) * &next[FRAME_PC]
        - ((&one - &curr[F_PC_ABS] - &curr[F_PC_REL] - &curr[F_PC_JNZ])
            * (&curr[FRAME_PC] + frame_inst_size(curr))
            + &curr[F_PC_ABS] * &curr[FRAME_RES]
            + &curr[F_PC_REL] * (&curr[FRAME_PC] + &curr[FRAME_RES]));
    constraints[T0] = &curr[F_PC_JNZ] * &curr[FRAME_DST] - &curr[FRAME_T0];
    constraints[T1] = &curr[FRAME_T0] * &curr[FRAME_RES] - &curr[FRAME_T1];
}

fn generate_opcode_constraints(
    constraints: &mut [FE],
    frame: &Frame<Stark252PrimeField>,
) {
    let curr = frame.get_row(0);
    let one = FieldElement::one();
    constraints[MUL_1] = &curr[FRAME_MUL] - (&curr[FRAME_OP0] * &curr[FRAME_OP1]);
    constraints[MUL_2] = &curr[F_RES_ADD] * (&curr[FRAME_OP0] + &curr[FRAME_OP1])
        + &curr[F_RES_MUL] * &curr[FRAME_MUL]
        + (&one - &curr[F_RES_ADD] - &curr[F_RES_MUL] - &curr[F_PC_JNZ]) * &curr[FRAME_OP1]
        - (&one - &curr[F_PC_JNZ]) * &curr[FRAME_RES];
    constraints[CALL_1] = &curr[F_OPC_CALL] * (&curr[FRAME_DST] - &curr[FRAME_AP]);
    constraints[CALL_2] =
        &curr[F_OPC_CALL] * (&curr[FRAME_OP0] - (&curr[FRAME_PC] + frame_inst_size(curr)));
    constraints[ASSERT_EQ] = &curr[F_OPC_AEQ] * (&curr[FRAME_DST] - &curr[FRAME_RES]);
}

fn enforce_selector(
    constraints: &mut [FE],
    frame: &Frame<Stark252PrimeField>,
) {
    let curr = frame.get_row(0);
    for result_cell in constraints.iter_mut().take(ASSERT_EQ + 1).skip(INST) {
        *result_cell = result_cell.clone() * curr[FRAME_SELECTOR].clone();
    }
}

fn frame_inst_size(
    frame_row: &[FE],
) -> FE {
    &frame_row[F_OP_1_VAL] + FieldElement::one()
}
