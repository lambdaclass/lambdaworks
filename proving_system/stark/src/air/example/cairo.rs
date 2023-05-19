use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    traits::{IsFFTField, IsPrimeField},
};

use crate::{
    air::{
        constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
        context::{AirContext, ProofOptions},
        frame::Frame,
        trace::TraceTable,
        AIR,
    },
    cairo_vm::{
        cairo_mem::CairoMemory, cairo_trace::CairoTrace,
        execution_trace::build_cairo_execution_trace,
    },
    transcript_to_field, FE,
};

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

// Auxiliary constraint identifiers
const MEMORY_INCREASING_0: usize = 31;
const MEMORY_INCREASING_1: usize = 32;
const MEMORY_INCREASING_2: usize = 33;
const MEMORY_INCREASING_3: usize = 34;

const MEMORY_CONSISTENCY_0: usize = 35;
const MEMORY_CONSISTENCY_1: usize = 36;
const MEMORY_CONSISTENCY_2: usize = 37;
const MEMORY_CONSISTENCY_3: usize = 38;

const PERMUTATION_ARGUMENT_0: usize = 39;
const PERMUTATION_ARGUMENT_1: usize = 40;
const PERMUTATION_ARGUMENT_2: usize = 41;
const PERMUTATION_ARGUMENT_3: usize = 42;

const RANGE_CHECK_INCREASING_0: usize = 43;
const RANGE_CHECK_INCREASING_1: usize = 44;
const RANGE_CHECK_INCREASING_2: usize = 45;

const RANGE_CHECK_0: usize = 46;
const RANGE_CHECK_1: usize = 47;
const RANGE_CHECK_2: usize = 48;

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
// TODO: These should probably be in the TraceTable module.
pub const FRAME_RES: usize = 16;
pub const FRAME_AP: usize = 17;
pub const FRAME_FP: usize = 18;
pub const FRAME_PC: usize = 19;
pub const FRAME_DST_ADDR: usize = 20;
pub const FRAME_OP0_ADDR: usize = 21;
pub const FRAME_OP1_ADDR: usize = 22;
pub const FRAME_INST: usize = 23;
pub const FRAME_DST: usize = 24;
pub const FRAME_OP0: usize = 25;
pub const FRAME_OP1: usize = 26;
pub const OFF_DST: usize = 27;
pub const OFF_OP0: usize = 28;
pub const OFF_OP1: usize = 29;
pub const FRAME_T0: usize = 30;
pub const FRAME_T1: usize = 31;
pub const FRAME_MUL: usize = 32;
pub const FRAME_SELECTOR: usize = 33;

// Auxiliary range check columns
pub const RANGE_CHECK_COL_1: usize = 34;
pub const RANGE_CHECK_COL_2: usize = 35;
pub const RANGE_CHECK_COL_3: usize = 36;

// Auxiliary memory columns
pub const MEMORY_ADDR_SORTED_0: usize = 37;
pub const MEMORY_ADDR_SORTED_1: usize = 38;
pub const MEMORY_ADDR_SORTED_2: usize = 39;
pub const MEMORY_ADDR_SORTED_3: usize = 40;

pub const MEMORY_VALUES_SORTED_0: usize = 41;
pub const MEMORY_VALUES_SORTED_1: usize = 42;
pub const MEMORY_VALUES_SORTED_2: usize = 43;
pub const MEMORY_VALUES_SORTED_3: usize = 44;

pub const PERMUTATION_ARGUMENT_COL_0: usize = 45;
pub const PERMUTATION_ARGUMENT_COL_1: usize = 46;
pub const PERMUTATION_ARGUMENT_COL_2: usize = 47;
pub const PERMUTATION_ARGUMENT_COL_3: usize = 48;

pub const PERMUTATION_ARGUMENT_RANGE_CHECK_COL_1: usize = 49;
pub const PERMUTATION_ARGUMENT_RANGE_CHECK_COL_2: usize = 50;
pub const PERMUTATION_ARGUMENT_RANGE_CHECK_COL_3: usize = 51;

pub const MEMORY_COLUMNS: [usize; 8] = [
    FRAME_PC,
    FRAME_DST_ADDR,
    FRAME_OP0_ADDR,
    FRAME_OP1_ADDR,
    FRAME_INST,
    FRAME_DST,
    FRAME_OP0,
    FRAME_OP1,
];

// Trace layout
pub const MEM_P_TRACE_OFFSET: usize = 17;
pub const MEM_A_TRACE_OFFSET: usize = 19;

// TODO: For memory constraints and builtins, the commented fields may be useful.
#[derive(Clone)]
pub struct PublicInputs {
    pub pc_init: FE,
    pub ap_init: FE,
    pub fp_init: FE,
    pub pc_final: FE,
    pub ap_final: FE,
    pub range_check_min: Option<u16>, // minimum range check value (0 < rc_min < rc_max < 2^16)
    pub range_check_max: Option<u16>, // maximum range check value
    // pub builtins: Vec<Builtin>, // list of builtins
    pub program: Vec<FE>,
    pub num_steps: usize, // number of execution steps
}

#[derive(Clone)]
pub struct CairoAIR {
    pub context: AirContext,
    pub number_steps: usize,
}

impl CairoAIR {
    pub fn new(proof_options: ProofOptions, full_trace_length: usize, number_steps: usize) -> Self {
        let context = AirContext {
            options: proof_options,
            trace_length: full_trace_length,
            trace_columns: 34 + 3 + 12 + 3,
            transition_degrees: vec![
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // Flags 0-14.
                1, // Flag 15
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // Other constraints.
                2, 2, 2, 2, // Increasing memory auxiliary constraints.
                2, 2, 2, 2, // Consistent memory auxiliary constraints.
                2, 2, 2, 2, // Permutation auxiliary constraints.
                2, 2, 2, // Permutation auxiliary constraints.
            ],
            transition_exemptions: vec![
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1,
            ],
            transition_offsets: vec![0, 1],
            num_transition_constraints: 49,
        };

        Self {
            context,
            number_steps,
        }
    }
}

pub struct CairoRAPChallenges {
    pub alpha_memory: FieldElement<Stark252PrimeField>,
    pub z_memory: FieldElement<Stark252PrimeField>,
    pub z_range_check: FieldElement<Stark252PrimeField>,
}

fn add_program_in_public_input_section(
    addresses: &Vec<FE>,
    values: &[FE],
    public_input: &PublicInputs,
) -> (Vec<FE>, Vec<FE>) {
    let mut a_aux = addresses.clone();
    let mut v_aux = values.to_owned();

    let public_input_section = addresses.len() - public_input.program.len();
    let continous_memory = (1..=public_input.program.len() as u64).map(FieldElement::from);

    a_aux.splice(public_input_section.., continous_memory);
    v_aux.splice(public_input_section.., public_input.program.clone());

    (a_aux, v_aux)
}

fn sort_columns_by_memory_address(adresses: Vec<FE>, values: Vec<FE>) -> (Vec<FE>, Vec<FE>) {
    let mut tuples: Vec<_> = adresses.into_iter().zip(values).collect();
    tuples.sort_by(|(x, _), (y, _)| x.representative().cmp(&y.representative()));
    tuples.into_iter().unzip()
    // (adresses, values)
}

fn generate_memory_permutation_argument_column(
    addresses_original: Vec<FE>,
    values_original: Vec<FE>,
    addresses_sorted: &[FE],
    values_sorted: &[FE],
    rap_challenges: &CairoRAPChallenges,
) -> Vec<FE> {
    let z = &rap_challenges.z_memory;
    let alpha = &rap_challenges.alpha_memory;
    let f = |a, v, ap, vp| (z - (a + alpha * v)) / (z - (ap + alpha * vp));

    let mut permutation_col = Vec::with_capacity(addresses_sorted.len());
    permutation_col.push(f(
        &addresses_original[0],
        &values_original[0],
        &addresses_sorted[0],
        &values_sorted[0],
    ));

    for i in 1..addresses_sorted.len() {
        let last = permutation_col.last().unwrap();
        permutation_col.push(
            last * f(
                &addresses_original[i],
                &values_original[i],
                &addresses_sorted[i],
                &values_sorted[i],
            ),
        );
    }

    permutation_col
}
fn generate_range_check_permutation_argument_column(
    offset_column_original: &[FE],
    offset_column_sorted: &[FE],
    rap_challenges: &CairoRAPChallenges,
) -> Vec<FE> {
    let z = &rap_challenges.z_range_check;
    let f = |a, ap| (z - a) / (z - ap);

    let mut permutation_col = Vec::with_capacity(offset_column_original.len());
    permutation_col.push(f(&offset_column_original[0], &offset_column_sorted[0]));

    for i in 1..offset_column_sorted.len() {
        let last = permutation_col.last().unwrap();
        permutation_col.push(last * f(&offset_column_original[i], &offset_column_sorted[i]));
    }
    permutation_col
}

fn pad_with_last_row<F: IsFFTField>(
    trace: &mut TraceTable<F>,
    number_rows: usize,
    exception_columns: &[usize],
) {
    let mut last_row = trace.last_row().to_vec();
    for excemption_column in exception_columns.iter() {
        last_row[*excemption_column] = FieldElement::zero();
    }
    let mut pad: Vec<_> = std::iter::repeat(&last_row)
        .take(number_rows)
        .flatten()
        .cloned()
        .collect();
    trace.table.append(&mut pad);
}

fn get_missing_values_offset_columns<F>(
    trace: &TraceTable<F>,
    columns_indices: &[usize],
) -> (Vec<FieldElement<F>>, u16, u16)
where
    F: IsFFTField + IsPrimeField,
    u16: From<F::RepresentativeType>,
{
    let offset_columns = trace.get_cols(columns_indices).table;

    let mut sorted_offset_representatives: Vec<u16> = offset_columns
        .iter()
        .map(|x| x.representative().into())
        .collect();
    sorted_offset_representatives.sort();

    let mut all_missing_values: Vec<FieldElement<F>> = Vec::new();

    for window in sorted_offset_representatives.windows(2) {
        if window[1] != window[0] {
            let mut missing_range: Vec<_> = ((window[0] + 1)..window[1])
                .map(|x| FieldElement::from(x as u64))
                .collect();
            all_missing_values.append(&mut missing_range);
        }
    }

    let multiple_of_three_padding =
        ((all_missing_values.len() + 2) / 3) * 3 - all_missing_values.len();
    let padding_element = FieldElement::from(*sorted_offset_representatives.last().unwrap() as u64);
    all_missing_values.append(&mut vec![padding_element; multiple_of_three_padding]);

    (
        all_missing_values,
        sorted_offset_representatives[0],
        sorted_offset_representatives.last().cloned().unwrap(),
    )
}

fn add_missing_values_to_offsets_column<F: IsFFTField>(
    trace: &mut TraceTable<F>,
    missing_values: Vec<FieldElement<F>>,
) {
    let zeros_left = vec![FieldElement::zero(); OFF_DST];
    let zeros_right = vec![FieldElement::zero(); trace.n_cols - OFF_OP1 - 1];

    for i in (0..missing_values.len()).step_by(3) {
        trace.table.append(&mut zeros_left.clone());
        trace.table.append(&mut missing_values[i..(i + 3)].to_vec());
        trace.table.append(&mut zeros_right.clone());
    }
}

impl AIR for CairoAIR {
    type Field = Stark252PrimeField;
    type RawTrace = (CairoTrace, CairoMemory);
    type RAPChallenges = CairoRAPChallenges;
    type PublicInput = PublicInputs;

    fn build_main_trace(
        &self,
        raw_trace: &Self::RawTrace,
        public_input: &mut Self::PublicInput,
    ) -> TraceTable<Self::Field> {
        let mut main_trace = build_cairo_execution_trace(&raw_trace.0, &raw_trace.1);

        pad_with_last_row(
            &mut main_trace,
            (public_input.program.len() >> 2) + 1,
            &MEMORY_COLUMNS,
        );

        let (missing_values, rc_min, rc_max) =
            get_missing_values_offset_columns(&main_trace, &[OFF_DST, OFF_OP0, OFF_OP1]);
        public_input.range_check_min = Some(rc_min);
        public_input.range_check_max = Some(rc_max);

        add_missing_values_to_offsets_column(&mut main_trace, missing_values);

        let padding = self.context().trace_length - main_trace.n_rows();
        pad_with_last_row(&mut main_trace, padding, &MEMORY_COLUMNS);

        main_trace
    }

    fn build_auxiliary_trace(
        &self,
        main_trace: &TraceTable<Self::Field>,
        rap_challenges: &Self::RAPChallenges,
        public_input: &Self::PublicInput,
    ) -> TraceTable<Self::Field> {
        let addresses_original = main_trace
            .get_cols(&[FRAME_PC, FRAME_DST_ADDR, FRAME_OP0_ADDR, FRAME_OP1_ADDR])
            .table;
        let values_original = main_trace
            .get_cols(&[FRAME_INST, FRAME_DST, FRAME_OP0, FRAME_OP1])
            .table;

        let (addresses, values) = add_program_in_public_input_section(
            &addresses_original,
            &values_original,
            public_input,
        );
        let (addresses, values) = sort_columns_by_memory_address(addresses, values);
        let permutation_col = generate_memory_permutation_argument_column(
            addresses_original,
            values_original,
            &addresses,
            &values,
            rap_challenges,
        );

        // Range Check
        let offsets_original = main_trace.get_cols(&[OFF_DST, OFF_OP0, OFF_OP1]).table;
        let mut offsets_sorted: Vec<u16> = offsets_original
            .iter()
            .map(|x| x.representative().into())
            .collect();
        offsets_sorted.sort();
        let offsets_sorted: Vec<_> = offsets_sorted
            .iter()
            .map(|x| FieldElement::from(*x as u64))
            .collect();

        let range_check_permutation_col = generate_range_check_permutation_argument_column(
            &offsets_original,
            &offsets_sorted,
            rap_challenges,
        );

        // Convert from long-format to wide-format again
        let mut aux_table = Vec::new();
        for i in 0..main_trace.n_rows() {
            aux_table.push(offsets_sorted[3 * i].clone());
            aux_table.push(offsets_sorted[3 * i + 1].clone());
            aux_table.push(offsets_sorted[3 * i + 2].clone());
            aux_table.push(addresses[4 * i].clone());
            aux_table.push(addresses[4 * i + 1].clone());
            aux_table.push(addresses[4 * i + 2].clone());
            aux_table.push(addresses[4 * i + 3].clone());
            aux_table.push(values[4 * i].clone());
            aux_table.push(values[4 * i + 1].clone());
            aux_table.push(values[4 * i + 2].clone());
            aux_table.push(values[4 * i + 3].clone());
            aux_table.push(permutation_col[4 * i].clone());
            aux_table.push(permutation_col[4 * i + 1].clone());
            aux_table.push(permutation_col[4 * i + 2].clone());
            aux_table.push(permutation_col[4 * i + 3].clone());
            aux_table.push(range_check_permutation_col[3 * i].clone());
            aux_table.push(range_check_permutation_col[3 * i + 1].clone());
            aux_table.push(range_check_permutation_col[3 * i + 2].clone());
        }

        TraceTable::new(aux_table, self.number_auxiliary_rap_columns())
    }

    fn build_rap_challenges<T: Transcript>(&self, transcript: &mut T) -> Self::RAPChallenges {
        CairoRAPChallenges {
            alpha_memory: transcript_to_field(transcript),
            z_memory: transcript_to_field(transcript),
            z_range_check: transcript_to_field(transcript),
        }
    }

    fn number_auxiliary_rap_columns(&self) -> usize {
        12 + 3 + 3
    }

    fn compute_transition(
        &self,
        frame: &Frame<Self::Field>,
        rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Self::Field>> {
        let mut constraints: Vec<FieldElement<Self::Field>> =
            vec![FE::zero(); self.num_transition_constraints()];

        compute_instr_constraints(&mut constraints, frame);
        compute_operand_constraints(&mut constraints, frame);
        compute_register_constraints(&mut constraints, frame);
        compute_opcode_constraints(&mut constraints, frame);
        enforce_selector(&mut constraints, frame);
        memory_is_increasing(&mut constraints, frame);
        permutation_argument(&mut constraints, frame, rap_challenges);
        permutation_argument_range_check(&mut constraints, frame, rap_challenges);

        constraints
    }

    /// From the Cairo whitepaper, section 9.10.
    /// These are part of the register constraints.
    ///
    /// Boundary constraints:
    ///  * ap_0 = fp_0 = ap_i
    ///  * ap_t = ap_f
    ///  * pc_0 = pc_i
    ///  * pc_t = pc_f
    fn boundary_constraints(
        &self,
        rap_challenges: &Self::RAPChallenges,
        public_input: &Self::PublicInput,
    ) -> BoundaryConstraints<Self::Field> {
        let initial_pc =
            BoundaryConstraint::new(MEM_A_TRACE_OFFSET, 0, public_input.pc_init.clone());
        let initial_ap =
            BoundaryConstraint::new(MEM_P_TRACE_OFFSET, 0, public_input.ap_init.clone());

        let final_pc = BoundaryConstraint::new(
            MEM_A_TRACE_OFFSET,
            self.number_steps - 1,
            public_input.pc_final.clone(),
        );
        let final_ap = BoundaryConstraint::new(
            MEM_P_TRACE_OFFSET,
            self.number_steps - 1,
            public_input.ap_final.clone(),
        );

        // Auxiliary constraint: permutation argument final value
        let final_index = self.context.trace_length - 1;

        let mut cumulative_product = FieldElement::one();
        for (i, value) in public_input.program.iter().enumerate() {
            cumulative_product = cumulative_product
                * (&rap_challenges.z_memory
                    - (FieldElement::from(i as u64 + 1) + &rap_challenges.alpha_memory * value));
        }
        let permutation_final =
            rap_challenges.z_memory.pow(public_input.program.len()) / cumulative_product;
        let permutation_final_constraint =
            BoundaryConstraint::new(PERMUTATION_ARGUMENT_COL_3, final_index, permutation_final);

        let one: FieldElement<Self::Field> = FieldElement::one();
        let range_check_final_constraint =
            BoundaryConstraint::new(PERMUTATION_ARGUMENT_RANGE_CHECK_COL_3, final_index, one);

        let range_check_min = BoundaryConstraint::new(
            RANGE_CHECK_COL_1,
            0,
            FieldElement::from(public_input.range_check_min.unwrap() as u64),
        );
        let range_check_max = BoundaryConstraint::new(
            RANGE_CHECK_COL_3,
            final_index,
            FieldElement::from(public_input.range_check_max.unwrap() as u64),
        );

        let constraints = vec![
            initial_pc,
            initial_ap,
            final_pc,
            final_ap,
            permutation_final_constraint,
            range_check_final_constraint,
            range_check_min,
            range_check_max,
        ];

        BoundaryConstraints::from_constraints(constraints)
    }

    fn context(&self) -> AirContext {
        self.context.clone()
    }

    fn composition_poly_degree_bound(&self) -> usize {
        2 * self.context().trace_length
    }
}

/// From the Cairo whitepaper, section 9.10
fn compute_instr_constraints(constraints: &mut [FE], frame: &Frame<Stark252PrimeField>) {
    // These constraints are only applied over elements of the same row.
    let curr = frame.get_row(0);

    // Bit constraints
    for (i, flag) in curr[0..16].iter().enumerate() {
        constraints[i] = match i {
            0..=14 => flag * (flag - FE::one()),
            15 => flag.clone(),
            _ => panic!("Unknown flag offset"),
        };
    }

    // Instruction unpacking
    let two = FE::from(2);
    let b16 = two.pow(16u32);
    let b32 = two.pow(32u32);
    let b48 = two.pow(48u32);

    // Named like this to match the Cairo whitepaper's notation.
    let f0_squiggle = &curr[0..15]
        .iter()
        .rev()
        .fold(FE::zero(), |acc, flag| flag + &two * acc);

    constraints[INST] =
        (&curr[OFF_DST]) + b16 * (&curr[OFF_OP0]) + b32 * (&curr[OFF_OP1]) + b48 * f0_squiggle
            - &curr[FRAME_INST];
}

fn compute_operand_constraints(constraints: &mut [FE], frame: &Frame<Stark252PrimeField>) {
    // These constraints are only applied over elements of the same row.
    let curr = frame.get_row(0);

    let ap = &curr[FRAME_AP];
    let fp = &curr[FRAME_FP];
    let pc = &curr[FRAME_PC];

    let one = FE::one();
    let b15 = FE::from(2).pow(15u32);

    constraints[DST_ADDR] =
        &curr[F_DST_FP] * fp + (&one - &curr[F_DST_FP]) * ap + (&curr[OFF_DST] - &b15)
            - &curr[FRAME_DST_ADDR];

    constraints[OP0_ADDR] =
        &curr[F_OP_0_FP] * fp + (&one - &curr[F_OP_0_FP]) * ap + (&curr[OFF_OP0] - &b15)
            - &curr[FRAME_OP0_ADDR];

    constraints[OP1_ADDR] = &curr[F_OP_1_VAL] * pc
        + &curr[F_OP_1_AP] * ap
        + &curr[F_OP_1_FP] * fp
        + (&one - &curr[F_OP_1_VAL] - &curr[F_OP_1_AP] - &curr[F_OP_1_FP]) * &curr[FRAME_OP0]
        + (&curr[OFF_OP1] - &b15)
        - &curr[FRAME_OP1_ADDR];
}

fn compute_register_constraints(constraints: &mut [FE], frame: &Frame<Stark252PrimeField>) {
    let curr = frame.get_row(0);
    let next = frame.get_row(1);

    let one = FE::one();
    let two = FE::from(2);

    // ap and fp constraints
    constraints[NEXT_AP] = &curr[FRAME_AP]
        + &curr[F_AP_ADD] * &curr[FRAME_RES]
        + &curr[F_AP_ONE]
        + &curr[F_OPC_CALL] * &two
        - &next[FRAME_AP];

    constraints[NEXT_FP] = &curr[F_OPC_RET] * &curr[FRAME_DST]
        + &curr[F_OPC_CALL] * (&curr[FRAME_AP] + &two)
        + (&one - &curr[F_OPC_RET] - &curr[F_OPC_CALL]) * &curr[FRAME_FP]
        - &next[FRAME_FP];

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

fn compute_opcode_constraints(constraints: &mut [FE], frame: &Frame<Stark252PrimeField>) {
    let curr = frame.get_row(0);
    let one = FE::one();

    constraints[MUL_1] = &curr[FRAME_MUL] - (&curr[FRAME_OP0] * &curr[FRAME_OP1]);

    constraints[MUL_2] = &curr[F_RES_ADD] * (&curr[FRAME_OP0] + &curr[FRAME_OP1])
        + &curr[F_RES_MUL] * &curr[FRAME_MUL]
        + (&one - &curr[F_RES_ADD] - &curr[F_RES_MUL] - &curr[F_PC_JNZ]) * &curr[FRAME_OP1]
        - (&one - &curr[F_PC_JNZ]) * &curr[FRAME_RES];

    constraints[CALL_1] = &curr[F_OPC_CALL] * (&curr[FRAME_DST] - &curr[FRAME_FP]);

    constraints[CALL_2] =
        &curr[F_OPC_CALL] * (&curr[FRAME_OP0] - (&curr[FRAME_PC] + frame_inst_size(curr)));

    constraints[ASSERT_EQ] = &curr[F_OPC_AEQ] * (&curr[FRAME_DST] - &curr[FRAME_RES]);
}

fn enforce_selector(constraints: &mut [FE], frame: &Frame<Stark252PrimeField>) {
    let curr = frame.get_row(0);
    for result_cell in constraints.iter_mut().take(ASSERT_EQ + 1).skip(INST) {
        *result_cell = result_cell.clone() * curr[FRAME_SELECTOR].clone();
    }
}

fn memory_is_increasing(constraints: &mut [FE], frame: &Frame<Stark252PrimeField>) {
    let curr = frame.get_row(0);
    let next = frame.get_row(1);
    let one = FieldElement::one();

    constraints[MEMORY_INCREASING_0] = (&curr[MEMORY_ADDR_SORTED_0] - &curr[MEMORY_ADDR_SORTED_1])
        * (&curr[MEMORY_ADDR_SORTED_1] - &curr[MEMORY_ADDR_SORTED_0] - &one);

    constraints[MEMORY_INCREASING_1] = (&curr[MEMORY_ADDR_SORTED_1] - &curr[MEMORY_ADDR_SORTED_2])
        * (&curr[MEMORY_ADDR_SORTED_2] - &curr[MEMORY_ADDR_SORTED_1] - &one);

    constraints[MEMORY_INCREASING_2] = (&curr[MEMORY_ADDR_SORTED_2] - &curr[MEMORY_ADDR_SORTED_3])
        * (&curr[MEMORY_ADDR_SORTED_3] - &curr[MEMORY_ADDR_SORTED_2] - &one);

    constraints[MEMORY_INCREASING_3] = (&curr[MEMORY_ADDR_SORTED_3] - &next[MEMORY_ADDR_SORTED_0])
        * (&next[MEMORY_ADDR_SORTED_0] - &curr[MEMORY_ADDR_SORTED_3] - &one);

    constraints[MEMORY_CONSISTENCY_0] = (&curr[MEMORY_VALUES_SORTED_0]
        - &curr[MEMORY_VALUES_SORTED_1])
        * (&curr[MEMORY_ADDR_SORTED_1] - &curr[MEMORY_ADDR_SORTED_0] - &one);

    constraints[MEMORY_CONSISTENCY_1] = (&curr[MEMORY_VALUES_SORTED_1]
        - &curr[MEMORY_VALUES_SORTED_2])
        * (&curr[MEMORY_ADDR_SORTED_2] - &curr[MEMORY_ADDR_SORTED_1] - &one);

    constraints[MEMORY_CONSISTENCY_2] = (&curr[MEMORY_VALUES_SORTED_2]
        - &curr[MEMORY_VALUES_SORTED_3])
        * (&curr[MEMORY_ADDR_SORTED_3] - &curr[MEMORY_ADDR_SORTED_2] - &one);

    constraints[MEMORY_CONSISTENCY_3] = (&curr[MEMORY_VALUES_SORTED_3]
        - &next[MEMORY_VALUES_SORTED_0])
        * (&next[MEMORY_ADDR_SORTED_0] - &curr[MEMORY_ADDR_SORTED_3] - &one);
}

fn permutation_argument(
    constraints: &mut [FE],
    frame: &Frame<Stark252PrimeField>,
    rap_challenges: &CairoRAPChallenges,
) {
    let curr = frame.get_row(0);
    let next = frame.get_row(1);
    let z = &rap_challenges.z_memory;
    let alpha = &rap_challenges.alpha_memory;

    let p0 = &curr[PERMUTATION_ARGUMENT_COL_0];
    let p0_next = &next[PERMUTATION_ARGUMENT_COL_0];
    let p1 = &curr[PERMUTATION_ARGUMENT_COL_1];
    let p2 = &curr[PERMUTATION_ARGUMENT_COL_2];
    let p3 = &curr[PERMUTATION_ARGUMENT_COL_3];

    let ap0_next = &next[MEMORY_ADDR_SORTED_0];
    let ap1 = &curr[MEMORY_ADDR_SORTED_1];
    let ap2 = &curr[MEMORY_ADDR_SORTED_2];
    let ap3 = &curr[MEMORY_ADDR_SORTED_3];

    let vp0_next = &next[MEMORY_VALUES_SORTED_0];
    let vp1 = &curr[MEMORY_VALUES_SORTED_1];
    let vp2 = &curr[MEMORY_VALUES_SORTED_2];
    let vp3 = &curr[MEMORY_VALUES_SORTED_3];

    let a0_next = &next[FRAME_PC];
    let a1 = &curr[FRAME_DST_ADDR];
    let a2 = &curr[FRAME_OP0_ADDR];
    let a3 = &curr[FRAME_OP1_ADDR];

    let v0_next = &next[FRAME_INST];
    let v1 = &curr[FRAME_DST];
    let v2 = &curr[FRAME_OP0];
    let v3 = &curr[FRAME_OP1];

    constraints[PERMUTATION_ARGUMENT_0] =
        (z - (ap1 + alpha * vp1)) * p1 - (z - (a1 + alpha * v1)) * p0;
    constraints[PERMUTATION_ARGUMENT_1] =
        (z - (ap2 + alpha * vp2)) * p2 - (z - (a2 + alpha * v2)) * p1;
    constraints[PERMUTATION_ARGUMENT_2] =
        (z - (ap3 + alpha * vp3)) * p3 - (z - (a3 + alpha * v3)) * p2;
    constraints[PERMUTATION_ARGUMENT_3] =
        (z - (ap0_next + alpha * vp0_next)) * p0_next - (z - (a0_next + alpha * v0_next)) * p3;
}

fn permutation_argument_range_check(
    constraints: &mut [FE],
    frame: &Frame<Stark252PrimeField>,
    rap_challenges: &CairoRAPChallenges,
) {
    let curr = frame.get_row(0);
    let next = frame.get_row(1);
    let one = FieldElement::one();
    let z = &rap_challenges.z_range_check;

    constraints[RANGE_CHECK_INCREASING_0] = (&curr[RANGE_CHECK_COL_1] - &curr[RANGE_CHECK_COL_2])
        * (&curr[RANGE_CHECK_COL_2] - &curr[RANGE_CHECK_COL_1] - &one);
    constraints[RANGE_CHECK_INCREASING_1] = (&curr[RANGE_CHECK_COL_2] - &curr[RANGE_CHECK_COL_3])
        * (&curr[RANGE_CHECK_COL_3] - &curr[RANGE_CHECK_COL_2] - &one);
    constraints[RANGE_CHECK_INCREASING_2] = (&curr[RANGE_CHECK_COL_3] - &next[RANGE_CHECK_COL_1])
        * (&next[RANGE_CHECK_COL_1] - &curr[RANGE_CHECK_COL_3] - &one);

    let p0 = &curr[PERMUTATION_ARGUMENT_RANGE_CHECK_COL_1];
    let p0_next = &next[PERMUTATION_ARGUMENT_RANGE_CHECK_COL_1];
    let p1 = &curr[PERMUTATION_ARGUMENT_RANGE_CHECK_COL_2];
    let p2 = &curr[PERMUTATION_ARGUMENT_RANGE_CHECK_COL_3];

    let ap0_next = &next[RANGE_CHECK_COL_1];
    let ap1 = &curr[RANGE_CHECK_COL_2];
    let ap2 = &curr[RANGE_CHECK_COL_3];

    let a0_next = &next[OFF_DST];
    let a1 = &curr[OFF_OP0];
    let a2 = &curr[OFF_OP1];

    constraints[RANGE_CHECK_0] = (z - ap1) * p1 - (z - a1) * p0;
    constraints[RANGE_CHECK_1] = (z - ap2) * p2 - (z - a2) * p1;
    constraints[RANGE_CHECK_2] = (z - ap0_next) * p0_next - (z - a0_next) * p2;
}
fn frame_inst_size(frame_row: &[FE]) -> FE {
    &frame_row[F_OP_1_VAL] + FE::one()
}

#[cfg(test)]
#[cfg(debug_assertions)]
mod test {
    use cairo_vm::{cairo_run, types::program::Program};
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };

    use crate::{
        air::{
            context::ProofOptions,
            debug::validate_trace,
            example::cairo::{
                add_program_in_public_input_section, CairoAIR, PublicInputs, OFF_DST, OFF_OP1,
            },
            trace::TraceTable,
            AIR,
        },
        cairo_run::run::Error,
        cairo_vm::{cairo_mem::CairoMemory, cairo_trace::CairoTrace},
        Domain,
    };

    use super::{
        add_missing_values_to_offsets_column, generate_memory_permutation_argument_column,
        get_missing_values_offset_columns, sort_columns_by_memory_address, CairoRAPChallenges,
    };

    #[test]
    fn check_simple_cairo_trace_evaluates_to_zero() {
        let base_dir = env!("CARGO_MANIFEST_DIR");

        let cairo_run_config = cairo_run::CairoRunConfig {
            entrypoint: "main",
            trace_enabled: true,
            relocate_mem: true,
            layout: "all_cairo",
            proof_mode: false,
            secure_run: None,
        };
        let json_filename = base_dir.to_owned() + "/src/cairo_vm/test_data/simple_program.json";
        let program_content = std::fs::read(json_filename).map_err(Error::IO).unwrap();
        let cairo_program =
            Program::from_bytes(&program_content, Some(cairo_run_config.entrypoint)).unwrap();

        let dir_trace = base_dir.to_owned() + "/src/cairo_vm/test_data/simple_program.trace";
        let dir_memory = base_dir.to_owned() + "/src/cairo_vm/test_data/simple_program.mem";

        let raw_trace = CairoTrace::from_file(&dir_trace).unwrap();
        let memory = CairoMemory::from_file(&dir_memory).unwrap();

        let mut program = Vec::new();
        for i in 1..=cairo_program.data_len() as u64 {
            program.push(memory.get(&i).unwrap().clone());
        }

        let proof_options = ProofOptions {
            blowup_factor: 4,
            fri_number_of_queries: 1,
            coset_offset: 3,
        };

        let cairo_air = CairoAIR::new(proof_options, 128, raw_trace.steps());

        // PC FINAL AND AP FINAL are not computed correctly since they are extracted after padding to
        // power of two and therefore are zero
        let last_register_state = &raw_trace.rows[raw_trace.steps() - 1];
        let mut public_input = PublicInputs {
            program,
            ap_final: FieldElement::from(last_register_state.ap),
            pc_final: FieldElement::from(last_register_state.pc),
            pc_init: FieldElement::from(raw_trace.rows[0].pc),
            ap_init: FieldElement::from(raw_trace.rows[0].ap),
            fp_init: FieldElement::from(raw_trace.rows[0].fp),
            range_check_max: None,
            range_check_min: None,
            num_steps: raw_trace.steps(),
        };

        let main_trace = cairo_air.build_main_trace(&(raw_trace, memory), &mut public_input);
        let mut trace_polys = main_trace.compute_trace_polys();
        let mut transcript = DefaultTranscript::new();
        let rap_challenges = cairo_air.build_rap_challenges(&mut transcript);

        let aux_trace =
            cairo_air.build_auxiliary_trace(&main_trace, &rap_challenges, &public_input);
        let aux_polys = aux_trace.compute_trace_polys();

        trace_polys.extend_from_slice(&aux_polys);

        let domain = Domain::new(&cairo_air);

        assert!(validate_trace(
            &cairo_air,
            &trace_polys,
            &domain,
            &public_input,
            &rap_challenges
        ));
    }

    #[test]
    fn test_build_auxiliary_trace_add_program_in_public_input_section_works() {
        let dummy_public_input = PublicInputs {
            pc_init: FieldElement::zero(),
            ap_init: FieldElement::zero(),
            fp_init: FieldElement::zero(),
            pc_final: FieldElement::zero(),
            ap_final: FieldElement::zero(),
            program: vec![
                FieldElement::from(10),
                FieldElement::from(20),
                FieldElement::from(30),
            ],
            range_check_max: None,
            range_check_min: None,
            num_steps: 1,
        };

        let a = vec![
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ];
        let v = vec![
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ];
        let (ap, vp) = add_program_in_public_input_section(&a, &v, &dummy_public_input);
        assert_eq!(
            ap,
            vec![
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::zero(),
                FieldElement::one(),
                FieldElement::from(2),
                FieldElement::from(3)
            ]
        );
        assert_eq!(
            vp,
            vec![
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::zero(),
                FieldElement::from(10),
                FieldElement::from(20),
                FieldElement::from(30)
            ]
        );
    }

    #[test]
    fn test_build_auxiliary_trace_sort_columns_by_memory_address() {
        let a = vec![
            FieldElement::from(2),
            FieldElement::one(),
            FieldElement::from(3),
            FieldElement::from(2),
        ];
        let v = vec![
            FieldElement::from(6),
            FieldElement::from(4),
            FieldElement::from(5),
            FieldElement::from(6),
        ];
        let (ap, vp) = sort_columns_by_memory_address(a, v);
        assert_eq!(
            ap,
            vec![
                FieldElement::one(),
                FieldElement::from(2),
                FieldElement::from(2),
                FieldElement::from(3)
            ]
        );
        assert_eq!(
            vp,
            vec![
                FieldElement::from(4),
                FieldElement::from(6),
                FieldElement::from(6),
                FieldElement::from(5),
            ]
        );
    }

    #[test]
    fn test_build_auxiliary_trace_generate_permutation_argument_column() {
        let a = vec![
            FieldElement::from(3),
            FieldElement::one(),
            FieldElement::from(2),
        ];
        let v = vec![
            FieldElement::from(5),
            FieldElement::one(),
            FieldElement::from(2),
        ];
        let ap = vec![
            FieldElement::one(),
            FieldElement::from(2),
            FieldElement::from(3),
        ];
        let vp = vec![
            FieldElement::one(),
            FieldElement::from(2),
            FieldElement::from(5),
        ];
        let rap_challenges = CairoRAPChallenges {
            alpha_memory: FieldElement::from(15),
            z_memory: FieldElement::from(10),
            z_range_check: FieldElement::zero(),
        };
        let p = generate_memory_permutation_argument_column(a, v, &ap, &vp, &rap_challenges);
        assert_eq!(
            p,
            vec![
                FieldElement::from_hex(
                    "2aaaaaaaaaaaab0555555555555555555555555555555555555555555555561"
                )
                .unwrap(),
                FieldElement::from_hex(
                    "1745d1745d174602e8ba2e8ba2e8ba2e8ba2e8ba2e8ba2e8ba2e8ba2e8ba2ec"
                )
                .unwrap(),
                FieldElement::one(),
            ]
        );
    }

    #[test]
    fn test_fill_range_check_values() {
        let columns = vec![
            vec![FieldElement::from(1); 3],
            vec![FieldElement::from(4); 3],
            vec![FieldElement::from(7); 3],
        ];
        let expected_col = vec![
            FieldElement::from(2),
            FieldElement::from(3),
            FieldElement::from(5),
            FieldElement::from(6),
            FieldElement::from(7),
            FieldElement::from(7),
        ];
        let table = TraceTable::<Stark252PrimeField>::new_from_cols(&columns);

        let (col, rc_min, rc_max) = get_missing_values_offset_columns(&table, &[0, 1, 2]);
        assert_eq!(col, expected_col);
        assert_eq!(rc_min, 1);
        assert_eq!(rc_max, 7);
    }

    #[test]
    fn test_add_missing_values_to_offsets_column() {
        let mut main_trace = TraceTable::<Stark252PrimeField> {
            table: (0..34 * 2).map(FieldElement::from).collect(),
            n_cols: 34,
        };
        let missing_values = vec![
            FieldElement::from(1),
            FieldElement::from(2),
            FieldElement::from(3),
            FieldElement::from(4),
            FieldElement::from(5),
            FieldElement::from(6),
        ];
        add_missing_values_to_offsets_column(&mut main_trace, missing_values);

        let mut expected: Vec<_> = (0..34 * 2).map(FieldElement::from).collect();
        expected.append(&mut vec![FieldElement::zero(); OFF_DST]);
        expected.append(&mut vec![
            FieldElement::from(1),
            FieldElement::from(2),
            FieldElement::from(3),
        ]);
        expected.append(&mut vec![FieldElement::zero(); 34 - OFF_OP1 - 1]);
        expected.append(&mut vec![FieldElement::zero(); OFF_DST]);
        expected.append(&mut vec![
            FieldElement::from(4),
            FieldElement::from(5),
            FieldElement::from(6),
        ]);
        expected.append(&mut vec![FieldElement::zero(); 34 - OFF_OP1 - 1]);
        assert_eq!(main_trace.table, expected);
        assert_eq!(main_trace.n_cols, 34);
        assert_eq!(main_trace.table.len(), 34 * 4);
    }
}
