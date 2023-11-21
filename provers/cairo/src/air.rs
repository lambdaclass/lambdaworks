use std::ops::Range;

use cairo_vm::without_std::collections::HashMap;
use lambdaworks_math::{
    errors::DeserializationError,
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
    traits::{ByteConversion, Deserializable, Serializable},
};
use stark_platinum_prover::{
    constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
    context::AirContext,
    frame::Frame,
    proof::{options::ProofOptions, stark::StarkProof},
    prover::{IsStarkProver, Prover, ProvingError},
    trace::{StepView, TraceTable},
    traits::AIR,
    transcript::{IsStarkTranscript, StoneProverTranscript},
    verifier::{IsStarkVerifier, Verifier},
};

use crate::Felt252;
use stark_platinum_prover::table::Table;

use super::{cairo_mem::CairoMemory, register_states::RegisterStates};

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
const MEMORY_INCREASING_4: usize = 35;

const MEMORY_CONSISTENCY_0: usize = 36;
const MEMORY_CONSISTENCY_1: usize = 37;
const MEMORY_CONSISTENCY_2: usize = 38;
const MEMORY_CONSISTENCY_3: usize = 39;
const MEMORY_CONSISTENCY_4: usize = 40;

const PERMUTATION_ARGUMENT_0: usize = 41;
const PERMUTATION_ARGUMENT_1: usize = 42;
const PERMUTATION_ARGUMENT_2: usize = 43;
const PERMUTATION_ARGUMENT_3: usize = 44;
const PERMUTATION_ARGUMENT_4: usize = 45;

const RANGE_CHECK_INCREASING_0: usize = 46;
const RANGE_CHECK_INCREASING_1: usize = 47;
const RANGE_CHECK_INCREASING_2: usize = 48;
const RANGE_CHECK_INCREASING_3: usize = 49;

const RANGE_CHECK_0: usize = 50;
const RANGE_CHECK_1: usize = 51;
const RANGE_CHECK_2: usize = 52;
const RANGE_CHECK_3: usize = 53;

const FLAG_OP1_BASE_OP0_BIT: usize = 54;
const FLAG_RES_OP1_BIT: usize = 55;
const FLAG_PC_UPDATE_REGULAR_BIT: usize = 56;
const FLAG_FP_UPDATE_REGULAR_BIT: usize = 57;
const OPCODES_CALL_OFF0: usize = 58;

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
pub const EXTRA_ADDR: usize = 33;
pub const EXTRA_VAL: usize = 34;
pub const RC_HOLES: usize = 35;

// Auxiliary range check columns
pub const RANGE_CHECK_COL_1: usize = 36;
pub const RANGE_CHECK_COL_2: usize = 37;
pub const RANGE_CHECK_COL_3: usize = 38;
pub const RANGE_CHECK_COL_4: usize = 39;

// Auxiliary memory columns
pub const MEMORY_ADDR_SORTED_0: usize = 40;
pub const MEMORY_ADDR_SORTED_1: usize = 41;
pub const MEMORY_ADDR_SORTED_2: usize = 42;
pub const MEMORY_ADDR_SORTED_3: usize = 43;
pub const MEMORY_ADDR_SORTED_4: usize = 44;

pub const MEMORY_VALUES_SORTED_0: usize = 45;
pub const MEMORY_VALUES_SORTED_1: usize = 46;
pub const MEMORY_VALUES_SORTED_2: usize = 47;
pub const MEMORY_VALUES_SORTED_3: usize = 48;
pub const MEMORY_VALUES_SORTED_4: usize = 49;

pub const PERMUTATION_ARGUMENT_COL_0: usize = 50;
pub const PERMUTATION_ARGUMENT_COL_1: usize = 51;
pub const PERMUTATION_ARGUMENT_COL_2: usize = 52;
pub const PERMUTATION_ARGUMENT_COL_3: usize = 53;
pub const PERMUTATION_ARGUMENT_COL_4: usize = 54;

pub const PERMUTATION_ARGUMENT_RANGE_CHECK_COL_1: usize = 55;
pub const PERMUTATION_ARGUMENT_RANGE_CHECK_COL_2: usize = 56;
pub const PERMUTATION_ARGUMENT_RANGE_CHECK_COL_3: usize = 57;
pub const PERMUTATION_ARGUMENT_RANGE_CHECK_COL_4: usize = 58;

// Trace layout
pub const MEM_P_TRACE_OFFSET: usize = 17;
pub const MEM_A_TRACE_OFFSET: usize = 19;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MemorySegment {
    RangeCheck,
    Output,
}

pub type MemorySegmentMap = HashMap<MemorySegment, Range<u64>>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PublicInputs {
    pub pc_init: Felt252,
    pub ap_init: Felt252,
    pub fp_init: Felt252,
    pub pc_final: Felt252,
    pub ap_final: Felt252,
    // These are Option because they're not known until
    // the trace is obtained. They represent the minimum
    // and maximum offsets used during program execution.
    // TODO: A possible refactor is moving them to the proof.
    // minimum range check value (0 < range_check_min < range_check_max < 2^16)
    pub range_check_min: Option<u16>,
    // maximum range check value
    pub range_check_max: Option<u16>,
    // Range-check builtin address range
    pub memory_segments: MemorySegmentMap,
    pub public_memory: HashMap<Felt252, Felt252>,
    pub num_steps: usize, // number of execution steps
    pub codelen: usize,   // length of the program segment
}

impl PublicInputs {
    /// Creates a Public Input from register states and memory
    /// - In the future we should use the output of the Cairo Runner. This is not currently supported in Cairo RS
    /// - RangeChecks are not filled, and the prover mutates them inside the prove function. This works but also should be loaded from the Cairo RS output
    pub fn from_regs_and_mem(
        register_states: &RegisterStates,
        memory: &CairoMemory,
        codelen: usize,
    ) -> Self {
        let public_memory = (1..=codelen as u64)
            .map(|i| (Felt252::from(i), *memory.get(&i).unwrap()))
            .collect::<HashMap<Felt252, Felt252>>();

        let last_step = &register_states.rows[register_states.steps() - 1];

        PublicInputs {
            pc_init: Felt252::from(register_states.rows[0].pc),
            ap_init: Felt252::from(register_states.rows[0].ap),
            fp_init: Felt252::from(register_states.rows[0].fp),
            pc_final: FieldElement::from(last_step.pc),
            ap_final: FieldElement::from(last_step.ap),
            range_check_min: None,
            range_check_max: None,
            memory_segments: MemorySegmentMap::new(),
            public_memory,
            num_steps: register_states.steps(),
            codelen,
        }
    }
}

impl Serializable for PublicInputs {
    fn serialize(&self) -> Vec<u8> {
        let mut bytes = vec![];
        let pc_init_bytes = self.pc_init.to_bytes_be();
        let felt_length = pc_init_bytes.len();
        bytes.extend(felt_length.to_be_bytes());
        bytes.extend(pc_init_bytes);
        bytes.extend(self.ap_init.to_bytes_be());
        bytes.extend(self.fp_init.to_bytes_be());
        bytes.extend(self.pc_final.to_bytes_be());
        bytes.extend(self.ap_final.to_bytes_be());

        if let Some(range_check_min) = self.range_check_min {
            bytes.extend(1u8.to_be_bytes());
            bytes.extend(range_check_min.to_be_bytes());
        } else {
            bytes.extend(0u8.to_be_bytes());
        }

        if let Some(range_check_max) = self.range_check_max {
            bytes.extend(1u8.to_be_bytes());
            bytes.extend(range_check_max.to_be_bytes());
        } else {
            bytes.extend(0u8.to_be_bytes());
        }

        let mut memory_segment_bytes = vec![];
        for (segment, range) in self.memory_segments.iter() {
            let segment_type = match segment {
                MemorySegment::RangeCheck => 0u8,
                MemorySegment::Output => 1u8,
            };
            memory_segment_bytes.extend(segment_type.to_be_bytes());
            memory_segment_bytes.extend(range.start.to_be_bytes());
            memory_segment_bytes.extend(range.end.to_be_bytes());
        }
        let memory_segment_length = self.memory_segments.len();
        bytes.extend(memory_segment_length.to_be_bytes());
        bytes.extend(memory_segment_bytes);

        let mut public_memory_bytes = vec![];
        for (address, value) in self.public_memory.iter() {
            public_memory_bytes.extend(address.to_bytes_be());
            public_memory_bytes.extend(value.to_bytes_be());
        }
        let public_memory_length = self.public_memory.len();
        bytes.extend(public_memory_length.to_be_bytes());
        bytes.extend(public_memory_bytes);

        bytes.extend(self.num_steps.to_be_bytes());
        bytes.extend(self.codelen.to_be_bytes());

        bytes
    }
}

impl Deserializable for PublicInputs {
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        let mut bytes = bytes;
        let felt_len = usize::from_be_bytes(
            bytes
                .get(0..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );
        bytes = &bytes[8..];
        let pc_init = Felt252::from_bytes_be(
            bytes
                .get(..felt_len)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?,
        )?;
        bytes = &bytes[felt_len..];
        let ap_init = Felt252::from_bytes_be(
            bytes
                .get(..felt_len)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?,
        )?;
        bytes = &bytes[felt_len..];
        let fp_init = Felt252::from_bytes_be(
            bytes
                .get(..felt_len)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?,
        )?;
        bytes = &bytes[felt_len..];
        let pc_final = Felt252::from_bytes_be(
            bytes
                .get(..felt_len)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?,
        )?;
        bytes = &bytes[felt_len..];
        let ap_final = Felt252::from_bytes_be(
            bytes
                .get(..felt_len)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?,
        )?;
        bytes = &bytes[felt_len..];

        if bytes.is_empty() {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }
        let range_check_min = match bytes[0] {
            0 => {
                bytes = &bytes[1..];
                None
            }
            1 => {
                bytes = &bytes[1..];
                let range_check_min = u16::from_be_bytes(
                    bytes[..2]
                        .try_into()
                        .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
                );
                bytes = &bytes[2..];
                Some(range_check_min)
            }
            _ => return Err(DeserializationError::FieldFromBytesError),
        };

        if bytes.is_empty() {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }
        let range_check_max = match bytes[0] {
            0 => {
                bytes = &bytes[1..];
                None
            }
            1 => {
                bytes = &bytes[1..];
                let range_check_max = u16::from_be_bytes(
                    bytes[..2]
                        .try_into()
                        .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
                );
                bytes = &bytes[2..];
                Some(range_check_max)
            }
            _ => return Err(DeserializationError::FieldFromBytesError),
        };

        let mut memory_segments = MemorySegmentMap::new();
        let memory_segment_length = usize::from_be_bytes(
            bytes
                .get(0..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );
        bytes = &bytes[8..];
        for _ in 0..memory_segment_length {
            if bytes.is_empty() {
                return Err(DeserializationError::InvalidAmountOfBytes);
            }
            let segment_type = match bytes[0] {
                0 => MemorySegment::RangeCheck,
                1 => MemorySegment::Output,
                _ => return Err(DeserializationError::FieldFromBytesError),
            };
            bytes = &bytes[1..];
            let start = u64::from_be_bytes(
                bytes
                    .get(0..8)
                    .ok_or(DeserializationError::InvalidAmountOfBytes)?
                    .try_into()
                    .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
            );
            bytes = &bytes[8..];
            let end = u64::from_be_bytes(
                bytes
                    .get(0..8)
                    .ok_or(DeserializationError::InvalidAmountOfBytes)?
                    .try_into()
                    .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
            );
            bytes = &bytes[8..];
            memory_segments.insert(segment_type, start..end);
        }

        let mut public_memory = HashMap::new();
        let public_memory_length = usize::from_be_bytes(
            bytes
                .get(0..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );
        bytes = &bytes[8..];
        for _ in 0..public_memory_length {
            let address = Felt252::from_bytes_be(
                bytes
                    .get(..felt_len)
                    .ok_or(DeserializationError::InvalidAmountOfBytes)?,
            )?;
            bytes = &bytes[felt_len..];
            let value = Felt252::from_bytes_be(
                bytes
                    .get(..felt_len)
                    .ok_or(DeserializationError::InvalidAmountOfBytes)?,
            )?;
            bytes = &bytes[felt_len..];
            public_memory.insert(address, value);
        }

        let num_steps = usize::from_be_bytes(
            bytes
                .get(0..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );

        let codelen = usize::from_be_bytes(
            bytes
                .get(0..8)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );

        Ok(Self {
            pc_init,
            ap_init,
            fp_init,
            pc_final,
            ap_final,
            range_check_min,
            range_check_max,
            memory_segments,
            public_memory,
            num_steps,
            codelen,
        })
    }
}

#[derive(Clone)]
pub struct CairoAIR {
    pub context: AirContext,
    pub trace_length: usize,
    pub pub_inputs: PublicInputs,
}

pub struct CairoRAPChallenges {
    pub alpha_memory: FieldElement<Stark252PrimeField>,
    pub z_memory: FieldElement<Stark252PrimeField>,
    pub z_range_check: FieldElement<Stark252PrimeField>,
}

/// Receives two slices corresponding to the accessed addresses and values, filled with
/// the memory holes and with the (0, 0) public memory dummy accesses.
/// Each (address, value) public memory pair is written in a (0, 0) dummy access until
/// there is no one left.
///
/// NOTE: At the end of this process there might be some additional (0, 0) dummy accesses
/// that were not overwritten. This is not a problem as long as all the public memory pairs
/// have been written.
fn add_pub_memory_in_public_input_section(
    addresses: &[Felt252],
    values: &[Felt252],
    public_input: &PublicInputs,
) -> (Vec<Felt252>, Vec<Felt252>) {
    let mut a_aux = addresses.to_owned();
    let mut v_aux = values.to_owned();

    let output_range = public_input.memory_segments.get(&MemorySegment::Output);

    let pub_addrs = get_pub_memory_addrs(output_range, public_input);
    let mut pub_addrs_iter = pub_addrs.iter();

    // Iterate over addresses
    for (i, a) in a_aux.iter_mut().enumerate() {
        // When address `0` is found, it means it corresponds to a dummy access.
        if a == &Felt252::zero() {
            // While there are public memory addresses left, overwrite the dummy
            // (addr, value) accesses with the real public memory pairs.
            if let Some(pub_addr) = pub_addrs_iter.next() {
                *a = *pub_addr;
                v_aux[i] = *public_input.public_memory.get(pub_addr).unwrap();
            } else {
                // When there are no public memory pairs left to write, break the
                // loop and return the (addr, value) pairs with dummy accesses
                // overwritten.
                break;
            }
        }
    }

    (a_aux, v_aux)
}

/// Gets public memory addresses of a program. First, this function builds a `Vec` of `FieldElement`s, filling it
/// incrementally with addresses from `1` to `program_len - 1`, where `program_len` is the length of the program.
/// If the output builtin is used, `output_range` is `Some(...)` and this function adds incrementally to the resulting
/// `Vec` addresses from the start to the end of the unwrapped `output_range`.
fn get_pub_memory_addrs(
    output_range: Option<&Range<u64>>,
    public_input: &PublicInputs,
) -> Vec<FieldElement<Stark252PrimeField>> {
    let public_memory_len = public_input.public_memory.len() as u64;

    if let Some(output_range) = output_range {
        let output_section = output_range.end - output_range.start;
        let program_section = public_memory_len - output_section;

        (1..=program_section)
            .map(FieldElement::from)
            .chain(output_range.clone().map(FieldElement::from))
            .collect()
    } else {
        (1..=public_memory_len).map(FieldElement::from).collect()
    }
}

fn sort_columns_by_memory_address(
    adresses: Vec<Felt252>,
    values: Vec<Felt252>,
) -> (Vec<Felt252>, Vec<Felt252>) {
    let mut tuples: Vec<_> = adresses.into_iter().zip(values).collect();
    tuples.sort_by(|(x, _), (y, _)| x.representative().cmp(&y.representative()));
    tuples.into_iter().unzip()
}

fn generate_memory_permutation_argument_column(
    addresses_original: Vec<Felt252>,
    values_original: Vec<Felt252>,
    addresses_sorted: &[Felt252],
    values_sorted: &[Felt252],
    rap_challenges: &CairoRAPChallenges,
) -> Vec<Felt252> {
    let z = &rap_challenges.z_memory;
    let alpha = &rap_challenges.alpha_memory;

    let mut denom: Vec<_> = addresses_sorted
        .iter()
        .zip(values_sorted)
        .map(|(ap, vp)| z - (ap + alpha * vp))
        .collect();
    FieldElement::inplace_batch_inverse(&mut denom).unwrap();
    // Returns the cumulative products of the numerators and denominators
    addresses_original
        .iter()
        .zip(&values_original)
        .zip(&denom)
        .scan(Felt252::one(), |product, ((a_i, v_i), den_i)| {
            let ret = *product;
            *product = ret * ((z - (a_i + alpha * v_i)) * den_i);
            Some(*product)
        })
        .collect::<Vec<Felt252>>()
}

fn generate_range_check_permutation_argument_column(
    offset_column_original: &[Felt252],
    offset_column_sorted: &[Felt252],
    rap_challenges: &CairoRAPChallenges,
) -> Vec<Felt252> {
    let z = &rap_challenges.z_range_check;

    let mut denom: Vec<_> = offset_column_sorted.iter().map(|x| z - x).collect();
    FieldElement::inplace_batch_inverse(&mut denom).unwrap();

    offset_column_original
        .iter()
        .zip(&denom)
        .scan(Felt252::one(), |product, (num_i, den_i)| {
            let ret = *product;
            *product = ret * (z - num_i) * den_i;
            Some(*product)
        })
        .collect::<Vec<Felt252>>()
}

impl AIR for CairoAIR {
    type Field = Stark252PrimeField;
    type RAPChallenges = CairoRAPChallenges;
    type PublicInputs = PublicInputs;

    const STEP_SIZE: usize = 1;

    /// Creates a new CairoAIR from proof_options
    ///
    /// # Arguments
    ///
    /// * `trace_length` - Length of the Cairo execution trace. Must be a power fo two.
    /// * `pub_inputs` - Public inputs sent by the Cairo runner.
    /// * `proof_options` - STARK proving configuration options.
    #[rustfmt::skip]
    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions
    ) -> Self {
        debug_assert!(trace_length.is_power_of_two());

        let trace_columns = 59;
        let transition_degrees = vec![
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // Flags 0-14.
            1, // Flag 15
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // Other constraints.
            2, 2, 2, 2, 2, // Increasing memory auxiliary constraints.
            2, 2, 2, 2, 2, // Consistent memory auxiliary constraints.
            2, 2, 2, 2, 2, // Permutation auxiliary constraints.
            2, 2, 2, 2, // range-check increasing constraints.
            2, 2, 2, 2, // range-check permutation argument constraints.
            2, // f_op1_imm_bit constraint
            2, // flag_res_op1_bit constraint
            2, // flag_pc_update_regular_bit constraint
            2, // flag_fp_update_regular_bit constraint
            2, // opcodes/call/off0 constraint
        ];
        let transition_exemptions = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // flags (16)
            0, // inst (1)
            0, 0, 0, // operand consraints (3)
            1, 1, 1, 1, 0, 0, // register constraints (6)
            0, 0, 0, 0, 0, // opcode constraints (5)
            0, 0, 0, 0, 1, // memory continuous (4)
            0, 0, 0, 0, 1, // memory value consistency (4)
            0, 0, 0, 0, 1, // memory permutation argument (4)
            0, 0, 0, 1, // range check continuous (3)
            0, 0, 0, 0, // range check permutation argument (3)
            0, // f_op1_imm_bit constraint
            0, // flag_res_op1_bit constraint
            0, // flag_pc_update_regular_bit constraint
            0, // flag_fp_update_regular_bit constraint
            0, // opcodes/call/off0 constraint
        ];
        let num_transition_constraints = 59;

        let num_transition_exemptions = 1_usize;

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns,
            transition_degrees,
            transition_exemptions,
            transition_offsets: vec![0, 1],
            num_transition_constraints,
            num_transition_exemptions,
        };

        // The number of the transition constraints and the lengths of transition degrees
        // and transition exemptions should be the same always.
        debug_assert_eq!(
            context.transition_degrees.len(),
            context.num_transition_constraints
        );
        debug_assert_eq!(
            context.transition_exemptions.len(),
            context.num_transition_constraints
        );

        Self {
            context,
            pub_inputs: pub_inputs.clone(),
            trace_length,
        }
    }

    fn build_auxiliary_trace(
        &self,
        main_trace: &TraceTable<Self::Field>,
        rap_challenges: &Self::RAPChallenges,
    ) -> TraceTable<Self::Field> {
        let addresses_original = main_trace.merge_columns(&[
            FRAME_PC,
            FRAME_DST_ADDR,
            FRAME_OP0_ADDR,
            FRAME_OP1_ADDR,
            EXTRA_ADDR,
        ]);

        let values_original =
            main_trace.merge_columns(&[FRAME_INST, FRAME_DST, FRAME_OP0, FRAME_OP1, EXTRA_VAL]);

        let (addresses, values) = add_pub_memory_in_public_input_section(
            &addresses_original,
            &values_original,
            &self.pub_inputs,
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
        let offsets_original = main_trace.merge_columns(&[OFF_DST, OFF_OP0, OFF_OP1, RC_HOLES]);

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
        let mut aux_data = Vec::new();
        for i in 0..main_trace.n_rows() {
            aux_data.push(offsets_sorted[4 * i]);
            aux_data.push(offsets_sorted[4 * i + 1]);
            aux_data.push(offsets_sorted[4 * i + 2]);
            aux_data.push(offsets_sorted[4 * i + 3]);
            aux_data.push(addresses[5 * i]);
            aux_data.push(addresses[5 * i + 1]);
            aux_data.push(addresses[5 * i + 2]);
            aux_data.push(addresses[5 * i + 3]);
            aux_data.push(addresses[5 * i + 4]);
            aux_data.push(values[5 * i]);
            aux_data.push(values[5 * i + 1]);
            aux_data.push(values[5 * i + 2]);
            aux_data.push(values[5 * i + 3]);
            aux_data.push(values[5 * i + 4]);
            aux_data.push(permutation_col[5 * i]);
            aux_data.push(permutation_col[5 * i + 1]);
            aux_data.push(permutation_col[5 * i + 2]);
            aux_data.push(permutation_col[5 * i + 3]);
            aux_data.push(permutation_col[5 * i + 4]);
            aux_data.push(range_check_permutation_col[4 * i]);
            aux_data.push(range_check_permutation_col[4 * i + 1]);
            aux_data.push(range_check_permutation_col[4 * i + 2]);
            aux_data.push(range_check_permutation_col[4 * i + 3]);
        }

        let aux_table = Table::new(aux_data, self.number_auxiliary_rap_columns());

        TraceTable {
            table: aux_table,
            step_size: Self::STEP_SIZE,
        }
    }

    fn build_rap_challenges(
        &self,
        transcript: &mut impl IsStarkTranscript<Self::Field>,
    ) -> Self::RAPChallenges {
        CairoRAPChallenges {
            alpha_memory: transcript.sample_field_element(),
            z_memory: transcript.sample_field_element(),
            z_range_check: transcript.sample_field_element(),
        }
    }

    fn number_auxiliary_rap_columns(&self) -> usize {
        // RANGE_CHECK_COL_i + MEMORY_INCREASING_i + MEMORY_CONSISTENCY_i + PERMUTATION_ARGUMENT_COL_i +
        // + PERMUTATION_ARGUMENT_RANGE_CHECK_COL_i
        23
    }

    fn compute_transition(
        &self,
        frame: &Frame<Self::Field>,
        rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Self::Field>> {
        let mut constraints: Vec<FieldElement<Self::Field>> =
            vec![Felt252::zero(); self.num_transition_constraints()];

        compute_instr_constraints(&mut constraints, frame);
        compute_operand_constraints(&mut constraints, frame);
        compute_register_constraints(&mut constraints, frame);
        compute_opcode_constraints(&mut constraints, frame);
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
    ) -> BoundaryConstraints<Self::Field> {
        let initial_pc = BoundaryConstraint::new(MEM_A_TRACE_OFFSET, 0, self.pub_inputs.pc_init);
        let initial_ap = BoundaryConstraint::new(MEM_P_TRACE_OFFSET, 0, self.pub_inputs.ap_init);

        let final_pc = BoundaryConstraint::new(
            MEM_A_TRACE_OFFSET,
            self.pub_inputs.num_steps - 1,
            self.pub_inputs.pc_final,
        );
        let final_ap = BoundaryConstraint::new(
            MEM_P_TRACE_OFFSET,
            self.pub_inputs.num_steps - 1,
            self.pub_inputs.ap_final,
        );

        // Auxiliary constraint: permutation argument final value
        let final_index = self.trace_length - 1;

        let cumulative_product = self
            .pub_inputs
            .public_memory
            .iter()
            .fold(FieldElement::one(), |product, (address, value)| {
                product
                    * (rap_challenges.z_memory - (address + rap_challenges.alpha_memory * value))
            })
            .inv()
            .unwrap();

        let permutation_final = rap_challenges
            .z_memory
            .pow(self.pub_inputs.public_memory.len())
            * cumulative_product;

        let permutation_final_constraint =
            BoundaryConstraint::new(PERMUTATION_ARGUMENT_COL_4, final_index, permutation_final);

        let one: FieldElement<Self::Field> = FieldElement::one();
        let range_check_final_constraint =
            BoundaryConstraint::new(PERMUTATION_ARGUMENT_RANGE_CHECK_COL_4, final_index, one);

        let range_check_min = BoundaryConstraint::new(
            RANGE_CHECK_COL_1,
            0,
            FieldElement::from(self.pub_inputs.range_check_min.unwrap() as u64),
        );

        let range_check_max = BoundaryConstraint::new(
            RANGE_CHECK_COL_4,
            final_index,
            FieldElement::from(self.pub_inputs.range_check_max.unwrap() as u64),
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

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self) -> usize {
        2 * self.trace_length
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }
}

/// From the Cairo whitepaper, section 9.10
fn compute_instr_constraints(constraints: &mut [Felt252], frame: &Frame<Stark252PrimeField>) {
    // These constraints are only applied over elements of the same row.
    let curr = frame.get_evaluation_step(0);

    // Bit-prefixes constraints.
    // See section 9.4 of Cairo whitepaper https://eprint.iacr.org/2021/1063.pdf.
    let flags: Vec<&Felt252> = (0..16)
        .map(|col_idx| curr.get_evaluation_element(0, col_idx))
        .collect();

    let one = Felt252::one();
    let two = Felt252::from(2);

    let bit_flags: Vec<Felt252> = (0..15)
        .map(|idx| flags[idx] - two * flags[idx + 1])
        .collect();

    (0..15).for_each(|idx| {
        constraints[idx] = match idx {
            0..=14 => bit_flags[idx] * (bit_flags[idx] - one),
            15 => *flags[idx],
            _ => panic!("Unknown flag offset"),
        }
    });

    // flag_op1_base_op0_bit constraint
    let f_op1_imm = bit_flags[2];
    let f_op1_fp = bit_flags[3];
    let f_op1_ap = bit_flags[4];
    let f_op1_base_op0_bit = one - f_op1_imm - f_op1_fp - f_op1_ap;
    constraints[FLAG_OP1_BASE_OP0_BIT] = f_op1_base_op0_bit * (f_op1_base_op0_bit - one);

    // flag_res_op1_bit constraint
    let f_res_add = bit_flags[5];
    let f_res_mul = bit_flags[6];
    let f_pc_jnz = bit_flags[9];
    let f_res_op1_bit = one - f_res_add - f_res_mul - f_pc_jnz;
    constraints[FLAG_RES_OP1_BIT] = f_res_op1_bit * (f_res_op1_bit - one);

    // flag_pc_update_regular_bit constraint
    let f_jump_abs = bit_flags[7];
    let f_jump_rel = bit_flags[8];
    let flag_pc_update_regular_bit = one - f_jump_abs - f_jump_rel - f_pc_jnz;
    constraints[FLAG_PC_UPDATE_REGULAR_BIT] =
        flag_pc_update_regular_bit * (flag_pc_update_regular_bit - one);

    // flag_fp_update_regular_bit constraint
    let f_opcode_call = bit_flags[12];
    let f_opcode_ret = bit_flags[13];
    let flag_fp_update_regular_bit = one - f_opcode_call - f_opcode_ret;
    constraints[FLAG_FP_UPDATE_REGULAR_BIT] =
        flag_fp_update_regular_bit * (flag_fp_update_regular_bit - one);

    // Instruction unpacking
    let b15 = two.pow(15u32);
    let b16 = two.pow(16u32);
    let b32 = two.pow(32u32);
    let b48 = two.pow(48u32);

    // Named like this to match the Cairo whitepaper's notation.
    let f0_squiggle = flags[0];

    let off_dst = curr.get_evaluation_element(0, OFF_DST);
    let off_op0 = curr.get_evaluation_element(0, OFF_OP0);
    let off_op1 = curr.get_evaluation_element(0, OFF_OP1);
    let instruction = curr.get_evaluation_element(0, FRAME_INST);

    constraints[INST] = off_dst + b16 * off_op0 + b32 * off_op1 + b48 * f0_squiggle - instruction;

    // cpu/opcodes/call/off0 constraint
    constraints[OPCODES_CALL_OFF0] = f_opcode_call * (off_dst - b15);
}

fn compute_operand_constraints(constraints: &mut [Felt252], frame: &Frame<Stark252PrimeField>) {
    // These constraints are only applied over elements of the same row.
    let curr = frame.get_evaluation_step(0);

    let ap = curr.get_evaluation_element(0, FRAME_AP);
    let fp = curr.get_evaluation_element(0, FRAME_FP);
    let pc = curr.get_evaluation_element(0, FRAME_PC);

    let dst_fp = into_bit_flag(curr, F_DST_FP);
    let off_dst = curr.get_evaluation_element(0, OFF_DST);
    let dst_addr = curr.get_evaluation_element(0, FRAME_DST_ADDR);

    let op0_fp = into_bit_flag(curr, F_OP_0_FP);
    let off_op0 = curr.get_evaluation_element(0, OFF_OP0);
    let op0_addr = curr.get_evaluation_element(0, FRAME_OP0_ADDR);

    let op1_val = into_bit_flag(curr, F_OP_1_VAL);
    let op1_ap = into_bit_flag(curr, F_OP_1_AP);
    let op1_fp = into_bit_flag(curr, F_OP_1_FP);
    let op0 = curr.get_evaluation_element(0, FRAME_OP0);
    let off_op1 = curr.get_evaluation_element(0, OFF_OP1);
    let op1_addr = curr.get_evaluation_element(0, FRAME_OP1_ADDR);

    let one = Felt252::one();
    let b15 = Felt252::from(2).pow(15u32);

    constraints[DST_ADDR] = dst_fp * fp + (one - dst_fp) * ap + (off_dst - b15) - dst_addr;

    constraints[OP0_ADDR] = op0_fp * fp + (one - op0_fp) * ap + (off_op0 - b15) - op0_addr;

    constraints[OP1_ADDR] = op1_val * pc
        + op1_ap * ap
        + op1_fp * fp
        + (one - op1_val - op1_ap - op1_fp) * op0
        + (off_op1 - b15)
        - op1_addr;
}

/// Given a step and the index of the bit-prefix format flag, gives the bit representation
/// of that flag, needed for the evaluation of some constraints.
#[inline(always)]
fn into_bit_flag(step: &StepView<Stark252PrimeField>, element_idx: usize) -> Felt252 {
    step.get_evaluation_element(0, element_idx)
        - Felt252::from(2) * step.get_evaluation_element(0, element_idx + 1)
}

fn compute_register_constraints(constraints: &mut [Felt252], frame: &Frame<Stark252PrimeField>) {
    let curr = frame.get_evaluation_step(0);
    let next = frame.get_evaluation_step(1);

    let one = Felt252::one();
    let two = Felt252::from(2);

    let ap = curr.get_evaluation_element(0, FRAME_AP);
    let next_ap = next.get_evaluation_element(0, FRAME_AP);
    let ap_add = into_bit_flag(curr, F_AP_ADD);
    let res = curr.get_evaluation_element(0, FRAME_RES);
    let ap_one = into_bit_flag(curr, F_AP_ONE);

    let opc_ret = into_bit_flag(curr, F_OPC_RET);
    let opc_call = into_bit_flag(curr, F_OPC_CALL);
    let dst = curr.get_evaluation_element(0, FRAME_DST);
    let fp = curr.get_evaluation_element(0, FRAME_FP);
    let next_fp = next.get_evaluation_element(0, FRAME_FP);

    let t1 = curr.get_evaluation_element(0, FRAME_T1);
    let pc_jnz = into_bit_flag(curr, F_PC_JNZ);
    let pc = curr.get_evaluation_element(0, FRAME_PC);
    let next_pc = next.get_evaluation_element(0, FRAME_PC);

    let t0 = curr.get_evaluation_element(0, FRAME_T0);
    let op1 = curr.get_evaluation_element(0, FRAME_OP1);
    let pc_abs = into_bit_flag(curr, F_PC_ABS);
    let pc_rel = into_bit_flag(curr, F_PC_REL);

    // ap and fp constraints
    constraints[NEXT_AP] = ap + ap_add * res + ap_one + opc_call * two - next_ap;

    constraints[NEXT_FP] =
        opc_ret * dst + opc_call * (ap + two) + (one - opc_ret - opc_call) * fp - next_fp;

    // pc constraints
    constraints[NEXT_PC_1] = (t1 - pc_jnz) * (next_pc - (pc + frame_inst_size(curr)));

    constraints[NEXT_PC_2] = t0 * (next_pc - (pc + op1)) + (one - pc_jnz) * next_pc
        - ((one - pc_abs - pc_rel - pc_jnz) * (pc + frame_inst_size(curr))
            + pc_abs * res
            + pc_rel * (pc + res));

    constraints[T0] = pc_jnz * dst - t0;
    constraints[T1] = t0 * res - t1;
}

fn compute_opcode_constraints(constraints: &mut [Felt252], frame: &Frame<Stark252PrimeField>) {
    let curr = frame.get_evaluation_step(0);
    let one = Felt252::one();

    let mul = curr.get_evaluation_element(0, FRAME_MUL);
    let op0 = curr.get_evaluation_element(0, FRAME_OP0);
    let op1 = curr.get_evaluation_element(0, FRAME_OP1);

    let res_add = into_bit_flag(curr, F_RES_ADD);
    let res_mul = into_bit_flag(curr, F_RES_MUL);
    let pc_jnz = into_bit_flag(curr, F_PC_JNZ);
    let res = curr.get_evaluation_element(0, FRAME_RES);

    let opc_call = into_bit_flag(curr, F_OPC_CALL);
    let dst = curr.get_evaluation_element(0, FRAME_DST);
    let fp = curr.get_evaluation_element(0, FRAME_FP);
    let pc = curr.get_evaluation_element(0, FRAME_PC);

    let opc_aeq = into_bit_flag(curr, F_OPC_AEQ);

    constraints[MUL_1] = mul - op0 * op1;

    constraints[MUL_2] =
        res_add * (op0 + op1) + res_mul * mul + (one - res_add - res_mul - pc_jnz) * op1
            - (one - pc_jnz) * res;

    constraints[CALL_1] = opc_call * (dst - fp);

    constraints[CALL_2] = opc_call * (op0 - (pc + frame_inst_size(curr)));

    constraints[ASSERT_EQ] = opc_aeq * (dst - res);
}

fn memory_is_increasing(constraints: &mut [Felt252], frame: &Frame<Stark252PrimeField>) {
    let curr = frame.get_evaluation_step(0);
    let next = frame.get_evaluation_step(1);
    let one = FieldElement::one();

    let mem_addr_sorted_0 = curr.get_evaluation_element(0, MEMORY_ADDR_SORTED_0);
    let mem_addr_sorted_1 = curr.get_evaluation_element(0, MEMORY_ADDR_SORTED_1);
    let mem_addr_sorted_2 = curr.get_evaluation_element(0, MEMORY_ADDR_SORTED_2);
    let mem_addr_sorted_3 = curr.get_evaluation_element(0, MEMORY_ADDR_SORTED_3);
    let mem_addr_sorted_4 = curr.get_evaluation_element(0, MEMORY_ADDR_SORTED_4);
    let next_mem_addr_sorted_0 = next.get_evaluation_element(0, MEMORY_ADDR_SORTED_0);

    let mem_val_sorted_0 = curr.get_evaluation_element(0, MEMORY_VALUES_SORTED_0);
    let mem_val_sorted_1 = curr.get_evaluation_element(0, MEMORY_VALUES_SORTED_1);
    let mem_val_sorted_2 = curr.get_evaluation_element(0, MEMORY_VALUES_SORTED_2);
    let mem_val_sorted_3 = curr.get_evaluation_element(0, MEMORY_VALUES_SORTED_3);
    let mem_val_sorted_4 = curr.get_evaluation_element(0, MEMORY_VALUES_SORTED_4);
    let next_mem_val_sorted_0 = next.get_evaluation_element(0, MEMORY_VALUES_SORTED_0);

    constraints[MEMORY_INCREASING_0] =
        (mem_addr_sorted_0 - mem_addr_sorted_1) * (mem_addr_sorted_1 - mem_addr_sorted_0 - one);

    constraints[MEMORY_INCREASING_1] =
        (mem_addr_sorted_1 - mem_addr_sorted_2) * (mem_addr_sorted_2 - mem_addr_sorted_1 - one);

    constraints[MEMORY_INCREASING_2] =
        (mem_addr_sorted_2 - mem_addr_sorted_3) * (mem_addr_sorted_3 - mem_addr_sorted_2 - one);

    constraints[MEMORY_INCREASING_3] =
        (mem_addr_sorted_3 - mem_addr_sorted_4) * (mem_addr_sorted_4 - mem_addr_sorted_3 - one);

    constraints[MEMORY_INCREASING_4] = (mem_addr_sorted_4 - next_mem_addr_sorted_0)
        * (next_mem_addr_sorted_0 - mem_addr_sorted_4 - one);

    constraints[MEMORY_CONSISTENCY_0] =
        (mem_val_sorted_0 - mem_val_sorted_1) * (mem_addr_sorted_1 - mem_addr_sorted_0 - one);

    constraints[MEMORY_CONSISTENCY_1] =
        (mem_val_sorted_1 - mem_val_sorted_2) * (mem_addr_sorted_2 - mem_addr_sorted_1 - one);

    constraints[MEMORY_CONSISTENCY_2] =
        (mem_val_sorted_2 - mem_val_sorted_3) * (mem_addr_sorted_3 - mem_addr_sorted_2 - one);

    constraints[MEMORY_CONSISTENCY_3] =
        (mem_val_sorted_3 - mem_val_sorted_4) * (mem_addr_sorted_4 - mem_addr_sorted_3 - one);

    constraints[MEMORY_CONSISTENCY_4] = (mem_val_sorted_4 - next_mem_val_sorted_0)
        * (next_mem_addr_sorted_0 - mem_addr_sorted_4 - one);
}

fn permutation_argument(
    constraints: &mut [Felt252],
    frame: &Frame<Stark252PrimeField>,
    rap_challenges: &CairoRAPChallenges,
) {
    let curr = frame.get_evaluation_step(0);
    let next = frame.get_evaluation_step(1);

    let z = &rap_challenges.z_memory;
    let alpha = &rap_challenges.alpha_memory;

    let p0 = curr.get_evaluation_element(0, PERMUTATION_ARGUMENT_COL_0);
    let next_p0 = next.get_evaluation_element(0, PERMUTATION_ARGUMENT_COL_0);
    let p1 = curr.get_evaluation_element(0, PERMUTATION_ARGUMENT_COL_1);
    let p2 = curr.get_evaluation_element(0, PERMUTATION_ARGUMENT_COL_2);
    let p3 = curr.get_evaluation_element(0, PERMUTATION_ARGUMENT_COL_3);
    let p4 = curr.get_evaluation_element(0, PERMUTATION_ARGUMENT_COL_4);

    let next_ap0 = next.get_evaluation_element(0, MEMORY_ADDR_SORTED_0);
    let ap1 = curr.get_evaluation_element(0, MEMORY_ADDR_SORTED_1);
    let ap2 = curr.get_evaluation_element(0, MEMORY_ADDR_SORTED_2);
    let ap3 = curr.get_evaluation_element(0, MEMORY_ADDR_SORTED_3);
    let ap4 = curr.get_evaluation_element(0, MEMORY_ADDR_SORTED_4);

    let next_vp0 = next.get_evaluation_element(0, MEMORY_VALUES_SORTED_0);
    let vp1 = curr.get_evaluation_element(0, MEMORY_VALUES_SORTED_1);
    let vp2 = curr.get_evaluation_element(0, MEMORY_VALUES_SORTED_2);
    let vp3 = curr.get_evaluation_element(0, MEMORY_VALUES_SORTED_3);
    let vp4 = curr.get_evaluation_element(0, MEMORY_VALUES_SORTED_4);

    let next_a0 = next.get_evaluation_element(0, FRAME_PC);
    let a1 = curr.get_evaluation_element(0, FRAME_DST_ADDR);
    let a2 = curr.get_evaluation_element(0, FRAME_OP0_ADDR);
    let a3 = curr.get_evaluation_element(0, FRAME_OP1_ADDR);
    let a4 = curr.get_evaluation_element(0, EXTRA_ADDR);

    let next_v0 = next.get_evaluation_element(0, FRAME_INST);
    let v1 = curr.get_evaluation_element(0, FRAME_DST);
    let v2 = curr.get_evaluation_element(0, FRAME_OP0);
    let v3 = curr.get_evaluation_element(0, FRAME_OP1);
    let v4 = curr.get_evaluation_element(0, EXTRA_VAL);

    constraints[PERMUTATION_ARGUMENT_0] =
        (z - (ap1 + alpha * vp1)) * p1 - (z - (a1 + alpha * v1)) * p0;
    constraints[PERMUTATION_ARGUMENT_1] =
        (z - (ap2 + alpha * vp2)) * p2 - (z - (a2 + alpha * v2)) * p1;
    constraints[PERMUTATION_ARGUMENT_2] =
        (z - (ap3 + alpha * vp3)) * p3 - (z - (a3 + alpha * v3)) * p2;
    constraints[PERMUTATION_ARGUMENT_3] =
        (z - (ap4 + alpha * vp4)) * p4 - (z - (a4 + alpha * v4)) * p3;
    constraints[PERMUTATION_ARGUMENT_4] =
        (z - (next_ap0 + alpha * next_vp0)) * next_p0 - (z - (next_a0 + alpha * next_v0)) * p4;
}

fn permutation_argument_range_check(
    constraints: &mut [Felt252],
    frame: &Frame<Stark252PrimeField>,
    rap_challenges: &CairoRAPChallenges,
) {
    let curr = frame.get_evaluation_step(0);
    let next = frame.get_evaluation_step(1);
    let one = FieldElement::one();
    let z = &rap_challenges.z_range_check;

    let rc_col_1 = curr.get_evaluation_element(0, RANGE_CHECK_COL_1);
    let rc_col_2 = curr.get_evaluation_element(0, RANGE_CHECK_COL_2);
    let rc_col_3 = curr.get_evaluation_element(0, RANGE_CHECK_COL_3);
    let rc_col_4 = curr.get_evaluation_element(0, RANGE_CHECK_COL_4);
    let next_rc_col_1 = next.get_evaluation_element(0, RANGE_CHECK_COL_1);

    constraints[RANGE_CHECK_INCREASING_0] = (rc_col_1 - rc_col_2) * (rc_col_2 - rc_col_1 - one);
    constraints[RANGE_CHECK_INCREASING_1] = (rc_col_2 - rc_col_3) * (rc_col_3 - rc_col_2 - one);
    constraints[RANGE_CHECK_INCREASING_2] = (rc_col_3 - rc_col_4) * (rc_col_4 - rc_col_3 - one);
    constraints[RANGE_CHECK_INCREASING_3] =
        (rc_col_4 - next_rc_col_1) * (next_rc_col_1 - rc_col_4 - one);

    let p0 = curr.get_evaluation_element(0, PERMUTATION_ARGUMENT_RANGE_CHECK_COL_1);
    let next_p0 = next.get_evaluation_element(0, PERMUTATION_ARGUMENT_RANGE_CHECK_COL_1);
    let p1 = curr.get_evaluation_element(0, PERMUTATION_ARGUMENT_RANGE_CHECK_COL_2);
    let p2 = curr.get_evaluation_element(0, PERMUTATION_ARGUMENT_RANGE_CHECK_COL_3);
    let p3 = curr.get_evaluation_element(0, PERMUTATION_ARGUMENT_RANGE_CHECK_COL_4);

    let next_ap0 = next.get_evaluation_element(0, RANGE_CHECK_COL_1);
    let ap1 = curr.get_evaluation_element(0, RANGE_CHECK_COL_2);
    let ap2 = curr.get_evaluation_element(0, RANGE_CHECK_COL_3);
    let ap3 = curr.get_evaluation_element(0, RANGE_CHECK_COL_4);

    let a0_next = next.get_evaluation_element(0, OFF_DST);
    let a1 = curr.get_evaluation_element(0, OFF_OP0);
    let a2 = curr.get_evaluation_element(0, OFF_OP1);
    let a3 = curr.get_evaluation_element(0, RC_HOLES);

    constraints[RANGE_CHECK_0] = (z - ap1) * p1 - (z - a1) * p0;
    constraints[RANGE_CHECK_1] = (z - ap2) * p2 - (z - a2) * p1;
    constraints[RANGE_CHECK_2] = (z - ap3) * p3 - (z - a3) * p2;
    constraints[RANGE_CHECK_3] = (z - next_ap0) * next_p0 - (z - a0_next) * p3;
}

fn frame_inst_size(step: &StepView<Stark252PrimeField>) -> Felt252 {
    let op1_val = into_bit_flag(step, F_OP_1_VAL);
    op1_val + Felt252::one()
}

/// Wrapper function for generating Cairo proofs without the need to specify
/// concrete types.
/// The field is set to Stark252PrimeField and the AIR to CairoAIR.
pub fn generate_cairo_proof(
    trace: &TraceTable<Stark252PrimeField>,
    pub_input: &PublicInputs,
    proof_options: &ProofOptions,
) -> Result<StarkProof<Stark252PrimeField>, ProvingError> {
    Prover::prove::<CairoAIR>(
        trace,
        pub_input,
        proof_options,
        StoneProverTranscript::new(&[]),
    )
}

/// Wrapper function for verifying Cairo proofs without the need to specify
/// concrete types.
/// The field is set to Stark252PrimeField and the AIR to CairoAIR.
pub fn verify_cairo_proof(
    proof: &StarkProof<Stark252PrimeField>,
    pub_input: &PublicInputs,
    proof_options: &ProofOptions,
) -> bool {
    Verifier::verify::<CairoAIR>(
        proof,
        pub_input,
        proof_options,
        StoneProverTranscript::new(&[]),
    )
}

#[cfg(test)]
#[cfg(debug_assertions)]
mod test {
    use super::*;
    use lambdaworks_math::field::element::FieldElement;

    #[test]
    fn test_build_auxiliary_trace_add_program_in_public_input_section_works() {
        let dummy_public_input = PublicInputs {
            pc_init: FieldElement::zero(),
            ap_init: FieldElement::zero(),
            fp_init: FieldElement::zero(),
            pc_final: FieldElement::zero(),
            ap_final: FieldElement::zero(),
            public_memory: HashMap::from([
                (FieldElement::one(), FieldElement::from(10)),
                (FieldElement::from(2), FieldElement::from(20)),
            ]),
            range_check_max: None,
            range_check_min: None,
            num_steps: 1,
            memory_segments: MemorySegmentMap::new(),
            codelen: 3,
        };

        let a = vec![
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::zero(),
            FieldElement::from(2),
            FieldElement::from(2),
            FieldElement::from(2),
            FieldElement::from(2),
            FieldElement::zero(),
        ];
        let v = vec![
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ];
        let (ap, vp) = add_pub_memory_in_public_input_section(&a, &v, &dummy_public_input);
        assert_eq!(
            ap,
            vec![
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::from(2),
                FieldElement::from(2),
                FieldElement::from(2),
                FieldElement::from(2),
                FieldElement::from(2),
            ]
        );
        assert_eq!(
            vp,
            vec![
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::from(10),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::from(20),
            ]
        );
    }

    #[test]
    fn test_build_auxiliary_trace_add_program_with_output_in_public_input_section_works() {
        let dummy_public_input = PublicInputs {
            pc_init: FieldElement::zero(),
            ap_init: FieldElement::zero(),
            fp_init: FieldElement::zero(),
            pc_final: FieldElement::zero(),
            ap_final: FieldElement::zero(),
            public_memory: HashMap::from([
                (FieldElement::one(), FieldElement::from(10)),
                (FieldElement::from(2), FieldElement::from(20)),
                (FieldElement::from(20), FieldElement::from(40)),
            ]),
            range_check_max: None,
            range_check_min: None,
            num_steps: 1,
            memory_segments: MemorySegmentMap::from([(MemorySegment::Output, 20..21)]),
            codelen: 3,
        };

        let a = vec![
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::zero(),
        ];

        let v = vec![
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ];

        let (ap, vp) = add_pub_memory_in_public_input_section(&a, &v, &dummy_public_input);
        assert_eq!(
            ap,
            vec![
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::from(2),
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::from(20),
            ]
        );
        assert_eq!(
            vp,
            vec![
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::from(10),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::from(20),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::zero(),
                FieldElement::from(40),
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
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[cfg(test)]
mod prop_test {
    use lambdaworks_math::{
        field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        traits::{Deserializable, Serializable},
    };
    use proptest::{prelude::*, prop_compose, proptest};
    use stark_platinum_prover::proof::{options::ProofOptions, stark::StarkProof};

    use crate::{
        air::{generate_cairo_proof, verify_cairo_proof},
        cairo_layout::CairoLayout,
        runner::run::generate_prover_args,
        tests::utils::cairo0_program_path,
        Felt252,
    };

    use super::{MemorySegment, MemorySegmentMap, PublicInputs};

    prop_compose! {
        fn some_felt()(base in any::<u64>(), exponent in any::<u128>()) -> Felt252 {
            Felt252::from(base).pow(exponent)
        }
    }

    prop_compose! {
        fn some_public_inputs()(
            pc_init in some_felt(),
            ap_init in some_felt(),
            fp_init in some_felt(),
            pc_final in some_felt(),
            ap_final in some_felt(),
            public_memory in proptest::collection::hash_map(any::<u64>(), any::<u64>(), (8_usize, 16_usize)),
            range_check_max in proptest::option::of(any::<u16>()),
            range_check_min in proptest::option::of(any::<u16>()),
            num_steps in any::<usize>(),
            codelen in any::<usize>(),
        ) -> PublicInputs {
            let public_memory = public_memory.iter().map(|(k, v)| (Felt252::from(*k), Felt252::from(*v))).collect();
            let memory_segments = MemorySegmentMap::from([(MemorySegment::Output, 10u64..16u64), (MemorySegment::RangeCheck, 20u64..71u64)]);
            PublicInputs {
                pc_init,
                ap_init,
                fp_init,
                pc_final,
                ap_final,
                public_memory,
                range_check_max,
                range_check_min,
                num_steps,
                memory_segments,
                codelen,
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {cases: 5, .. ProptestConfig::default()})]
        #[test]
        fn test_public_inputs_serialization(
            public_inputs in some_public_inputs(),
        ){
            let serialized = Serializable::serialize(&public_inputs);
            let deserialized: PublicInputs = Deserializable::deserialize(&serialized).unwrap();
            prop_assert_eq!(public_inputs.pc_init, deserialized.pc_init);
            prop_assert_eq!(public_inputs.ap_init, deserialized.ap_init);
            prop_assert_eq!(public_inputs.fp_init, deserialized.fp_init);
            prop_assert_eq!(public_inputs.pc_final, deserialized.pc_final);
            prop_assert_eq!(public_inputs.ap_final, deserialized.ap_final);
            prop_assert_eq!(public_inputs.public_memory, deserialized.public_memory);
            prop_assert_eq!(public_inputs.range_check_max, deserialized.range_check_max);
            prop_assert_eq!(public_inputs.range_check_min, deserialized.range_check_min);
            prop_assert_eq!(public_inputs.num_steps, deserialized.num_steps);
            prop_assert_eq!(public_inputs.memory_segments, deserialized.memory_segments);
        }
    }

    #[test]
    fn deserialize_and_verify() {
        let program_content = std::fs::read(cairo0_program_path("fibonacci_10.json")).unwrap();
        let (main_trace, pub_inputs) =
            generate_prover_args(&program_content, CairoLayout::Plain).unwrap();

        let proof_options = ProofOptions::default_test_options();

        // The proof is generated and serialized.
        let proof = generate_cairo_proof(&main_trace, &pub_inputs, &proof_options).unwrap();
        let proof_bytes: Vec<u8> = serde_cbor::to_vec(&proof).unwrap();

        // The trace and original proof are dropped to show that they are decoupled from
        // the verifying process.
        drop(main_trace);
        drop(proof);

        // At this point, the verifier only knows about the serialized proof, the proof options
        // and the public inputs.
        let proof: StarkProof<Stark252PrimeField> = serde_cbor::from_slice(&proof_bytes).unwrap();

        // The proof is verified successfully.
        assert!(verify_cairo_proof(&proof, &pub_inputs, &proof_options));
    }
}
