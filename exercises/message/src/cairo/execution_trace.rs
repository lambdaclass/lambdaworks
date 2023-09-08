use std::{iter, ops::Range};

use crate::{cairo::air::*, starks::trace::TraceTable, FE};
use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        traits::{IsFFTField, IsPrimeField},
    },
    unsigned_integer::element::UnsignedInteger,
};
use num_integer::div_ceil;

use super::{
    cairo_mem::CairoMemory,
    decode::{
        instruction_flags::{
            aux_get_last_nim_of_field_element, ApUpdate, CairoInstructionFlags, CairoOpcode,
            DstReg, Op0Reg, Op1Src, PcUpdate, ResLogic,
        },
        instruction_offsets::InstructionOffsets,
    },
    register_states::RegisterStates,
};

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

pub const ADDR_COLUMNS: [usize; 4] = [FRAME_PC, FRAME_DST_ADDR, FRAME_OP0_ADDR, FRAME_OP1_ADDR];

// MAIN TRACE LAYOUT
// -----------------------------------------------------------------------------------------
//  A.  flags   (16) : Decoded instruction flags
//  B.  res     (1)  : Res value
//  C.  mem_p   (2)  : Temporary memory pointers (ap and fp)
//  D.  mem_a   (4)  : Memory addresses (pc, dst_addr, op0_addr, op1_addr)
//  E.  mem_v   (4)  : Memory values (inst, dst, op0, op1)
//  F.  offsets (3)  : (off_dst, off_op0, off_op1)
//  G.  derived (3)  : (t0, t1, mul)
//
//  A                B C  D    E    F   G
// ├xxxxxxxxxxxxxxxx|x|xx|xxxx|xxxx|xxx|xxx┤
//

/// Builds the Cairo main trace (i.e. the trace without the auxiliary columns).
/// Builds the execution trace, fills the offset range-check holes and memory holes, adds
/// public memory dummy accesses (See section 9.8 of the Cairo whitepaper) and pads the result
/// so that it has a trace length equal to the closest power of two.
pub fn build_main_trace(
    register_states: &RegisterStates,
    memory: &CairoMemory,
    public_input: &mut PublicInputs,
) -> TraceTable<Stark252PrimeField> {
    let mut main_trace = build_cairo_execution_trace(register_states, memory, public_input);

    let mut address_cols = main_trace
        .get_cols(&[FRAME_PC, FRAME_DST_ADDR, FRAME_OP0_ADDR, FRAME_OP1_ADDR])
        .table;
    address_cols.sort_by_key(|x| x.representative());

    let (rc_holes, rc_min, rc_max) = get_rc_holes(&main_trace, &[OFF_DST, OFF_OP0, OFF_OP1]);
    public_input.range_check_min = Some(rc_min);
    public_input.range_check_max = Some(rc_max);
    fill_rc_holes(&mut main_trace, rc_holes);

    let mut memory_holes = get_memory_holes(&address_cols, public_input.public_memory.len());

    if !memory_holes.is_empty() {
        fill_memory_holes(&mut main_trace, &mut memory_holes);
    }

    add_pub_memory_dummy_accesses(&mut main_trace, public_input.public_memory.len());

    let trace_len_next_power_of_two = main_trace.n_rows().next_power_of_two();
    let padding = trace_len_next_power_of_two - main_trace.n_rows();
    pad_with_last_row(&mut main_trace, padding);

    main_trace
}

/// Artificial `(0, 0)` dummy memory accesses must be added for the public memory.
/// See section 9.8 of the Cairo whitepaper.
fn add_pub_memory_dummy_accesses<F: IsFFTField>(
    main_trace: &mut TraceTable<F>,
    pub_memory_len: usize,
) {
    pad_with_last_row_and_zeros(main_trace, (pub_memory_len >> 2) + 1, &MEMORY_COLUMNS)
}

fn pad_with_last_row<F: IsFFTField>(trace: &mut TraceTable<F>, number_rows: usize) {
    let last_row = trace.last_row().to_vec();
    let mut pad: Vec<_> = std::iter::repeat(&last_row)
        .take(number_rows)
        .flatten()
        .cloned()
        .collect();
    trace.table.append(&mut pad);
}

/// Pads trace with its last row, with the exception of the columns specified in
/// `zero_pad_columns`, where the pad is done with zeros.
/// If the last row is [2, 1, 4, 1] and the zero pad columns are [0, 1], then the
/// padding will be [0, 0, 4, 1].
fn pad_with_last_row_and_zeros<F: IsFFTField>(
    trace: &mut TraceTable<F>,
    number_rows: usize,
    zero_pad_columns: &[usize],
) {
    let mut last_row = trace.last_row().to_vec();
    for exemption_column in zero_pad_columns.iter() {
        last_row[*exemption_column] = FieldElement::zero();
    }
    let mut pad: Vec<_> = std::iter::repeat(&last_row)
        .take(number_rows)
        .flatten()
        .cloned()
        .collect();
    trace.table.append(&mut pad);
}

/// Gets holes from the range-checked columns. These holes must be filled for the
/// permutation range-checks, as can be read in section 9.9 of the Cairo whitepaper.
/// Receives the trace and the indexes of the range-checked columns.
/// Outputs the holes that must be filled to make the range continuous and the extreme
/// values rc_min and rc_max, corresponding to the minimum and maximum values of the range.
/// NOTE: These extreme values should be received as public inputs in the future and not
/// calculated here.
fn get_rc_holes<F>(
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

/// Fills holes found in the range-checked columns.
fn fill_rc_holes<F: IsFFTField>(trace: &mut TraceTable<F>, holes: Vec<FieldElement<F>>) {
    let zeros_left = vec![FieldElement::zero(); OFF_DST];
    let zeros_right = vec![FieldElement::zero(); trace.n_cols - OFF_OP1 - 1];

    for i in (0..holes.len()).step_by(3) {
        trace.table.append(&mut zeros_left.clone());
        trace.table.append(&mut holes[i..(i + 3)].to_vec());
        trace.table.append(&mut zeros_right.clone());
    }
}

/// Get memory holes from accessed addresses. These memory holes appear
/// as a consequence of interaction with builtins.
/// Returns a vector of addresses that were not present in the input vector (holes)
///
/// # Arguments
///
/// * `sorted_addrs` - Vector of sorted memory addresses.
/// * `codelen` - the length of the Cairo program instructions.
fn get_memory_holes(sorted_addrs: &[FE], codelen: usize) -> Vec<FE> {
    let mut memory_holes = Vec::new();
    let mut prev_addr = &sorted_addrs[0];

    for addr in sorted_addrs.iter() {
        let addr_diff = addr - prev_addr;

        // If the candidate memory hole has an address belonging to the program segment (public
        // memory), that is not accounted here since public memory is added in a posterior step of
        // the protocol.
        if addr_diff != FE::one()
            && addr_diff != FE::zero()
            && addr.representative() > (codelen as u64).into()
        {
            let mut hole_addr = prev_addr + FE::one();

            while hole_addr.representative() < addr.representative() {
                if hole_addr.representative() > (codelen as u64).into() {
                    memory_holes.push(hole_addr);
                }
                hole_addr += FE::one();
            }
        }
        prev_addr = addr;
    }

    memory_holes
}

/// Fill memory holes in each address column of the trace with the missing address, depending on the
/// trace column. If the trace column refers to memory addresses, it will be filled with the missing
/// addresses.
fn fill_memory_holes(trace: &mut TraceTable<Stark252PrimeField>, memory_holes: &mut [FE]) {
    let last_row = trace.last_row().to_vec();

    // This number represents the amount of times we have to pad to fill the memory
    // holes into the trace.
    // There are a total of ADDR_COLUMNS.len() address columns, and we have to append
    // hole addresses in each column until there are no more holes.
    // If we have that memory_holes = [1, 2, 3, 4, 5] and there are 4 address columns,
    // we will have to pad with 2 rows, one for the first [1, 2, 3, 4] and one for the
    // 5 value.
    let padding_size = div_ceil(memory_holes.len(), ADDR_COLUMNS.len());

    let padding_row_iter = iter::repeat(last_row).take(padding_size);
    let addr_columns_iter = iter::repeat(ADDR_COLUMNS).take(padding_size);
    let mut memory_holes_iter = memory_holes.iter();

    for (addr_cols, mut padding_row) in iter::zip(addr_columns_iter, padding_row_iter) {
        // The particular placement of the holes in each column is not important,
        // the only thing that matters is that the addresses are put somewhere in the address
        // columns.
        addr_cols.iter().for_each(|a_col| {
            if let Some(hole) = memory_holes_iter.next() {
                padding_row[*a_col] = *hole;
            }
        });

        trace.table.append(&mut padding_row);
    }
}

/// Receives the raw Cairo trace and memory as outputted from the Cairo VM and returns
/// the trace table used to feed the Cairo STARK prover.
/// The constraints of the Cairo AIR are defined over this trace rather than the raw trace
/// obtained from the Cairo VM, this is why this function is needed.
pub fn build_cairo_execution_trace(
    raw_trace: &RegisterStates,
    memory: &CairoMemory,
    public_inputs: &PublicInputs,
) -> TraceTable<Stark252PrimeField> {
    let n_steps = raw_trace.steps();

    // Instruction flags and offsets are decoded from the raw instructions and represented
    // by the CairoInstructionFlags and InstructionOffsets as an intermediate representation
    let (flags, offsets): (Vec<CairoInstructionFlags>, Vec<InstructionOffsets>) = raw_trace
        .flags_and_offsets(memory)
        .unwrap()
        .into_iter()
        .unzip();

    // dst, op0, op1 and res are computed from flags and offsets
    let (dst_addrs, mut dsts): (Vec<FE>, Vec<FE>) =
        compute_dst(&flags, &offsets, raw_trace, memory);
    let (op0_addrs, mut op0s): (Vec<FE>, Vec<FE>) =
        compute_op0(&flags, &offsets, raw_trace, memory);
    let (op1_addrs, op1s): (Vec<FE>, Vec<FE>) =
        compute_op1(&flags, &offsets, raw_trace, memory, &op0s);
    let mut res = compute_res(&flags, &op0s, &op1s, &dsts);

    // In some cases op0, dst or res may need to be updated from the already calculated values
    update_values(&flags, raw_trace, &mut op0s, &mut dsts, &mut res);

    // Flags and offsets are transformed to a bit representation. This is needed since
    // the flag constraints of the Cairo AIR are defined over bit representations of these
    let trace_repr_flags: Vec<[FE; 16]> = flags
        .iter()
        .map(CairoInstructionFlags::to_trace_representation)
        .collect();
    let trace_repr_offsets: Vec<[FE; 3]> = offsets
        .iter()
        .map(InstructionOffsets::to_trace_representation)
        .collect();

    // ap, fp, pc and instruction columns are computed
    let aps: Vec<FE> = raw_trace.rows.iter().map(|t| FE::from(t.ap)).collect();
    let fps: Vec<FE> = raw_trace.rows.iter().map(|t| FE::from(t.fp)).collect();
    let pcs: Vec<FE> = raw_trace.rows.iter().map(|t| FE::from(t.pc)).collect();
    let instructions: Vec<FE> = raw_trace
        .rows
        .iter()
        .map(|t| *memory.get(&t.pc).unwrap())
        .collect();

    // t0, t1 and mul derived values are constructed. For details refer to
    // section 9.1 of the Cairo whitepaper
    let t0: Vec<FE> = trace_repr_flags
        .iter()
        .zip(&dsts)
        .map(|(repr_flags, dst)| repr_flags[9] * dst)
        .collect();
    let t1: Vec<FE> = t0.iter().zip(&res).map(|(t, r)| t * r).collect();
    let mul: Vec<FE> = op0s.iter().zip(&op1s).map(|(op0, op1)| op0 * op1).collect();

    // A structure change of the flags and offsets representations to fit into the arguments
    // expected by the TraceTable constructor. A vector of columns of the representations
    // is obtained from the rows representation.
    let trace_repr_flags = rows_to_cols(&trace_repr_flags);
    let trace_repr_offsets = rows_to_cols(&trace_repr_offsets);

    let mut selector = vec![FE::one(); n_steps];
    selector[n_steps - 1] = FE::zero();

    // Build Cairo trace columns to instantiate TraceTable struct as defined in the trace layout
    let mut trace_cols: Vec<Vec<FE>> = Vec::new();
    (0..trace_repr_flags.len()).for_each(|n| trace_cols.push(trace_repr_flags[n].clone()));
    trace_cols.push(res);
    trace_cols.push(aps);
    trace_cols.push(fps);
    trace_cols.push(pcs);
    trace_cols.push(dst_addrs);
    trace_cols.push(op0_addrs);
    trace_cols.push(op1_addrs);
    trace_cols.push(instructions);
    trace_cols.push(dsts);
    trace_cols.push(op0s);
    trace_cols.push(op1s);
    (0..trace_repr_offsets.len()).for_each(|n| trace_cols.push(trace_repr_offsets[n].clone()));
    trace_cols.push(t0);
    trace_cols.push(t1);
    trace_cols.push(mul);
    trace_cols.push(selector);

    if let Some(range_check_builtin_range) = public_inputs
        .memory_segments
        .get(&MemorySegment::RangeCheck)
    {
        add_rc_builtin_columns(&mut trace_cols, range_check_builtin_range.clone(), memory);
    }

    TraceTable::new_from_cols(&trace_cols)
}

// Build range-check builtin columns: rc_0, rc_1, ... , rc_7, rc_value
fn add_rc_builtin_columns(
    trace_cols: &mut Vec<Vec<FE>>,
    range_check_builtin_range: Range<u64>,
    memory: &CairoMemory,
) {
    let range_checked_values: Vec<&FE> = range_check_builtin_range
        .map(|addr| memory.get(&addr).unwrap())
        .collect();
    let mut rc_trace_columns = decompose_rc_values_into_trace_columns(&range_checked_values);

    // rc decomposition columns are appended with zeros and then pushed to the trace table
    rc_trace_columns.iter_mut().for_each(|column| {
        column.resize(trace_cols[0].len(), FE::zero());
        trace_cols.push(column.to_vec())
    });

    let mut rc_values_dereferenced: Vec<FE> = range_checked_values.iter().map(|&x| *x).collect();
    rc_values_dereferenced.resize(trace_cols[0].len(), FE::zero());

    trace_cols.push(rc_values_dereferenced);
}

/// Returns the vector of res values.
fn compute_res(flags: &[CairoInstructionFlags], op0s: &[FE], op1s: &[FE], dsts: &[FE]) -> Vec<FE> {
    /*
    Cairo whitepaper, page 33 - https://eprint.iacr.org/2021/1063.pdf
    # Compute res.
    if pc_update == 4:
        if res_logic == 0 && opcode == 0 && ap_update != 1:
            res = Unused
        else:
            Undefined Behavior
    else if pc_update = 0, 1 or 2:
        switch res_logic:
            case 0: res = op1
            case 1: res = op0 + op1
            case 2: res = op0 * op1
            default: Undefined Behavior
    else: Undefined Behavior
    */
    flags
        .iter()
        .zip(op0s)
        .zip(op1s)
        .zip(dsts)
        .map(|(((f, op0), op1), dst)| {
            match f.pc_update {
                PcUpdate::Jnz => {
                    match (&f.res_logic, &f.opcode, &f.ap_update) {
                        (
                            ResLogic::Op1,
                            CairoOpcode::NOp,
                            ApUpdate::Regular | ApUpdate::Add1 | ApUpdate::Add2,
                        ) => {
                            // In a `jnz` instruction, res is not used, so it is used
                            // to hold the value v = dst^(-1) as an optimization.
                            // This is important for the calculation of the `t1` virtual column
                            // values later on.
                            // See section 9.5 of the Cairo whitepaper, page 53.
                            if dst == &FE::zero() {
                                *dst
                            } else {
                                dst.inv()
                            }
                        }
                        _ => {
                            panic!("Undefined Behavior");
                        }
                    }
                }
                PcUpdate::Regular | PcUpdate::Jump | PcUpdate::JumpRel => match f.res_logic {
                    ResLogic::Op1 => *op1,
                    ResLogic::Add => op0 + op1,
                    ResLogic::Mul => op0 * op1,
                    ResLogic::Unconstrained => {
                        panic!("Undefined Behavior");
                    }
                },
            }
        })
        .collect()
}

/// Returns the vector of:
/// - dst_addrs
/// - dsts
fn compute_dst(
    flags: &[CairoInstructionFlags],
    offsets: &[InstructionOffsets],
    register_states: &RegisterStates,
    memory: &CairoMemory,
) -> (Vec<FE>, Vec<FE>) {
    /* Cairo whitepaper, page 33 - https://eprint.iacr.org/2021/1063.pdf

    # Compute dst
    if dst_reg == 0:
        dst = m(ap + offdst)
    else:
        dst = m(fp + offdst)
    */
    flags
        .iter()
        .zip(offsets)
        .zip(register_states.rows.iter())
        .map(|((f, o), t)| match f.dst_reg {
            DstReg::AP => {
                let addr = t.ap.checked_add_signed(o.off_dst.into()).unwrap();
                (FE::from(addr), *memory.get(&addr).unwrap())
            }
            DstReg::FP => {
                let addr = t.fp.checked_add_signed(o.off_dst.into()).unwrap();
                (FE::from(addr), *memory.get(&addr).unwrap())
            }
        })
        .unzip()
}

/// Returns the vector of:
/// - op0_addrs
/// - op0s
fn compute_op0(
    flags: &[CairoInstructionFlags],
    offsets: &[InstructionOffsets],
    register_states: &RegisterStates,
    memory: &CairoMemory,
) -> (Vec<FE>, Vec<FE>) {
    /* Cairo whitepaper, page 33 - https://eprint.iacr.org/2021/1063.pdf

    # Compute op0.
    if op0_reg == 0:
        op0 = m(ap + offop0)
    else:
        op0 = m(fp + offop0)
    */
    flags
        .iter()
        .zip(offsets)
        .zip(register_states.rows.iter())
        .map(|((f, o), t)| match f.op0_reg {
            Op0Reg::AP => {
                let addr = t.ap.checked_add_signed(o.off_op0.into()).unwrap();
                (FE::from(addr), *memory.get(&addr).unwrap())
            }
            Op0Reg::FP => {
                let addr = t.fp.checked_add_signed(o.off_op0.into()).unwrap();
                (FE::from(addr), *memory.get(&addr).unwrap())
            }
        })
        .unzip()
}

/// Returns the vector of:
/// - op1_addrs
/// - op1s
fn compute_op1(
    flags: &[CairoInstructionFlags],
    offsets: &[InstructionOffsets],
    register_states: &RegisterStates,
    memory: &CairoMemory,
    op0s: &[FE],
) -> (Vec<FE>, Vec<FE>) {
    /* Cairo whitepaper, page 33 - https://eprint.iacr.org/2021/1063.pdf
    # Compute op1 and instruction_size.
    switch op1_src:
        case 0:
            instruction_size = 1
            op1 = m(op0 + offop1)
        case 1:
            instruction_size = 2
            op1 = m(pc + offop1)
            # If offop1 = 1, we have op1 = immediate_value.
        case 2:
            instruction_size = 1
            op1 = m(fp + offop1)
        case 4:
            instruction_size = 1
            op1 = m(ap + offop1)
        default:
            Undefined Behavior
    */
    flags
        .iter()
        .zip(offsets)
        .zip(op0s)
        .zip(register_states.rows.iter())
        .map(|(((flag, offset), op0), trace_state)| match flag.op1_src {
            Op1Src::Op0 => {
                let addr = aux_get_last_nim_of_field_element(op0)
                    .checked_add_signed(offset.off_op1.into())
                    .unwrap();
                (FE::from(addr), *memory.get(&addr).unwrap())
            }
            Op1Src::Imm => {
                let pc = trace_state.pc;
                let addr = pc.checked_add_signed(offset.off_op1.into()).unwrap();
                (FE::from(addr), *memory.get(&addr).unwrap())
            }
            Op1Src::AP => {
                let ap = trace_state.ap;
                let addr = ap.checked_add_signed(offset.off_op1.into()).unwrap();
                (FE::from(addr), *memory.get(&addr).unwrap())
            }
            Op1Src::FP => {
                let fp = trace_state.fp;
                let addr = fp.checked_add_signed(offset.off_op1.into()).unwrap();
                (FE::from(addr), *memory.get(&addr).unwrap())
            }
        })
        .unzip()
}

/// Depending on the instruction opcodes, some values should be updated.
/// This function updates op0s, dst, res in place when the conditions hold.
fn update_values(
    flags: &[CairoInstructionFlags],
    register_states: &RegisterStates,
    op0s: &mut [FE],
    dst: &mut [FE],
    res: &mut [FE],
) {
    for (i, f) in flags.iter().enumerate() {
        if f.opcode == CairoOpcode::Call {
            let instruction_size = if flags[i].op1_src == Op1Src::Imm {
                2
            } else {
                1
            };
            op0s[i] = (register_states.rows[i].pc + instruction_size).into();
            dst[i] = register_states.rows[i].fp.into();
        } else if f.opcode == CairoOpcode::AssertEq {
            res[i] = dst[i];
        }
    }
}

/// Utility function to change from a rows representation to a columns
/// representation of a slice of arrays.   
fn rows_to_cols<const N: usize>(rows: &[[FE; N]]) -> Vec<Vec<FE>> {
    let n_cols = rows[0].len();

    (0..n_cols)
        .map(|col_idx| rows.iter().map(|elem| elem[col_idx]).collect::<Vec<FE>>())
        .collect::<Vec<Vec<FE>>>()
}

fn decompose_rc_values_into_trace_columns(rc_values: &[&FE]) -> [Vec<FE>; 8] {
    let mask = UnsignedInteger::from_hex("FFFF").unwrap();
    let mut rc_base_types: Vec<UnsignedInteger<4>> =
        rc_values.iter().map(|x| x.representative()).collect();

    let mut decomposition_columns: Vec<Vec<FE>> = Vec::new();

    for _ in 0..8 {
        decomposition_columns.push(
            rc_base_types
                .iter()
                .map(|&x| FE::from(&(x & mask)))
                .collect(),
        );

        rc_base_types = rc_base_types.iter().map(|&x| x >> 16).collect();
    }

    // This can't fail since we have 8 pushes
    decomposition_columns.try_into().unwrap()
}

#[cfg(test)]
mod test {
    use crate::cairo::{
        cairo_layout::CairoLayout,
        runner::run::{cairo0_program_path, run_program, CairoVersion},
    };

    use super::*;
    use lambdaworks_math::field::element::FieldElement;

    #[test]
    fn test_rc_decompose() {
        let fifteen = FE::from_hex("000F000F000F000F000F000F000F000F").unwrap();
        let sixteen = FE::from_hex("00100010001000100010001000100010").unwrap();
        let one_two_three = FE::from_hex("00010002000300040005000600070008").unwrap();

        let decomposition_columns =
            decompose_rc_values_into_trace_columns(&[&fifteen, &sixteen, &one_two_three]);

        for row in &decomposition_columns {
            assert_eq!(row[0], FE::from_hex("F").unwrap());
            assert_eq!(row[1], FE::from_hex("10").unwrap());
        }

        assert_eq!(decomposition_columns[0][2], FE::from_hex("8").unwrap());
        assert_eq!(decomposition_columns[1][2], FE::from_hex("7").unwrap());
        assert_eq!(decomposition_columns[2][2], FE::from_hex("6").unwrap());
        assert_eq!(decomposition_columns[3][2], FE::from_hex("5").unwrap());
        assert_eq!(decomposition_columns[4][2], FE::from_hex("4").unwrap());
        assert_eq!(decomposition_columns[5][2], FE::from_hex("3").unwrap());
        assert_eq!(decomposition_columns[6][2], FE::from_hex("2").unwrap());
        assert_eq!(decomposition_columns[7][2], FE::from_hex("1").unwrap());
    }

    #[test]
    fn test_build_main_trace_simple_program() {
        /*
        The following trace and memory files are obtained running the following Cairo program:
        ```
        func main() {
            let x = 1;
            let y = 2;
            assert x + y = 3;
            return ();
        }

        ```
        */

        let program_content = std::fs::read(cairo0_program_path("simple_program.json")).unwrap();
        let (register_states, memory, program_size, _rangecheck_base_end) = run_program(
            None,
            CairoLayout::AllCairo,
            &program_content,
            &CairoVersion::V0,
        )
        .unwrap();
        let pub_inputs = PublicInputs::from_regs_and_mem(
            &register_states,
            &memory,
            program_size,
            &MemorySegmentMap::new(),
        );
        let execution_trace = build_cairo_execution_trace(&register_states, &memory, &pub_inputs);

        // This trace is obtained from Giza when running the prover for the mentioned program.
        let expected_trace = TraceTable::new_from_cols(&vec![
            // col 0
            vec![FE::zero(), FE::zero(), FE::one()],
            // col 1
            vec![FE::one(), FE::one(), FE::one()],
            // col 2
            vec![FE::one(), FE::one(), FE::zero()],
            // col 3
            vec![FE::zero(), FE::zero(), FE::one()],
            // col 4
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 5
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 6
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 7
            vec![FE::zero(), FE::zero(), FE::one()],
            // col 8
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 9
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 10
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 11
            vec![FE::one(), FE::zero(), FE::zero()],
            // col 12
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 13
            vec![FE::zero(), FE::zero(), FE::one()],
            // col 14
            vec![FE::one(), FE::one(), FE::zero()],
            // col 15
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 16
            vec![FE::from(3), FE::from(3), FE::from(9)],
            // col 17
            vec![FE::from(8), FE::from(9), FE::from(9)],
            // col 18
            vec![FE::from(8), FE::from(8), FE::from(8)],
            // col 19
            vec![FE::from(1), FE::from(3), FE::from(5)],
            // col 20
            vec![FE::from(8), FE::from(8), FE::from(6)],
            // col 21
            vec![FE::from(7), FE::from(7), FE::from(7)],
            // col 22
            vec![FE::from(2), FE::from(4), FE::from(7)],
            // col 23
            vec![
                FE::from(0x480680017fff8000),
                FE::from(0x400680017fff7fff),
                FE::from(0x208b7fff7fff7ffe),
            ],
            // col 24
            vec![FE::from(3), FE::from(3), FE::from(9)],
            // col 25
            vec![FE::from(9), FE::from(9), FE::from(9)],
            // col 26
            vec![FE::from(3), FE::from(3), FE::from(9)],
            // col 27
            vec![FE::from(0x8000), FE::from(0x7fff), FE::from(0x7ffe)],
            // col 28
            vec![FE::from(0x7fff), FE::from(0x7fff), FE::from(0x7fff)],
            // col 29
            vec![FE::from(0x8001), FE::from(0x8001), FE::from(0x7fff)],
            // col 30
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 31
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 32
            vec![FE::from(0x1b), FE::from(0x1b), FE::from(0x51)],
            // col 33 - Selector column
            vec![FE::one(), FE::one(), FE::zero()],
        ]);

        assert_eq!(execution_trace.cols(), expected_trace.cols());
    }

    #[test]
    fn test_build_main_trace_call_func_program() {
        /*
        The following trace and memory files are obtained running the following Cairo program:
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

        let program_content = std::fs::read(cairo0_program_path("call_func.json")).unwrap();
        let (register_states, memory, program_size, _rangecheck_base_end) = run_program(
            None,
            CairoLayout::AllCairo,
            &program_content,
            &CairoVersion::V0,
        )
        .unwrap();

        let pub_inputs = PublicInputs::from_regs_and_mem(
            &register_states,
            &memory,
            program_size,
            &MemorySegmentMap::new(),
        );

        let execution_trace = build_cairo_execution_trace(&register_states, &memory, &pub_inputs);

        // This trace is obtained from Giza when running the prover for the mentioned program.
        let expected_trace = TraceTable::new_from_cols(&[
            // col 0
            vec![
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::one(),
            ],
            // col 1
            vec![
                FE::one(),
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::one(),
                FE::one(),
                FE::one(),
            ],
            // col 2
            vec![
                FE::one(),
                FE::one(),
                FE::one(),
                FE::zero(),
                FE::zero(),
                FE::one(),
                FE::zero(),
            ],
            // col 3
            vec![
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::one(),
                FE::one(),
                FE::zero(),
                FE::one(),
            ],
            // col 4
            vec![
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            // col 5
            vec![
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            // col 6
            vec![
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            // col 7
            vec![
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::one(),
            ],
            // col 8
            vec![
                FE::zero(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            // col 9
            vec![
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            // col 10
            vec![
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            // col 11
            vec![
                FE::one(),
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            // col 12
            vec![
                FE::zero(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            // col 13
            vec![
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::one(),
            ],
            // col 14
            vec![
                FE::one(),
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::zero(),
            ],
            // col 15
            vec![
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            // col 16
            vec![
                FE::from(2),
                FE::from(3),
                FE::from_hex_unchecked(
                    "0800000000000010fffffffffffffffffffffffffffffffffffffffffffffffb",
                ),
                FE::from(6),
                FE::from(9),
                FE::from(6),
                FE::from(19),
            ],
            // col 17
            vec![
                FE::from(14),
                FE::from(15),
                FE::from(16),
                FE::from(18),
                FE::from(19),
                FE::from(19),
                FE::from(19),
            ],
            // col 18
            vec![
                FE::from(14),
                FE::from(14),
                FE::from(14),
                FE::from(18),
                FE::from(18),
                FE::from(14),
                FE::from(14),
            ],
            // col 19
            vec![
                FE::from(3),
                FE::from(5),
                FE::from(7),
                FE::from(1),
                FE::from(2),
                FE::from(9),
                FE::from(11),
            ],
            // col 20
            vec![
                FE::from(14),
                FE::from(15),
                FE::from(16),
                FE::from(18),
                FE::from(16),
                FE::from(18),
                FE::from(12),
            ],
            // col 21
            vec![
                FE::from(13),
                FE::from(13),
                FE::from(17),
                FE::from(14),
                FE::from(17),
                FE::from(13),
                FE::from(13),
            ],
            // col 22
            vec![
                FE::from(4),
                FE::from(6),
                FE::from(8),
                FE::from(15),
                FE::from(17),
                FE::from(10),
                FE::from(13),
            ],
            // col 23
            vec![
                FE::from(0x480680017fff8000),
                FE::from(0x480680017fff8000),
                FE::from(0x1104800180018000),
                FE::from(0x484a7ffd7ffc8000),
                FE::from(0x208b7fff7fff7ffe),
                FE::from(0x400680017fff7fff),
                FE::from(0x208b7fff7fff7ffe),
            ],
            // col 24
            vec![
                FE::from(2),
                FE::from(3),
                FE::from(14),
                FE::from(6),
                FE::from(14),
                FE::from(6),
                FE::from(19),
            ],
            // col 25
            vec![
                FE::from(19),
                FE::from(19),
                FE::from(9),
                FE::from(2),
                FE::from(9),
                FE::from(19),
                FE::from(19),
            ],
            // col 26
            vec![
                FE::from(2),
                FE::from(3),
                FE::from_hex_unchecked(
                    "0800000000000010fffffffffffffffffffffffffffffffffffffffffffffffb",
                ),
                FE::from(3),
                FE::from(9),
                FE::from(6),
                FE::from(19),
            ],
            // col 27
            vec![
                FE::from(0x8000),
                FE::from(0x8000),
                FE::from(0x8000),
                FE::from(0x8000),
                FE::from(0x7ffe),
                FE::from(0x7fff),
                FE::from(0x7ffe),
            ],
            // col 28
            vec![
                FE::from(0x7fff),
                FE::from(0x7fff),
                FE::from(0x8001),
                FE::from(0x7ffc),
                FE::from(0x7fff),
                FE::from(0x7fff),
                FE::from(0x7fff),
            ],
            // col 29
            vec![
                FE::from(0x8001),
                FE::from(0x8001),
                FE::from(0x8001),
                FE::from(0x7ffd),
                FE::from(0x7fff),
                FE::from(0x8001),
                FE::from(0x7fff),
            ],
            // col 30
            vec![
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            // col 31
            vec![
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
                FE::zero(),
            ],
            // col 32
            vec![
                FE::from(38),
                FE::from(57),
                FE::from_hex_unchecked(
                    "0800000000000010ffffffffffffffffffffffffffffffffffffffffffffffcb",
                ),
                FE::from(6),
                FE::from(81),
                FE::from(114),
                FE::from(0x169),
            ],
            // col 33 - Selector column
            vec![
                FE::one(),
                FE::one(),
                FE::one(),
                FE::one(),
                FE::one(),
                FE::one(),
                FE::zero(),
            ],
        ]);

        assert_eq!(execution_trace.cols(), expected_trace.cols());
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

        let (col, rc_min, rc_max) = get_rc_holes(&table, &[0, 1, 2]);
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
        fill_rc_holes(&mut main_trace, missing_values);

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

    #[test]
    fn test_get_memory_holes_no_codelen() {
        // We construct a sorted addresses list [1, 2, 3, 6, 7, 8, 9, 13, 14, 15], and
        // set codelen = 0. With this value of codelen, any holes present between
        // the min and max addresses should be returned by the function.
        let mut addrs: Vec<FE> = (1..4).map(FE::from).collect();
        let addrs_extension: Vec<FE> = (6..10).map(FE::from).collect();
        addrs.extend_from_slice(&addrs_extension);
        let addrs_extension: Vec<FE> = (13..16).map(FE::from).collect();
        addrs.extend_from_slice(&addrs_extension);
        let codelen = 0;

        let expected_memory_holes = vec![
            FE::from(4),
            FE::from(5),
            FE::from(10),
            FE::from(11),
            FE::from(12),
        ];
        let calculated_memory_holes = get_memory_holes(&addrs, codelen);

        assert_eq!(expected_memory_holes, calculated_memory_holes);
    }

    #[test]
    fn test_get_memory_holes_inside_program_section() {
        // We construct a sorted addresses list [1, 2, 3, 8, 9] and we
        // set a codelen of 9. Since all the holes will be inside the
        // program segment (meaning from addresses 1 to 9), the function
        // should not return any of them.
        let mut addrs: Vec<FE> = (1..4).map(FE::from).collect();
        let addrs_extension: Vec<FE> = (8..10).map(FE::from).collect();
        addrs.extend_from_slice(&addrs_extension);
        let codelen = 9;

        let calculated_memory_holes = get_memory_holes(&addrs, codelen);
        let expected_memory_holes: Vec<FE> = Vec::new();

        assert_eq!(expected_memory_holes, calculated_memory_holes);
    }

    #[test]
    fn test_get_memory_holes_outside_program_section() {
        // We construct a sorted addresses list [1, 2, 3, 8, 9] and we
        // set a codelen of 6. The holes found inside the program section,
        // i.e. in the address range between 1 to 6, should not be returned.
        // So addresses 4, 5 and 6 will no be returned, only address 7.
        let mut addrs: Vec<FE> = (1..4).map(FE::from).collect();
        let addrs_extension: Vec<FE> = (8..10).map(FE::from).collect();
        addrs.extend_from_slice(&addrs_extension);
        let codelen = 6;

        let calculated_memory_holes = get_memory_holes(&addrs, codelen);
        let expected_memory_holes = vec![FE::from(7)];

        assert_eq!(expected_memory_holes, calculated_memory_holes);
    }

    #[test]
    fn test_fill_memory_holes() {
        const TRACE_COL_LEN: usize = 2;
        const NUM_TRACE_COLS: usize = FRAME_SELECTOR + 1;

        let mut trace_cols = vec![vec![FE::zero(); TRACE_COL_LEN]; NUM_TRACE_COLS];
        trace_cols[FRAME_PC][0] = FE::one();
        trace_cols[FRAME_DST_ADDR][0] = FE::from(2);
        trace_cols[FRAME_OP0_ADDR][0] = FE::from(3);
        trace_cols[FRAME_OP1_ADDR][0] = FE::from(5);
        trace_cols[FRAME_PC][1] = FE::from(6);
        trace_cols[FRAME_DST_ADDR][1] = FE::from(9);
        trace_cols[FRAME_OP0_ADDR][1] = FE::from(10);
        trace_cols[FRAME_OP1_ADDR][1] = FE::from(11);
        let mut trace = TraceTable::new_from_cols(&trace_cols);

        let mut memory_holes = vec![FE::from(4), FE::from(7), FE::from(8)];
        fill_memory_holes(&mut trace, &mut memory_holes);

        let frame_pc = &trace.cols()[FRAME_PC];
        let dst_addr = &trace.cols()[FRAME_DST_ADDR];
        let op0_addr = &trace.cols()[FRAME_OP0_ADDR];
        let op1_addr = &trace.cols()[FRAME_OP1_ADDR];
        assert_eq!(frame_pc[0], FE::one());
        assert_eq!(dst_addr[0], FE::from(2));
        assert_eq!(op0_addr[0], FE::from(3));
        assert_eq!(op1_addr[0], FE::from(5));
        assert_eq!(frame_pc[1], FE::from(6));
        assert_eq!(dst_addr[1], FE::from(9));
        assert_eq!(op0_addr[1], FE::from(10));
        assert_eq!(op1_addr[1], FE::from(11));
    }
}
