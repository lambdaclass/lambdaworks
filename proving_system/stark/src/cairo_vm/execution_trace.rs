use super::{
    cairo_mem::CairoMemory,
    cairo_trace::CairoTrace,
    instruction_flags::{
        aux_get_last_nim_of_field_element, ApUpdate, CairoOpcode, DstReg, Op0Reg, Op1Src, PcUpdate,
        ResLogic,
    },
};
use crate::{
    air::trace::TraceTable,
    cairo_vm::{instruction_flags::CairoInstructionFlags, instruction_offsets::InstructionOffsets},
    FE,
};
use lambdaworks_math::{
    field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, helpers,
};

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

/// Receives the raw Cairo trace and memory as outputted from the Cairo VM and returns
/// the trace table used to feed the Cairo STARK prover.
/// The constraints of the Cairo AIR are defined over this trace rather than the raw trace
/// obtained from the Cairo VM, this is why this function is needed.
pub fn build_cairo_execution_trace(
    raw_trace: &CairoTrace,
    memory: &CairoMemory,
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
        .map(|t| memory.get(&t.pc).unwrap().clone())
        .collect();

    // t0, t1 and mul derived values are constructed. For details refer to
    // section 9.1 of the Cairo whitepaper
    let t0: Vec<FE> = trace_repr_flags
        .iter()
        .zip(&dsts)
        .map(|(repr_flags, dst)| repr_flags[9].clone() * dst)
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

    TraceTable::new_from_cols(&trace_cols)
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
                                dst.clone()
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
                    ResLogic::Op1 => op1.clone(),
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
    raw_trace: &CairoTrace,
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
        .zip(raw_trace.rows.iter())
        .map(|((f, o), t)| match f.dst_reg {
            DstReg::AP => {
                let addr = t.ap.checked_add_signed(o.off_dst.into()).unwrap();
                (FE::from(addr), memory.get(&addr).unwrap().clone())
            }
            DstReg::FP => {
                let addr = t.fp.checked_add_signed(o.off_dst.into()).unwrap();
                (FE::from(addr), memory.get(&addr).unwrap().clone())
            }
        })
        .unzip()
}

/// Returns the vector of:
/// - op0_addrs
/// - op0s
pub fn compute_op0(
    flags: &[CairoInstructionFlags],
    offsets: &[InstructionOffsets],
    raw_trace: &CairoTrace,
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
        .zip(raw_trace.rows.iter())
        .map(|((f, o), t)| match f.op0_reg {
            Op0Reg::AP => {
                let addr = t.ap.checked_add_signed(o.off_op0.into()).unwrap();
                (FE::from(addr), memory.get(&addr).unwrap().clone())
            }
            Op0Reg::FP => {
                let addr = t.fp.checked_add_signed(o.off_op0.into()).unwrap();
                (FE::from(addr), memory.get(&addr).unwrap().clone())
            }
        })
        .unzip()
}

/// Returns the vector of:
/// - op1_addrs
/// - op1s
pub fn compute_op1(
    flags: &[CairoInstructionFlags],
    offsets: &[InstructionOffsets],
    raw_trace: &CairoTrace,
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
        .zip(raw_trace.rows.iter())
        .map(|(((flag, offset), op0), trace_state)| match flag.op1_src {
            Op1Src::Op0 => {
                let addr = aux_get_last_nim_of_field_element(op0)
                    .checked_add_signed(offset.off_op1.into())
                    .unwrap();
                (FE::from(addr), memory.get(&addr).unwrap().clone())
            }
            Op1Src::Imm => {
                let pc = trace_state.pc;
                let addr = pc.checked_add_signed(offset.off_op1.into()).unwrap();
                (FE::from(addr), memory.get(&addr).unwrap().clone())
            }
            Op1Src::AP => {
                let ap = trace_state.ap;
                let addr = ap.checked_add_signed(offset.off_op1.into()).unwrap();
                (FE::from(addr), memory.get(&addr).unwrap().clone())
            }
            Op1Src::FP => {
                let fp = trace_state.fp;
                let addr = fp.checked_add_signed(offset.off_op1.into()).unwrap();
                (FE::from(addr), memory.get(&addr).unwrap().clone())
            }
        })
        .unzip()
}

/// Depending on the instruction opcodes, some values should be updated.
/// This function updates op0s, dst, res in place when the conditions hold.
fn update_values(
    flags: &[CairoInstructionFlags],
    raw_trace: &CairoTrace,
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
            op0s[i] = (raw_trace.rows[i].pc + instruction_size).into();
            dst[i] = raw_trace.rows[i].fp.into();
        } else if f.opcode == CairoOpcode::AssertEq {
            res[i] = dst[i].clone();
        }
    }
}

/// Utility function to change from a rows representation to a columns
/// representation of a slice of arrays.   
fn rows_to_cols<const N: usize>(rows: &[[FE; N]]) -> Vec<Vec<FE>> {
    let mut cols = Vec::new();
    let n_cols = rows[0].len();

    for col_idx in 0..n_cols {
        let mut col = Vec::new();
        for row in rows {
            col.push(row[col_idx].clone());
        }
        cols.push(col);
    }
    cols
}

#[cfg(test)]
mod test {
    use super::*;

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

        let base_dir = env!("CARGO_MANIFEST_DIR");
        let dir_trace = base_dir.to_owned() + "/src/cairo_vm/test_data/simple_program.trace";
        let dir_memory = base_dir.to_owned() + "/src/cairo_vm/test_data/simple_program.mem";

        let raw_trace = CairoTrace::from_file(&dir_trace).unwrap();
        let memory = CairoMemory::from_file(&dir_memory).unwrap();

        let execution_trace = build_cairo_execution_trace(&raw_trace, &memory);

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

        let base_dir = env!("CARGO_MANIFEST_DIR");
        let dir_trace = base_dir.to_owned() + "/src/cairo_vm/test_data/call_func.trace";
        let dir_memory = base_dir.to_owned() + "/src/cairo_vm/test_data/call_func.mem";

        let raw_trace = CairoTrace::from_file(&dir_trace).unwrap();
        let memory = CairoMemory::from_file(&dir_memory).unwrap();

        let execution_trace = build_cairo_execution_trace(&raw_trace, &memory);

        // This trace is obtained from Giza when running the prover for the mentioned program.
        let expected_trace = TraceTable::new_from_cols(&vec![
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
}
