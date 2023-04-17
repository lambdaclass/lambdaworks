use super::{
    cairo_mem::CairoMemory,
    cairo_trace::CairoTrace,
    instruction_flags::{
        aux_get_last_nim_of_FE, ApUpdate, CairoOpcode, DstReg, Op0Reg, Op1Src, PcUpdate, ResLogic,
    },
};
use crate::{
    air::trace::TraceTable,
    cairo_vm::{instruction_flags::CairoInstructionFlags, instruction_offsets::InstructionOffsets},
    FE,
};
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

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

pub fn build_cairo_execution_trace(
    raw_trace: &CairoTrace,
    memory: &CairoMemory,
) -> TraceTable<Stark252PrimeField> {
    let (flags, offsets): (Vec<CairoInstructionFlags>, Vec<InstructionOffsets>) = raw_trace
        .flags_and_offsets(memory)
        .unwrap()
        .into_iter()
        .unzip();

    let (dst_addrs, mut dsts): (Vec<FE>, Vec<FE>) =
        compute_dst(&flags, &offsets, raw_trace, memory);
    let (op0_addrs, mut op0s): (Vec<FE>, Vec<FE>) =
        compute_op0(&flags, &offsets, raw_trace, memory);
    let (op1_addrs, op1s): (Vec<FE>, Vec<FE>) =
        compute_op1(&flags, &offsets, raw_trace, memory, &op0s);
    let mut res = compute_res(&flags, &op0s, &op1s);

    update_values(&flags, raw_trace, &mut op0s, &mut dsts, &mut res);

    let trace_repr_flags: Vec<[FE; 16]> =
        flags.iter().map(|f| f.to_trace_representation()).collect();

    let trace_repr_offsets: Vec<[FE; 3]> = offsets
        .iter()
        .map(|off| off.to_trace_representation())
        .collect();

    let aps: Vec<FE> = raw_trace.rows.iter().map(|t| FE::from(t.ap)).collect();
    let fps: Vec<FE> = raw_trace.rows.iter().map(|t| FE::from(t.fp)).collect();
    let pcs: Vec<FE> = raw_trace.rows.iter().map(|t| FE::from(t.pc)).collect();
    let instructions: Vec<FE> = raw_trace
        .rows
        .iter()
        .map(|t| memory.get(&t.pc).unwrap().clone())
        .collect();

    let t0: Vec<FE> = trace_repr_flags[9]
        .iter()
        .zip(&dsts)
        .map(|(f_pc_jnz, dst)| f_pc_jnz * dst)
        .collect();
    let t1: Vec<FE> = t0.iter().zip(&res).map(|(t, r)| t * r).collect();
    let mul: Vec<FE> = op0s.iter().zip(&op1s).map(|(op0, op1)| op0 * op1).collect();

    // Build Cairo trace columns to instantiate TraceTable struct
    let mut trace_cols: Vec<Vec<_>> = Vec::new();
    (0..trace_repr_flags.len()).for_each(|n| trace_cols.push(trace_repr_flags[n].to_vec()));
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
    (0..trace_repr_offsets.len()).for_each(|n| trace_cols.push(trace_repr_offsets[n].to_vec()));
    trace_cols.push(t0);
    trace_cols.push(t1);
    trace_cols.push(mul);

    TraceTable::new_from_cols(&trace_cols)
}

pub fn compute_res(flags: &[CairoInstructionFlags], op0s: &[FE], op1s: &[FE]) -> Vec<FE> {
    /*
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
        .map(|((f, op0), op1)| {
            match f.pc_update {
                PcUpdate::Jnz => {
                    match (&f.res_logic, &f.opcode, &f.ap_update) {
                        (
                            ResLogic::Op1,
                            CairoOpcode::NOp,
                            ApUpdate::Regular | ApUpdate::Add1 | ApUpdate::Add2,
                        ) => {
                            // res = Unused
                            FE::zero()
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

pub fn compute_dst(
    flags: &[CairoInstructionFlags],
    offsets: &[InstructionOffsets],
    raw_trace: &CairoTrace,
    memory: &CairoMemory,
) -> (Vec<FE>, Vec<FE>) {
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

pub fn compute_op0(
    flags: &[CairoInstructionFlags],
    offsets: &[InstructionOffsets],
    raw_trace: &CairoTrace,
    memory: &CairoMemory,
) -> (Vec<FE>, Vec<FE>) {
    flags
        .iter()
        .zip(offsets)
        .zip(raw_trace.rows.iter())
        .map(|((f, o), t)| match f.op0_reg {
            Op0Reg::AP => {
                let addr = t.ap.checked_add_signed(o.off_dst.into()).unwrap();
                (FE::from(addr), memory.get(&addr).unwrap().clone())
            }
            Op0Reg::FP => {
                let addr = t.fp.checked_add_signed(o.off_dst.into()).unwrap();
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
    /*
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
                let addr = aux_get_last_nim_of_FE(op0)
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

pub fn update_values(
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

#[cfg(test)]
mod test {}
