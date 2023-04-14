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

use lambdaworks_math::traits::ByteConversion;

use super::{
    cairo_mem::CairoMemory,
    cairo_trace::CairoTrace,
    instruction_flags::{
        aux_get_last_nim_of_FE, ApUpdate, CairoOpcode, DstReg, Op0Reg, Op1Src, PcUpdate, ResLogic,
    },
};
use crate::{
    air::trace::TraceTable,
    cairo_vm::{instruction_flags::{CairoInstructionFlags, self}, instruction_offsets::InstructionOffsets},
    IsField, FE,
};

pub const FLAG_TRACE_OFFSET: usize = 0;
pub const FLAG_TRACE_WIDTH: usize = 16;
// pub const FLAG_TRACE_RANGE: Range<usize> = range(FLAG_TRACE_OFFSET, FLAG_TRACE_WIDTH);

pub const RES_TRACE_OFFSET: usize = 16;
pub const RES_TRACE_WIDTH: usize = 1;
// pub const RES_TRACE_RANGE: Range<usize> = range(RES_TRACE_OFFSET, RES_TRACE_WIDTH);

pub const MEM_P_TRACE_OFFSET: usize = 17;
pub const MEM_P_TRACE_WIDTH: usize = 2;
// pub const MEM_P_TRACE_RANGE: Range<usize> = range(MEM_P_TRACE_OFFSET, MEM_P_TRACE_WIDTH);

pub const MEM_A_TRACE_OFFSET: usize = 19;
pub const MEM_A_TRACE_WIDTH: usize = 4;
// pub const MEM_A_TRACE_RANGE: Range<usize> = range(MEM_A_TRACE_OFFSET, MEM_A_TRACE_WIDTH);

pub const MEM_V_TRACE_OFFSET: usize = 23;
pub const MEM_V_TRACE_WIDTH: usize = 4;
// pub const MEM_V_TRACE_RANGE: Range<usize> = range(MEM_V_TRACE_OFFSET, MEM_V_TRACE_WIDTH);

pub const OFF_X_TRACE_OFFSET: usize = 27;
pub const OFF_X_TRACE_WIDTH: usize = 3;
// pub const OFF_X_TRACE_RANGE: Range<usize> = range(OFF_X_TRACE_OFFSET, OFF_X_TRACE_WIDTH);

pub const DERIVED_TRACE_OFFSET: usize = 30;
pub const DERIVED_TRACE_WIDTH: usize = 3;
// pub const DERIVED_TRACE_RANGE: Range<usize> = range(DERIVED_TRACE_OFFSET, DERIVED_TRACE_WIDTH);

pub const SELECTOR_TRACE_OFFSET: usize = 33;
pub const SELECTOR_TRACE_WIDTH: usize = 1;
// pub const SELECTOR_TRACE_RANGE: Range<usize> = range(SELECTOR_TRACE_OFFSET, SELECTOR_TRACE_WIDTH);

pub const TRACE_WIDTH: usize = 34;

pub fn build_cairo_execution_trace<F: IsField>(
    raw_trace: &CairoTrace,
    memory: &CairoMemory,
) -> TraceTable<F> {
    let (flags, offsets): (Vec<CairoInstructionFlags>, Vec<InstructionOffsets>) = raw_trace
        .flags_and_offsets(memory)
        .unwrap()
        .into_iter()
        .unzip();

    // TODO: Get res, op0_addr and op1_addr
    let (dst_addrs, dsts): (Vec<FE>, Vec<FE>) = compute_dst(&flags, &offsets, raw_trace, memory);
    let (op0_addrs, op0s): (Vec<FE>, Vec<FE>) = compute_op0(&flags, &offsets, raw_trace, memory);
    let (op1_addrs, op1s, instruction_size): (Vec<FE>, Vec<FE>, Vec<FE>) =
        compute_op1(&flags, &offsets, raw_trace, memory, &op0s);
    let res = compute_res(&flags, &op0s, &op1s);

    let trace_repr_flags: Vec<[FE; 16]> =
        flags.iter().map(|f| f.to_trace_representation()).collect();

    let trace_repr_offsets: Vec<[FE; 3]> = offsets
        .iter()
        .map(|off| off.to_trace_representation())
        .collect();

    todo!()
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
/// - instruction_size
pub fn compute_op1(
    flags: &[CairoInstructionFlags],
    offsets: &[InstructionOffsets],
    raw_trace: &CairoTrace,
    memory: &CairoMemory,
    op0s: &[FE],
) -> (Vec<FE>, Vec<FE>, Vec<FE>) {
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
    let ret = flags
        .iter()
        .zip(offsets)
        .zip(op0s)
        .zip(raw_trace.rows.iter())
        .map(|(((flag, offset), op0), trace_state)| match flag.op1_src {
            Op1Src::Op0 => {
                let instruction_size = FE::one();
                let addr = aux_get_last_nim_of_FE(op0)
                    .checked_add_signed(offset.off_op1.into())
                    .unwrap();
                (
                    FE::from(addr),
                    memory.get(&addr).unwrap().clone(),
                    instruction_size,
                )
            }
            Op1Src::Imm => {
                let instruction_size = FE::from(2);
                let pc = trace_state.pc;
                let addr = pc.checked_add_signed(offset.off_op1.into()).unwrap();
                (
                    FE::from(addr),
                    memory.get(&addr).unwrap().clone(),
                    instruction_size,
                )
            }
            Op1Src::AP => {
                let instruction_size = FE::one();
                let ap = trace_state.ap;
                let addr = ap.checked_add_signed(offset.off_op1.into()).unwrap();
                (
                    FE::from(addr),
                    memory.get(&addr).unwrap().clone(),
                    instruction_size,
                )
            }
            Op1Src::FP => {
                let instruction_size = FE::one();
                let fp = trace_state.fp;
                let addr = fp.checked_add_signed(offset.off_op1.into()).unwrap();
                (
                    FE::from(addr),
                    memory.get(&addr).unwrap().clone(),
                    instruction_size,
                )
            }
        })
        .collect::<(FE, FE, FE)>();
    //ret
        todo!()
}
