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

use crate::{
    air::trace::TraceTable,
    cairo_vm::{instruction_flags::CairoInstructionFlags, instruction_offsets::InstructionOffsets},
    IsField, FE,
};

use super::{
    cairo_mem::CairoMemory,
    cairo_trace::CairoTrace,
    instruction_flags::{ApUpdate, CairoOpcode, DstReg, Op0Reg, PcUpdate, ResLogic},
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
    let (op1_addrs, op1s): (Vec<FE>, Vec<FE>) = compute_op1(&flags, &offsets, raw_trace, memory);

    let trace_repr_flags: Vec<[FE; 16]> =
        flags.iter().map(|f| f.to_trace_representation()).collect();

    let trace_repr_offsets: Vec<[FE; 3]> = offsets
        .iter()
        .map(|off| off.to_trace_representation())
        .collect();

    todo!()
}

pub fn compute_res(
    flags: &[CairoInstructionFlags],
    offsets: &[InstructionOffsets],
    raw_trace: &CairoTrace,
    memory: &CairoMemory,
) -> Vec<FE> {
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

    flags.iter().map(|f| {
        match f.pc_update {
            PcUpdate::Jnz => {
                match (f.res_logic, f.opcode, f.ap_update) {
                    (
                        ResLogic::Op1,
                        CairoOpcode::NOp,
                        ApUpdate::Regular | ApUpdate::Add1 | ApUpdate::Add2,
                    ) => {
                        // res = Unused
                    }
                    _ => {
                        panic!("Undefined Behavior");
                    }
                }
            }
            PcUpdate::Regular | PcUpdate::Jump | PcUpdate::JumpRel => {
                match f.res_logic {
                    ResLogic::Op1 => {
                        // res = op1
                    }
                    ResLogic::Add => {
                        // res = op0 + op1
                    }
                    ResLogic::Mul => {
                        // res = op0 * op1
                    }
                    ResLogic::Unconstrained => {
                        panic!("Undefined Behavior");
                    }
                }
            }
        };
    });

    todo!()
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

pub fn compute_op1(
    flags: &[CairoInstructionFlags],
    offsets: &[InstructionOffsets],
    raw_trace: &CairoTrace,
    memory: &CairoMemory,
) -> (Vec<FE>, Vec<FE>) {
    todo!()
}
