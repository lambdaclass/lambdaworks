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

use lambdaworks_math::field::traits::IsField;

use crate::air::trace::TraceTable;

use super::{cairo_mem::CairoMemory, cairo_trace::CairoTrace};

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
    // CREATE FLAG COLUMS
    // let in = .

    // CREATE

    todo!()
}
