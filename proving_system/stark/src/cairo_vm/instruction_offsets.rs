use super::cairo_mem::CairoMemoryCell;

const OFF_DST_OFF: u32 = 0;
const OFF_OP0_OFF: u32 = 16;
const OFF_OP1_OFF: u32 = 32;
const OFFX_MASK: u64 = 0xFFFF;

pub struct InstructionOffsets {
    pub off_dst: i32,
    pub off_op0: i32,
    pub off_op1: i32,
}

impl InstructionOffsets {
    pub fn new(cell: &CairoMemoryCell) -> Self {
        Self {
            off_dst: Self::decode_offset(cell, OFF_DST_OFF),
            off_op0: Self::decode_offset(cell, OFF_OP0_OFF),
            off_op1: Self::decode_offset(cell, OFF_OP1_OFF),
        }
    }

    fn decode_offset(cell: &CairoMemoryCell, instruction_offset: u32) -> i32 {
        let offset = cell.value.limbs[3] >> instruction_offset & OFFX_MASK;
        let vectorized_offset = offset.to_le_bytes();
        let offset_16b_encoded = u16::from_le_bytes([vectorized_offset[0], vectorized_offset[1]]);
        let complement_const = 0x8000u16;
        let (offset_16b, _) = offset_16b_encoded.overflowing_sub(complement_const);
        i32::from(offset_16b)
    }
}
