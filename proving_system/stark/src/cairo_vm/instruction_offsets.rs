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
        let aux = [
            vectorized_offset[0],
            vectorized_offset[1].overflowing_sub(128).0,
        ];
        i32::from(i16::from_le_bytes(aux))
    }
}

#[cfg(test)]
mod tests {
    use super::super::cairo_mem::CairoMemoryCell;
    use super::InstructionOffsets;

    use lambdaworks_math::unsigned_integer::element::U256;
    #[test]
    fn assert_opcode_flag_is_correct_1() {
        // Instruction A
        let value = U256::from_limbs([0, 0, 0, 0x480680017fff8000]);
        let addr: u64 = 1;

        let mem_cell = CairoMemoryCell {
            address: addr,
            value,
        };

        let instruction_offsets = InstructionOffsets::new(&mem_cell);

        assert_eq!(instruction_offsets.off_dst, 0);
        assert_eq!(instruction_offsets.off_op0, -1);
        assert_eq!(instruction_offsets.off_op1, 1);
    }

    #[test]
    fn assert_opcode_flag_is_correct_2() {
        // Instruction A
        let value = U256::from_limbs([0, 0, 0, 0x208b7fff7fff7ffe]);
        let addr: u64 = 1;

        let mem_cell = CairoMemoryCell {
            address: addr,
            value,
        };

        let instruction_offsets = InstructionOffsets::new(&mem_cell);

        assert_eq!(instruction_offsets.off_dst, -2);
        assert_eq!(instruction_offsets.off_op0, -1);
        assert_eq!(instruction_offsets.off_op1, -1);
    }

    #[test]
    fn assert_opcode_flag_is_correct_3() {
        // Instruction A
        let value = U256::from_limbs([0, 0, 0, 0x48327ffc7ffa8000]);
        let addr: u64 = 1;

        let mem_cell = CairoMemoryCell {
            address: addr,
            value,
        };

        let instruction_offsets = InstructionOffsets::new(&mem_cell);

        assert_eq!(instruction_offsets.off_dst, 0);
        assert_eq!(instruction_offsets.off_op0, -6);
        assert_eq!(instruction_offsets.off_op1, -4);
    }
}
