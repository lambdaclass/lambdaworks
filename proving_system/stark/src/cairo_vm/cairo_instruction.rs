/*
    TO DO: Find a better name
    This is a "pre trace", with everything that's needed for the Stark Trace

    It starts by joining the register pointers with their instructions 
 */

use super::cairo_mem::CairoMemoryCell;


// Consts copied from Cairo RS
const DST_REG_MASK: u64 = 0x0001;
const DST_REG_OFF: u64 = 0;
const OP0_REG_MASK: u64 = 0x0002;
const OP0_REG_OFF: u64 = 1;
const OP1_SRC_MASK: u64 = 0x001C;
const OP1_SRC_OFF: u64 = 2;
const RES_LOGIC_MASK: u64 = 0x0060;
const RES_LOGIC_OFF: u64 = 5;
const PC_UPDATE_MASK: u64 = 0x0380;
const PC_UPDATE_OFF: u64 = 7;
const AP_UPDATE_MASK: u64 = 0x0C00;
const AP_UPDATE_OFF: u64 = 10;
const OPCODE_MASK: u64 = 0x7000;
const OPCODE_OFF: u64 = 12;
const FLAGS_OFFSET: u64 = 48;

#[derive(Clone,Debug,PartialEq)]
pub struct CairoInstructionFlags {
    pub op_code: u8,
    pub ap_update: u8,
    pub pc_update: u8,
    pub res_logic: u8,
    pub op1_src: u8,
    pub op0_reg: u8,
    pub dst_src: u8
}

impl From<&CairoMemoryCell> for CairoInstructionFlags {
    fn from(cell: &CairoMemoryCell) -> Self {
        CairoInstructionFlags { 
            op_code: op_code(cell), 
            ap_update: ap_update(cell),
            pc_update: pc_update(cell), 
            res_logic: res_logic(cell), 
            op1_src: op1_src(cell), 
            op0_reg: op0_reg(cell), 
            dst_src: dst_src(cell) }
    }
}

pub fn op_code(cell: &CairoMemoryCell) -> u8 {
    let flags = cell.value.limbs[3] >> FLAGS_OFFSET;
    ((flags & OPCODE_MASK)  >> OPCODE_OFF) as u8
}

pub fn ap_update(cell: &CairoMemoryCell) -> u8 {
    let flags = cell.value.limbs[3] >> FLAGS_OFFSET;
    ((flags & AP_UPDATE_MASK)  >> AP_UPDATE_OFF) as u8
}

pub fn pc_update(cell: &CairoMemoryCell) -> u8 {
    let flags = cell.value.limbs[3] >> FLAGS_OFFSET;
    ((flags & PC_UPDATE_MASK)  >> PC_UPDATE_OFF) as u8
}

pub fn res_logic(cell: &CairoMemoryCell) -> u8 {
    let flags = cell.value.limbs[3] >> FLAGS_OFFSET;
    ((flags & RES_LOGIC_MASK)  >> RES_LOGIC_OFF) as u8
}

pub fn op1_src(cell: &CairoMemoryCell) -> u8 {
    let flags = cell.value.limbs[3] >> FLAGS_OFFSET;
    ((flags & OP1_SRC_MASK)  >> OP1_SRC_OFF) as u8
}

pub fn op0_reg(cell: &CairoMemoryCell) -> u8 {
    let flags = cell.value.limbs[3] >> FLAGS_OFFSET;
    ((flags & OP0_REG_MASK)  >> OP0_REG_OFF) as u8
}

pub fn dst_src(cell: &CairoMemoryCell) -> u8 {
    let flags = cell.value.limbs[3] >> FLAGS_OFFSET;
    ((flags & DST_REG_MASK)  >> DST_REG_OFF) as u8
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::unsigned_integer::element::U256;

    use super::*;

    #[test]
    fn assert_op_flag_is_correct() {
        // This is an assert eq
        let value = U256::from_limbs([0,0,0,0x480680017fff8000u64]);
        let addr: u64 = 1;

        let mem_cell = CairoMemoryCell {
            address: addr,
            value
        };

        assert_eq!(op_code(&mem_cell),4);
    }

    #[test]
    fn decoded_flags_of_assert_are_correct() {

        let value = U256::from_limbs([0,0,0,0x480680017fff8000u64]);
        let addr: u64 = 1;

        let mem_cell = CairoMemoryCell {
            address: addr,
            value
        };
        let flags = CairoInstructionFlags::from(&mem_cell);
        dbg!(flags);
    }
}
