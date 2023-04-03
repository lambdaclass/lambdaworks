/*
   TO DO: Find a better name
   This is a "pre trace", with everything that's needed for the Stark Trace

   It starts by joining the register pointers with their instructions
*/

use super::{cairo_mem::CairoMemoryCell, errors::InstructionDecodingError};

// Consts copied from cairo-rs
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

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Op0Reg {
    AP,
    FP,
}

impl TryFrom<&CairoMemoryCell> for Op0Reg {
    type Error = InstructionDecodingError;

    fn try_from(cell: &CairoMemoryCell) -> Result<Self, Self::Error> {
        let flags = cell.value.limbs[3] >> FLAGS_OFFSET;
        let op0_reg = ((flags & OP0_REG_MASK) >> OP0_REG_OFF) as u8;

        if op0_reg == 0 {
            Ok(Op0Reg::AP)
        } else if op0_reg == 1 {
            Ok(Op0Reg::FP)
        } else {
            Err(InstructionDecodingError::InvalidOp0Reg)
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum DstReg {
    AP,
    FP,
}

impl TryFrom<&CairoMemoryCell> for DstReg {
    type Error = InstructionDecodingError;

    fn try_from(cell: &CairoMemoryCell) -> Result<Self, Self::Error> {
        let flags = cell.value.limbs[3] >> FLAGS_OFFSET;
        let dst_reg = ((flags & DST_REG_MASK) >> DST_REG_OFF) as u8;

        if dst_reg == 0 {
            Ok(DstReg::AP)
        } else if dst_reg == 1 {
            Ok(DstReg::FP)
        } else {
            Err(InstructionDecodingError::InvalidDstReg)
        }
    }
}

// #[derive(Clone, Debug, PartialEq)]
// pub struct CairoInstruction {
//     pub off0: isize,
//     pub off1: isize,
//     pub off2: isize,
//     pub imm: Option<U256>,
//     pub dst_register: Register,
//     pub op0_register: Register,
//     pub op1_addr: Op1Addr,
//     pub res: Res,
//     pub pc_update: PcUpdate,
//     pub ap_update: ApUpdate,
//     pub fp_update: FpUpdate,
//     pub opcode: CairoOpcode,
// }

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op1Src {
    Op0,
    Imm,
    AP,
    FP,
}

impl TryFrom<&CairoMemoryCell> for Op1Src {
    type Error = InstructionDecodingError;

    fn try_from(cell: &CairoMemoryCell) -> Result<Self, Self::Error> {
        let flags = cell.value.limbs[3] >> FLAGS_OFFSET;
        let op1_src = ((flags & OP1_SRC_MASK) >> OP1_SRC_OFF) as u8;

        match op1_src {
            0 => Ok(Op1Src::Op0),
            1 => Ok(Op1Src::Imm),
            2 => Ok(Op1Src::AP),
            4 => Ok(Op1Src::FP),
            _ => Err(InstructionDecodingError::InvalidOp1Src),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResLogic {
    Op1,
    Add,
    Mul,
    Unconstrained,
}

impl TryFrom<&CairoMemoryCell> for ResLogic {
    type Error = InstructionDecodingError;

    fn try_from(cell: &CairoMemoryCell) -> Result<Self, Self::Error> {
        let flags = cell.value.limbs[3] >> FLAGS_OFFSET;
        let res_logic = ((flags & RES_LOGIC_MASK) >> RES_LOGIC_OFF) as u8;

        match res_logic {
            0 => Ok(ResLogic::Op1),
            1 => Ok(ResLogic::Add),
            2 => Ok(ResLogic::Mul),
            4 => Ok(ResLogic::Unconstrained),
            _ => Err(InstructionDecodingError::InvalidResLogic),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PcUpdate {
    Regular,
    Jump,
    JumpRel,
    Jnz,
}

impl TryFrom<&CairoMemoryCell> for PcUpdate {
    type Error = InstructionDecodingError;

    fn try_from(cell: &CairoMemoryCell) -> Result<Self, Self::Error> {
        let flags = cell.value.limbs[3] >> FLAGS_OFFSET;
        let pc_update = ((flags & PC_UPDATE_MASK) >> PC_UPDATE_OFF) as u8;

        match pc_update {
            0 => Ok(PcUpdate::Regular),
            1 => Ok(PcUpdate::Jump),
            2 => Ok(PcUpdate::JumpRel),
            4 => Ok(PcUpdate::Jnz),
            _ => Err(InstructionDecodingError::InvalidPcUpdate),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApUpdate {
    Regular,
    Add,
    Add1,
    Add2,
}

impl TryFrom<&CairoMemoryCell> for ApUpdate {
    type Error = InstructionDecodingError;

    fn try_from(cell: &CairoMemoryCell) -> Result<Self, Self::Error> {
        let flags = cell.value.limbs[3] >> FLAGS_OFFSET;
        let ap_update = ((flags & AP_UPDATE_MASK) >> AP_UPDATE_OFF) as u8;

        match ap_update {
            0 => Ok(ApUpdate::Regular),
            1 => Ok(ApUpdate::Add),
            2 => Ok(ApUpdate::Add1),
            4 => Ok(ApUpdate::Add2),
            _ => Err(InstructionDecodingError::InvalidApUpdate),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FpUpdate {
    Regular,
    APPlus2,
    Dst,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CairoInstructionFlags {
    pub opcode: CairoOpcode,
    pub ap_update: ApUpdate,
    pub pc_update: PcUpdate,
    pub res_logic: ResLogic,
    pub op1_src: Op1Src,
    pub op0_reg: Op0Reg,
    pub dst_reg: DstReg,
}

impl TryFrom<&CairoMemoryCell> for CairoInstructionFlags {
    type Error = InstructionDecodingError;

    fn try_from(cell: &CairoMemoryCell) -> Result<Self, Self::Error> {
        Ok(CairoInstructionFlags {
            opcode: CairoOpcode::try_from(cell)?,
            pc_update: PcUpdate::try_from(cell)?,
            ap_update: ApUpdate::try_from(cell)?,
            res_logic: ResLogic::try_from(cell)?,
            op1_src: Op1Src::try_from(cell)?,
            op0_reg: Op0Reg::try_from(cell)?,
            dst_reg: DstReg::try_from(cell)?,
        })
    }
}
#[derive(Clone, Debug, PartialEq)]
pub enum CairoOpcode {
    NOp,
    Call,
    Ret,
    AssertEq,
}

impl TryFrom<&CairoMemoryCell> for CairoOpcode {
    type Error = InstructionDecodingError;

    fn try_from(cell: &CairoMemoryCell) -> Result<Self, Self::Error> {
        let flags = cell.value.limbs[3] >> FLAGS_OFFSET;
        let opcode = ((flags & OPCODE_MASK) >> OPCODE_OFF) as u8;

        match opcode {
            0 => Ok(CairoOpcode::NOp),
            1 => Ok(CairoOpcode::Call),
            2 => Ok(CairoOpcode::Ret),
            4 => Ok(CairoOpcode::AssertEq),
            _ => Err(InstructionDecodingError::InvalidOpcode),
        }
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::unsigned_integer::element::U256;

    use super::*;

    #[test]
    fn assert_op_flag_is_correct() {
        // This is an assert eq
        let value = U256::from_limbs([0, 0, 0, 0x480680017fff8000u64]);
        let addr: u64 = 1;

        let mem_cell = CairoMemoryCell {
            address: addr,
            value,
        };

        assert_eq!(CairoOpcode::try_from(&mem_cell), Ok(CairoOpcode::AssertEq));
    }

    #[test]
    fn call_op_flag_is_correct() {
        let value = U256::from_limbs([0, 0, 0, 0x1104800180018000]);
        let addr: u64 = 1;

        let mem_cell = CairoMemoryCell {
            address: addr,
            value,
        };

        assert_eq!(CairoOpcode::try_from(&mem_cell), Ok(CairoOpcode::Call));
    }

    #[test]
    fn ret_op_flag_is_correct() {
        let value = U256::from_limbs([0, 0, 0, 0x208b7fff7fff7ffe]);
        let addr: u64 = 1;

        let mem_cell = CairoMemoryCell {
            address: addr,
            value,
        };

        assert_eq!(CairoOpcode::try_from(&mem_cell), Ok(CairoOpcode::Ret));
    }

    #[test]
    fn nop_op_flag_is_correct() {
        let value = U256::from_limbs([0, 0, 0, 0xa0680017fff7ffe]);
        let addr: u64 = 1;

        let mem_cell = CairoMemoryCell {
            address: addr,
            value,
        };

        assert_eq!(CairoOpcode::try_from(&mem_cell), Ok(CairoOpcode::NOp));
    }

    #[test]
    fn decoded_flags_of_assert_are_correct() {
        let value = U256::from_limbs([0, 0, 0, 0x480680017fff8000u64]);
        let addr: u64 = 1;

        let mem_cell = CairoMemoryCell {
            address: addr,
            value,
        };
        let flags = CairoInstructionFlags::try_from(&mem_cell).unwrap();
        dbg!(flags);
    }
}
