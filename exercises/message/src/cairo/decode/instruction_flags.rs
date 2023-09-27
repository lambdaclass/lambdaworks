use crate::{cairo::errors::InstructionDecodingError, FE};
// Constants for instructions decoding
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

// TODO: This is just an auxiliary function done to get out of the way.
// It should be deleted afterwards
pub(crate) fn aux_get_last_nim_of_field_element(value: &FE) -> u64 {
    let mem_value_bytes = value.to_bytes_be();

    // we are taking the last nim of the field element,
    // since it is a U256
    u64::from_be_bytes([
        mem_value_bytes[24],
        mem_value_bytes[25],
        mem_value_bytes[26],
        mem_value_bytes[27],
        mem_value_bytes[28],
        mem_value_bytes[29],
        mem_value_bytes[30],
        mem_value_bytes[31],
    ])
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

impl CairoInstructionFlags {
    /// Gives a bit trace representation of all flags.
    /// Altough the flags can be interpreted as bits, they are
    /// represented by field elements: FE::zero() for bit 0 and
    /// FE::one() for bit 1.
    #[rustfmt::skip]
    pub fn to_trace_representation(&self) -> [FE; 16] {
        let b0 = self.dst_reg.to_trace_representation();
        let b1 = self.op0_reg.to_trace_representation();
        let [b2, b3, b4] = self.op1_src.to_trace_representation();
        let [b5, b6] = self.res_logic.to_trace_representation();
        let [b7, b8, b9] = self.pc_update.to_trace_representation();
        let [b10, b11] = self.ap_update.to_trace_representation();
        let [b12, b13, b14] = self.opcode.to_trace_representation();

        // In the paper, a little-endian format for the bit flags is
        // mentioned. That is why they are arranged in this way (section 4.4
        // of the Cairo whitepaper - https://eprint.iacr.org/2021/1063.pdf)
        [
            b0,             // dst_reg bits
            b1,             // op0_reg bits
            b4, b3, b2,     // op1_src bits
            b6, b5,         // res_logic bits
            b9, b8, b7,     // pc_update bits
            b11, b10,       // ap_update bits
            b14, b13, b12,  // opcode bits
            FE::zero(),
        ]
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Op0Reg {
    AP = 0,
    FP = 1,
}

impl Op0Reg {
    pub fn to_trace_representation(&self) -> FE {
        match self {
            Op0Reg::AP => FE::zero(),
            Op0Reg::FP => FE::one(),
        }
    }
}

impl TryFrom<&FE> for Op0Reg {
    type Error = InstructionDecodingError;

    fn try_from(mem_value: &FE) -> Result<Self, Self::Error> {
        let flags = aux_get_last_nim_of_field_element(mem_value) >> FLAGS_OFFSET;

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
    AP = 0,
    FP = 1,
}
impl DstReg {
    pub fn to_trace_representation(&self) -> FE {
        match self {
            DstReg::AP => FE::zero(),
            DstReg::FP => FE::one(),
        }
    }
}

impl TryFrom<&FE> for DstReg {
    type Error = InstructionDecodingError;

    fn try_from(mem_value: &FE) -> Result<Self, Self::Error> {
        let flags = aux_get_last_nim_of_field_element(mem_value) >> FLAGS_OFFSET;
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op1Src {
    Op0 = 0,
    Imm = 1,
    FP = 2,
    AP = 4,
}

impl Op1Src {
    pub fn to_trace_representation(&self) -> [FE; 3] {
        match self {
            Op1Src::Op0 => [FE::zero(), FE::zero(), FE::zero()],
            Op1Src::Imm => [FE::zero(), FE::zero(), FE::one()],
            Op1Src::FP => [FE::zero(), FE::one(), FE::zero()],
            Op1Src::AP => [FE::one(), FE::zero(), FE::zero()],
        }
    }
}

impl TryFrom<&FE> for Op1Src {
    type Error = InstructionDecodingError;

    fn try_from(mem_value: &FE) -> Result<Self, Self::Error> {
        let flags = aux_get_last_nim_of_field_element(mem_value) >> FLAGS_OFFSET;
        let op1_src = ((flags & OP1_SRC_MASK) >> OP1_SRC_OFF) as u8;

        match op1_src {
            0 => Ok(Op1Src::Op0),
            1 => Ok(Op1Src::Imm),
            2 => Ok(Op1Src::FP),
            4 => Ok(Op1Src::AP),
            _ => Err(InstructionDecodingError::InvalidOp1Src),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResLogic {
    Op1 = 0,
    Add = 1,
    Mul = 2,
    // TODO: Check if this is correct
    Unconstrained,
}

impl ResLogic {
    pub fn to_trace_representation(&self) -> [FE; 2] {
        match self {
            ResLogic::Op1 => [FE::zero(), FE::zero()],
            ResLogic::Add => [FE::zero(), FE::one()],
            ResLogic::Mul => [FE::one(), FE::zero()],
            ResLogic::Unconstrained => todo!(),
        }
    }
}

impl TryFrom<&FE> for ResLogic {
    type Error = InstructionDecodingError;

    fn try_from(mem_value: &FE) -> Result<Self, Self::Error> {
        let flags = aux_get_last_nim_of_field_element(mem_value) >> FLAGS_OFFSET;
        let res_logic = ((flags & RES_LOGIC_MASK) >> RES_LOGIC_OFF) as u8;

        match res_logic {
            0 => Ok(ResLogic::Op1),
            1 => Ok(ResLogic::Add),
            2 => Ok(ResLogic::Mul),
            // TODO: Check this is correct
            4 => Ok(ResLogic::Unconstrained),
            _ => Err(InstructionDecodingError::InvalidResLogic),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PcUpdate {
    Regular = 0,
    Jump = 1,
    JumpRel = 2,
    Jnz = 4,
}

impl PcUpdate {
    pub fn to_trace_representation(&self) -> [FE; 3] {
        match self {
            PcUpdate::Regular => [FE::zero(), FE::zero(), FE::zero()],
            PcUpdate::Jump => [FE::zero(), FE::zero(), FE::one()],
            PcUpdate::JumpRel => [FE::zero(), FE::one(), FE::zero()],
            PcUpdate::Jnz => [FE::one(), FE::zero(), FE::zero()],
        }
    }
}

impl TryFrom<&FE> for PcUpdate {
    type Error = InstructionDecodingError;

    fn try_from(mem_value: &FE) -> Result<Self, Self::Error> {
        let flags = aux_get_last_nim_of_field_element(mem_value) >> FLAGS_OFFSET;
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
    Regular = 0,
    Add = 1,
    Add1 = 2,
    // TODO: Check if this is correct
    Add2,
}

impl ApUpdate {
    pub fn to_trace_representation(&self) -> [FE; 2] {
        match self {
            ApUpdate::Regular => [FE::zero(), FE::zero()],
            ApUpdate::Add => [FE::zero(), FE::one()],
            ApUpdate::Add1 => [FE::one(), FE::zero()],
            ApUpdate::Add2 => todo!(),
        }
    }
}

impl TryFrom<&FE> for ApUpdate {
    type Error = InstructionDecodingError;

    fn try_from(mem_value: &FE) -> Result<Self, Self::Error> {
        let flags = aux_get_last_nim_of_field_element(mem_value) >> FLAGS_OFFSET;
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

impl TryFrom<&FE> for CairoInstructionFlags {
    type Error = InstructionDecodingError;

    fn try_from(mem_value: &FE) -> Result<Self, Self::Error> {
        Ok(CairoInstructionFlags {
            opcode: CairoOpcode::try_from(mem_value)?,
            pc_update: PcUpdate::try_from(mem_value)?,
            ap_update: ApUpdate::try_from(mem_value)?,
            res_logic: ResLogic::try_from(mem_value)?,
            op1_src: Op1Src::try_from(mem_value)?,
            op0_reg: Op0Reg::try_from(mem_value)?,
            dst_reg: DstReg::try_from(mem_value)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CairoOpcode {
    NOp = 0,
    Call = 1,
    Ret = 2,
    AssertEq = 4,
}

impl CairoOpcode {
    pub fn to_trace_representation(&self) -> [FE; 3] {
        match self {
            CairoOpcode::NOp => [FE::zero(), FE::zero(), FE::zero()],
            CairoOpcode::Call => [FE::zero(), FE::zero(), FE::one()],
            CairoOpcode::Ret => [FE::zero(), FE::one(), FE::zero()],
            CairoOpcode::AssertEq => [FE::one(), FE::zero(), FE::zero()],
        }
    }
}

impl TryFrom<&FE> for CairoOpcode {
    type Error = InstructionDecodingError;

    fn try_from(mem_value: &FE) -> Result<Self, Self::Error> {
        let flags = aux_get_last_nim_of_field_element(mem_value) >> FLAGS_OFFSET;
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
    use super::*;
    /*
    For the purpose of testing the decoding, we are going to use instructions obtained
    directly from valid Cairo programs. The decoding shown here is obtained by inspecting
    cairo-rs:
        * Instruction A:  0x480680017fff8000 ->
            Instruction {
                off0: 0,
                off1: -1,
                off2: 1,
                imm: Some(3618502680826344545094760424199446925499834509564823019178951359862461693953),
                dst_register: AP,
                op0_register: FP,
                op1_addr: Imm,
                res: Op1,
                pc_update: Regular,
                ap_update: Add1,
                fp_update: Regular,
                opcode: AssertEq
            }

        * Instruction B: 0x1104800180018000 ->
             Instruction {
                off0: 0,
                off1: 1,
                off2: 1,
                imm: Some(3618502788666131213697322783095070105623107215331596699973092056135872020275),
                dst_register: AP,
                op0_register: AP,
                op1_addr: Imm,
                res: Op1,
                pc_update: JumpRel,
                ap_update: Add2,
                fp_update: APPlus2,
                opcode: Call
            }

        * Instruction C: 0x208b7fff7fff7ffe ->
            Instruction {
                off0: -2,
                off1: -1,
                off2: -1,
                imm: None,
                dst_register: FP,
                op0_register: FP,
                op1_addr: FP,
                res: Op1,
                pc_update: Jump,
                ap_update: Regular,
                fp_update: Dst,
                opcode: Ret
            }

        * Instruction D: 0xa0680017fff7fff ->
            Instruction { off0: -1,
                off1: -1,
                off2: 1,
                imm: Some(7),
                dst_register: AP,
                op0_register: FP,
                op1_addr: Imm,
                res: Unconstrained,
                pc_update: Jnz,
                ap_update: Add1,
                fp_update: Regular,
                opcode: NOp
            }

        * Instruction E: 0x48327ffc7ffa8000 ->
            Instruction {
                off0: 0,
                off1: -6,
                off2: -4,
                imm: None,
                dst_register: AP,
                op0_register: FP,
                op1_addr: AP,
                res: Add,
                pc_update: Regular,
                ap_update: Add1,
                fp_update: Regular,
                opcode: AssertEq
            }

        * Instruction F: 0x4000800d7ff07fff ->
            Instruction {
                off0: -1,
                off1: -16,
                off2: 13,
                imm: None,
                dst_register: AP,
                op0_register: AP,
                op1_addr: Op0,
                res: Op1,
                pc_update: Regular,
                ap_update: Regular,
                fp_update: Regular,
                opcode: AssertEq
            }

        * Instruction G: 0x48507fff7ffe8000 ->
            Instruction {
                off0: 0,
                off1: -1,
                off2: 1,
                imm: Some(3),
                dst_register: AP,
                op0_register: AP,
                op1_addr: Imm,
                res: Mul,
                pc_update: Regular,
                ap_update: Add1,
                fp_update: Regular,
                opcode: AssertEq
            }

        * Instruction H: 0x40780017fff7fff ->
            Instruction {
                off0: -1,
                off1: -1,
                off2: 1,
                imm: Some(2),
                dst_register: FP,
                op0_register: FP,
                op1_addr: Imm,
                res: Op1,
                pc_update: Regular,
                ap_update: Add,
                fp_update: Regular,
                opcode: NOp
            }
    */

    #[test]
    fn assert_opcode_flag_is_correct() {
        // Instruction A
        let value = FE::from(0x480680017fff8000);

        assert_eq!(CairoOpcode::try_from(&value), Ok(CairoOpcode::AssertEq));
    }

    #[test]
    fn call_opcode_flag_is_correct() {
        // Instruction B
        let value = FE::from(0x1104800180018000);

        assert_eq!(CairoOpcode::try_from(&value), Ok(CairoOpcode::Call));
    }

    #[test]
    fn ret_opcode_flag_is_correct() {
        // Instruction C
        let value = FE::from(0x208b7fff7fff7ffe);

        assert_eq!(CairoOpcode::try_from(&value), Ok(CairoOpcode::Ret));
    }

    #[test]
    fn nop_opcode_flag_is_correct() {
        // Instruction D
        let value = FE::from(0xa0680017fff7fff);

        assert_eq!(CairoOpcode::try_from(&value), Ok(CairoOpcode::NOp));
    }

    #[test]
    fn regular_pc_update_flag_is_correct() {
        // Instruction A
        let value = FE::from(0x480680017fff8000);

        assert_eq!(PcUpdate::try_from(&value), Ok(PcUpdate::Regular));
    }

    #[test]
    fn jump_pc_update_flag_is_correct() {
        // Instruction C
        let value = FE::from(0x208b7fff7fff7ffe);

        assert_eq!(PcUpdate::try_from(&value), Ok(PcUpdate::Jump));
    }

    #[test]
    fn jumprel_pc_update_flag_is_correct() {
        // Instruction B
        let value = FE::from(0x1104800180018000);

        assert_eq!(PcUpdate::try_from(&value), Ok(PcUpdate::JumpRel));
    }

    #[test]
    fn jnz_pc_update_flag_is_correct() {
        // Instruction D
        let value = FE::from(0xa0680017fff7fff);

        assert_eq!(PcUpdate::try_from(&value), Ok(PcUpdate::Jnz));
    }

    #[test]
    fn regular_ap_update_flag_is_correct() {
        // Instruction C
        let value = FE::from(0x208b7fff7fff7ffe);

        assert_eq!(ApUpdate::try_from(&value), Ok(ApUpdate::Regular));
    }

    #[test]
    fn add_ap_update_flag_is_correct() {
        // Instruction H
        let value = FE::from(0x40780017fff7fff);

        assert_eq!(ApUpdate::try_from(&value), Ok(ApUpdate::Add));
    }

    #[test]
    fn add1_ap_update_flag_is_correct() {
        // Instruction A
        let value = FE::from(0x480680017fff8000);

        assert_eq!(ApUpdate::try_from(&value), Ok(ApUpdate::Add1));
    }

    #[test]
    fn op1_res_logic_flag_is_correct() {
        // Instruction A
        let value = FE::from(0x480680017fff8000);

        assert_eq!(ResLogic::try_from(&value), Ok(ResLogic::Op1));
    }

    #[test]
    fn add_res_logic_flag_is_correct() {
        // Instruction E
        let value = FE::from(0x48327ffc7ffa8000);

        assert_eq!(ResLogic::try_from(&value), Ok(ResLogic::Add));
    }

    #[test]
    fn mul_res_logic_flag_is_correct() {
        // Instruction G
        let value = FE::from(0x48507fff7ffe8000);

        assert_eq!(ResLogic::try_from(&value), Ok(ResLogic::Mul));
    }

    #[test]
    fn op0_op1_src_flag_is_correct() {
        // Instruction F
        let value = FE::from(0x4000800d7ff07fff);

        assert_eq!(Op1Src::try_from(&value), Ok(Op1Src::Op0));
    }

    #[test]
    fn imm_op1_src_flag_is_correct() {
        // Instruction A
        let value = FE::from(0x480680017fff8000);

        assert_eq!(Op1Src::try_from(&value), Ok(Op1Src::Imm));
    }

    #[test]
    fn ap_op1_src_flag_is_correct() {
        // Instruction E
        let value = FE::from(0x48327ffc7ffa8000);

        assert_eq!(Op1Src::try_from(&value), Ok(Op1Src::AP));
    }

    #[test]
    fn fp_op1_src_flag_is_correct() {
        // Instruction C
        let value = FE::from(0x208b7fff7fff7ffe);

        assert_eq!(Op1Src::try_from(&value), Ok(Op1Src::FP));
    }

    #[test]
    fn ap_op0_reg_flag_is_correct() {
        // Instruction B
        let value = FE::from(0x1104800180018000);

        assert_eq!(Op0Reg::try_from(&value), Ok(Op0Reg::AP));
    }

    #[test]
    fn fp_op0_reg_flag_is_correct() {
        // Instruction A
        let value = FE::from(0x480680017fff8000);

        assert_eq!(Op0Reg::try_from(&value), Ok(Op0Reg::FP));
    }

    #[test]
    fn ap_dst_reg_flag_is_correct() {
        // Instruction A
        let value = FE::from(0x480680017fff8000);

        assert_eq!(DstReg::try_from(&value), Ok(DstReg::AP));
    }

    #[test]
    fn fp_dst_reg_flag_is_correct() {
        // Instruction C
        let value = FE::from(0x208b7fff7fff7ffe);

        assert_eq!(DstReg::try_from(&value), Ok(DstReg::FP));
    }

    #[test]
    fn decoded_flags_of_assert_are_correct() {
        let value = FE::from(0x400380837ffb8000);
        let expected_flags = CairoInstructionFlags {
            opcode: CairoOpcode::AssertEq,
            pc_update: PcUpdate::Regular,
            ap_update: ApUpdate::Regular,
            op0_reg: Op0Reg::FP,
            op1_src: Op1Src::Op0,
            res_logic: ResLogic::Op1,
            dst_reg: DstReg::FP,
        };

        let flags = CairoInstructionFlags::try_from(&value).unwrap();

        assert_eq!(expected_flags, flags);
    }

    #[test]
    fn flags_trace_representation() {
        // Bit-trace representation for each flag:
        //    DstReg::FP = 1
        //    Op0Reg::FP = 1
        //    Op1Src::Op0 = 0 0 0
        //    ResLogic::Op1 = 0 0
        //    PcUpdate::Regular = 0 0 0
        //    ApUpdate::Regular = 0 0
        //    CairoOpcode::AssertEq = 0 0 1

        let flags = CairoInstructionFlags {
            opcode: CairoOpcode::AssertEq,
            pc_update: PcUpdate::Regular,
            ap_update: ApUpdate::Regular,
            op0_reg: Op0Reg::FP,
            op1_src: Op1Src::Op0,
            res_logic: ResLogic::Op1,
            dst_reg: DstReg::FP,
        };

        #[rustfmt::skip]
        let expected_representation = [
            FE::one(),
            FE::one(),
            FE::zero(), FE::zero(), FE::zero(),
            FE::zero(), FE::zero(),
            FE::zero(), FE::zero(), FE::zero(),
            FE::zero(), FE::zero(),
            FE::zero(), FE::zero(), FE::one(),
            FE::zero(),
        ];

        let representation = flags.to_trace_representation();

        assert_eq!(representation, expected_representation);
    }
}
