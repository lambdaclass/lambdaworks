use super::{
    cairo_mem::CairoMemory,
    decode::{instruction_flags::CairoInstructionFlags, instruction_offsets::InstructionOffsets},
    errors::{CairoImportError, InstructionDecodingError},
};
use std::fs;

#[derive(PartialEq, Clone, Debug)]
pub struct RegistersState {
    pub pc: u64,
    pub fp: u64,
    pub ap: u64,
}

impl RegistersState {
    fn instruction_flags_and_offsets(
        &self,
        memory: &CairoMemory,
    ) -> Result<(CairoInstructionFlags, InstructionOffsets), InstructionDecodingError> {
        let instruction = memory
            .get(&self.pc)
            .ok_or(InstructionDecodingError::InstructionNotFound)?;

        let flags = CairoInstructionFlags::try_from(instruction)?;
        let offsets = InstructionOffsets::new(instruction);

        Ok((flags, offsets))
    }
}

#[derive(PartialEq, Clone, Debug)]
pub struct RegisterStates {
    pub rows: Vec<RegistersState>,
}

impl RegisterStates {
    pub fn steps(&self) -> usize {
        self.rows.len()
    }

    pub fn flags_and_offsets(
        &self,
        memory: &CairoMemory,
    ) -> Result<Vec<(CairoInstructionFlags, InstructionOffsets)>, InstructionDecodingError> {
        self.rows
            .iter()
            .map(|state| state.instruction_flags_and_offsets(memory))
            .collect()
    }

    pub fn from_bytes_le(bytes: &[u8]) -> Result<Self, CairoImportError> {
        // Each row of the trace is a RegisterState
        // ap, fp, pc, each 8 bytes long (u64)
        const ROW_SIZE: usize = 8 * 3;

        if bytes.len() % ROW_SIZE != 0 {
            return Err(CairoImportError::IncorrectNumberOfBytes);
        }
        let num_rows = bytes.len() / ROW_SIZE;

        let rows = (0..num_rows)
            .map(|i| RegistersState {
                ap: u64::from_le_bytes(bytes[i * ROW_SIZE..i * ROW_SIZE + 8].try_into().unwrap()),
                fp: u64::from_le_bytes(
                    bytes[i * ROW_SIZE + 8..i * ROW_SIZE + 16]
                        .try_into()
                        .unwrap(),
                ),
                pc: u64::from_le_bytes(
                    bytes[i * ROW_SIZE + 16..i * 24 + ROW_SIZE]
                        .try_into()
                        .unwrap(),
                ),
            })
            .collect::<Vec<RegistersState>>();

        Ok(Self { rows })
    }

    pub fn from_file(path: &str) -> Result<Self, CairoImportError> {
        let data = fs::read(path)?;
        Self::from_bytes_le(&data)
    }
}

#[cfg(test)]
mod tests {
    use crate::{cairo::decode::instruction_flags::*, FE};

    use super::*;
    use std::collections::HashMap;

    #[test]
    fn mul_program_gives_expected_trace() {
        /*
        Hex from the trace of the following cairo program

        func main() {
            let x = 2;
            let y = 3;
            assert x * y = 6;
            return();
        }

        Generated with:

        cairo-compile multiply.cairo --output multiply.out

        cairo-run --layout all --trace_file trace.out --memory_file mem.out --program multiply.out

        xxd -p trace.out
         */

        let bytes = hex::decode("080000000000000008000000000000000100000000000000090000000000000008000000000000000300000000000000090000000000000008000000000000000500000000000000").unwrap();

        let register_states = RegisterStates::from_bytes_le(&bytes);

        let expected_state0 = RegistersState {
            ap: 8,
            fp: 8,
            pc: 1,
        };

        let expected_state1 = RegistersState {
            ap: 9,
            fp: 8,
            pc: 3,
        };

        let expected_state2 = RegistersState {
            ap: 9,
            fp: 8,
            pc: 5,
        };

        let expected_reg_states = RegisterStates {
            rows: [expected_state0, expected_state1, expected_state2].to_vec(),
        };

        assert_eq!(register_states.unwrap(), expected_reg_states)
    }

    #[test]
    fn wrong_amount_of_bytes_gives_err() {
        let bytes = hex::decode("080000000000").unwrap();

        match RegisterStates::from_bytes_le(&bytes) {
            Err(CairoImportError::IncorrectNumberOfBytes) => (),
            Err(_) => panic!(),
            Ok(_) => panic!(),
        }
    }

    #[test]
    fn loads_mul_trace_from_file_correctly() {
        let base_dir = env!("CARGO_MANIFEST_DIR");
        dbg!(base_dir);
        let dir = base_dir.to_owned() + "/tests/data/mul_trace.out";

        let register_states = RegisterStates::from_file(&dir).unwrap();

        let expected_state0 = RegistersState {
            ap: 8,
            fp: 8,
            pc: 1,
        };

        let expected_state1 = RegistersState {
            ap: 9,
            fp: 8,
            pc: 3,
        };

        let expected_state2 = RegistersState {
            ap: 9,
            fp: 8,
            pc: 5,
        };

        let expected_reg_states = RegisterStates {
            rows: [expected_state0, expected_state1, expected_state2].to_vec(),
        };

        assert_eq!(register_states, expected_reg_states);
    }

    #[test]
    fn decode_instruction_flags_and_offsets() {
        let data = HashMap::from([
            (1u64, FE::from(0x480680017fff8000)),
            (2u64, FE::from(0x1104800180018000)),
        ]);

        let memory = CairoMemory::new(data);
        let state1 = RegistersState {
            ap: 8,
            fp: 8,
            pc: 1,
        };
        let state2 = RegistersState {
            ap: 9,
            fp: 8,
            pc: 2,
        };

        let trace = RegisterStates {
            rows: [state1, state2].to_vec(),
        };

        let expected_flags1 = CairoInstructionFlags {
            opcode: CairoOpcode::AssertEq,
            pc_update: PcUpdate::Regular,
            ap_update: ApUpdate::Add1,
            op0_reg: Op0Reg::FP,
            op1_src: Op1Src::Imm,
            res_logic: ResLogic::Op1,
            dst_reg: DstReg::AP,
        };

        let expected_offsets1 = InstructionOffsets {
            off_dst: 0,
            off_op0: -1,
            off_op1: 1,
        };

        let expected_flags2 = CairoInstructionFlags {
            opcode: CairoOpcode::Call,
            pc_update: PcUpdate::JumpRel,
            ap_update: ApUpdate::Regular,
            op0_reg: Op0Reg::AP,
            op1_src: Op1Src::Imm,
            res_logic: ResLogic::Op1,
            dst_reg: DstReg::AP,
        };

        let expected_offsets2 = InstructionOffsets {
            off_dst: 0,
            off_op0: 1,
            off_op1: 1,
        };

        let flags_and_offsets = trace.flags_and_offsets(&memory).unwrap();

        assert_eq!(
            flags_and_offsets,
            vec![
                (expected_flags1, expected_offsets1),
                (expected_flags2, expected_offsets2)
            ]
        );
    }
}
