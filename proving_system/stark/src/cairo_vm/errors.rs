use thiserror::Error;

#[derive(Error, Debug)]
pub enum CairoImportError {
    #[error("Bytes should be a multiple of 24 for trace or 40 for memory")]
    IncorrectNumberOfBytes,
    #[error("IO Error")]
    FileError(#[from] std::io::Error),
}

#[derive(Error, Debug, PartialEq)]
pub enum InstructionDecodingError {
    #[error("Invalid opcode value")]
    InvalidOpcode,
    #[error("Invalid pc_update value")]
    InvalidPcUpdate,
    #[error("Invalid ap_update value")]
    InvalidApUpdate,
    #[error("Invalid res_logic value")]
    InvalidResLogic,
    #[error("Invalid op1_src value")]
    InvalidOp1Src,
    #[error("Invalid op0_reg value")]
    InvalidOp0Reg,
    #[error("Invalid dst_reg value")]
    InvalidDstReg,
    #[error("Instruction not found in memory")]
    InstructionNotFound,
}
