#[derive(Debug)]
pub enum CairoImportError {
    /// Bytes should be a multiple of 24 for trace or 40 for memory
    IncorrectNumberOfBytes,
    FileError(std::io::Error),
}

impl From<std::io::Error> for CairoImportError {
    fn from(err: std::io::Error) -> CairoImportError {
        CairoImportError::FileError(err)
    }
}

#[derive(Debug, PartialEq)]
pub enum InstructionDecodingError {
    InvalidOpcode,
    InvalidPcUpdate,
    InvalidApUpdate,
    InvalidResLogic,
    InvalidOp1Src,
    InvalidOp0Reg,
    InvalidDstReg,
    InstructionNotFound,
}
