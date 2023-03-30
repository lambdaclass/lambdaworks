use thiserror::Error;

#[derive(Error, Debug)]
pub enum CairoImportError {
    #[error("Bytes should be a multiple of 24 for trace or 40 for memory")]
    IncorrectNumberOfBytes,
    #[error("IO Error")]
    FileError(#[from] std::io::Error),
}
