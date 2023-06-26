use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("The order of polynomial + 1 should a be power of 2. Got: {0}")]
    InvalidOrder(usize),
    #[error("Couldn't load compiled PTX: {0}")]
    PtxError(String),
    #[error("Couldn't get CUDA function: {0}")]
    FunctionError(String),
    #[error("Couldn't find a CUDA device: {0}")]
    DeviceNotFound(String),
    #[error("Couldn't allocate memory for copying: {0}")]
    AllocateMemory(String),
    #[error("Couldn't retrieve information from GPU: {0}")]
    RetrieveMemory(String),
    #[error("Couldn't launch CUDA function: {0}")]
    Launch(String),
    #[error("Index out of bounds: {0}. Length of buffer is {0}")]
    IndexOutOfBounds(usize, usize),
}
