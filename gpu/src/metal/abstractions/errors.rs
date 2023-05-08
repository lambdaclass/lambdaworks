use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Couldn't find a system default device for Metal")]
    DeviceNotFound(),
    #[error("Couldn't create a new Metal library: {0}")]
    LibraryError(String),
    #[error("Couldn't create a new Metal function object: {0}")]
    FunctionError(String),
    #[error("Couldn't create a new Metal compute pipeline: {0}")]
    PipelineError(String),
}
