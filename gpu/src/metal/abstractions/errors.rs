use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Couldn't find a system default device for Metal")]
    MetalDeviceNotFound(),
    #[error("Couldn't create a new Metal library: {0}")]
    MetalLibraryError(String),
    #[error("Couldn't create a new Metal function object: {0}")]
    MetalFunctionError(String),
    #[error("Couldn't create a new Metal compute pipeline: {0}")]
    MetalPipelineError(String),
    #[error("Could not calculate {1} root of unity")]
    RootOfUnityError(String, u64),
    #[error("Input length is {0}, which is not a power of two")]
    InputError(usize),
}
