//! Metal error types for GPU operations.
//!
//! Inspired by the original lambdaworks Metal implementation and ICICLE's
//! multi-backend error handling approach.

use thiserror::Error;

/// Errors that can occur during Metal GPU operations.
#[derive(Debug, Error)]
pub enum MetalError {
    /// No Metal-compatible GPU device was found.
    #[error("Metal device not found - ensure you're running on Apple Silicon or a Mac with a compatible GPU")]
    DeviceNotFound,

    /// Failed to load the Metal shader library.
    #[error("Failed to load Metal library: {0}")]
    LibraryError(String),

    /// Failed to get a function from the Metal library.
    #[error("Failed to get Metal function '{0}'")]
    FunctionError(String),

    /// Failed to create a compute pipeline.
    #[error("Failed to create Metal pipeline: {0}")]
    PipelineError(String),

    /// Invalid input provided to a Metal operation.
    #[error("Invalid input: length {0} is not a power of two")]
    InputError(usize),

    /// Failed to allocate GPU memory.
    #[error("Failed to allocate Metal buffer: {0}")]
    AllocationError(String),

    /// A Metal command failed to execute.
    #[error("Metal command execution failed: {0}")]
    ExecutionError(String),
}
