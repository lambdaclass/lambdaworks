use thiserror::Error;

/// Errors that can occur when initializing or using the Metal GPU state.
#[derive(Debug, Error)]
pub enum GpuStateError {
    /// Metal is not available on this platform.
    #[error("Metal GPU is not available on this platform")]
    NotAvailable,

    /// Failed to find a Metal-compatible GPU device.
    #[error("No Metal-compatible GPU device found")]
    DeviceNotFound,

    /// Failed to initialize the underlying MetalState.
    #[error("Failed to initialize Metal state: {0}")]
    InitError(String),
}

/// GPU state wrapper for the STARK prover.
///
/// On macOS with the `metal` feature enabled, this wraps the `MetalState`
/// from `lambdaworks-gpu` and provides access to Metal GPU resources.
///
/// On other platforms (or without the `metal` feature), constructing this
/// type will return an error.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct StarkMetalState {
    inner: lambdaworks_gpu::metal::abstractions::state::MetalState,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl StarkMetalState {
    /// Creates a new `StarkMetalState` by initializing the default Metal device.
    ///
    /// # Errors
    ///
    /// Returns `GpuStateError::DeviceNotFound` if no Metal device is available.
    /// Returns `GpuStateError::InitError` if the Metal shader library fails to load.
    pub fn new() -> Result<Self, GpuStateError> {
        let inner = lambdaworks_gpu::metal::abstractions::state::MetalState::new(None)
            .map_err(|e| GpuStateError::InitError(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Returns a reference to the underlying `MetalState`.
    pub fn inner(&self) -> &lambdaworks_gpu::metal::abstractions::state::MetalState {
        &self.inner
    }
}

/// Stub for non-macOS platforms or when the `metal` feature is disabled.
#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub struct StarkMetalState {
    _private: (),
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
impl StarkMetalState {
    /// Always returns an error on non-Metal platforms.
    pub fn new() -> Result<Self, GpuStateError> {
        Err(GpuStateError::NotAvailable)
    }
}
