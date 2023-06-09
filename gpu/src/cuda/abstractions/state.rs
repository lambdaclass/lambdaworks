use super::errors::CudaError;
use cudarc::{
    driver::{safe::CudaSlice, CudaDevice, CudaFunction, DeviceRepr},
    nvrtc::safe::Ptx,
};
use std::sync::Arc;

const STARK256_PTX: &str =
    include_str!("../../../../math/src/gpu/cuda/shaders/fields/stark256.ptx");

/// Structure for abstracting basic calls to a Metal device and saving the state. Used for
/// implementing GPU parallel computations in Apple machines.
pub struct CudaState {
    device: Arc<CudaDevice>,
}

impl CudaState {
    /// Creates a new CUDA state with the first GPU.
    pub fn new() -> Result<Self, CudaError> {
        let device =
            CudaDevice::new(0).map_err(|err| CudaError::DeviceNotFound(err.to_string()))?;
        let state = Self { device };

        // Load PTX libraries
        state.load_library(STARK256_PTX, "stark256")?;

        Ok(state)
    }

    pub fn load_library(&self, src: &str, mod_name: &str) -> Result<(), CudaError> {
        let functions = ["radix2_dit_butterfly"];
        self.device
            .load_ptx(Ptx::from_src(src), mod_name, &functions)
            .map_err(|err| CudaError::PtxError(err.to_string()))
    }

    pub fn get_function(&self, mod_name: &str, func_name: &str) -> Result<CudaFunction, CudaError> {
        self.device
            .get_func(mod_name, func_name)
            .ok_or_else(|| CudaError::FunctionError(func_name.to_string()))
    }

    /// Allocates a buffer in the GPU and copies `data` into it. Returns its handle.
    pub fn alloc_buffer_with_data<T: DeviceRepr>(
        &self,
        data: &[T],
    ) -> Result<CudaSlice<T>, CudaError> {
        self.device
            .htod_sync_copy(data)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))
    }

    pub fn retrieve_result<T>(&self, src: CudaSlice<T>) -> Result<Vec<T>, CudaError>
    where
        T: Clone + Default + DeviceRepr + Unpin,
    {
        self.device
            .sync_reclaim(src)
            .map_err(|err| CudaError::RetrieveMemory(err.to_string()))
    }
}
