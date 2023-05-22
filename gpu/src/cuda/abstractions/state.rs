use crate::cuda::abstractions::{element::CUDAFieldElement, errors::CudaError};
use cudarc::{
    driver::{
        safe::{CudaSlice, DeviceSlice},
        CudaDevice, CudaFunction, LaunchAsync, LaunchConfig,
    },
    nvrtc::safe::Ptx,
};
use lambdaworks_math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField},
};
use std::sync::Arc;

const FFT_PTX: &str = include_str!("../shaders/fft.ptx");

/// Structure for abstracting basic calls to a Metal device and saving the state. Used for
/// implementing GPU parallel computations in Apple machines.
pub struct CudaState {
    device: Arc<CudaDevice>,
}

impl CudaState {
    /// Creates a new CUDA state with the first GPU.
    pub(crate) fn new() -> Result<Self, CudaError> {
        let device =
            CudaDevice::new(0).map_err(|err| CudaError::DeviceNotFound(err.to_string()))?;
        let state = Self { device };

        // Load PTX libraries
        state.load_library(FFT_PTX, "fft", &["radix2_dit_butterfly"])?;

        Ok(state)
    }

    fn load_library(
        &self,
        src: &'static str,          // Library code
        module: &'static str,       // Module name
        functions: &'static [&str], // List of functions
    ) -> Result<(), CudaError> {
        self.device
            .load_ptx(Ptx::from_src(src), module, functions)
            .map_err(|err| CudaError::PtxError(err.to_string()))
    }

    /// Allocates a buffer in the GPU and copies `data` into it. Returns its handle.
    fn alloc_buffer_with_data<F: IsField>(
        &self,
        data: &[FieldElement<F>],
    ) -> Result<CudaSlice<CUDAFieldElement<F>>, CudaError> {
        self.device
            .htod_sync_copy(&data.iter().map(CUDAFieldElement::from).collect::<Vec<_>>())
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))
    }

    /// Returns a wrapper object over the `radix2_dit_butterfly` function defined in `fft.cu`
    pub(crate) fn get_radix2_dit_butterfly<F: IsFFTField>(
        &self,
        input: &[FieldElement<F>],
        twiddles: &[FieldElement<F>],
    ) -> Result<Radix2DitButterflyFunction<F>, CudaError> {
        let function = self
            .device
            .get_func("fft", "radix2_dit_butterfly")
            .ok_or_else(|| CudaError::FunctionError("fft::radix2_dit_butterfly".to_string()))?;

        let input_buffer = self.alloc_buffer_with_data(input)?;
        let twiddles_buffer = self.alloc_buffer_with_data(twiddles)?;

        Ok(Radix2DitButterflyFunction::new(
            Arc::clone(&self.device),
            function,
            input_buffer,
            twiddles_buffer,
        ))
    }
}

pub(crate) struct Radix2DitButterflyFunction<F: IsField> {
    device: Arc<CudaDevice>,
    function: CudaFunction,
    input: CudaSlice<CUDAFieldElement<F>>,
    twiddles: CudaSlice<CUDAFieldElement<F>>,
}

impl<F: IsField> Radix2DitButterflyFunction<F> {
    fn new(
        device: Arc<CudaDevice>,
        function: CudaFunction,
        input: CudaSlice<CUDAFieldElement<F>>,
        twiddles: CudaSlice<CUDAFieldElement<F>>,
    ) -> Self {
        Self {
            device,
            function,
            input,
            twiddles,
        }
    }

    pub(crate) fn launch(
        &mut self,
        group_count: usize,
        group_size: usize,
    ) -> Result<(), CudaError> {
        let grid_dim = (group_count as u32, 1, 1); // in blocks
        let block_dim = ((group_size / 2) as u32, 1, 1);

        if block_dim.0 as usize > DeviceSlice::len(&self.twiddles) {
            return Err(CudaError::IndexOutOfBounds(
                block_dim.0 as usize,
                self.twiddles.len(),
            ));
        } else if (grid_dim.0 * block_dim.0) as usize > DeviceSlice::len(&self.input) {
            return Err(CudaError::IndexOutOfBounds(
                (grid_dim.0 * block_dim.0) as usize,
                self.input.len(),
            ));
        }

        let config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };
        unsafe {
            self.function
                .clone()
                .launch(config, (&mut self.input, &self.twiddles))
        }
        .map_err(|err| CudaError::Launch(err.to_string()))
    }

    pub(crate) fn retrieve_result(self) -> Result<Vec<FieldElement<F>>, CudaError> {
        let Self { device, input, .. } = self;
        let output = device
            .sync_reclaim(input)
            .map_err(|err| CudaError::RetrieveMemory(err.to_string()))?
            .into_iter()
            .map(FieldElement::from)
            .collect();

        Ok(output)
    }
}
