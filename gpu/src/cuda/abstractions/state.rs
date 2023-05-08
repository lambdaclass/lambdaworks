use crate::cuda::abstractions::{element::CUDAFieldElement, errors::CudaError};
use cudarc::{
    driver::{CudaDevice, LaunchAsync, LaunchConfig},
    nvrtc::safe::Ptx,
};

const FFT_PTX: &str = include_str!("../shaders/fft.ptx");

/// Structure for abstracting basic calls to a Metal device and saving the state. Used for
/// implementing GPU parallel computations in Apple machines.
pub(crate) struct CudaState {
    pub(crate) device: CudaDevice,
}

impl CudaState {
    /// Creates a new CUDA state with the first GPU.
    pub(crate) fn new() -> Result<Self, CudaError> {
        let device =
            CudaDevice::new(0).map_err(|err| CudaError::DeviceNotFound(err.to_string()))?;
        let self = Self { device };

        // Load PTX libraries
        self.load_library(FFT_PTX, "fft", &["radix2_dit_butterfly"])?;

        Ok(self)
    }

    fn load_library(
        &self,
        src: &'static str,          // Library code
        module: &'static str,       // Module name
        functions: &'static [&str], // List of functions
    ) -> Result<(), CudaError> {
        self.device
            .load_ptx(Ptx::from_src(src), module, functions)
            .map_err(|err| CudaError::PtxError(err.to_string()))?;
    }

    /// Allocates a buffer in the GPU and copies `data` into it. Returns its handle.
    fn alloc_buffer_with_data<F: IsField>(
        &self,
        data: &[F::BaseType],
    ) -> Result<CudaSlice<CUDAFieldElement<F>>, CudaError> {
        self.device
            .htod_sync_copy(data.iter().map(CUDAFieldElement::from).collect::<Vec<_>>())
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?
    }

    /// Returns a wrapper object over the `radix2_dit_butterfly` function defined in `fft.cu`
    pub(crate) fn get_radix2_dit_butterfly<F: IsFFTField>(
        &self,
        input: &[FieldElement<F>],
        twiddles: &[FieldElement<F>],
    ) -> Result<Radix2DitButterflyFunction, CudaError> {
        let function = self
            .device
            .get_func("fft", "radix2_dit_butterfly")
            .ok_or_else(|| CudaError::FunctionError("fft::radix2_dit_butterfly".to_string()))?;

        let input_buffer = self.alloc_buffer_with_data(input)?;
        let twiddles_buffer = self.alloc_buffer_with_data(twiddles)?;

        Radix2DitButterflyFunction::new(function, input_buffer, twiddles_buffer)
    }
}

pub(crate) struct Radix2DitButterflyFunction<F: IsField> {
    function: CudaFunction,
    input: CudaSlice<CUDAFieldElement<F>>,
    twiddles: CudaSlice<CUDAFieldElement<F>>,
}

impl<F: IsField> Radix2DitButterflyFunction<F> {
    fn new(
        function: CudaFunction,
        input: CudaSlice<CUDAFieldElement<F>>,
        twiddles: CudaSlice<CUDAFieldElement<F>>,
    ) -> Result<Self, CudaError> {
        Self {
            function,
            input,
            twiddles,
        }
    }

    pub(crate) fn launch(group_count: u32, group_size: u32) -> Result<(), CudaError> {
        let grid_dim = (group_count, 1, 1); // in blocks
        let block_dim = (group_size / 2, 1, 1);

        if block_dim.0 > twiddles.len() {
            return Err(CudaError::IndexOutOfBounds(
                block_dim.0 as usize,
                twiddles.len(),
            ));
        } else if grid_dim.0 * block_dim.0 > input.len() {
            return Err(CudaError::IndexOutOfBounds(
                grid_dim.0 * block_dim.0 as usize,
                twiddles.len(),
            ));
        }

        let config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };
        unsafe { kernel.clone().launch(config, (&mut d_input, &d_twiddles)) }
            .map_err(|err| CudaError::Launch(err.to_string()))?
    }

    pub(crate) fn retrieve_result() -> Result<Vec<FieldElement<F>>, CudaError> {
        let output = device
            .sync_reclaim(d_input)
            .map_err(|err| CudaError::RetrieveMemory(err.to_string()))?
            .into_iter()
            .map(FieldElement::<F>::from)
            .collect();

        Ok(output)
    }
}
