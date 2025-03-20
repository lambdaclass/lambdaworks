use crate::{
    field::{
        element::FieldElement,
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        traits::{IsFFTField, IsField, RootsConfig},
    },
    gpu::cuda::field::element::CUDAFieldElement,
};
use cudarc::{
    driver::{
        safe::CudaSlice, safe::DeviceSlice, CudaDevice, CudaFunction, LaunchAsync, LaunchConfig,
    },
    nvrtc::safe::Ptx,
};
use lambdaworks_gpu::cuda::abstractions::errors::CudaError;
use std::sync::Arc;

const STARK256_PTX: &str = include_str!("../../../gpu/cuda/shaders/field/stark256.ptx");
const WARP_SIZE: usize = 32; // the implementation will spawn threadblocks of this size.

/// Structure for abstracting basic calls to a CUDA device and saving the state. Used for
/// implementing GPU parallel computations in CUDA.
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
        state.load_library::<Stark252PrimeField>(STARK256_PTX)?;

        Ok(state)
    }

    fn load_library<F: IsFFTField>(&self, src: &'static str) -> Result<(), CudaError> {
        let mod_name: &'static str = F::field_name();
        let functions = [
            "radix2_dit_butterfly",
            "calc_twiddles",
            "calc_twiddles_bitrev",
            "bitrev_permutation",
        ];
        self.device
            .load_ptx(Ptx::from_src(src), mod_name, &functions)
            .map_err(|err| CudaError::PtxError(err.to_string()))
    }

    fn get_function<F: IsFFTField>(&self, func_name: &str) -> Result<CudaFunction, CudaError> {
        let mod_name = F::field_name();
        self.device
            .get_func(mod_name, func_name)
            .ok_or_else(|| CudaError::FunctionError(func_name.to_string()))
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
        let function = self.get_function::<F>("radix2_dit_butterfly")?;

        let input_buffer = self.alloc_buffer_with_data(input)?;
        let twiddles_buffer = self.alloc_buffer_with_data(twiddles)?;

        Ok(Radix2DitButterflyFunction::new(
            Arc::clone(&self.device),
            function,
            input_buffer,
            twiddles_buffer,
        ))
    }

    /// Returns a wrapper object over the `calc_twiddles` function defined in `twiddles.cu`
    pub(crate) fn get_calc_twiddles<F: IsFFTField>(
        &self,
        order: u64,
        config: RootsConfig,
    ) -> Result<CalcTwiddlesFunction<F>, CudaError> {
        let root: FieldElement<F> = F::get_primitive_root_of_unity(order).map_err(|_| {
            CudaError::FunctionError(format!(
                "Couldn't get primitive root of unity of order {}",
                order
            ))
        })?;

        let (root, function_name) = match config {
            RootsConfig::Natural => (root, "calc_twiddles"),
            RootsConfig::NaturalInversed => (root.inv(), "calc_twiddles"),
            RootsConfig::BitReverse => (root, "calc_twiddles_bitrev"),
            RootsConfig::BitReverseInversed => (root.inv(), "calc_twiddles_bitrev"),
        };

        let function = self.get_function::<F>(function_name)?;

        let count = (1 << order) / 2;
        let omega_buffer = self.alloc_buffer_with_data(&[root])?;
        let twiddles: &[FieldElement<F>] = &(0..count)
            .map(|_| FieldElement::one())
            .collect::<Vec<FieldElement<F>>>();
        let twiddles_buffer = self.alloc_buffer_with_data(twiddles)?;

        Ok(CalcTwiddlesFunction::new(
            Arc::clone(&self.device),
            function,
            omega_buffer,
            twiddles_buffer,
        ))
    }

    /// Returns a wrapper object over the `bitrev_permutation` function defined in `bitrev_permutation.cu`
    pub(crate) fn get_bitrev_permutation<F: IsFFTField>(
        &self,
        input: &[FieldElement<F>],
        result: &[FieldElement<F>],
    ) -> Result<BitrevPermutationFunction<F>, CudaError> {
        let function = self.get_function::<F>("bitrev_permutation")?;

        let input_buffer = self.alloc_buffer_with_data(input)?;
        let result_buffer = self.alloc_buffer_with_data(result)?;

        Ok(BitrevPermutationFunction::new(
            Arc::clone(&self.device),
            function,
            input_buffer,
            result_buffer,
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
        block_count: usize,
        block_size: usize,
        stage: u32,
        butterfly_count: u32,
    ) -> Result<(), CudaError> {
        let grid_dim = (block_count as u32, 1, 1); // in blocks
        let block_dim = (block_size as u32, 1, 1);

        let config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };
        // Launching kernels must be done in an unsafe block.
        // Calling a kernel is similar to calling a foreign-language function,
        // as the kernel itself could be written in C or unsafe Rust.
        unsafe {
            self.function.clone().launch(
                config,
                (&mut self.input, &self.twiddles, stage, butterfly_count),
            )
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

pub(crate) struct CalcTwiddlesFunction<F: IsField> {
    device: Arc<CudaDevice>,
    function: CudaFunction,
    omega: CudaSlice<CUDAFieldElement<F>>,
    twiddles: CudaSlice<CUDAFieldElement<F>>,
}

impl<F: IsField> CalcTwiddlesFunction<F> {
    fn new(
        device: Arc<CudaDevice>,
        function: CudaFunction,
        omega: CudaSlice<CUDAFieldElement<F>>,
        twiddles: CudaSlice<CUDAFieldElement<F>>,
    ) -> Self {
        Self {
            device,
            function,
            omega,
            twiddles,
        }
    }

    pub(crate) fn launch(&mut self, count: usize) -> Result<(), CudaError> {
        let block_size = WARP_SIZE;
        let block_count = (count + block_size - 1) / block_size;

        let grid_dim = (block_count as u32, 1, 1); // in blocks
        let block_dim = (block_size as u32, 1, 1);

        let config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };
        // Launching kernels must be done in an unsafe block.
        // Calling a kernel is similar to calling a foreign-language function,
        // as the kernel itself could be written in C or unsafe Rust.
        unsafe {
            self.function
                .clone()
                .launch(config, (&mut self.twiddles, &self.omega, count as u32))
        }
        .map_err(|err| CudaError::Launch(err.to_string()))
    }

    pub(crate) fn retrieve_result(self) -> Result<Vec<FieldElement<F>>, CudaError> {
        let Self {
            device, twiddles, ..
        } = self;
        let output = device
            .sync_reclaim(twiddles)
            .map_err(|err| CudaError::RetrieveMemory(err.to_string()))?
            .into_iter()
            .map(FieldElement::from)
            .collect();

        Ok(output)
    }
}

pub(crate) struct BitrevPermutationFunction<F: IsField> {
    device: Arc<CudaDevice>,
    function: CudaFunction,
    input: CudaSlice<CUDAFieldElement<F>>,
    result: CudaSlice<CUDAFieldElement<F>>,
}

impl<F: IsField> BitrevPermutationFunction<F> {
    fn new(
        device: Arc<CudaDevice>,
        function: CudaFunction,
        input: CudaSlice<CUDAFieldElement<F>>,
        result: CudaSlice<CUDAFieldElement<F>>,
    ) -> Self {
        Self {
            device,
            function,
            input,
            result,
        }
    }

    pub(crate) fn launch(&mut self) -> Result<(), CudaError> {
        let len = self.input.len();
        let block_size = WARP_SIZE;
        let block_count = (len + block_size - 1) / block_size;

        let grid_dim = (block_count as u32, 1, 1); // in blocks
        let block_dim = (block_size as u32, 1, 1);

        let config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };
        // Launching kernels must be done in an unsafe block.
        // Calling a kernel is similar to calling a foreign-language function,
        // as the kernel itself could be written in C or unsafe Rust.
        unsafe {
            self.function
                .clone()
                .launch(config, (&mut self.input, &self.result, len))
        }
        .map_err(|err| CudaError::Launch(err.to_string()))
    }

    pub(crate) fn retrieve_result(self) -> Result<Vec<FieldElement<F>>, CudaError> {
        let Self { device, result, .. } = self;
        let output = device
            .sync_reclaim(result)
            .map_err(|err| CudaError::RetrieveMemory(err.to_string()))?
            .into_iter()
            .map(FieldElement::from)
            .collect();

        Ok(output)
    }
}
