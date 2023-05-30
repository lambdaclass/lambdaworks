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
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    traits::{IsFFTField, IsField, RootsConfig},
};
use std::sync::Arc;

const FFT_PTX: &str = include_str!("../shaders/fft.ptx");
const GEN_TWIDDLES_PTX: &str = include_str!("../shaders/twiddles.ptx");

const STARK256_PTX: &str = include_str!("../shaders/fields/stark256.ptx");

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
        state._load_library(
            GEN_TWIDDLES_PTX,
            "twiddles",
            &["calc_twiddles", "calc_twiddles_bitrev"],
        )?;
        state.load_library::<Stark252PrimeField>(STARK256_PTX)?;

        Ok(state)
    }

    fn _load_library(
        &self,
        src: &'static str,          // Library code
        module: &'static str,       // Module name
        functions: &'static [&str], // List of functions
    ) -> Result<(), CudaError> {
        self.device
            .load_ptx(Ptx::from_src(src), module, functions)
            .map_err(|err| CudaError::PtxError(err.to_string()))
    }

    fn load_library<F: IsFFTField>(&self, src: &'static str) -> Result<(), CudaError> {
        let mod_name: &'static str = F::field_name();
        let functions = ["radix2_dit_butterfly"];
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
        let root: FieldElement<F> = F::get_primitive_root_of_unity(order)?;

        let (root, function_name) = match config {
            RootsConfig::Natural => (root, "calc_twiddles"),
            RootsConfig::NaturalInversed => (root.inv(), "calc_twiddles"),
            RootsConfig::BitReverse => (root, "calc_twiddles_bitrev"),
            RootsConfig::BitReverseInversed => (root.inv(), "calc_twiddles_bitrev"),
        };

        let function = self
            .device
            .get_func("twiddles", function_name)
            .ok_or_else(|| CudaError::FunctionError(format!("twiddles::{function_name}")))?;

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

    pub(crate) fn launch(&mut self, group_size: usize) -> Result<(), CudaError> {
        let grid_dim = (1, 1, 1); // in blocks
        let block_dim = (group_size as u32, 1, 1);

        if block_dim.0 as usize > DeviceSlice::len(&self.twiddles) {
            return Err(CudaError::IndexOutOfBounds(
                block_dim.0 as usize,
                self.twiddles.len(),
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
                .launch(config, (&mut self.twiddles, &self.omega))
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
