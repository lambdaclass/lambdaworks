use crate::field::{element::FieldElement, traits::IsTwoAdicField};
use metal::{Device, FunctionConstantValues, MTLDataType, MTLResourceOptions, MTLSize};

use super::{errors::FFTError, helpers::void_ptr};

const TWIDDLE_LIB_DATA: &[u8] = include_bytes!("metal/fft.metallib");

/// Generates `2^k` twiddle factors for a field of `modulus` using the GPU via Metal.
pub fn gen_twiddles<F>(k: u64, modulus: u64, metal_device: Option<Device>) -> Result<Vec<FieldElement<F>>, FFTError> 
where
    F: IsTwoAdicField
{
    let metal_device: Device =
        metal_device.unwrap_or(Device::system_default().ok_or(FFTError::MetalDeviceNotFound())?);

    let library = metal_device
        .new_library_with_data(TWIDDLE_LIB_DATA)
        .map_err(FFTError::MetalLibraryError)?;

    // the field's modulus and root of unity will be passed as constants to the kernel
    let omega = F::get_root_of_unity(k)?;
    let constants = FunctionConstantValues::new();
    // FIXME: the 'metal' crate doesn't seem to have an enum for 64-bit uints, which are supported
    // by Metal. This may yield UB. A possible workaround is to use a uint2 and get the constants
    // as composite.
    constants.set_constant_value_at_index(void_ptr(&omega), MTLDataType::UInt, 0);
    constants.set_constant_value_at_index(void_ptr(&modulus), MTLDataType::UInt, 1);

    let gen_twiddles = library
        .get_function("gen_twiddles", Some(constants))
        .map_err(FFTError::MetalFunctionError)?;

    let pipeline = metal_device
        .new_compute_pipeline_state_with_function(&gen_twiddles)
        .map_err(FFTError::MetalPipelineError)?;

    let result_length = 1 << k;
    let size = result_length * std::mem::size_of::<F::BaseType>() as u64;
    let result_buffer = metal_device.new_buffer(size, MTLResourceOptions::StorageModeShared);

    let command_queue = metal_device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_encoder = command_buffer.new_compute_command_encoder();
    
    compute_encoder.set_compute_pipeline_state(&pipeline);
    compute_encoder.set_buffer(2, Some(&result_buffer), 0);

    let grid_size = MTLSize::new(result_length, 1, 1);

    let thread_count = {
        let max = pipeline.max_total_threads_per_threadgroup();
        let width = pipeline.thread_execution_width();
        (max / width) * width
    };
    let threads_per_threadgroup = MTLSize::new(thread_count, 1, 1);

    compute_encoder.dispatch_threads(grid_size, threads_per_threadgroup);
    compute_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // TODO: deref the result buffer, this will be unsafe.
    todo!()
}
