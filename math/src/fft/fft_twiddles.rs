use crate::field::{element::FieldElement, traits::IsTwoAdicField};
use metal::{Device, MTLResourceOptions, MTLSize};

use super::{
    errors::FFTError,
    helpers::{log2, void_ptr},
};

const TWIDDLE_LIB_DATA: &[u8] = include_bytes!("metal/fft.metallib");

// WARN: the K const is temporary, in the future this will return a vector of variable size K.
/// Generates `K` twiddle factors for a **u64**-field of `modulus` using the GPU via Metal.
/// `K` should be a power of two.
pub fn gen_twiddles<F, const K: usize>(
    modulus: u64,
    metal_device: Option<Device>,
) -> Result<[FieldElement<F>; K], FFTError>
where
    F: IsTwoAdicField,
{
    let metal_device: Device =
        metal_device.unwrap_or(Device::system_default().ok_or(FFTError::MetalDeviceNotFound())?);

    let library = metal_device
        .new_library_with_data(TWIDDLE_LIB_DATA)
        .map_err(FFTError::MetalLibraryError)?;

    let kernel = library
        .get_function("gen_twiddles", None)
        .map_err(FFTError::MetalFunctionError)?;

    let pipeline = metal_device
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(FFTError::MetalPipelineError)?;

    let basetype_size = std::mem::size_of::<F::BaseType>() as u64;

    // the field's modulus and omega will be passed with a buffer to the kernel for now
    // (every two-adic field should be implemented in its own Metal shader).
    // a buffer was chosen instead of a function constant because the latter don't support
    // 64 bit integers.
    let modulus_buffer = metal_device.new_buffer_with_data(
        void_ptr(&modulus),
        basetype_size,
        MTLResourceOptions::StorageModeShared,
    );

    let omega = F::get_root_of_unity(log2(K)?)?;
    let omega_buffer = metal_device.new_buffer_with_data(
        void_ptr(&omega),
        basetype_size,
        MTLResourceOptions::StorageModeShared,
    );

    let result_size = K as u64 * basetype_size;
    let result_buffer = metal_device.new_buffer(result_size, MTLResourceOptions::StorageModeShared);
    // TODO: a better practice is to have managed buffers.

    let command_queue = metal_device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_encoder = command_buffer.new_compute_command_encoder();

    compute_encoder.set_compute_pipeline_state(&pipeline);
    compute_encoder.set_buffer(0, Some(&modulus_buffer), 0);
    compute_encoder.set_buffer(1, Some(&omega_buffer), 0);
    compute_encoder.set_buffer(2, Some(&result_buffer), 0);

    // SIMD group size:
    let threads = pipeline.thread_execution_width();
    // the idea is to have a SIMD per threadgroup, so:
    let thread_group_size = MTLSize::new(threads, 1, 1);
    // then the amount of groups should be the minimum that make K fit:
    let thread_group_count = MTLSize::new((K as u64 + threads - 1) / threads, 1, 1);

    compute_encoder.dispatch_threads(thread_group_size, thread_group_count);
    compute_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // FIXME: A shared (or managed) vector should be returned instead to avoid this unsafe.
    let results = unsafe { *(result_buffer.contents() as *const [u64; K]) };
    Ok(results.map(FieldElement::from))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::test_fields::u64_test_field::U64TestField;

    const MODULUS: u64 = 0xFFFFFFFF00000001;
    const K: usize = 4;
    type F = U64TestField<MODULUS>;

    fn gen_twiddles_cpu<F, const K: usize>() -> Result<Vec<FieldElement<F>>, FFTError>
    where
        F: IsTwoAdicField,
    {
        let omega = F::get_root_of_unity(log2(K)?)?;
        let mut twiddles = vec![FieldElement::zero(); K];

        for i in 0..K - 1 {
            twiddles[i] = omega.pow(i as u64);
        }

        Ok(twiddles)
    }

    #[test]
    fn test_gpu_twiddles_match_cpu() {
        let cpu_twiddles = gen_twiddles_cpu::<F, K>().unwrap();
        let gpu_twiddles = gen_twiddles::<F, K>(MODULUS, None).unwrap().to_vec();

        assert_eq!(cpu_twiddles, gpu_twiddles);
    }
}
