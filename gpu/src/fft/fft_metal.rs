use std::mem;

use lambdaworks_math::field::{element::FieldElement, traits::IsTwoAdicField};

use super::helpers::void_ptr;
use lambdaworks_math::fft::errors::FFTError;
use metal::{CommandQueue, Device, Library, MTLResourceOptions, MTLSize};

const FFT_LIB_DATA: &[u8] = include_bytes!("../metal/fft.metallib");

pub struct FFTMetalState {
    pub device: Device,
    pub library: Library,
    pub command_queue: CommandQueue,
}

impl FFTMetalState {
    pub fn new(device: Option<Device>) -> Result<Self, FFTError> {
        let device: Device =
            device.unwrap_or(Device::system_default().ok_or(FFTError::MetalDeviceNotFound())?);

        let library = device
            .new_library_with_data(FFT_LIB_DATA)
            .map_err(FFTError::MetalLibraryError)?;
        let command_queue = device.new_command_queue();

        Ok(FFTMetalState {
            device,
            library,
            command_queue,
        })
    }

    pub fn setup<F: IsTwoAdicField>(
        &self,
        kernel: &str,
        twiddles: &[FieldElement<F>],
    ) -> Result<(&metal::CommandBufferRef, &metal::ComputeCommandEncoderRef), FFTError> {
        let butterfly_kernel = self
            .library
            .get_function(kernel, None)
            .map_err(FFTError::MetalFunctionError)?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&butterfly_kernel)
            .map_err(FFTError::MetalPipelineError)?;

        let twiddles_buffer = {
            let twiddles: Vec<_> = twiddles.iter().map(|elem| elem.value().clone()).collect();
            let basetype_size = std::mem::size_of::<F::BaseType>();

            self.device.new_buffer_with_data(
                unsafe { mem::transmute(twiddles.as_ptr()) }, // reinterprets into a void pointer
                (twiddles.len() * basetype_size) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_compute_pipeline_state(&pipeline);
        compute_encoder.set_buffer(1, Some(&twiddles_buffer), 0);

        Ok((command_buffer, compute_encoder))
    }

    pub fn execute<F: IsTwoAdicField>(
        &self,
        input: &[FieldElement<F>],
        (command_buffer, compute_encoder): (
            &metal::CommandBufferRef,
            &metal::ComputeCommandEncoderRef,
        ),
    ) -> Result<Vec<FieldElement<F>>, FFTError> {
        let order = input.len().trailing_zeros() as u64;
        let basetype_size = std::mem::size_of::<F::BaseType>();

        let input_buffer = {
            let input: Vec<_> = input.iter().map(|elem| elem.value()).cloned().collect();
            // without that .cloned() the UB monster is summoned, not sure why.
            self.device.new_buffer_with_data(
                unsafe { mem::transmute(input.as_ptr()) }, // reinterprets into a void pointer
                (input.len() * basetype_size) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        compute_encoder.set_buffer(0, Some(&input_buffer), 0);

        for stage in 0..order {
            let group_count = stage + 1;
            let group_size = input.len() as u64 / (1 << stage);

            let group_size_buffer = self.device.new_buffer_with_data(
                void_ptr(&group_size),
                basetype_size as u64,
                MTLResourceOptions::StorageModeShared,
            );

            compute_encoder.set_buffer(2, Some(&group_size_buffer), 0);

            let threadgroup_size = MTLSize::new(group_size / 2, 1, 1);
            let threadgroup_count = MTLSize::new(group_count, 1, 1);

            compute_encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
        }
        compute_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let results_ptr = input_buffer.contents() as *const F::BaseType;
        let results_len = input_buffer.length() as usize / basetype_size;
        let results_slice = unsafe { std::slice::from_raw_parts(results_ptr, results_len) };

        Ok(results_slice.iter().map(FieldElement::from).collect())
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::{
        fft::bit_reversing::in_place_bit_reverse_permute,
        field::{test_fields::u32_test_field::U32TestField, traits::RootsConfig},
        polynomial::Polynomial,
    };
    use proptest::prelude::*;

    use super::*;

    type F = U32TestField;
    type FE = FieldElement<F>;

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
        // max_exp cannot be multiple of the bits that represent a usize, generally 64 or 32.
        // also it can't exceed the test field's two-adicity.
    }
    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null polynomial", |x| x != &0)) -> FE {
            FE::from(num)
        }
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(elem in field_element(), size in powers_of_two(max_exp)) -> Vec<FE> {
            vec![elem; size]
        }
    }
    prop_compose! {
        fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<FE> {
            Polynomial::new(&coeffs)
        }
    }

    proptest! {
        // Property-based test that ensures NR Radix-2 FFT gives same result as a naive polynomial evaluation.
        #[test]
        fn test_metal_fft_matches_sequential(poly in poly(8)) {
            objc::rc::autoreleasepool(|| {
                let order = poly.coefficients().len().trailing_zeros() as u64;
                let expected = poly.evaluate_fft().unwrap();

                let metal_state = FFTMetalState::new(None).unwrap();
                let twiddles: Vec<_> = F::get_twiddles(order, RootsConfig::BitReverse).unwrap();
                let command_buff_encoder = metal_state.setup("radix2_dit_butterfly", &twiddles).unwrap();

                let mut result = metal_state.execute(poly.coefficients(), command_buff_encoder).unwrap();
                in_place_bit_reverse_permute(&mut result);

                prop_assert_eq!(&result[..], &expected[..]);

                Ok(())
            }).unwrap();
        }
    }
}
