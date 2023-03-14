use lambdaworks_math::fft::bit_reversing::in_place_bit_reverse_permute;
use lambdaworks_math::field::{element::FieldElement, traits::IsTwoAdicField};

use super::{fft_twiddles::gen_twiddles, helpers::void_ptr};
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

    pub fn execute_fft<F: IsTwoAdicField, const K: usize>(
        &self,
        input: [FieldElement<F>; K],
    ) -> Result<[FieldElement<F>; K], FFTError> {
        let butterfly_kernel = self
            .library
            .get_function("radix2_dit_butterfly", None)
            .map_err(FFTError::MetalFunctionError)?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&butterfly_kernel)
            .map_err(FFTError::MetalPipelineError)?;

        let basetype_size = std::mem::size_of::<u32>();

        let input = input.map(|elem| elem.value().clone());
        let input_buffer = self.device.new_buffer_with_data(
            void_ptr(&input),
            (input.len() * basetype_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let mut twiddles =
            gen_twiddles::<F, K>(Some(self.device.clone()))?.map(|elem| elem.value().clone());
        in_place_bit_reverse_permute(&mut twiddles[..]); // required for NR
        let twiddles_buffer = self.device.new_buffer_with_data(
            void_ptr(&twiddles),
            (twiddles.len() * basetype_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let mut group_count = 1_u64;
        let mut group_size = input.len() as u64;

        while group_count < input.len() as u64 {
            let command_buffer = self.command_queue.new_command_buffer();
            let compute_encoder = command_buffer.new_compute_command_encoder();

            compute_encoder.set_compute_pipeline_state(&pipeline);
            compute_encoder.set_buffer(0, Some(&input_buffer), 0);
            compute_encoder.set_buffer(1, Some(&twiddles_buffer), 0);

            let threadgroup_size = MTLSize::new(group_size, 1, 1);
            let threadgroup_count = MTLSize::new(group_count, 1, 1);
            compute_encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            compute_encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            group_count *= 2;
            group_size /= 2;
        }
        let results = unsafe { *(input_buffer.contents() as *const [u32; K]) };
        Ok(results.map(|x| FieldElement::from(x as u64)))
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::{
        fft::bit_reversing::in_place_bit_reverse_permute,
        field::test_fields::u64_test_field::U32TestField, polynomial::Polynomial,
    };

    use super::*;

    type F = U32TestField;
    type FE = FieldElement<F>;

    // Property-based test that ensures NR Radix-2 FFT gives same result as a naive polynomial evaluation.
    #[test]
    fn test_metal_fft_matches_naive_eval() {
        // random:
        let mut coeffs = [
            3924592,
            923482529,
            82348923529,
            9235825395,
            9235823905238,
            9823592358,
            98239283599,
            1391831318,
        ]
        .map(FE::from);

        let root = F::get_primitive_root_of_unity(3).unwrap();
        let mut twiddles = (0..coeffs.len() as u64)
            .map(|i| root.pow(i))
            .collect::<Vec<FE>>();
        in_place_bit_reverse_permute(&mut twiddles[..]); // required for NR

        let poly = Polynomial::new(&coeffs[..]);
        let expected: Vec<FE> = twiddles.iter().map(|x| poly.evaluate(x)).collect();

        let metal_state = FFTMetalState::new(None).unwrap();

        let mut result = metal_state.execute_fft(coeffs).unwrap();
        in_place_bit_reverse_permute(&mut result);

        dbg!(result);
        dbg!(expected);

        assert!(false);
    }
}
