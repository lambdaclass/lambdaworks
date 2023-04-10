use rust_gpu_tools::{cuda, program_closures, Device, GPUError, Program};
use lambdaworks_math::{
    fft::bit_reversing::in_place_bit_reverse_permute,
    field::{element::FieldElement, traits::IsTwoAdicField},
    field::{test_fields::u32_test_field::U32TestField, traits::RootsConfig},
};
use lambdaworks_math::fft::errors::FFTError;

type F = U32TestField;

/// Returns a `Program` that runs on CUDA.
pub fn cuda(device: &Device) -> Program {
    let cuda_kernel = include_bytes!("../cuda/fft.fatbin");
    let cuda_device = device.cuda_device().unwrap();
    let cuda_program = cuda::Program::from_bytes(cuda_device, cuda_kernel).unwrap();
    Program::Cuda(cuda_program)
}

pub fn run_main(input: &[FieldElement<F>]) -> Result<Vec<FieldElement<F>>, FFTError> {
    let order: u64 = input.len().trailing_zeros() as u64;
    let twiddles = F::get_twiddles(order, RootsConfig::BitReverse).unwrap();
    let size = input.len();

    let mut input_s = vec![0u32; size];
    let mut twiddles_s = vec![0u32; (size / 2) as usize];
    let mut twiddles_permut = vec![0u32; (size.clone() / 2) as usize];

    for i in 0..input.len(){
        input_s[i] = *input[i].value();
    }

    for i in 0..twiddles.len(){
        twiddles_s[i] = *twiddles[i].value();
        twiddles_permut[i] = *twiddles[i].value();
    }

    in_place_bit_reverse_permute(&mut twiddles_permut);

    let closures = program_closures!(|program, _args| -> Result<Vec<u32>, GPUError> {
        // Copy the data to the GPU.
        let input_c = program.create_buffer_from_slice(&input_s)?;
        let twiddles_c = program.create_buffer_from_slice(&twiddles_s)?;

        let kernel = program.create_kernel("main_fft", 1, 1)?;
        kernel
            .arg(&input_c)
            .arg(&twiddles_c)
            .arg(&(size as u32))
            .run()?;

        // Get the resulting data.
        let mut result = vec![0u32; size];
        program.read_into_buffer(&input_c, &mut result)?;

        Ok(result)
    });

    // Get the first available device.
    let device = *Device::all().first().unwrap();

    // First we run it on CUDA.
    let cuda_program = cuda(device);
    let mut cuda_result = cuda_program.run(closures, ()).unwrap();
    in_place_bit_reverse_permute(&mut cuda_result);
    Ok(cuda_result.iter().map(FieldElement::from).collect())
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::{
        field::{test_fields::u32_test_field::U32TestField},
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
        #[test]
        fn test_very_basic(poly in poly(8)) {
            // let order = poly.coefficients().len().trailing_zeros() as u64;
            let expected = poly.evaluate_fft().unwrap();
            // let twiddles = F::get_twiddles(order, RootsConfig::BitReverse).unwrap();

            let result = run_main(poly.coefficients()).unwrap();
            prop_assert_eq!(&result[..], &expected[..]);
        }
    }
}