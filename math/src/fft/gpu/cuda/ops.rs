use crate::{
    fft::{errors::FFTError, gpu::cuda::state::CudaState},
    field::{element::FieldElement, traits::IsFFTField},
    gpu::cuda::field::element::CUDAFieldElement,
};
use cudarc::driver::{LaunchAsync, LaunchConfig};
use lambdaworks_gpu::cuda::abstractions::errors::CudaError;

/// Executes parallel ordered FFT over a slice of two-adic field elements, in CUDA.
/// Twiddle factors are required to be in bit-reverse order.
///
/// "Ordered" means that the input is required to be in natural order, and the output will be
/// in this order too. Natural order means that input[i] corresponds to the i-th coefficient,
/// as opposed to bit-reverse order in which input[bit_rev(i)] corresponds to the i-th
/// coefficient.
pub fn fft<F>(
    input: &[FieldElement<F>],
    twiddles: &[FieldElement<F>],
    state: &CudaState,
) -> Result<Vec<FieldElement<F>>, FFTError>
where
    F: IsFFTField,
    F::BaseType: Unpin,
{
    // TODO: make a twiddle factor abstraction for handling invalid twiddles
    if !input.len().is_power_of_two() {
        return Err(FFTError::InputError(input.len()));
    }

    let function = state.get_function(F::field_name(), "radix2_dit_butterfly")?;

    let input: Vec<_> = input.iter().map(CUDAFieldElement::from).collect();
    let twiddles: Vec<_> = twiddles.iter().map(CUDAFieldElement::from).collect();

    let mut input_buffer = state.alloc_buffer_with_data(&input)?;
    let twiddles_buffer = state.alloc_buffer_with_data(&twiddles)?;

    let order = input.len().trailing_zeros();
    for stage in 0..order {
        let group_count = 1 << stage;
        let group_size = input.len() / group_count;

        let config = LaunchConfig {
            grid_dim: (group_count as u32, 1, 1),
            block_dim: (group_size as u32 / 2, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            function
                .clone()
                .launch(config, (&mut input_buffer, &twiddles_buffer))
        }
        .map_err(|err| CudaError::Launch(err.to_string()))?;
    }

    let output = function.retrieve_result()?;

    bitrev_permutation(output, state)
}

pub fn gen_twiddles<F: IsFFTField>(
    order: u64,
    config: RootsConfig,
    state: &CudaState,
) -> Result<Vec<FieldElement<F>>, CudaError> {
    let count = (1 << order) / 2;
    if count == 0 {
        return Ok(Vec::new());
    }

    let mut function = state.get_calc_twiddles::<F>(order, config)?;

    function.launch(count)?;

    function.retrieve_result()
}

pub fn bitrev_permutation<F: IsFFTField>(
    input: Vec<FieldElement<F>>,
    state: &CudaState,
) -> Result<Vec<FieldElement<F>>, CudaError> {
    let mut function = state.get_bitrev_permutation(&input, &input)?;

    function.launch(input.len())?;

    function.retrieve_result()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::roots_of_unity::get_twiddles;
    use crate::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        traits::RootsConfig,
    };
    use proptest::{collection, prelude::*};

    // FFT related tests
    type F = Stark252PrimeField;
    type FE = FieldElement<F>;

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
        // max_exp cannot be multiple of the bits that represent a usize, generally 64 or 32.
        // also it can't exceed the test field's two-adicity.
    }

    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FE {
            FE::from(num)
        }
    }

    fn field_vec(max_exp: u8) -> impl Strategy<Value = Vec<FE>> {
        powers_of_two(max_exp).prop_flat_map(|size| collection::vec(field_element(), size))
    }

    proptest! {
        #[test]
        fn test_cuda_fft_matches_sequential_fft(input in field_vec(4)) {
            let state = CudaState::new().unwrap();
            let order = input.len().trailing_zeros();
            let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse).unwrap();

            let cuda_fft = fft(&input, &twiddles, &state).unwrap();
            let fft = crate::fft::ops::fft(&input, &twiddles).unwrap();

            prop_assert_eq!(cuda_fft, fft);
        }
    }
}
