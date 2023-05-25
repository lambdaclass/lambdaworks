use crate::{
    fft::errors::FFTError,
    field::{element::FieldElement, traits::IsFFTField},
    gpu::cuda::field::element::CUDAFieldElement,
};
use cudarc::driver::{LaunchAsync, LaunchConfig};
use lambdaworks_gpu::cuda::abstractions::{errors::CudaError, state::CudaState};

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

    let mut output: Vec<FieldElement<F>> = state
        .retrieve_result(input_buffer)?
        .into_iter()
        .map(FieldElement::from)
        .collect();

    in_place_bit_reverse_permute(&mut output);
    Ok(output)
}

// TODO: remove after implementing in cuda
pub(crate) fn in_place_bit_reverse_permute<E>(input: &mut [E]) {
    for i in 0..input.len() {
        let bit_reversed_index = reverse_index(&i, input.len() as u64);
        if bit_reversed_index > i {
            input.swap(i, bit_reversed_index);
        }
    }
}

// TODO: remove after implementing in cuda
pub(crate) fn reverse_index(i: &usize, size: u64) -> usize {
    if size == 1 {
        *i
    } else {
        i.reverse_bits() >> (usize::BITS - size.trailing_zeros())
    }
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
