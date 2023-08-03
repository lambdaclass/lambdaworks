use crate::{
    fft::gpu::cuda::state::CudaState,
    field::{
        element::FieldElement,
        traits::{IsFFTField, RootsConfig},
    },
};
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
) -> Result<Vec<FieldElement<F>>, CudaError>
where
    F: IsFFTField,
    F::BaseType: Unpin,
{
    let mut function = state.get_radix2_dit_butterfly(input, twiddles)?;

    const WARP_SIZE: usize = 32;

    let block_size = WARP_SIZE;
    let butterfly_count = input.len() / 2;
    let block_count = (butterfly_count + block_size - 1) / block_size;

    let order = input.len().trailing_zeros();

    for stage in 0..order {
        function.launch(block_count, block_size, stage, butterfly_count as u32)?;
    }

    let output = function.retrieve_result()?;

    bitrev_permutation(output, state)
}

pub fn gen_twiddles<F: IsFFTField>(
    order: u64,
    config: RootsConfig,
    state: &CudaState,
) -> Result<Vec<FieldElement<F>>, CudaError> {
    if order > 63 {
        return Err(CudaError::FunctionError(
            "Order should be less than or equal to 63".to_string(),
        ));
    }

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

    function.launch()?;

    function.retrieve_result()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::cpu::roots_of_unity::get_twiddles;
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
            let fft = crate::fft::cpu::ops::fft(&input, &twiddles).unwrap();

            prop_assert_eq!(cuda_fft, fft);
        }
    }

    #[test]
    fn test_cuda_fft_matches_sequential_large_input() {
        const ORDER: usize = 20;
        let input = vec![FE::one(); 1 << ORDER];

        let state = CudaState::new().unwrap();
        let order = input.len().trailing_zeros();
        let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse).unwrap();

        let cuda_result = fft(&input, &twiddles, &state).unwrap();
        let sequential_result = crate::fft::cpu::ops::fft(&input, &twiddles).unwrap();

        assert_eq!(&cuda_result, &sequential_result);
    }

    #[test]
    fn gen_twiddles_with_order_greater_than_63_should_fail() {
        let state = CudaState::new().unwrap();
        let twiddles = gen_twiddles::<F>(64, RootsConfig::Natural, &state);

        assert!(matches!(twiddles, Err(CudaError::FunctionError(_))));
    }
}
