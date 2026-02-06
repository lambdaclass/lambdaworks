use crate::{
    fft::gpu::cuda::state::{CudaState, HasCudaExtFft},
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

/// Extension field FFT: input in E (e.g. Fp2/Fp3), twiddles in base field F (e.g. Fp).
pub(crate) fn fft_ext<F, E>(
    input: &[FieldElement<E>],
    twiddles: &[FieldElement<F>],
    state: &CudaState,
) -> Result<Vec<FieldElement<E>>, CudaError>
where
    F: IsFFTField,
    E: HasCudaExtFft,
{
    if input.is_empty() || !input.len().is_power_of_two() {
        return Err(CudaError::InvalidOrder(input.len()));
    }

    let mut function = state.get_radix2_dit_butterfly_ext::<F, E>(input, twiddles)?;

    const WARP_SIZE: usize = 32;

    let block_size = WARP_SIZE;
    let butterfly_count = input.len() / 2;
    let block_count = (butterfly_count + block_size - 1) / block_size;

    let order = input.len().trailing_zeros();

    for stage in 0..order {
        function.launch(block_count, block_size, stage, butterfly_count as u32)?;
    }

    let output = function.retrieve_result()?;

    bitrev_permutation_ext(output, state)
}

pub(crate) fn bitrev_permutation_ext<E: HasCudaExtFft>(
    input: Vec<FieldElement<E>>,
    state: &CudaState,
) -> Result<Vec<FieldElement<E>>, CudaError> {
    let mut function = state.get_bitrev_permutation_ext(&input)?;

    function.launch()?;

    function.retrieve_result()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::cpu::roots_of_unity::get_twiddles;
    use crate::fft::errors::FFTError;
    use crate::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        fields::u64_goldilocks_field::Goldilocks64Field, traits::RootsConfig,
    };
    use proptest::{collection, prelude::*};

    // FFT related tests for Stark252
    type F = Stark252PrimeField;
    type FE = FieldElement<F>;

    // FFT tests for Goldilocks
    type GF = Goldilocks64Field;
    type GFE = FieldElement<GF>;

    fn to_test_err<E: std::fmt::Debug>(e: E) -> TestCaseError {
        TestCaseError::fail(format!("{:?}", e))
    }

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
            let state = CudaState::new().map_err(to_test_err)?;
            let order = input.len().trailing_zeros();
            let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse).map_err(to_test_err)?;

            let cuda_fft = fft(&input, &twiddles, &state).map_err(to_test_err)?;
            let cpu_fft = crate::fft::cpu::ops::fft(&input, &twiddles).map_err(to_test_err)?;

            prop_assert_eq!(cuda_fft, cpu_fft);
        }
    }

    #[test]
    fn test_cuda_fft_matches_sequential_large_input() -> Result<(), FFTError> {
        const ORDER: usize = 20;
        let input = vec![FE::one(); 1 << ORDER];

        let state = CudaState::new()?;
        let order = input.len().trailing_zeros();
        let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse)?;

        let cuda_result = fft(&input, &twiddles, &state)?;
        let sequential_result = crate::fft::cpu::ops::fft(&input, &twiddles)?;

        assert_eq!(&cuda_result, &sequential_result);
        Ok(())
    }

    #[test]
    fn gen_twiddles_with_order_greater_than_63_should_fail() -> Result<(), FFTError> {
        let state = CudaState::new()?;
        let twiddles = gen_twiddles::<F>(64, RootsConfig::Natural, &state);

        assert!(matches!(twiddles, Err(CudaError::FunctionError(_))));
        Ok(())
    }

    // =====================================================
    // GOLDILOCKS FIELD CUDA FFT TESTS
    // =====================================================

    prop_compose! {
        fn goldilocks_powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
    }

    prop_compose! {
        fn goldilocks_field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> GFE {
            GFE::from(num)
        }
    }

    fn goldilocks_field_vec(max_exp: u8) -> impl Strategy<Value = Vec<GFE>> {
        goldilocks_powers_of_two(max_exp)
            .prop_flat_map(|size| collection::vec(goldilocks_field_element(), size))
    }

    proptest! {
        #[test]
        fn test_goldilocks_cuda_fft_matches_cpu_fft(input in goldilocks_field_vec(4)) {
            let state = CudaState::new().map_err(to_test_err)?;
            let order = input.len().trailing_zeros();
            let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse).map_err(to_test_err)?;

            let cuda_fft = fft(&input, &twiddles, &state).map_err(to_test_err)?;
            let cpu_fft = crate::fft::cpu::ops::fft(&input, &twiddles).map_err(to_test_err)?;

            prop_assert_eq!(cuda_fft, cpu_fft);
        }

        #[test]
        fn test_goldilocks_cuda_twiddles_match_cpu_twiddles(order in 1u64..10) {
            let state = CudaState::new().map_err(to_test_err)?;

            let cuda_twiddles: Vec<GFE> = gen_twiddles(order, RootsConfig::Natural, &state).map_err(to_test_err)?;
            let cpu_twiddles: Vec<GFE> = get_twiddles(order, RootsConfig::Natural).map_err(to_test_err)?;

            prop_assert_eq!(cuda_twiddles, cpu_twiddles);
        }
    }

    #[test]
    fn test_goldilocks_cuda_fft_simple() -> Result<(), FFTError> {
        // Test with a simple known input
        let input: Vec<GFE> = vec![
            GFE::from(1u64),
            GFE::from(2u64),
            GFE::from(3u64),
            GFE::from(4u64),
        ];

        let state = CudaState::new()?;
        let order = input.len().trailing_zeros();
        let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse)?;

        let cuda_result = fft(&input, &twiddles, &state)?;
        let cpu_result = crate::fft::cpu::ops::fft(&input, &twiddles)?;

        assert_eq!(cuda_result, cpu_result);
        Ok(())
    }

    #[test]
    fn test_goldilocks_cuda_fft_large_input() -> Result<(), FFTError> {
        const ORDER: usize = 16;
        let input: Vec<GFE> = (0..(1 << ORDER)).map(|i| GFE::from(i as u64 + 1)).collect();

        let state = CudaState::new()?;
        let order = input.len().trailing_zeros();
        let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse)?;

        let cuda_result = fft(&input, &twiddles, &state)?;
        let cpu_result = crate::fft::cpu::ops::fft(&input, &twiddles)?;

        assert_eq!(cuda_result, cpu_result);
        Ok(())
    }

    #[test]
    fn test_goldilocks_cuda_bitrev_permutation() -> Result<(), FFTError> {
        let input: Vec<GFE> = (0..8).map(|i| GFE::from(i + 1)).collect();
        let state = CudaState::new()?;

        let cuda_result = bitrev_permutation(input.clone(), &state)?;

        // Expected bit-reversed order for 8 elements:
        // 0 (000) -> 0 (000)
        // 1 (001) -> 4 (100)
        // 2 (010) -> 2 (010)
        // 3 (011) -> 6 (110)
        // 4 (100) -> 1 (001)
        // 5 (101) -> 5 (101)
        // 6 (110) -> 3 (011)
        // 7 (111) -> 7 (111)
        let expected: Vec<GFE> = vec![
            GFE::from(1u64),
            GFE::from(5u64),
            GFE::from(3u64),
            GFE::from(7u64),
            GFE::from(2u64),
            GFE::from(6u64),
            GFE::from(4u64),
            GFE::from(8u64),
        ];

        assert_eq!(cuda_result, expected);
        Ok(())
    }

    // =====================================================
    // GOLDILOCKS EXTENSION FIELD (Fp2) CUDA FFT TESTS
    // =====================================================

    use crate::field::fields::u64_goldilocks_field::{
        Degree2GoldilocksExtensionField, Degree3GoldilocksExtensionField,
    };

    type Fp2 = Degree2GoldilocksExtensionField;
    type Fp2E = FieldElement<Fp2>;
    type Fp3 = Degree3GoldilocksExtensionField;
    type Fp3E = FieldElement<Fp3>;

    #[test]
    fn test_goldilocks_fp2_cuda_fft_simple() -> Result<(), FFTError> {
        let input: Vec<Fp2E> = vec![
            Fp2E::from(&[GFE::from(1u64), GFE::from(2u64)]),
            Fp2E::from(&[GFE::from(3u64), GFE::from(4u64)]),
            Fp2E::from(&[GFE::from(5u64), GFE::from(6u64)]),
            Fp2E::from(&[GFE::from(7u64), GFE::from(8u64)]),
        ];

        let state = CudaState::new()?;
        let order = input.len().trailing_zeros();
        let twiddles: Vec<GFE> = get_twiddles(order.into(), RootsConfig::BitReverse)?;

        let cuda_result = fft_ext(&input, &twiddles, &state)?;
        let cpu_result = crate::fft::cpu::ops::fft(&input, &twiddles)?;

        assert_eq!(cuda_result, cpu_result);
        Ok(())
    }

    #[test]
    fn test_goldilocks_fp2_cuda_fft_large() -> Result<(), FFTError> {
        const ORDER: usize = 14;
        let input: Vec<Fp2E> = (0..(1u64 << ORDER))
            .map(|i| Fp2E::from(&[GFE::from(i + 1), GFE::from(i + 2)]))
            .collect();

        let state = CudaState::new()?;
        let order = input.len().trailing_zeros();
        let twiddles: Vec<GFE> = get_twiddles(order.into(), RootsConfig::BitReverse)?;

        let cuda_result = fft_ext(&input, &twiddles, &state)?;
        let cpu_result = crate::fft::cpu::ops::fft(&input, &twiddles)?;

        assert_eq!(cuda_result, cpu_result);
        Ok(())
    }

    // =====================================================
    // GOLDILOCKS EXTENSION FIELD (Fp3) CUDA FFT TESTS
    // =====================================================

    #[test]
    fn test_goldilocks_fp3_cuda_fft_simple() -> Result<(), FFTError> {
        let input: Vec<Fp3E> = vec![
            Fp3E::from(&[GFE::from(1u64), GFE::from(2u64), GFE::from(3u64)]),
            Fp3E::from(&[GFE::from(4u64), GFE::from(5u64), GFE::from(6u64)]),
            Fp3E::from(&[GFE::from(7u64), GFE::from(8u64), GFE::from(9u64)]),
            Fp3E::from(&[GFE::from(10u64), GFE::from(11u64), GFE::from(12u64)]),
        ];

        let state = CudaState::new()?;
        let order = input.len().trailing_zeros();
        let twiddles: Vec<GFE> = get_twiddles(order.into(), RootsConfig::BitReverse)?;

        let cuda_result = fft_ext(&input, &twiddles, &state)?;
        let cpu_result = crate::fft::cpu::ops::fft(&input, &twiddles)?;

        assert_eq!(cuda_result, cpu_result);
        Ok(())
    }

    #[test]
    fn test_goldilocks_fp3_cuda_fft_large() -> Result<(), FFTError> {
        const ORDER: usize = 14;
        let input: Vec<Fp3E> = (0..(1u64 << ORDER))
            .map(|i| Fp3E::from(&[GFE::from(i + 1), GFE::from(i + 2), GFE::from(i + 3)]))
            .collect();

        let state = CudaState::new()?;
        let order = input.len().trailing_zeros();
        let twiddles: Vec<GFE> = get_twiddles(order.into(), RootsConfig::BitReverse)?;

        let cuda_result = fft_ext(&input, &twiddles, &state)?;
        let cpu_result = crate::fft::cpu::ops::fft(&input, &twiddles)?;

        assert_eq!(cuda_result, cpu_result);
        Ok(())
    }

    // =====================================================
    // PROPTEST FOR EXTENSION FIELDS
    // =====================================================

    prop_compose! {
        fn fp2_element()(a in any::<u64>().prop_filter("nonzero", |x| x != &0),
                         b in any::<u64>()) -> Fp2E {
            Fp2E::from(&[GFE::from(a), GFE::from(b)])
        }
    }

    fn fp2_field_vec(max_exp: u8) -> impl Strategy<Value = Vec<Fp2E>> {
        goldilocks_powers_of_two(max_exp).prop_flat_map(|size| collection::vec(fp2_element(), size))
    }

    prop_compose! {
        fn fp3_element()(a in any::<u64>().prop_filter("nonzero", |x| x != &0),
                         b in any::<u64>(),
                         c in any::<u64>()) -> Fp3E {
            Fp3E::from(&[GFE::from(a), GFE::from(b), GFE::from(c)])
        }
    }

    fn fp3_field_vec(max_exp: u8) -> impl Strategy<Value = Vec<Fp3E>> {
        goldilocks_powers_of_two(max_exp).prop_flat_map(|size| collection::vec(fp3_element(), size))
    }

    proptest! {
        #[test]
        fn test_goldilocks_fp2_cuda_fft_matches_cpu(input in fp2_field_vec(4)) {
            let state = CudaState::new().map_err(to_test_err)?;
            let order = input.len().trailing_zeros();
            let twiddles: Vec<GFE> = get_twiddles(order.into(), RootsConfig::BitReverse).map_err(to_test_err)?;

            let cuda_fft = fft_ext(&input, &twiddles, &state).map_err(to_test_err)?;
            let cpu_fft = crate::fft::cpu::ops::fft(&input, &twiddles).map_err(to_test_err)?;

            prop_assert_eq!(cuda_fft, cpu_fft);
        }

        #[test]
        fn test_goldilocks_fp3_cuda_fft_matches_cpu(input in fp3_field_vec(4)) {
            let state = CudaState::new().map_err(to_test_err)?;
            let order = input.len().trailing_zeros();
            let twiddles: Vec<GFE> = get_twiddles(order.into(), RootsConfig::BitReverse).map_err(to_test_err)?;

            let cuda_fft = fft_ext(&input, &twiddles, &state).map_err(to_test_err)?;
            let cpu_fft = crate::fft::cpu::ops::fft(&input, &twiddles).map_err(to_test_err)?;

            prop_assert_eq!(cuda_fft, cpu_fft);
        }
    }
}
