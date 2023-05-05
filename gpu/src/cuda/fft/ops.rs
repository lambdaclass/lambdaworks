use lambdaworks_math::field::{
    element::FieldElement,
    errors::FieldError,
    traits::{IsFFTField, RootsConfig},
};

use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::safe::Ptx,
};

use crate::cuda::field::element::CUDAFieldElement;

const SHADER_PTX_FFT: &str = include_str!("../shaders/fft.ptx");
const SHADER_PTX_TWIDDLES: &str = include_str!("../shaders/twiddles.ptx");

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
) -> Result<Vec<FieldElement<F>>, DriverError>
where
    F: IsFFTField,
    F::BaseType: Unpin,
{
    let device = CudaDevice::new(0)?;

    // d_ prefix is used to indicate device memory.
    let mut d_input =
        device.htod_sync_copy(&input.iter().map(CUDAFieldElement::from).collect::<Vec<_>>())?;
    let d_twiddles = device.htod_sync_copy(
        &twiddles
            .iter()
            .map(CUDAFieldElement::from)
            .collect::<Vec<_>>(),
    )?;

    device.load_ptx(
        Ptx::from_src(SHADER_PTX_FFT),
        "fft",
        &["radix2_dit_butterfly"],
    )?;
    let kernel = device.get_func("fft", "radix2_dit_butterfly").unwrap();

    let order = input.len().trailing_zeros();
    for stage in 0..order {
        let group_count = 1 << stage;
        let group_size = input.len() / group_count;

        let grid_dim = (group_count as u32, 1, 1); // in blocks
        let block_dim = (group_size as u32 / 2, 1, 1);

        let config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };

        unsafe { kernel.clone().launch(config, (&mut d_input, &d_twiddles)) }?;
    }

    let output = device.sync_reclaim(d_input).unwrap();
    let mut output: Vec<_> = output
        .into_iter()
        .map(|cuda_elem| cuda_elem.into())
        .collect();

    in_place_bit_reverse_permute(&mut output);
    Ok(output)
}

//TODO: Should return CudaError
//TODO: This should mimic metal helpers module
pub fn log2(n: usize) -> Result<u64, FieldError> {
    if !n.is_power_of_two() {
        panic!("The order of polynomial + 1 should a be power of 2");
    }
    Ok(n.trailing_zeros() as u64)
}

pub fn gen_twiddles<F: IsFFTField>(
    order: u64,
    config: RootsConfig,
) -> Result<Vec<FieldElement<F>>, DriverError> {
    let count = (1 << order) / 2;
    let root: FieldElement<F> = F::get_primitive_root_of_unity(order).unwrap();

    let (root, function_name) = match config {
        RootsConfig::Natural => (root, "calc_twiddles"),
        RootsConfig::NaturalInversed => (root.inv(), "calc_twiddles"),
        RootsConfig::BitReverse => (root, "calc_twiddles_bitrev"),
        RootsConfig::BitReverseInversed => (root.inv(), "calc_twiddles_bitrev"),
    };

    let device = CudaDevice::new(0)?;

    // d_ prefix is used to indicate device memory.
    let mut d_twiddles = device.htod_sync_copy(
        (0..count)
            .map(|i| CUDAFieldElement::from(&FieldElement::from(i)))
            .collect::<Vec<_>>()
            .as_slice(),
    )?;
    let d_root = device.htod_sync_copy(&[CUDAFieldElement::from(&root)])?;

    device.load_ptx(
        Ptx::from_src(SHADER_PTX_TWIDDLES),
        "twiddles",
        &[function_name],
    )?;
    let kernel = device.get_func("twiddles", function_name).unwrap();

    for stage in 0..order {
        let group_count = 1 << stage;
        let group_size = count / group_count;

        let grid_dim = (group_count as u32, 1, 1); // in blocks
        let block_dim = (group_size as u32 / 2, 1, 1);

        let config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };

        unsafe { kernel.clone().launch(config, (&mut d_twiddles, &d_root)) }?;
    }

    let output = device.sync_reclaim(d_twiddles).unwrap();
    let mut output: Vec<_> = output
        .into_iter()
        .map(|cuda_elem| cuda_elem.into())
        .collect();

    Ok(output)
}

//TODO remove after implementing in cuda
pub fn get_powers_of_primitive_root<F: IsFFTField>(
    n: u64,
    count: usize,
    config: RootsConfig,
) -> Result<Vec<FieldElement<F>>, FieldError> {
    let root = F::get_primitive_root_of_unity(n).unwrap();

    let calc = |i| match config {
        RootsConfig::Natural => root.pow(i),
        RootsConfig::NaturalInversed => root.pow(i).inv(),
        RootsConfig::BitReverse => root.pow(reverse_index(&i, count as u64)),
        RootsConfig::BitReverseInversed => root.pow(reverse_index(&i, count as u64)).inv(),
    };

    let results = (0..count).map(calc);
    Ok(results.collect())
}

//TODO remove after implementing in cuda
pub fn in_place_bit_reverse_permute<E>(input: &mut [E]) {
    for i in 0..input.len() {
        let bit_reversed_index = reverse_index(&i, input.len() as u64);
        if bit_reversed_index > i {
            input.swap(i, bit_reversed_index);
        }
    }
}

//TODO remove after implementing in cuda
pub fn reverse_index(i: &usize, size: u64) -> usize {
    if size == 1 {
        *i
    } else {
        i.reverse_bits() >> (usize::BITS - size.trailing_zeros())
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_fft::roots_of_unity::get_twiddles;
    use lambdaworks_math::field::{
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
    prop_compose! {
        fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 2..1<<max_exp).prop_filter("Avoid polynomials of size not power of two", |vec| vec.len().is_power_of_two())) -> Vec<FE> {
            vec
        }
    }

    proptest! {
        #[test]
        fn test_cuda_fft_matches_sequential_fft(input in field_vec(4)) {
            let order = input.len().trailing_zeros();
            let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse).unwrap();

            let cuda_fft = super::fft(&input, &twiddles).unwrap();
            let fft = lambdaworks_fft::ops::fft(&input).unwrap();

            assert_eq!(cuda_fft, fft);
        }
    }
}
