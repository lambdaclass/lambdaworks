use lambdaworks_math::field::{element::FieldElement, traits::IsTwoAdicField};

use cudarc::{
    driver::{CudaDevice, LaunchAsync, LaunchConfig},
    nvrtc::safe::Ptx,
};

use crate::cuda::field::element::CUDAFieldElement;

const SHADER_PTX: &str = include_str!("../shaders/fft.ptx");

/// Executes parallel ordered FFT over a slice of two-adic field elements, in CUDA.
/// Twiddle factors are required to be in bit-reverse order.
///
/// "Ordered" means that the input is required to be in natural order, and the output will be
/// in this order too. Natural order means that input[i] corresponds to the i-th coefficient,
/// as opposed to bit-reverse order in which input[bit_rev(i)] corresponds to the i-th
/// coefficient.
pub fn fft<F>(input: &[FieldElement<F>], twiddles: &[FieldElement<F>]) -> Vec<FieldElement<F>>
where
    F: IsTwoAdicField,
    F::BaseType: Unpin,
{
    let device = CudaDevice::new(0).unwrap();

    // d_ prefix is used to indicate device memory.
    let mut d_input = device
        .htod_sync_copy(
            &input
                .iter()
                .map(CUDAFieldElement::from)
                .collect::<Vec<CUDAFieldElement<F>>>(),
        )
        .unwrap();
    let d_twiddles = device
        .htod_sync_copy(
            &twiddles
                .iter()
                .map(CUDAFieldElement::from)
                .collect::<Vec<CUDAFieldElement<F>>>(),
        )
        .unwrap();

    device
        .load_ptx(Ptx::from_src(SHADER_PTX), "fft", &["radix2_dit_butterfly"])
        .unwrap();

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

        unsafe { kernel.clone().launch(config, (&mut d_input, &d_twiddles)) }.unwrap();
    }

    let output: Vec<CUDAFieldElement<F>> = device.sync_reclaim(d_input).unwrap();
    let output: Vec<FieldElement<F>> = output
        .into_iter()
        .map(|cuda_elem| cuda_elem.into())
        .collect();
    output
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
        fn test_cuda_fft_matches_sequential_fft(input in field_vec(2)) {
            let order = input.len().trailing_zeros();
            let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse).unwrap();

            let mut cuda_fft = super::fft(&input, &twiddles);
            lambdaworks_fft::bit_reversing::in_place_bit_reverse_permute(&mut cuda_fft);
            let fft = lambdaworks_fft::ops::fft(&input).unwrap();

            assert_eq!(cuda_fft, fft);
        }
    }
}
