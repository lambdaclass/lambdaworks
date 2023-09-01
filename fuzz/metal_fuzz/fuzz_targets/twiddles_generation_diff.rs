#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_math::{
    fft::{
        gpu::metal::ops::gen_twiddles,
        cpu::roots_of_unity::get_twiddles,
        errors::FFTError
    },
    field::{
        traits::RootsConfig,
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
};
use lambdaworks_gpu::metal::abstractions::{state::MetalState, errors::MetalError};

fn compare_errors(metal_error: MetalError, fft_error: FFTError) -> bool {
    match (metal_error, fft_error) {
        (MetalError::RootOfUnityError(_, nth_metal), FFTError::RootOfUnityError(nth_cpu)) => nth_metal == nth_cpu,
        (MetalError::InputError(len_metal), FFTError::InputError(len_cpu)) => len_metal == len_cpu,
        _ => false
    }
}

fuzz_target!(|input: u64| {
    let state = MetalState::new(None).unwrap();
    let roots_configurations = vec![RootsConfig::Natural, RootsConfig::BitReverseInversed, RootsConfig::BitReverse, RootsConfig::NaturalInversed];

    for roots_config in roots_configurations {
        let gen_twiddles_metal = gen_twiddles::<Stark252PrimeField>(input, roots_config, &state);
        let get_twiddles_cpu = get_twiddles::<Stark252PrimeField>(input, roots_config);

        match (gen_twiddles_metal, get_twiddles_cpu) {
            (Ok(twiddles_metal), Ok(twiddles_cpu)) => assert_eq!(twiddles_metal, twiddles_cpu),
            (Err(err_metal), Err(err_cpu)) => assert!(compare_errors(err_metal, err_cpu)),
            (metal, cpu) => panic!("Evaluate results didn't match. metal.is_err(): {}, cpu.is_err(): {}", metal.is_err(), cpu.is_err())
        }
    }
});
