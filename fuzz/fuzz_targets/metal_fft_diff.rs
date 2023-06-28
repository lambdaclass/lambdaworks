#![no_main]
use libfuzzer_sys::fuzz_target;
use lambdaworks_math::{
    fft::{
        gpu::metal::ops::fft as fft_metal,
        cpu::{
            roots_of_unity::get_twiddles,
            ops::fft as fft_cpu
        }
    },
    field::{
        traits::RootsConfig,
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        element::FieldElement
    },
};
use lambdaworks_gpu::metal::abstractions::state::MetalState;

fuzz_target!(|data: Vec<u64>| {
    let mut input_raw = data;
    let mut inputs = Vec::new();

    if input_raw.len() == 0 {
        input_raw.push(0u64);
    }

    for i in 0..input_raw.len() {
        let input_value = format!("{:x}", input_raw[i]);
        inputs.push(FieldElement::<Stark252PrimeField>::from_hex_unchecked(&input_value))
    }

    let twiddles = get_twiddles(
        inputs.len().trailing_zeros() as u64,
        RootsConfig::BitReverse,
    )
    .unwrap();

    let state = MetalState::new(None).unwrap();

    match fft_cpu(&inputs, &twiddles) {
        Ok(fft_result) => assert_eq!(fft_result, fft_metal(&inputs, &twiddles, &state).unwrap()),
        Err(_) => assert!(fft_metal(&inputs, &twiddles, &state).is_err())
    }
});
