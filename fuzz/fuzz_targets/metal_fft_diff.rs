#![no_main]
use libfuzzer_sys::fuzz_target;
use lambdaworks_math::{
    fft::{
        gpu::metal::ops::fft as fft_metal,
        cpu::ops::fft as fft_cpu
    },
    field::{
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        element::FieldElement
    },
    unsigned_integer::element::UnsignedInteger
};
use lambdaworks_gpu::metal::abstractions::state::MetalState;

fuzz_target!(|values: (Vec<[u64;4]>, Vec<[u64;4]>)| {
    let (mut input_raw, mut twiddles_raw) = values;
    let mut inputs = Vec::new();
    let mut twiddles = Vec::new();

    if input_raw.len() == 0 {
        input_raw.push([1u64;4]);
    }

    for i in 0..input_raw.len() {
        let input_value = UnsignedInteger::<4>::from_limbs(input_raw[i]);
        inputs.push(FieldElement::<Stark252PrimeField>::from_raw(&input_value))
    }

    if twiddles_raw.len() == 0 {
        twiddles_raw.push([1u64;4]);
    }


    for i in 0..twiddles_raw.len() {
        let twiddle_value = UnsignedInteger::<4>::from_limbs(twiddles_raw[i]);
        twiddles.push(FieldElement::<Stark252PrimeField>::from_raw(&twiddle_value))
    }

    let state = MetalState::new(None).unwrap();

    match fft_cpu(&inputs, &twiddles) {
        Ok(fft_result) => assert_eq!(fft_result, fft_metal(&inputs, &twiddles, &state).unwrap()),
        Err(_) => assert!(fft_metal(&inputs, &twiddles, &state).is_err())
    }
});

