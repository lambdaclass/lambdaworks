#![no_main]
use libfuzzer_sys::fuzz_target;
use lambdaworks_math::{
    gpu::metal::fft::{
        ops::fft as fft_metal, 
        polynomial::{evaluate_fft_metal, interpolate_fft_metal}
    },
    fft::{ops::fft as fft_cpu, polynomial::FFTPoly}, 
    polynomial::Polynomial,
    field::{
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        element::FieldElement
    }
};
use lambdaworks_gpu::metal::abstractions::state::MetalState;

fuzz_target!(|values: (Vec<u64>, Vec<u64>)| {
    let (mut input_raw, mut twiddles_raw) = values;
    let mut inputs = Vec::new();
    let mut twiddles = Vec::new();

    while input_raw.len() < 4 {
        input_raw.push(0);
    }

    for i in (0..input_raw.len()).step_by(4) {
        inputs.push(FieldElement::<Stark252PrimeField>::from_raw(input_raw[i..i + 4]))
    }

    while twiddles_raw.len() < 4 {
        twiddles_raw.push(0);
    }

    for i in (0..twiddles_raw.len()).step_by(4) {
        twiddles.push(FieldElement::<Stark252PrimeField>::from_raw(twiddles_raw[i..i + 4]))
    }

    let state = MetalState::new(None);

    match fft_cpu(&inputs, &twiddles) {
        Ok(fft_result) => assert_eq!(fft_result, fft_metal(&inputs, &twiddles, state)),
        Err(_) => assert!(fft_metal(&inputs, &twiddles, state).is_err())
    }
});

