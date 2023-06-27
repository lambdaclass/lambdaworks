#![no_main]
use libfuzzer_sys::fuzz_target;
use lambdaworks_math::{
    gpu::metal::fft::{
        polynomial::{evaluate_fft_metal, interpolate_fft_metal}
    },
    polynomial::Polynomial,
    field::{
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        element::FieldElement
    },
    unsigned_integer::element::UnsignedInteger
};

use lambdaworks_gpu::metal::abstractions::state::MetalState;

fuzz_target!(|values: (Vec<[u64;4]>, Vec<[u64;4]>)| {
    println!("{:?}", values);
    let (mut input_raw, mut twiddles_raw) = values;
    let mut inputs = Vec::new();
    let mut twiddles = Vec::new();

    while input_raw.len() < 4 {
        input_raw.push([1u64;4]);
    }


    for i in 0..input_raw.len() {
        let input_value = UnsignedInteger::<4>::from_limbs(input_raw[i]);
        inputs.push(FieldElement::<Stark252PrimeField>::from_raw(&input_value))
    }

    while twiddles_raw.len() < 4 {
        twiddles_raw.push([1u64;4]);
    }


    for i in 0..twiddles_raw.len() {
        let twiddle_value = UnsignedInteger::<4>::from_limbs(twiddles_raw[i]);
        twiddles.push(FieldElement::<Stark252PrimeField>::from_raw(&twiddle_value))
    }

    let evaluated_fields = evaluate_fft_metal(&inputs).unwrap();
    let interpolated_fields = interpolate_fft_metal(&evaluated_fields).unwrap();

    let polinomial_inputs =  Polynomial { coefficients: (*inputs).to_vec() };
    assert_eq!(interpolated_fields, polinomial_inputs);

});
