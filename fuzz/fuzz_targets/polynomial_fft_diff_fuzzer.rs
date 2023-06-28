#![no_main]
use libfuzzer_sys::fuzz_target;
use lambdaworks_math::{
    fft::{
        gpu::metal::polynomial::{evaluate_fft_metal, interpolate_fft_metal},
        polynomial::{evaluate_fft_cpu, interpolate_fft_cpu},
    },
    field::{
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        element::FieldElement,
    },
    unsigned_integer::element::UnsignedInteger
};

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

    let evaluated_fft_cpu = evaluate_fft_cpu(&inputs);
    let evaluated_fft_metal = evaluate_fft_metal(&inputs);

    match evaluated_fft_cpu {
        Ok(fft_evals_cpu) => {
            let fft_evals_metal = evaluated_fft_metal.unwrap();
            match interpolate_fft_cpu(&fft_evals_cpu) {
                Ok(interpolated_cpu) => {
                    assert_eq!(interpolate_fft_metal(&fft_evals_metal).unwrap(), interpolated_cpu)
                },
                Err(_) => {assert!(interpolate_fft_metal(&fft_evals_metal).is_err())}
            }
        },
        Err(_) => assert!(evaluated_fft_metal.is_err())
    };
});

