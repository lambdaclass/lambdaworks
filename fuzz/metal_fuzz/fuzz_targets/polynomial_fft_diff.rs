#![no_main]
use libfuzzer_sys::fuzz_target;
use lambdaworks_math::{
    fft::{
        gpu::metal::polynomial::{evaluate_fft_metal, interpolate_fft_metal},
        polynomial::{evaluate_fft_cpu, interpolate_fft_cpu},
        errors::FFTError
    },
    field::{
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        element::FieldElement,
    }
};

use lambdaworks_gpu::metal::abstractions::errors::MetalError;

fn compare_errors(metal_error: MetalError, fft_error: FFTError) -> bool {
    match (metal_error, fft_error) {
        (MetalError::RootOfUnityError(_, nth_metal), FFTError::RootOfUnityError(nth_cpu)) => nth_metal == nth_cpu,
        (MetalError::InputError(len_metal), FFTError::InputError(len_cpu)) => len_metal == len_cpu,
        _ => false
    }
}

fuzz_target!(|data: Vec<u64>| {
    let mut inputs_raw = data;
    let mut inputs = Vec::new();

    if inputs_raw.len() == 0 {
        inputs_raw.push(0u64);
    }

    for i in 0..inputs_raw.len() {
        let input_value = format!("{:x}", inputs_raw[i]);
        inputs.push(FieldElement::<Stark252PrimeField>::from_hex_unchecked(&input_value))
    }

    let (fft_eval_metal, fft_eval_cpu) = match (evaluate_fft_metal(&inputs), evaluate_fft_cpu(&inputs)) {
        (Ok(fft_eval_metal), Ok(fft_eval_cpu)) => {
            assert_eq!(fft_eval_metal, fft_eval_cpu);
            (fft_eval_metal.clone(), fft_eval_cpu.clone())
        },
        (Err(err_metal), Err(err_cpu)) => {
            assert!(compare_errors(err_metal, err_cpu));
            (inputs.clone(), inputs)
        },
        (metal, cpu) => panic!("Evaluate results didn't match. metal.is_err(): {}, cpu.is_err(): {}", metal.is_err(), cpu.is_err())
    };

    match (interpolate_fft_metal(&fft_eval_metal), interpolate_fft_cpu(&fft_eval_cpu)) {
        (Ok(interpolated_metal), Ok(interpolated_cpu)) => {
            assert_eq!(interpolated_metal, interpolated_cpu);
        },
        (Err(err_metal), Err(err_cpu)) => assert!(compare_errors(err_metal, err_cpu)),
        (metal, cpu) => panic!("Interpolate results didn't match. metal.is_err(): {}, cpu.is_err(): {}", metal.is_err(), cpu.is_err())
    };
});

