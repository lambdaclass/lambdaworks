#![no_main]
use libfuzzer_sys::fuzz_target;
use lambdaworks_math::{
    fft::{
        gpu::cuda::polynomial::{evaluate_fft_metal, interpolate_fft_metal},
        polynomial::{evaluate_fft_cpu, interpolate_fft_cpu},
        errors::FFTError
    },
    field::{
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        element::FieldElement,
    }
};

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

    let (fft_eval_cuda, fft_eval_cpu) = match (evaluate_fft_cuda(&inputs), evaluate_fft_cpu(&inputs)) {
        (Ok(fft_eval_cuda), Ok(fft_eval_cpu)) => {
            assert_eq!(fft_eval_cuda, fft_eval_cpu);
            (fft_eval_cuda.clone(), fft_eval_cpu.clone())
        },
        (Err(err_cuda), Err(err_cpu)) => {
            (inputs.clone(), inputs)
        },
        (cuda, cpu) => panic!("Evaluate results didn't match. cuda.is_err(): {}, cpu.is_err(): {}", cuda.is_err(), cpu.is_err())
    };

    match (interpolate_fft_cuda(&fft_eval_cuda), interpolate_fft_cpu(&fft_eval_cpu)) {
        (Ok(interpolated_cuda), Ok(interpolated_cpu)) => {
            assert_eq!(interpolated_cuda, interpolated_cpu);
        },
        (Err(err_cuda), Err(err_cpu)) => {},
        (cuda, cpu) => panic!("Interpolate results didn't match. cuda.is_err(): {}, cpu.is_err(): {}", cuda.is_err(), cpu.is_err())
    };
});

