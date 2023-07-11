#[macro_use]
extern crate honggfuzz;
use lambdaworks_math::{
    fft::{
        gpu::cuda::{ops::fft as fft_cuda, state::CudaState},
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


fn main() {
    loop {
        fuzz!(|data:  Vec<u64>| {
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

            let state = CudaState::new().unwrap();
            println!("inputs {:?}", &inputs);
            println!("fft cpu{:?}", fft_cpu(&inputs, &twiddles));

            match fft_cpu(&inputs, &twiddles) {
                Ok(fft_result) => assert_eq!(fft_result, fft_cuda(&inputs, &twiddles, &state).unwrap()),
                Err(_) => assert!(fft_cuda(&inputs, &twiddles, &state).is_err())
            }
        });
    }
}
