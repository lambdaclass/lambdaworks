use honggfuzz::fuzz;
use lambdaworks_math::{
    fft::{
        gpu::cuda::{ state::CudaState, ops::gen_twiddles },
        cpu::roots_of_unity::get_twiddles,
    },
    field::{
        traits::RootsConfig,
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
};

fn main() {
    loop {
        fuzz!(|input: u64| {
            let state = CudaState::new().unwrap();
            let roots_configurations = vec![RootsConfig::Natural, RootsConfig::BitReverseInversed, RootsConfig::BitReverse, RootsConfig::NaturalInversed];

            for roots_config in roots_configurations {
                let gen_twiddles_cuda = gen_twiddles::<Stark252PrimeField>(input, roots_config, &state);
                let get_twiddles_cpu = get_twiddles::<Stark252PrimeField>(input, roots_config);

                match (gen_twiddles_cuda, get_twiddles_cpu) {
                    (Ok(twiddles_cuda), Ok(twiddles_cpu)) => assert_eq!(twiddles_cuda, twiddles_cpu),
                    (Err(_), Err(_)) => {},
                    (cuda, cpu) => panic!("Evaluate results didn't match. cuda.is_err(): {}, cpu.is_err(): {}", cuda.is_err(), cpu.is_err())
                }
            }
        });
    }
}

