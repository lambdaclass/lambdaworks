#![no_main]
use libfuzzer_sys::fuzz_target;
use stark_platinum_prover::proof::stark::StarkProof;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use lambdaworks_math::traits::{Deserializable, Serializable};


fuzz_target!(|data: Vec<u8>| {
    
    let proof = StarkProof::<Stark252PrimeField>::deserialize(&data);

});
