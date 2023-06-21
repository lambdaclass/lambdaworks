#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_math::field::{
    element::FieldElement, 
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField
};


fuzz_target!(|data: &[u8]| {
    let a =  FieldElement::<Stark252PrimeField>::from(3_u64);
    let b =  FieldElement::<Stark252PrimeField>::from(3_u64);
    let c = a * b;
});
