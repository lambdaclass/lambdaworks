#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_crypto::hash::poseidon::starknet::PoseidonCairoStark252;
use lambdaworks_math::field::{
    element::FieldElement, 
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_crypto::hash::poseidon::Poseidon;
fuzz_target!(|data: ([u8; 32], [u8; 32])| {
    let (bytes_a, bytes_b) = data;

    let element_a = FieldElement::<Stark252PrimeField>::from_bytes_be(&bytes_a).unwrap();
    let element_b = FieldElement::<Stark252PrimeField>::from_bytes_be(&bytes_b).unwrap();

    let hash_result_1 = PoseidonCairoStark252::hash(&element_a, &element_b);

    let hash_result_2 = PoseidonCairoStark252::hash(&element_a, &element_b);
    assert_eq!(hash_result_1, hash_result_2, "Hashes don't match each other");

    let mut bytes_a_modified = bytes_a;
    //change the first byte
    bytes_a_modified[0] ^= 0x01; 
    let element_a_modified = FieldElement::<Stark252PrimeField>::from_bytes_be(&bytes_a_modified).unwrap();
    let hash_result_modified = PoseidonCairoStark252::hash(&element_a_modified, &element_b);
    assert_ne!(hash_result_1, hash_result_modified, "Collision found!");
});

