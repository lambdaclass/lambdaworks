#![no_main]
use libfuzzer_sys::fuzz_target;
use lambdaworks_crypto::hash::poseidon::starknet::PoseidonCairoStark252;
use lambdaworks_math::field::{
    element::FieldElement, 
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_crypto::hash::poseidon::Poseidon;

fn get_expected_hash(a: &FieldElement<Stark252PrimeField>, b: &FieldElement<Stark252PrimeField>) -> FieldElement<Stark252PrimeField> {
    PoseidonCairoStark252::hash(a, b)
}

fuzz_target!(|data: ([u8; 32], [u8; 32])| {
    let (bytes_a, bytes_b) = data;

    let element_a = FieldElement::<Stark252PrimeField>::from_bytes_be(&bytes_a).unwrap();
    let element_b = FieldElement::<Stark252PrimeField>::from_bytes_be(&bytes_b).unwrap();

    let hash_result = PoseidonCairoStark252::hash(&element_a, &element_b);

    let hash_expected = get_expected_hash(&element_a, &element_b);

    assert_eq!(hash_result, hash_expected, "Hashes don't match each other");
});
