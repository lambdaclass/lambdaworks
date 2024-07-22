#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_crypto::hash::poseidon::starknet::PoseidonCairoStark252;
use lambdaworks_math::field::{
    element::FieldElement, 
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_crypto::hash::poseidon::Poseidon;
use pathfinder_crypto::MontFelt;
use pathfinder_crypto::Felt;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use std::str::FromStr;

fn convert_dec_to_hex(dec_str: &str) -> String {
    let dec = BigInt::from_str(dec_str)
        .unwrap();  
    format!("0x{:x}", dec)  
}

fuzz_target!(|data: ([u8; 32], [u8; 32])| {
    let (bytes_a, bytes_b) = data;

    let lw_x = FieldElement::<Stark252PrimeField>::from_bytes_be(&bytes_a).unwrap();
    let lw_y = FieldElement::<Stark252PrimeField>::from_bytes_be(&bytes_b).unwrap();
    let lambdaworks_hash =PoseidonCairoStark252::hash(&lw_x, &lw_y).to_string();

    let mut mont_x = lw_x.value().limbs;
    let mut mont_y = lw_y.value().limbs;

    // In order use the same field elements for starknet-rs and pathfinder, we have to reverse
    // the limbs order respect to the lambdaworks implementation.
    
    mont_x.reverse();
    mont_y.reverse();
  
    let pf_x = MontFelt(mont_x);
    let pf_y = MontFelt(mont_y);

    let pathfinder_hash = Felt::from(pathfinder_crypto::hash::poseidon_hash(pf_x, pf_y)).to_hex_str();
    assert_eq!(lambdaworks_hash, pathfinder_hash, "Lambdaworks and Pathfinder hashes don't match each other");

// Starknet-rs 
 
    let sn_ff_x = starknet_ff::FieldElement::from_mont(mont_x);
    let sn_ff_y = starknet_ff::FieldElement::from_mont(mont_y);
    let starknet_hash =convert_dec_to_hex(&starknet_crypto::poseidon_hash(sn_ff_x, sn_ff_y).to_string());

   assert_eq!(lambdaworks_hash, starknet_hash, "Lambdaworks and Starknet hashes don't match each other");

});



