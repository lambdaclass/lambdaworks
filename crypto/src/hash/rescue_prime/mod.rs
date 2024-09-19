// Need to implement a Field over the prime field p
// p = 2^64 - 2^32 + 1

// I have GoldilockPrimeField

// MiniGoldilockPrimeField
// 2^32 - 2^16 + 1

use lambdaworks_math::field::{fields::fft_friendly::u64_goldilocks, traits::IsField};


fn get_round_constants(security_level:u32)-> Vec
