/// Implementation of the Babybear Prime Field p = 2^31 - 2^27 + 1 using u32
pub mod babybear;
/// Implementation of the KoalaBear Prime Field p = 2^31 - 2^24 + 1 using u32
pub mod koalabear;
/// Implementation of the extension of degree 4 of the babybear field
pub mod quartic_babybear;
/// Implementation of the extension of degree 4 of the KoalaBear field
pub mod quartic_koalabear;
/// Implementation of the prime field used in [Stark101](https://starkware.co/stark-101/) tutorial, p = 3 * 2^30 + 1
pub mod stark_101_prime_field;
/// Implementation of two-adic prime field over 256 bit unsigned integers.
pub mod stark_252_prime_field;
/// Implementation of the Goldilocks Prime Field p = 2^64 - 2^32 + 1
pub mod u64_goldilocks;
