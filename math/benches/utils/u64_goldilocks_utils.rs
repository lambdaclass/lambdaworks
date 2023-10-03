use const_random::const_random;
use lambdaworks_math::{
    field::element::FieldElement,
    field::fields::{
        fft_friendly::u64_goldilocks::U64GoldilocksPrimeField, u64_prime_field::U64FieldElement,
    },
    polynomial::Polynomial,
};
use rand::random;

// Mersenne prime numbers
// https://www.math.utah.edu/~pa/math/mersenne.html
const PRIMES: [u64; 39] = [
    13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941,
    11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049, 216091, 756839, 859433, 1257787,
    1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457,
];

const MODULUS: u64 = PRIMES[const_random!(usize) % PRIMES.len()];
pub type FE = U64FieldElement<MODULUS>;

#[inline(never)]
#[export_name = "u64_utils::fp_get_goldilocks_primes"]
pub fn get_field_elements() -> (
    FieldElement<U64GoldilocksPrimeField>,
    FieldElement<U64GoldilocksPrimeField>,
) {
    let x = FieldElement::<U64GoldilocksPrimeField>::from_hex(
        "0x03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb",
    )
    .unwrap();
    let y = FieldElement::<U64GoldilocksPrimeField>::from_hex(
        "0x0208a0a10250e382e1e4bbe2880906c2791bf6275695e02fbbc6aeff9cd8b31a",
    )
    .unwrap();
    (x, y)
}

#[inline(never)]
#[export_name = "u64_utils::fp_squared_goldilocks_prime"]
pub fn get_squared_field_element() -> FieldElement<U64GoldilocksPrimeField> {
    let (x, _) = get_field_elements();
    x * x
}

#[allow(dead_code)]
#[inline(never)]
#[export_name = "u64_utils::rand_field_goldilocks_elements"]
pub fn rand_field_elements(order: u64) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        result.push(FE::new(random()));
    }
    result
}

#[allow(dead_code)]
#[inline(never)]
#[export_name = "u64_utils::rand_goldilocks_field_elements_pair"]
pub fn rand_field_elements_pair() -> (FE, FE) {
    (FE::new(random()), FE::new(random()))
}

#[allow(dead_code)]
#[inline(never)]
#[export_name = "u64_utils::rand_goldilocks_poly"]
pub fn rand_poly(order: u64) -> Polynomial<FE> {
    Polynomial::new(&rand_field_elements(order))
}
