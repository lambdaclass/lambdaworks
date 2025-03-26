use lambdaworks_math::{
    field::element::FieldElement,
    field::fields::fft_friendly::u64_goldilocks::U64GoldilocksPrimeField, polynomial::Polynomial,
};
use rand::random;

pub type FE = FieldElement<U64GoldilocksPrimeField>;

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
        result.push(FE::from(random::<u64>()));
    }
    result
}

#[allow(dead_code)]
#[inline(never)]
#[export_name = "u64_utils::rand_goldilocks_field_elements_pair"]
pub fn rand_field_elements_pair() -> (FE, FE) {
    (FE::from(random::<u64>()), FE::from(random::<u64>()))
}

#[allow(dead_code)]
#[inline(never)]
#[export_name = "u64_utils::rand_goldilocks_poly"]
pub fn rand_poly(order: u64) -> Polynomial<FE> {
    Polynomial::new(&rand_field_elements(order))
}
