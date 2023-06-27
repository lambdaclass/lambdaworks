use lambdaworks_math::{
    fft::cpu::roots_of_unity::get_twiddles,
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        traits::RootsConfig,
    },
    polynomial::Polynomial,
    unsigned_integer::element::UnsignedInteger,
};
use rand::random;

pub type F = Stark252PrimeField;
pub type FE = FieldElement<F>;

pub fn rand_field_elements(order: u64) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        let rand_big = UnsignedInteger { limbs: random() };
        result.push(FE::new(rand_big));
    }
    result
}

pub fn rand_poly(order: u64) -> Polynomial<FE> {
    Polynomial::new(&rand_field_elements(order))
}

#[inline(never)]
#[no_mangle]
#[export_name = "util::get_twiddles"]
pub fn twiddles(order: u64, config: RootsConfig) -> Vec<FE> {
    get_twiddles(order, config).unwrap()
}
