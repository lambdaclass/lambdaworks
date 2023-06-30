use lambdaworks_math::{
    fft::cpu::{bit_reversing::in_place_bit_reverse_permute, roots_of_unity::get_twiddles},
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

// NOTE: intentional duplicate to help IAI skip setup code
#[inline(never)]
#[no_mangle]
#[export_name = "util::bitrev_permute"]
pub fn bitrev_permute(input: &mut [FE]) {
    in_place_bit_reverse_permute(input);
}

#[inline(never)]
#[no_mangle]
#[export_name = "util::rand_field_elements"]
pub fn rand_field_elements(order: u64) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        let rand_big = UnsignedInteger { limbs: random() };
        result.push(FE::new(rand_big));
    }
    result
}

#[inline(never)]
#[no_mangle]
#[export_name = "util::rand_poly"]
pub fn rand_poly(order: u64) -> Polynomial<FE> {
    Polynomial::new(&rand_field_elements(order))
}

#[inline(never)]
#[no_mangle]
#[export_name = "util::get_twiddles"]
pub fn twiddles(order: u64, config: RootsConfig) -> Vec<FE> {
    get_twiddles(order, config).unwrap()
}
