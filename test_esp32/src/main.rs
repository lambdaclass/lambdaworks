//use esp_idf_hal::peripherals::Peripherals;
//use esp_idf_hal::gpio::PinDriver;
use esp_idf_sys as _;
use lambdaworks_math::field::{
    element::FieldElement, fields::u64_prime_field::U64PrimeField,
    fields::montgomery_backed_prime_fields::{IsModulus, U256PrimeField},
    traits::IsTwoAdicField,
};
use lambdaworks_math::unsigned_integer::element::{UnsignedInteger, U256};
use std::thread;
use std::time::Duration;


#[derive(Clone, Debug)]
pub struct MontgomeryConfigStark252PrimeField;
impl IsModulus<U256> for MontgomeryConfigStark252PrimeField {
    const MODULUS: U256 =
        U256::from("800000000000011000000000000000000000000000000000000000000000001");
}

pub type Stark252PrimeField = U256PrimeField<MontgomeryConfigStark252PrimeField>;
type FE = FieldElement<Stark252PrimeField>;

fn main() {
    let elem_one = FE::one();
    let mut val = FE::zero();

    loop {
        println!("Test lambdaworks!");
        thread::sleep(Duration::from_millis(1000));
        val = val + &elem_one;
        println!("{val:?}");
    }
}
