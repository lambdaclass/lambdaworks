use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};
use wasm_bindgen_test::*;
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
#[test]
fn one_of_sqrt_roots_for_4_is_2() {
    type FrField = Stark252PrimeField;
    type FrElement = FieldElement<FrField>;

    let input = FrElement::from(4);
    let sqrt = input.sqrt().unwrap();
    let result = FrElement::from(2);
    assert_eq!(sqrt.0, result);
}
