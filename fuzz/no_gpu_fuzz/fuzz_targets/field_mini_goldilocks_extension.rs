#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_math::field::{
    element::FieldElement, 
    fields::u64_goldilocks_field::Goldilocks64ExtensionField,
};
use p3_goldilocks::Goldilocks;
use p3_field::{Field, PrimeField64, AbstractField, AbstractExtensionField as AEF, extension::{BinomiallyExtendable, binomial_extension::BinomialExtensionField}};

type LF = FieldElement<Goldilocks64ExtensionField>;
type PF = BinomialExtensionField::<Goldilocks, 2>;

fuzz_target!(|values: (u64, u64)| {

    let (value_u64_a, value_u64_b) = values;

    let a = LF::from(value_u64_a);
    let b = LF::from(value_u64_b);

    let a_expected = PF::from_canonical_u64(value_u64_a);
    let b_expected = PF::from_canonical_u64(value_u64_b);

    let add_u64 = &a + &b;
    let addition = a_expected + b_expected;

    assert_eq!(add_u64.value()[0].representative(), AEF::<PF>::as_base_slice(&addition).to;
    assert_eq!(add_u64.value()[1].representative(), AEF::<PF>::as_base_slice(&addition)[0]);


    let sub_u64 = &a - &b;
    let subtraction = a_expected - b_expected;

    let mul_u64 = &a * &b;
    let multiplication = a_expected  * b_expected;

    let pow = &a.pow(value_u64_b);
    let expected_pow = a_expected.exp_u64(value_u64_b);

    if value_u64_b != 0 && b.inv().is_ok() && b_expected.try_inverse().is_some() { 

        let div = &a / &b; 
        assert_eq!(&div * &b, a.clone());
        let expected_div = a_expected / b_expected;
    }

    // Axioms soundness

    let one = FieldElement::<Goldilocks64ExtensionField>::one();
    let zero = FieldElement::<Goldilocks64ExtensionField>::zero();

    assert_eq!(&a + &zero, a, "Neutral add element a failed");
    assert_eq!(&b + &zero, b, "Neutral mul element b failed");
    assert_eq!(&a * &one, a, "Neutral add element a failed");
    assert_eq!(&b * &one, b, "Neutral mul element b failed");

    assert_eq!(&a + &b, &b + &a, "Commutative add property failed");
    assert_eq!(&a * &b, &b * &a, "Commutative mul property failed");

    let c = &a * &b;
    assert_eq!((&a + &b) + &c, &a + (&b + &c), "Associative add property failed");
    assert_eq!((&a * &b) * &c, &a * (&b * &c), "Associative mul property failed");

    assert_eq!(&a * (&b + &c), &a * &b + &a * &c, "Distributive property failed");

    assert_eq!(&a - &a, zero, "Inverse add a failed");
    assert_eq!(&b - &b, zero, "Inverse add b failed");

    if a != zero {
        assert_eq!(&a * a.inv().unwrap(), one, "Inverse mul a failed");
    }
    if b != zero {
        assert_eq!(&b * b.inv().unwrap(), one, "Inverse mul b failed");
    }
});