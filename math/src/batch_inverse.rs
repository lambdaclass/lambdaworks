use crate::field::{element::FieldElement, traits::IsField};

// Source: https://en.wikipedia.org/wiki/Modular_multiplicative_inverse#Multiple_inverses
pub fn inplace_batch_inverse<F: IsField>(numbers: &mut [FieldElement<F>]) {
    if numbers.is_empty() {
        return;
    }
    let count = numbers.len();
    let mut prod_prefix = Vec::with_capacity(count);
    prod_prefix.push(numbers[0].clone());
    for i in 1..count {
        prod_prefix.push(&prod_prefix[i - 1] * &numbers[i]);
    }
    let mut bi_inv = prod_prefix[count - 1].inv();
    for i in (1..count).rev() {
        let ai_inv = &bi_inv * &prod_prefix[i - 1];
        bi_inv = &bi_inv * &numbers[i];
        numbers[i] = ai_inv;
    }
    numbers[0] = bi_inv;
}

#[cfg(test)]
mod tests {
    use proptest::{collection, prelude::*, prop_compose, proptest, strategy::Strategy};

    use crate::field::fields::u64_prime_field::U64PrimeField;

    // Some of these tests work when the finite field has order greater than 2.
    use super::*;
    const ORDER: u64 = 23;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FE {
            FE::from(num)
        }
    }

    prop_compose! {
        fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 0..1 << max_exp)) -> Vec<FE> {
            vec
        }
    }

    proptest! {
        #[test]
        fn test_inplace_batch_inverse_returns_inverses(vec in field_vec(10)) {
            let input: Vec<_> = vec.into_iter().filter(|x| x != &FE::zero()).collect();
            let mut inverses = input.clone();
            inplace_batch_inverse(&mut inverses);

            for (i, x) in inverses.into_iter().enumerate() {
                prop_assert_eq!(x * input[i], FE::one());
            }
        }
    }
}
