use lambdaworks_math::field::{element::FieldElement, traits::IsField};

/// Fill a field element slice with 0s until a power of two size is reached, unless it already is.
pub fn zero_padding<F: IsField>(input: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
    let mut input = input.to_vec();
    input.resize(input.len().next_power_of_two(), FieldElement::zero());
    input
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::{
        element::FieldElement, test_fields::u64_test_field::U64TestField,
    };
    use proptest::{collection, prelude::*};

    use super::*;

    type F = U64TestField;
    type FE = FieldElement<F>;

    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FE {
            FE::from(num)
        }
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(vec in collection::vec(field_element(), 2..1<<max_exp).prop_filter("Avoid polynomials of size not power of two", |vec| vec.len().is_power_of_two())) -> Vec<FE> {
            vec
        }
    }
    prop_compose! {
        fn non_power_of_two_sized_field_vec(max_exp: u8)(vec in collection::vec(field_element(), 2..1<<max_exp).prop_filter("Avoid polynomials of size power of two", |vec| !vec.len().is_power_of_two())) -> Vec<FE> {
            vec
        }
    }

    proptest! {
        #[test]
        fn test_zero_padding_works(input in non_power_of_two_sized_field_vec(8)) {
            prop_assert!(zero_padding(&input).len().is_power_of_two());
        }

        #[test]
        fn test_zero_padding_does_nothing_with_2pow(input in field_vec(8)) {
            prop_assert_eq!(zero_padding(&input).len(), input.len());
        }
    }
}
