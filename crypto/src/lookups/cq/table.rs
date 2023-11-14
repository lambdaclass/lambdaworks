use lambdaworks_math::{
    elliptic_curve::traits::IsPairing,
    field::{
        element::{self, FieldElement},
        traits::{IsFFTField, IsField},
    },
    polynomial::Polynomial,
};
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct Table<F: IsField> {
    pub(crate) size: usize,
    pub(crate) values: Vec<FieldElement<F>>,
    pub(crate) value_index_mapping: BTreeMap<FieldElement<F>, usize>,
}

impl<F: IsField> Table<F> {
    pub fn new(values: &Vec<FieldElement<F>>) -> Self {
        if !values.len().is_power_of_two() {
            panic!("values length is not power of two")
        }

        //TODO: implement Ord for FieldElement<F>
        let mut value_index_mapping = BTreeMap::<FieldElement<F>, usize>::default();
        for (i, &ti) in values.iter().enumerate() {
            let prev = value_index_mapping.insert(ti, i);
            if prev.is_some() {
                panic!("duplicate value in table")
            }
        }

        Self {
            size: values.len(),
            values: values.clone(),
            value_index_mapping,
        }
    }
}

#[cfg(test)]
pub mod table_tests {
    use super::*;
    use lambdaworks_math::elliptic_curve::short_weierstrass::{
        curves::bls12_381::curve::BLS12381FieldElement, point::ShortWeierstrassProjectivePoint,
    };
    use rand::random;

    use crate::lookups::cq::table::Table;

    type F = BLS12381FieldElement;
    #[test]
    fn test_correct_table() {
        let n = 32;

        let table_values: Vec<_> = (0..n).map(|_| F::from(random())).collect();
        let _ = Table::new(&table_values);
    }

    #[test]
    fn test_not_pow_2() {
        let n = 31;

        let table_values: Vec<_> = (0..n).map(|_| F::from(random())).collect();
        let res = Table::new(&table_values);

        assert_eq!(res.unwrap_err(), Error::TableSizeNotPow2(n));
    }

    #[test]
    fn test_dup_value() {
        let n = 32;

        let mut table_values: Vec<_> = (0..n).map(|_| F::from(random())).collect();
        table_values[5] = table_values[10];

        let res = Table::new(&table_values);

        assert_eq!(
            res.unwrap_err(),
            Error::DuplicateValueInTable(format!("{}", table_values[5]))
        );
    }
}
