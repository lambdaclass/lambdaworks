use lambdaworks_math::field::{
    element::FieldElement,
    traits::IsField,
};
use std::collections::BTreeMap;
use std::cmp::Ord;

use super::errors::{Error, TableError};

#[derive(Debug)]
pub struct Table<F: IsField> {
    pub(crate) size: usize,
    pub(crate) values: Vec<FieldElement<F>>,
    pub(crate) value_index_mapping: BTreeMap<FieldElement<F>, usize>,
}

impl<F: IsField + Ord> Table<F> 
where 
    <F as IsField>::BaseType: Ord
{
    pub fn new(values: &Vec<FieldElement<F>>) -> Result<Self, Error<F>> {
        if !values.len().is_power_of_two() {
            return Err(Error::Table(TableError::TableSizeNotPow2(values.len())))
        }

        let mut value_index_mapping = BTreeMap::<FieldElement<F>, usize>::default();
        for (i, ti) in values.iter().enumerate() {
            let prev = value_index_mapping.insert(ti.clone(), i);
            if prev.is_some() {
                return Err(Error::Table(TableError::DuplicateValueInTable(ti.clone())))
            }
        }

        Ok(Self {
            size: values.len(),
            values: values.clone(),
            value_index_mapping,
        })
    }
}

#[cfg(test)]
pub mod table_tests {
    use super::*;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381FieldElement;
    use rand::random;

    use crate::lookups::{
        errors::Error,
        table::Table
    };

    type F = BLS12381FieldElement;
    #[test]
    fn test_correct_table() {
        let n = 32;

        let table_values: Vec<_> = (0..n).map(|_| F::from(random::<u64>())).collect();
        let _ = Table::new(&table_values);
    }

    #[test]
    fn test_not_pow_2() {
        let n = 31;

        let table_values: Vec<_> = (0..n).map(|_| F::from(random::<u64>())).collect();
        let res = Table::new(&table_values);

        assert_eq!(res.unwrap_err(), Error::Table(TableError::TableSizeNotPow2(table_values.len())));
    }

    #[test]
    fn test_dup_value() {
        let n = 32;

        let mut table_values: Vec<_> = (0..n).map(|_| F::from(random::<u64>())).collect();
        table_values[5] = table_values[10].clone();

        let res = Table::new(&table_values);

        assert_eq!(
            res.unwrap_err(),
            Error::Table(TableError::DuplicateValueInTable(table_values[5].clone()))
        );
    }
}
