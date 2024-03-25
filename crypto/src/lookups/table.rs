use std::{hash::Hash, collections::HashMap};

use lambdaworks_math::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass,
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField},
    },
    unsigned_integer::element::UnsignedInteger,
};

use super::traits::IsLookUpTable;

/// Constructs a Generic lookup table over a bi-variate function
pub struct LookUpTable<F: IsField>(HashMap<(FieldElement<F>, FieldElement<F>), FieldElement<F>>);

impl<F: IsField> LookUpTable<F> 
where
    FieldElement<F>: Hash,
{
    pub fn with_fn(f: F, n: usize) -> Self
    where
        F: Fn(usize, usize) -> FieldElement<F>,
    {
        let mut table = LookUpTable(HashMap::new());

        for i in 0..n {
            //TODO: add support for u8 to field
            let f_i = FieldElement::from(i as u64);

            for k in 0..n {
                let k_i = FieldElement::from(k as u64);

                let result = f(i, k);
                table.0.insert((f_i.clone(), k_i), result);
            }
        }
        table
    }

    pub fn with_hashmap(map: HashMap<(FieldElement<F>, FieldElement<F>), FieldElement<F>>) -> Self {
        LookUpTable(map)
    }
}

impl<const NUM_LIMBS: usize, F: IsFFTField<BaseType = UnsignedInteger<NUM_LIMBS>>, C: IsShortWeierstrass> IsLookUpTable<NUM_LIMBS, F, C> for LookUpTable<F> 
where
    FieldElement<F>: Hash
{
    fn borrow_map(&self) -> &HashMap<(FieldElement<F>, FieldElement<F>), FieldElement<F>> {
        &self.0
    }
}