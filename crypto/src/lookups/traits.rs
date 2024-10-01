use std::{hash::Hash, collections::HashMap};

use lambdaworks_math::{
    elliptic_curve::short_weierstrass::{point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass}, field::{
        element::FieldElement,
        traits::{IsFFTField, IsField},
    }, msm::pippenger::msm, polynomial::Polynomial, unsigned_integer::element::UnsignedInteger
};

use super::multiset::MultiSet;

pub struct PreProcessedTable<F: IsField, C: IsShortWeierstrass> {
    pub n: usize,
    pub t_1: (
        MultiSet<F>,
        ShortWeierstrassProjectivePoint<C>,
        Polynomial<FieldElement<F>>,
    ),
    pub t_2: (
        MultiSet<F>,
        ShortWeierstrassProjectivePoint<C>,
        Polynomial<FieldElement<F>>,
    ),
    pub t_3: (
        MultiSet<F>,
        ShortWeierstrassProjectivePoint<C>,
        Polynomial<FieldElement<F>>,
    ),
}


pub trait IsLookUpTable<const NUM_LIMBS: usize, F: IsFFTField<BaseType = UnsignedInteger<NUM_LIMBS>>, C: IsShortWeierstrass> 
where
 (FieldElement<F>, FieldElement<F>): Hash
{

    // Returns the number of entries in the lookup table
    fn len(&self) -> usize {
        self.borrow_map().keys().len()
    }

    // Represent lookup table as map returns immutable copy of the map
    fn borrow_map(&self) -> &HashMap<(FieldElement<F>, FieldElement<F>), FieldElement<F>>;

    // Fetches the appropriate lookup table value given input
    fn read(&self, key: &(FieldElement<F>, FieldElement<F>)) -> Option<&FieldElement<F>> {
        self.borrow_map().get(key)
    }

    // Given a lookup table where each row contains three entries (a, b, c)
    // Create three multisets
    // a = {a_0, a_1, a_2, a_3,...,a_n}
    // b = {b_0, b_1, b_2, b_3,...,b_n}
    // c = {c_0, c_1, c_2, c_3,...,c_n}
    fn multiset(
        &self,
    ) -> (
        MultiSet<F>,
        MultiSet<F>,
        MultiSet<F>,
    ) {
        //TODO: Set capacity if possible to reduce allocations???
        let mut table_multiset_left = MultiSet(Vec::new());
        let mut table_multiset_right = MultiSet(Vec::new());
        let mut table_multiset_out = MultiSet(Vec::new());

        for (key, value) in self.borrow_map().iter() {
            let input_0 = key.0.clone();
            let input_1 = key.1.clone();
            let output = value.clone();

            table_multiset_left.push(input_0);
            table_multiset_right.push(input_1);
            table_multiset_out.push(output);
        }

        (
            table_multiset_left,
            table_multiset_right,
            table_multiset_out,
        )
    }

    fn preprocess(
        &self,
        commit_key: Vec<ShortWeierstrassProjectivePoint<C>>,
        n: usize,
    ) -> PreProcessedTable<F, C> {
        assert!(n.is_power_of_two());

        let (mut t_1, mut t_2, mut t_3) = self.multiset();

        let k = t_1.len();
        assert_eq!(t_1.len(), k);
        assert_eq!(t_2.len(), k);
        assert_eq!(t_3.len(), k);

        // Pad
        let pad_by = n - t_1.len();
        t_1.extend(pad_by, t_1.last());
        t_2.extend(pad_by, t_2.last());
        t_3.extend(pad_by, t_3.last());

        let t_1_poly = Polynomial::interpolate_fft::<F>(&t_1.0).unwrap();
        let t_2_poly = Polynomial::interpolate_fft::<F>(&t_2.0).unwrap();
        let t_3_poly = Polynomial::interpolate_fft::<F>(&t_3.0).unwrap();

        let t_1_commit = msm(&t_1_poly.coefficients, &commit_key).unwrap();
        let t_2_commit = msm(&t_2_poly.coefficients, &commit_key).unwrap();
        let t_3_commit = msm(&t_3_poly.coefficients, &commit_key).unwrap();

        PreProcessedTable {
            n,
            t_1: (t_1, t_1_commit, t_1_poly),
            t_2: (t_2, t_2_commit, t_2_poly),
            t_3: (t_3, t_3_commit, t_3_poly),
        }
    }
}
