
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::{point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass},
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField},
    },
    unsigned_integer::element::UnsignedInteger,
};
use core::marker::PhantomData;
use std::hash::Hash;

use crate::fiat_shamir::is_transcript::IsTranscript;

use super::traits::{IsLookUpTable, PreProcessedTable};

// Evaluations store the evaluations of different polynomial.
// `t` denotes that the polynomial was evaluated at t(z) for some random evaluation challenge `z`
// `t_omega` denotes the polynomial was evaluated at t(z * omega) where omega is the group generator
// In the FFT context, the normal terminology is that t(z*omega) means to evaluate a polynomial at the next root of unity from `z`.
pub struct LookUpProof<F: IsField, C: IsShortWeierstrass> {
    pub aggregate_witness_comm: ShortWeierstrassProjectivePoint<C>,
    pub shifted_aggregate_witness_comm: ShortWeierstrassProjectivePoint<C>,
    pub f_comm: ShortWeierstrassProjectivePoint<C>,
    pub q_comm: ShortWeierstrassProjectivePoint<C>,
    pub h_1_comm: ShortWeierstrassProjectivePoint<C>,
    pub h_2_comm: ShortWeierstrassProjectivePoint<C>,
    pub z_comm: ShortWeierstrassProjectivePoint<C>,
    pub f: FieldElement<F>,
    pub t: FieldElement<F>,
    pub t_omega: FieldElement<F>,
    pub h_1: FieldElement<F>,
    pub h_1_omega: FieldElement<F>,
    pub h_2: FieldElement<F>,
    pub h_2_omega: FieldElement<F>,
    pub z: FieldElement<F>,
    pub z_omega: FieldElement<F>,
}

pub struct LookUp<const NUM_LIMBS: usize, C: IsShortWeierstrass, F: IsFFTField<BaseType = UnsignedInteger<NUM_LIMBS>>, T: IsLookUpTable<NUM_LIMBS, F, C>>
where
    FieldElement<F>: Hash,
{
    table: T,
    left_wires: Vec<FieldElement<F>>,
    right_wires: Vec<FieldElement<F>>,
    output_wires: Vec<FieldElement<F>>,
    _phantom: PhantomData<C>
}

impl<const NUM_LIMBS: usize, C: IsShortWeierstrass, F: IsFFTField<BaseType = UnsignedInteger<NUM_LIMBS>>, T: IsLookUpTable<NUM_LIMBS, F, C>> LookUp<NUM_LIMBS, C, F, T> 
where
    FieldElement<F>: Hash
{
    pub fn new(table: T) -> LookUp<NUM_LIMBS, C, F, T> {
        LookUp {
            table,
            left_wires: Vec::new(),
            right_wires: Vec::new(),
            output_wires: Vec::new(),
            _phantom: PhantomData,
        }
    }

    pub fn read(&mut self, key: &(FieldElement<F>, FieldElement<F>)) -> bool {
        let option_output = self.table.read(key);
        if option_output.is_none() {
            return false;
        }

        let output = option_output.unwrap().clone();

        // Add (input, output) combination into the corresponding multisets
        self.left_wires.push(key.0.clone());
        self.right_wires.push(key.1.clone());
        self.output_wires.push(output);

        return true
    }

    pub fn prove(
        &mut self,
        proving_key: Vec<ShortWeierstrassProjectivePoint<C>>,
        preprocessed_table: &PreProcessedTable<F, C>,
        transcript: &impl IsTranscript<F>,
    ) -> LookUpProof<F, C> {
        /*
        
        // Gen alpha
        let alpha = transcript.sample_field_element();


        //Aggregate table and witness values into a one multiset and pad the witness to be correct size
        let merged_table = MultiSet::aggregate(
            vec![

            ],
            alpha
        );

        // Aggregate witness values into one multiset
        let mut merged_witness = MultiSet::aggregate(vec![f_1, f_2, f_3], alpha);

        // Pad merged Witness to be one less than 'n' -> TODO: change to error
        assert!(merged_witness.len() < preprocessed_table.n);
        merged_witness.extend(pad_by, merged_witness.last());

        // Perform a Multi-set equality proof

        LookUpProof {
            aggregate_witness_comm,
            shifted_aggregate_witness_comm,
            f_comm,
            q_comm,
            h_1_comm,
            h_2_comm,
            z_comm,
            f,
            t,
            t_omega,
            h_1,
            h_1_omega,
            h_2,
            h_2_omega,
            z,
            z_omega,
        }
        */
    }


}